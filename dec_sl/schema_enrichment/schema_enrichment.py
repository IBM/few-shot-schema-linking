import json
import logging
import sys

import numpy as np
from sentence_transformers import SentenceTransformer
import time
from typing import Any, Dict, List
import random
from sklearn.cluster import DBSCAN
from collections import Counter


class SchemaEnrichment:

    def __init__(self, fname: str):
        # Setup logging
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - p %(process)s - %(filename)s:%(lineno)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.ERROR)

        # load the model once and for all
        #embeddings = 'dunzhang/stella_en_400M_v5'
        embeddings = 'all-MiniLM-L12-v2'
        self.model = SentenceTransformer(embeddings, trust_remote_code=True).cuda()

        self.db_representations = {}
        self.oracle_values = []            # for testing/debugging

        self.__read_data(fname)
        self.__compute_embeddings()

    def __dict_stats(self, d):
        min_dist = float("inf")
        max_dist = 0
        avg_dist = 0
        for k in d.keys():
            avg_dist += d[k]
            if d[k] < min_dist:
                min_dist = d[k]
            if d[k] > max_dist:
                max_dist = d[k]
        if len(d) > 0:
            avg_dist /= float(len(d))
        return min_dist, max_dist, avg_dist

    def __find_min_distance(self, candidates, possible_candidates):
        distance = float("inf")
        candidate = None
        for k in candidates.keys():
            if k in possible_candidates:
                if candidates[k] < distance:
                    distance = candidates[k]
                    candidate = k
        return candidate, distance

    def __find_max_distance(self, candidates):
        distance = 0.0
        candidate = None
        for k in candidates:
            if k[1] > distance:
                distance = k[1]
                candidate = k[0]
        return candidate, distance

    def __read_data(self, fname: str):
        with open(fname, 'r') as f:
            d = json.load(f)
        for entry in d:
            if entry['db_id'] not in self.db_representations.keys():
                schema_elements = []
                for table in entry['schema']['schema_items']:
                    schema_elements.append(table)
                self.db_representations[entry['db_id']] = {}
                self.db_representations[entry['db_id']]['schema_elements'] = schema_elements
            oracle_record = {}
            oracle_record['db_id'] = entry['db_id']
            oracle_record['question'] = entry['question']
            oracle_record['selected'] = list(entry['matched_contents'].keys())
            self.oracle_values.append(oracle_record)
        self.logger.info(f"Read {len(self.db_representations)} databases")
        return None


    def __compute_embeddings(self):
        for db in self.db_representations:
            self.logger.debug(f"Processing db {db}")
            self.db_representations[db]['tables'] = {}
            for table in self.db_representations[db]['schema_elements']:
                table_entry = {}
                col_embeddings = {}
                # create the generic string
                start_time = time.time()
                table_str = f"{table['table_name']}: "
                for idx, name in enumerate(table['column_names']):
                    name_reformat = name.replace('_', ' ')
                    table_str += f"(Name:{name_reformat}"
                    if table['column_comments'][idx]:
                        table_str += f",Description:{table['column_comments'][idx]}),"
                    else:
                        table_str += "),"
                table_str = table_str[:-1]
                # prepend each colum and compute embeddings
                for idx, name in enumerate(table['column_names']):
                    name_reformat = name.replace('_', ' ')
                    col_str = f"{name_reformat} "
                    col_str += f"Type:{table['column_types'][idx]},"
                    if table['column_comments'][idx]:
                        col_str += f"Description:{table['column_comments'][idx]},"
                    col_str += f"Samples:{table['column_contents'][idx]} in the context of table "
                    col_str = col_str + f" {table_str}"
                    # compute embeddings
                    embeddings = self.model.encode(col_str)
                    col_embeddings[name] = embeddings
                table_entry['column_embeddings'] = col_embeddings
                self.logger.debug(f"computed {len(col_embeddings)} embeddings in {time.time() - start_time} secs")

                # compute all distances (brute force will optimize later)
                distances = {}
                start_time = time.time()
                for source in col_embeddings.keys():
                    source_distances = {}
                    for target in col_embeddings.keys():
                        if target != source:
                            source_distances[target] = float(np.linalg.norm(col_embeddings[source] - col_embeddings[target]))
                            if source_distances[target]  == 'nan':
                                self.logger.debug('!' * 50)
                                self.logger.debug(f"Nan!")
                    distances[source] = source_distances
                self.logger.debug(f"computed all distances {len(col_embeddings)} columns in {time.time() - start_time} secs")
                table_entry['column_distances'] = distances
                self.logger.debug(f"\tAdding table {table['table_name']}")
                self.db_representations[db]['tables'][table['table_name']] = table_entry
        return None

    # increase the representativeness of the columns by random sampling up to representation_percent value
    def enrich_representativeness(self, distances, already_selected, representation_percent) -> List[str]:
        result = []
        result += already_selected
        total_columns_num = len(distances)  # the number of the all columns per table
        already_selected_num = len(already_selected)  # the number of already selected columns
        if 1.0*already_selected_num/total_columns_num >= representation_percent:
            return result
        num_to_select = (int)(representation_percent*total_columns_num - already_selected_num)
        outside_columns = list(set(distances.keys()) - set(already_selected))
        result += random.sample(outside_columns, num_to_select)
        return result

    # increase representativeness by adding a point for each cluster of column names given their embeddings
    def enrich_representativenes_dbscan(self, columns, embeddings, already_selected, include_outliers=True, eps=0.5, min_samples=2) -> List[str]:
        result = []
        result += already_selected
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)
        cluster_labels = clustering.labels_
        print(f'Columns = {columns}')
        print(f'Cluster labels = {cluster_labels}')
        cluster_counts = Counter(cluster_labels)
        del cluster_counts[-1]
        num_clusters = len(cluster_counts) # the number of clusters detected by dbscan
        # output should contain the core indices from each class and all the outliers
        print(f'DBscan detected {num_clusters} clusters')
        covered_clusters = set()
        for col in already_selected:
            col_ind = columns.index(col)
            label = cluster_labels[col_ind]
            if label != -1: # if it is not an outlier
                covered_clusters.add(int(label))
        print(f'covered_clusters = {list(covered_clusters)}')
        if include_outliers:
            result += [columns[i] for i in range(len(columns)) if cluster_labels[i] == -1]
        for i,el in enumerate(cluster_labels):
            if el not in covered_clusters and columns[i] not in result:
                result.append(columns[i])
                covered_clusters.add(el)
        return result



    def enrich_schema(self, db_id: str, selected_columns: List[str], representation_percent=0.0, is_dbscan=False, plus_outliers=True) -> List[str]:
        enriched_columns = []
        enriched_columns += selected_columns
        table_dict = {}
        for entry in selected_columns:
            table, column = entry.split('.')
            if table in table_dict.keys():
                table_dict[table].append(column)
            else:
                table_dict[table] = [column]
        for table in table_dict.keys():
            distances = self.db_representations[db_id]['tables'][table]['column_distances']
            # increase the representativeness of the columns by random sampling up to representation_percent value
            if representation_percent > 0.0:
                #print("*" * 80)
                #print(f"Before ---> {table_dict[table]}")
                table_dict[table] = self.enrich_representativeness(distances, table_dict[table], representation_percent)
                #print(f"After ---> {table_dict[table]}")
            if is_dbscan:
                embeddings = self.db_representations[db_id]['tables'][table]['column_embeddings']
                cols, embs = zip(*embeddings.items())
                print("*" * 80)
                print(f"Before ---> {table_dict[table]}")
                table_dict[table] = self.enrich_representativenes_dbscan(cols, embs, table_dict[table], include_outliers=plus_outliers)
                print(f"After ---> {table_dict[table]}")

            # the following increases the diversity of the selected columns via the max in-out distances algorithm
            reiterate = True
            while reiterate :
                outside_columns = list(set(distances.keys()) - set(table_dict[table]))
                inside_columns = table_dict[table]
                inout_min_distances = []
                inside_min_distances = []

                self.logger.debug(f"Table: {table} -- columns: {distances.keys()}")
                self.logger.debug(f"\tLinked columns: {table_dict[table]}")
                self.logger.debug(f"\tOutside columns: {outside_columns}")

                self.logger.debug(f"****************************   Outside columns ")
                for column in outside_columns:
                    self.logger.debug(f"columns: {column}")
                    _, d = self.__find_min_distance(distances[column], inside_columns)
                    inout_min_distances.append((column, d))
                column_to_add, inout_max_distance = self.__find_max_distance(inout_min_distances)
                self.logger.debug('-' * 50)
                self.logger.debug(f"column to add {column_to_add} dist {inout_max_distance}")
                self.logger.debug('-' * 50)

                self.logger.debug(f"****************************   Inside columns ")
                for column in inside_columns:
                    _, d = self.__find_min_distance(distances[column], inside_columns)
                    inside_min_distances.append((column, d))
                # find the max among the min distances
                inside_max_column, inside_max_distance = self.__find_max_distance(inside_min_distances)
                self.logger.debug('-' * 50)
                self.logger.debug(f"column to add {inside_max_column} dist {inside_max_distance}")
                self.logger.debug('-' * 50)

                if column_to_add and inside_max_column and inout_max_distance > inside_max_distance:
                    self.logger.info(f"Adding column {column_to_add} to table {table}")
                    enriched_columns.append('.'.join((table, column_to_add)))
                    self.logger.debug(f"Add{column_to_add} to list: {table_dict[table]}")
                    table_dict[table].append(column_to_add)
                else:
                    reiterate = False


        self.logger.info(f"Original ({len(selected_columns)}): {selected_columns} -- Enriched ({len(enriched_columns)}): {enriched_columns}")
        return enriched_columns


    def test_output_distances_distribution(self, fname: str):
        with open(fname, 'w') as f:
            random.seed = 12345
            # pick 30 databases
            db_list = list(self.db_representations.keys())
            dbs = random.sample(db_list, min(30, len(db_list)))
            for db in dbs:
                # pick a table
                table = random.sample(list(self.db_representations[db]['tables'].keys()), 1)[0]
                distances = self.db_representations[db]['tables'][table]['column_distances']
                # pick a column
                column = random.sample(list(distances.keys()), 1)[0]
                min_dist, max_dist, avg_dist = self.__dict_stats(distances[column])
                f.write(f"{min_dist} {max_dist} {avg_dist}\n")

    def test_with_oracle(self):
        for entry in self.oracle_values:
            entry['enriched'] = self.enrich_schema(entry['db_id'], entry['selected'], representation_percent=0.0, is_dbscan=True, plus_outliers=False)
            print(len(entry['selected']), len(entry['enriched']))
        self.test_output_distances_distribution("distances-stats.txt")

if __name__ == '__main__':

    #enricher = SchemaEnrichment('bird_with_evidence_dev_text2sql.json')
    enricher = SchemaEnrichment('bird_with_evidence_train_text2sql.json')

    enricher.test_with_oracle()