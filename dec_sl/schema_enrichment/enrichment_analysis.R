require(gridExtra)
require(ggplot2)
require(dplyr)
require(reshape2)

enrichments <- read.csv('selected-enriched.txt', sep=" ")
names(enrichments) <- c("selected", "enriched")

# remove rows with 0 selected
enrichments <- enrichments %>%  filter(selected > 0)

enrichments$increase <- enrichments$enriched - enrichments$selected

increase.summary <- enrichments %>%
                      group_by(selected) %>%
                      summarise(min=min(increase, na.rm=TRUE), max=max(increase, na.rm=TRUE), average=mean(increase, na.rm=TRUE)) 

increase.summary.m <- melt(increase.summary, id.vars = "selected")

plot1 <- ggplot(data=enrichments, mapping=aes(x=selected, y=enriched)) + geom_point(color="dodgerblue") +  
            geom_abline(color='red')  +
            xlab("Number of Selected Columns") +
            ylab("Number of\nEnriched Columns")
plot2 <- ggplot(increase.summary.m , aes(x=selected, y=value, color=variable)) + geom_point()
plot2 <- ggplot(data=increase.summary, mapping=aes(x=selected, y=average)) + 
            geom_errorbar(mapping=aes(ymin=min, ymax=max), width=0.5, color="chartreuse3") + 
            geom_point(size=2, color='salmon') +
            xlab("Number of Selected Columns") +
            ylab("Average Number of Added\nColumns with Min/Max")
plot3 <- ggplot(data=enrichments, mapping=aes(x=selected)) + 
            geom_histogram(binwidth=1, position="identity",color='black', fill="salmon")  +
            xlab("Number of Selected Columns") +
            ylab("Count")

grid.arrange(plot1, plot2, plot3, ncol=1)
