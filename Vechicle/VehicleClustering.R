# Import libraries
library(readxl)
library(dplyr)
library(NbClust)
library(stats)
library(tidyverse)
library(ggplot2)
library(cluster)
library(factoextra)
library(dataset)
library(fpc)


getwd()
setwd("../Desktop/ML cwk")

vechicleset <- read_excel("Cwk/vehicles.xlsx")

##PREPROCCESING##
# Remove the final coulum
vechicles <- vechicleset[, -c(1, ncol(vechicleset))]

# data Type
class(vechicleset)

# Check for missing values
vechicles <- vechicles[complete.cases(vechicles), ]

# If there are missing values, remove them using the na.omit() function
vechicles <- na.omit(vechicles)

summary(vechicles)
# create a boxplot to visualize distribution in order to find outliers
boxplot(vechicles)

# Scale the data using the standardization method
vechicles <- scale(vechicles)

vechicle_outliers <- apply(vechicles,1,function(x) any(x > 3 | x < -3))
cleaned_vechicles <- subset(vechicles, !vechicle_outliers)

##Finding the Optimal Number of Clusterrs##
#NBClust Method
set.seed(26)
clust_no_nb = NbClust(cleaned_vechicles,distance="euclidean", min.nc=2,max.nc=10,method="kmeans",index="all")

#Elbow Method
set.seed(28)
k_val <- 2:10
WSS <- sapply(k_val,function(k_val){kmeans(cleaned_vechicles,centers = k_val)$tot.withinss})
plot(k_val, WSS, type = "b", xlab = "Number of K values", ylab = "WSS")


#Silouette Method
set.seed(32)
fviz_nbclust(cleaned_vechicles, kmeans, method = "silhouette")

#Gap-Stat Method
set.seed(34)
fviz_nbclust(cleaned_vechicles,kmeans,method = "gap_stat")

##K MEANS ##
k = 3
kmeans_vechicle <- kmeans(cleaned_vechicles,centers = k,nstart = 10)
kmeans_vechicle

#Calculating Centers
kmeans_vechicle$centers

fviz_cluster(kmeans_vechicle,data=cleaned_vechicles)

vechicle_cluster <- data.frame(cleaned_vechicles,cluster = as.factor(kmeans_vechicle$cluster))
head(vechicle_cluster)

vechicle_wss = kmeans_vechicle$tot.withinss
vechicle_wss

vechicle_bss = kmeans_vechicle$betweenss
vechicle_bss

# Calculating the ratio between WSS and BSS
vechicle_wss/vechicle_bss

# Calculate the total sum of squares (TSS)
TSS <- sum(apply(cleaned_vechicles, 2, var)) * nrow(cleaned_vechicles)

#Calculating the ratio of between_cluster_sums_of_squares (BSS) over total_sum_of_Squares (TSS)
TSS/vechicle_bss

#silhouette plot
vehicle_sil = silhouette(kmeans_vechicle$cluster,dist(cleaned_vechicles))
fviz_silhouette(vehicle_sil)

#AVG silhouette width score
silhouette_avg = mean(vehicle_sil[,3])

#//.................................................................................................//#

###PCA###
v_pca = prcomp(cleaned_vechicles)
summary(v_pca)

#EigenValues and Eigenvectors
v_eigenvalues <- v_pca$sdev^2
v_eigenvectors <- v_pca$rotation

v_eigenvalues
v_eigenvectors

#cumulative score per principal components
fviz_eig(v_pca, addlabels = TRUE, ylim = c(0, 60))

#Create Tranformed Dataset(PCA as attributes)#
vechicle_transformed <- predict(v_pca,cleaned_vechicles)
summary(vechicle_transformed)

#cumulative score per principal components
PVE <- v_pca$sdev^2/sum(v_pca$sdev^2)
PVE <- round(PVE,2)
cum_score = cumsum(PVE)
pc_num = sum(cum_score<0.92) + 1
pc_num

vechicle_pc_dataset = data.frame(vechicle_transformed[, 1:pc_num])

##Finding the Optimal Number of Clusterrs##
#NBClust Method
set.seed(26)
clust_no_pc = NbClust(vechicle_pc_dataset,distance="euclidean", min.nc=2,max.nc=6,method="kmeans",index="all")

#Elbow Method
set.seed(28)
k_val_pc <- 2:6
WSS_pc <- sapply(k_val,function(k_val){kmeans(vechicle_pc_dataset,centers = k_val)$tot.withinss})
plot(k_val_pc, WSS_pc, type = "b", xlab = "Number of K values", ylab = "WSS")



#Silouette Method
set.seed(32)
fviz_nbclust(vechicle_pc_dataset, kmeans, method = "silhouette")

#Gap-Stat Method
set.seed(34)
fviz_nbclust(vechicle_pc_dataset,kmeans,method = "gap_stat")


##K MEANS PC##
k_pc = 3
kmeans_pc <- kmeans(vechicle_pc_dataset,centers = k_pc,nstart = 10)
kmeans_pc

#Calculating Centers
kmeans_pc$centers

fviz_cluster(kmeans_pc,data=vechicle_pc_dataset)

vechicle_cluster_pc <- data.frame(vechicle_pc_dataset,cluster = as.factor(kmeans_pc$cluster))
head(vechicle_cluster_pc)

vechicle_wss_pc = kmeans_pc$tot.withinss
vechicle_bss_pc = kmeans_pc$betweenss

# Calculating the ratio between WSS and BSS
vechicle_wss_pc/vechicle_bss_pc

# Calculate the total sum of squares (TSS)
TSS_pc <- sum(apply(vechicle_pc_dataset, 2, var)) * nrow(vechicle_pc_dataset)

#Calculating the ratio of between_cluster_sums_of_squares (BSS) over total_sum_of_Squares (TSS)
TSS_pc/vechicle_bss_pc

#silhouette plot
vehicle_sil_pc = silhouette(kmeans_pc$cluster,dist(vechicle_pc_dataset))
fviz_silhouette(vehicle_sil_pc)

#AVG silhouette width score
silhouette_avg_pc = mean(vehicle_sil_pc[,2])

##Calinski-Harabasz Index##

set.seed(40)
# Compute the Calinski-Harabasz Index 
ch_index <- round(calinhara(vechicle_pc_dataset,kmeans_pc$cluster),digits=3)
ch_index

set.seed(42)
# Set the number of clusters to evaluate
k <- 2:10

# Initialize empty vector to store Calinski-Harabasz index values
ch_scores <- vector("numeric", length(k))

# Compute the Calinski-Harabasz index for each number of clusters
for (i in 1:length(k)) {
  km <- kmeans(vechicle_pc_dataset, centers = k[i], nstart = 10)
  ch_scores[i] <- calinhara(vechicle_pc_dataset, km$cluster)
}

## Visualize the Calinski-Harabasz Index##
# Plot the Calinski-Harabasz index values
plot(k, ch_scores, type = "b", xlab = "Number of Clusters", ylab = "Calinski-Harabasz Index", main = "Calinski-Harabasz Index against Number of Clusters")







