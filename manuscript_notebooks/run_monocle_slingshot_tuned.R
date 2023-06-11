library(monocle3, quietly = TRUE)
library(slingshot, quietly = TRUE)
library(mclust, quietly = TRUE)

args = commandArgs(trailingOnly=TRUE)
data_path = args[1]
slingshot_ncenters = as.integer(args[2])
mcle_ncenters = as.integer(args[3])
mcle_sigma = as.numeric(args[4])
mcle_gamma = as.numeric(args[5])
mcle_eps = as.numeric(args[6])
res_path = args[7]

### Run slingshot

X <- t(read.table(data_path,sep=','))

sce <- SingleCellExperiment(assays = List(counts = t(X)))
reducedDims(sce) <- SimpleList(PCA = X)

#GMM
cl1 <- Mclust(X)$classification
colData(sce)$GMM <- cl1
#K-means
cl2 <- kmeans(X, centers = slingshot_ncenters)$cluster
colData(sce)$kmeans <- cl2
sce <- slingshot(sce, clusterLabels = 'kmeans', reducedDim = 'PCA',omega = TRUE)
saveRDS(sce@colData$slingshot@metadata$curves,paste(res_path,'slingshot_curves_kmeans',slingshot_ncenters,'.rds',sep='_'))
saveRDS(colData(sce)$kmeans,paste(res_path,'slingshot_kmeans',slingshot_ncenters,'.rds',sep='_'))
sce <- slingshot(sce, clusterLabels = 'GMM', reducedDim = 'PCA',omega = TRUE)
saveRDS(sce@colData$slingshot@metadata$curves,paste(res_path,'slingshot_curves_gmm.rds',sep='_'))
saveRDS(colData(sce)$GMM,paste(res_path,'slingshot_gmm.rds',sep='_'))

### Run monocle3

X <- t(X)
cds <- new_cell_data_set(as.matrix(X))
#cds <- preprocess_cds(cds,norm_method = "none", scaling = FALSE)
#cds <- reduce_dimension(cds)
reducedDim(cds,"UMAP") <- as.matrix(t(X))
#cds <- cluster_cells(cds,reduction_method='PCA')
cds <- cluster_cells(cds)
cds <- learn_graph(cds,learn_graph_control=list(ncenter = mcle_ncenters,L1.sigma = mcle_sigma,L1.gamma = mcle_gamma, eps = mcle_eps))
saveRDS(cds@principal_graph_aux@listData$UMAP$stree,paste(res_path,'monocle_stree',mcle_ncenters,mcle_sigma,mcle_gamma,'.rds',sep='_'))
saveRDS(cds@principal_graph_aux@listData$UMAP$Q,paste(res_path,'monocle_Q',mcle_ncenters,mcle_sigma,mcle_gamma,'.rds',sep='_'))
Matrix::writeMM(cds@principal_graph_aux@listData$UMAP$R,paste(res_path,'monocle_R',mcle_ncenters,mcle_sigma,mcle_gamma,'.mm',sep='_'))
saveRDS(cds@principal_graph_aux@listData$UMAP$dp_mst,paste(res_path,'monocle_dp_mst',mcle_ncenters,mcle_sigma,mcle_gamma,'.rds',sep='_'))
Matrix::writeMM(cds@principal_graph_aux@listData$UMAP$stree,paste(res_path,'monocle_stree',mcle_ncenters,mcle_sigma,mcle_gamma,'.mm',sep='_'))
saveRDS(cds@clusters$UMAP,paste(res_path,'monocle_clus',mcle_ncenters,mcle_sigma,mcle_gamma,'.rds',sep='_'))