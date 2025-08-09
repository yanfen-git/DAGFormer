rm(list = ls())
gc()
if (interactive()) {
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
}
# 
# if (!requireNamespace("devtools", quietly = TRUE))
#   install.packages("devtools")
# devtools::install_github("sajuukLyu/ggunchull", type = "source")
library(Seurat)
library(ggplot2)
library(ggunchull)
library(SeuratObject)
library(paletteer)

Drug = 'Etoposide'
# Dataset = 'Target'
Dataset = 'Source'
# matrix_path = paste0("D:/Workspace/GT_ASN/preprocessNormData/", Drug, "/", "Target_expr_resp_z.", Drug, ".tsv")
matrix_path = paste0("D:/Workspace/GT_ASN/preprocessNormData/", Drug, "/", "Source_exprS_resp_z.", Drug, ".tsv")
mt = read.table(matrix_path, sep = "\t", header = T, row.names = 1)


output_dir_1 = "./scatterplot-data/"
if (!dir.exists(output_dir_1)) {
  dir.create(output_dir_1, recursive = TRUE)
}
output_dir_2 = "./scatterplot-pic/"
if (!dir.exists(output_dir_2)) {
  dir.create(output_dir_2, recursive = TRUE)
}

save_file_path = paste0("./raw-data/raw-", Dataset,'-', Drug, ".Rdata")
save(mt, file = save_file_path)



load(save_file_path)

label = mt['response']

label$response <- factor(label$response, levels = c(0, 1), labels = c("resistant", "sensitive"))

mt$response <- NULL
mt = t(mt)




seurat_object = CreateSeuratObject(counts = mt, assay = "RNA")

seurat_object[["response"]] <- label$response

# seurat_object <- NormalizeData(seurat_object)

# seurat_object <- FindVariableFeatures(seurat_object, selection.method = "vst", nfeatures = 2000)

# seurat_object <- ScaleData(seurat_object)
# seurat_object <- RunPCA(seurat_object, features = VariableFeatures(object = seurat_object))


seurat_object <- NormalizeData(seurat_object)



seurat_object@assays$RNA@data[is.na(seurat_object@assays$RNA@data)] <- 0


all_features <- rownames(seurat_object)

seurat_object@assays$RNA@scale.data <- as.matrix(seurat_object@assays$RNA@data)
seurat_object <- RunPCA(seurat_object, assay = "RNA", slot = "data", features = all_features)



seurat_object = RunUMAP(seurat_object, dims = 1:20)

plotData = as.data.frame(seurat_object[["umap"]]@cell.embeddings)
plotData$response <- seurat_object$response

custom_colors <- c("resistant" = "#7DA6C6", "sensitive" = "#E68B81")
# custom_colors_alpha <- setNames(paste0(custom_colors, "50"), names(custom_colors))

p = ggplot(plotData, aes(x = UMAP_1, y = UMAP_2, fill = response, color = response))+
    # geom_point(aes(color = response), shape = 21, size = 1, stroke = 0.5)+
    geom_point(size = 1)+
    guides(color = guide_legend(override.aes = list(size = 3)))+ 
    labs(x = "UMAP1", y = "UMAP2")+
    ggtitle(paste(Drug),"Raw Data")+
    theme(
      axis.text.y = element_blank(),   
      axis.ticks.y = element_blank(),   
      axis.text.x = element_blank(),   
      axis.ticks.x = element_blank(),
      aspect.ratio = 1,
      panel.border = element_rect(fill=NA,color="black", linewidth=1, linetype="solid"),
      panel.background = element_blank(),
      panel.grid = element_blank(),
      legend.title = element_blank(), 
      legend.key=element_rect(fill='white', color = NA), 
      legend.text = element_text(size=10), 
      legend.key.size=unit(0.5,'cm')
    )+
    # scale_fill_manual(values=custom_colors_alpha)+
    scale_color_manual(values = custom_colors)

save_filename <- paste0('./scatterplot-pic/RAW-', Dataset,'-', Drug, '.png') 
ggsave(filename = save_filename,
       plot = p,
       device = "png",
       width = 6,
       height = 6,
       dpi = 300,
       units = "in"
)



