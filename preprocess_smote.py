import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import scipy.sparse as sp
from collections import Counter
import dgl
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


##  smote采样和下采样相结合 ##
# 定义SMOTE和下采样策略
over = SMOTE(sampling_strategy=0.75)  # 少数类上采样到多数类的50%
under = RandomUnderSampler(sampling_strategy=0.75)  # 多数类下采样到少数类的50%
steps = [('over', over), ('under', under)]
pipeline = Pipeline(steps=steps)

def sample_data(data, sample_rate, random_state=None):
    """随机采样数据集中的样本。
    Args:
        data (DataFrame): 原始数据集。
        sample_rate (float): 采样比例，例如 0.2 表示 20%。
        random_state (int, optional): 随机数种子，用于结果复现。
    Returns:
        DataFrame: 采样后的数据。
    """
    # 计算需要采样的样本数量
    num_samples = int(len(data) * sample_rate)
    # 随机选择样本
    sampled_data = data.sample(n=num_samples, random_state=random_state)
    return sampled_data

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)



def load_data_drug(dataset,DRUG):
    k = 15
    # print(k)
    # DRUG = "Gefitinib"
    data = None

    if dataset == 'source':
        path_s = './split_norm/' + DRUG + '/' + 'Source_exprs_resp_z.' + DRUG + '.tsv'
        source_data = pd.read_csv(path_s, sep='\t', index_col=0)

        sample_rate = 1
        source_data = sample_data(source_data, sample_rate,random_state=42) # 随机采样
        # print(f'当前的采样率为: {sample_rate * 100}%')

        x_expression = source_data.iloc[:, 2:]  # gene expressions (features)
        y_logIC50 = source_data.iloc[:, 1]  # col index 1 of the source df is logIC50
        y_response = source_data.iloc[:, 0]
        threshold = source_data['logIC50'][source_data['response'] == 0].min()  # 计算 logIC50 列中响应为0的最小值，并将其设置为阈值。


        Counter(source_data['response'])[0] / len(source_data['response'])  # 计算 response 列中类别为0的样本占总样本的比例。
        Counter(source_data['response'])[1] / len(source_data['response'])  # 计算 response 列中类别为1的样本占总样本的比例。
        class_sample_count_s = np.array([Counter(source_data['response'])[0] / len(source_data['response']),
                                       Counter(source_data['response'])[1] / len(source_data['response'])])
        # print(class_sample_count_s)

        # # 如果药物不是 PLX4720_451Lu，则使用SMOTE和下采样
        # if DRUG != "PLX4720_451Lu":
        #     # 应用SMOTE和下采样
        #     x_resampled, y_resampled = pipeline.fit_resample(x_expression, y_response)
        #     class_counts = Counter(y_resampled)
        #     total_samples = len(y_resampled)
        #     class_1_proportion = class_counts[1] / total_samples
        #
        #     print(f"类别为1的样本占总样本的比例: {class_1_proportion:.4f}")
        #     print(f"类别为1的样本数: {class_counts[1]}")
        #
        #     # 标准化处理
        #     scaler = StandardScaler()
        #     x_scaled = scaler.fit_transform(x_resampled)
        #
        #     # 返回的标签是y_resampled
        #     label = y_resampled.values
        # else:
        #     # 如果是 PLX4720_451Lu，跳过SMOTE和下采样，直接标准化处理
        #     print(f"跳过SMOTE和下采样，直接对{DRUG}进行标准化处理")
        #     scaler = StandardScaler()
        #     x_scaled = scaler.fit_transform(x_expression)
        #
        #     # 手动计算并输出response列中类别为1的样本占总样本的比例
        #     class_counts = Counter(y_response)
        #     total_samples = len(y_response)
        #     class_1_proportion = class_counts[1] / total_samples
        #     print(f"类别为1的样本占总样本的比例: {class_1_proportion:.4f}")
        #     print(f"类别为1的样本数: {class_counts[1]}")
        #
        #     # 返回的标签是y_response
        #     label = y_response.values

        # 应用SMOTE和下采样
        x_resampled, y_resampled = pipeline.fit_resample(x_expression, y_response)

        class_counts = Counter(y_resampled)
        total_samples = len(y_resampled)
        class_1_proportion = class_counts[1] / total_samples

        # print(f"类别为1的样本占总样本的比例: {class_1_proportion:.4f}")
        # print(f"类别为1的样本数: {class_counts[1]}")

        # 标准化处理
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_resampled)

        # data （细胞，基因）细胞索引  0..... 基因名
        data = x_scaled
        label = y_resampled.values
        # # 使用 train_test_split 函数分割数据集
        # X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.4, random_state=42)


    elif dataset == 'target':
        path_s = './split_norm/' + DRUG + '/' + 'Target_expr_resp_z.' + DRUG + '.tsv'
        target_data = pd.read_csv(path_s, sep='\t', index_col=0)
        x_expression = target_data.iloc[:, 1:]  # gene expressions (features)
        y_response = target_data.iloc[:, 0]  # col index 1 of the source df is logIC50
        data = x_expression
        label = y_response.values

    scaled = data

    # 将数据转换为numpy矩阵
    matrix = scaled.to_numpy() if isinstance(scaled, pd.DataFrame) else scaled
    features = torch.from_numpy(matrix).float()
    # 生成k近邻图的稀疏矩阵表示
    # knn_adj = kneighbors_graph(features, n_neighbors=k, mode='distance', include_self=True,
    #                                metric='cosine')
    knn_adj = dgl.knn_graph(features, k=k, algorithm='kd-tree', dist='cosine') ## 使用DGL库生成k近邻图



    # 修改部分：确保返回的adj是DGLGraph对象
    g = dgl.add_self_loop(knn_adj) # 为图添加自环
    # adj = g.adjacency_matrix_scipy(return_edge_ids=False).tocoo()
    adj = g.adjacency_matrix()  # 获取邻接矩阵
    row, col= adj.coo()   ## 将邻接矩阵转换为COO格式
    row = row.numpy()
    col = col.numpy()
    coo_adj_data = adj.val.numpy()    # 获取COO格式的邻接矩阵数据

    # 创建 SciPy 的 COO 稀疏矩阵
    sp_adj = sp.coo_matrix((coo_adj_data, (row, col)), shape=(g.number_of_nodes(), g.number_of_nodes()))

    adj_normalized = normalize_adj(sp_adj)  # 归一化邻接矩阵
    adj = dgl.from_scipy(adj_normalized)   # 将归一化的SciPy稀疏矩阵转换为DGL图


    labels = torch.LongTensor(label)
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    # 返回特征数量
    n_features = features.shape[1]

    # 检查输入数据是否存在 NaN
    if np.isnan(features.numpy()).any():
        print("NaN detected in features")
    if np.isnan(labels.numpy()).any():
        print("NaN detected in labels")

    return adj, features, labels, knn_adj, n_features


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

class ConditionalEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(ConditionalEntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum(dim=1)
        return -1.0 * b.mean(dim=0)


def load_adj_label_drug(graph):
    # 提取邻接矩阵的边
    src, dst = graph.edges()
    # 创建一个SciPy稀疏矩阵
    A = sp.coo_matrix((np.ones(len(src)), (src.numpy(), dst.numpy())), shape=(graph.number_of_nodes(), graph.number_of_nodes()))
    adj_label = torch.FloatTensor(A.toarray())
    pos_weight = float(A.shape[0] * A.shape[0] - A.sum()) / A.sum()
    pos_weight = np.array(pos_weight).reshape(1, 1)
    pos_weight = torch.from_numpy(pos_weight)
    norm = A.shape[0] * A.shape[0] / float((A.shape[0] * A.shape[0] - A.sum()) * 2)
    return adj_label, pos_weight, norm


def compute_positional_encodings(graph, num_eigenvectors):
    """
    计算给定DGL图的Laplacian位置编码。

    参数:
    - graph: DGLGraph，输入的图。
    - num_eigenvectors: int，要计算的最小非平凡特征向量的数量。

    返回:
    - Tensor，形状为(N, k)，N是节点数，k是特征向量的数量。
    """
    # 使用dgl.lap_pe计算位置编码
    pos_enc = dgl.lap_pe(graph, k=num_eigenvectors, padding=True)
    # 标准化位置编码
    pos_enc_mean = pos_enc.mean(dim=0, keepdim=True)
    pos_enc_std = pos_enc.std(dim=0, keepdim=True)
    pos_enc_normalized = (pos_enc - pos_enc_mean) / (pos_enc_std + 1e-6)  # 添加小常数避免除以零
    return pos_enc_normalized