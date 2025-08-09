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

## Combine SMOTE sampling and undersampling ##
# Define SMOTE and undersampling strategy
over = SMOTE(sampling_strategy=0.75)  # Upsample the minority class to 75% of the majority class
under = RandomUnderSampler(sampling_strategy=0.75)  # Undersample the majority class to 75% of the minority class
steps = [('over', over), ('under', under)]
pipeline = Pipeline(steps=steps)

def sample_data(data, sample_rate, random_state=None):
    """Randomly sample from the dataset.
    Args:
        data (DataFrame): The original dataset.
        sample_rate (float): Sampling ratio, e.g., 0.2 means 20%.
        random_state (int, optional): Random seed for reproducibility.
    Returns:
        DataFrame: The sampled data.
    """
    # Calculate the number of samples to be taken
    num_samples = int(len(data) * sample_rate)
    # Randomly select samples
    sampled_data = data.sample(n=num_samples, random_state=random_state)
    return sampled_data

def normalize_adj(adj):
    """Symmetrically normalize the adjacency matrix."""
    adj = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


# Compute the Pearson correlation matrix
def compute_pearson_correlation(features):
    """
    Compute the Pearson correlation matrix
    Args:
        features (numpy.ndarray): Feature matrix, shape (n_samples, n_features)
    Returns:
        numpy.ndarray: Pearson correlation matrix
    """
    # Compute the Pearson correlation matrix
    corr_matrix = np.corrcoef(features)
    return corr_matrix


def build_graph_using_pearson(features, percentile=20):
    """
    Build a graph based on Pearson correlation
    Args:
        features (numpy.ndarray): Feature matrix, shape (n_samples, n_features)
        percentile (float): The threshold for Pearson correlation coefficient, only nodes with correlation above this threshold will be connected.
    Returns:
        dgl.DGLGraph: A graph constructed based on Pearson correlation coefficient
    """
    # Compute the Pearson correlation matrix
    corr_matrix = compute_pearson_correlation(features)

    # Get all non-diagonal elements of the correlation matrix
    corr_values = corr_matrix[np.triu_indices(corr_matrix.shape[0], k=1)]

    # Compute the specified percentile of the correlation matrix
    threshold = np.percentile(corr_values, 100 - percentile)  # Select the top percentile% correlations as the threshold

    # Convert the correlation matrix to an adjacency matrix, keeping connections where correlation is above the threshold
    adj_matrix = (corr_matrix > threshold).astype(int)

    # Get the non-zero elements of the adjacency matrix
    src_nodes, dst_nodes = np.where(adj_matrix)
    edge_weight = adj_matrix[src_nodes, dst_nodes]
    edges = (src_nodes, dst_nodes)

    # Create the DGL graph
    g = dgl.graph((torch.from_numpy(src_nodes), torch.from_numpy(dst_nodes)))

    # Add self-loops
    g = dgl.add_self_loop(g)

    adj = g.adjacency_matrix()  # Get the adjacency matrix
    row, col = adj.coo()
    row = row.numpy()
    col = col.numpy()
    coo_adj_data = adj.val.numpy()

    sp_adj = sp.coo_matrix((coo_adj_data, (row, col)), shape=(g.number_of_nodes(), g.number_of_nodes()))
    adj_normalized = normalize_adj(sp_adj)  # Normalize the adjacency matrix
    adj = dgl.from_scipy(adj_normalized)  # Convert the normalized SciPy sparse matrix back to DGL graph

    return adj, g


# Modified load_data_drug function
def load_data_drug(dataset, DRUG):
    data = None

    if dataset == 'source':
        path_s = './preprocessNormData/' + DRUG + '/' + 'Source_exprs_resp_z.' + DRUG + '.tsv'
        source_data = pd.read_csv(path_s, sep='\t', index_col=0)

        sample_rate = 1
        source_data = sample_data(source_data, sample_rate, random_state=42)  # Random sampling
        print(f'Current sampling rate: {sample_rate * 100}%')

        x_expression = source_data.iloc[:, 2:]  # Gene expressions (features)
        y_logIC50 = source_data.iloc[:, 1]  # Column index 1 of the source df is logIC50
        y_response = source_data.iloc[:, 0]
        threshold = source_data['logIC50'][source_data['response'] == 0].min()  # Calculate the minimum logIC50 value for response 0
        # print("Calculate the minimum logIC50 value for response 0, threshold is: " + str(threshold))

        # Apply SMOTE and undersampling
        x_resampled, y_resampled = pipeline.fit_resample(x_expression, y_response)

        # Standardization
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_resampled)

        # Data (cells, genes) Cell index 0..... Gene names
        data = x_scaled
        label = y_resampled.values

    elif dataset == 'target':
        path_s = './preprocessNormData/' + DRUG + '/' + 'Target_expr_resp_z.' + DRUG + '.tsv'
        target_data = pd.read_csv(path_s, sep='\t', index_col=0)
        x_expression = target_data.iloc[:, 1:]  # Gene expressions (features)
        y_response = target_data.iloc[:, 0]  # Column index 1 of the source df is logIC50
        data = x_expression
        label = y_response.values


    matrix = data.to_numpy() if isinstance(data, pd.DataFrame) else data
    features = torch.from_numpy(matrix).float()

    adj, graph = build_graph_using_pearson(features, percentile=20)

    labels = torch.LongTensor(label)
    n_features = features.shape[1]

    # Check for NaN in features or labels
    if np.isnan(features.numpy()).any():
        print("NaN detected in features")
    if np.isnan(labels.numpy()).any():
        print("NaN detected in labels")

    return adj, features, labels, graph, n_features

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
    # Extract edges from the adjacency matrix A
    src, dst = graph.edges()
    # Create a SciPy sparse matrix
    A = sp.coo_matrix((np.ones(len(src)), (src.numpy(), dst.numpy())), shape=(graph.number_of_nodes(), graph.number_of_nodes()))
    adj_label = torch.FloatTensor(A.toarray())
    pos_weight = float(A.shape[0] * A.shape[0] - A.sum()) / A.sum()
    pos_weight = np.array(pos_weight).reshape(1, 1)
    pos_weight = torch.from_numpy(pos_weight)
    norm = A.shape[0] * A.shape[0] / float((A.shape[0] * A.shape[0] - A.sum()) * 2)
    return adj_label, pos_weight, norm
