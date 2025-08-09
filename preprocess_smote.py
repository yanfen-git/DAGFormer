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
    """Randomly sample the dataset.
    Args:
        data (DataFrame): Original dataset.
        sample_rate (float): Sampling ratio, e.g., 0.2 means 20%.
        random_state (int, optional): Random seed for reproducibility.
    Returns:
        DataFrame: Sampled data.
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

def load_data_drug(dataset, DRUG):
    k = 15
    data = None

    if dataset == 'source':
        path_s = './preprocessNormData/' + DRUG + '/' + 'Source_exprs_resp_z.' + DRUG + '.tsv'
        source_data = pd.read_csv(path_s, sep='\t', index_col=0)

        sample_rate = 1
        source_data = sample_data(source_data, sample_rate, random_state=42)  # Random sampling
        x_expression = source_data.iloc[:, 2:]  # Gene expressions (features)
        y_logIC50 = source_data.iloc[:, 1]  # Column index 1 of the source df is logIC50
        y_response = source_data.iloc[:, 0]
        threshold = source_data['logIC50'][source_data['response'] == 0].min()  # Calculate the minimum logIC50 value for response 0

        # Calculate the proportion of samples in each class
        Counter(source_data['response'])[0] / len(source_data['response'])  # Proportion of class 0
        Counter(source_data['response'])[1] / len(source_data['response'])  # Proportion of class 1
        class_sample_count_s = np.array([Counter(source_data['response'])[0] / len(source_data['response']),
                                       Counter(source_data['response'])[1] / len(source_data['response'])])

        # Apply SMOTE and undersampling
        x_resampled, y_resampled = pipeline.fit_resample(x_expression, y_response)

        class_counts = Counter(y_resampled)
        total_samples = len(y_resampled)
        class_1_proportion = class_counts[1] / total_samples

        # Standardize the data
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_resampled)

        # Return the features and labels
        data = x_scaled
        label = y_resampled.values

    elif dataset == 'target':
        path_s = './preprocessNormData/' + DRUG + '/' + 'Target_expr_resp_z.' + DRUG + '.tsv'
        target_data = pd.read_csv(path_s, sep='\t', index_col=0)
        x_expression = target_data.iloc[:, 1:]  # Gene expressions (features)
        y_response = target_data.iloc[:, 0]  # Column index 1 of the source df is logIC50
        data = x_expression
        label = y_response.values

    scaled = data

    # Convert data to numpy matrix
    matrix = scaled.to_numpy() if isinstance(scaled, pd.DataFrame) else scaled
    features = torch.from_numpy(matrix).float()
    # Generate k-nearest neighbors graph sparse matrix representation
    knn_adj = dgl.knn_graph(features, k=k, algorithm='kd-tree', dist='cosine')  # Using DGL to generate k-NN graph

    # Modify part: Ensure the returned adj is a DGLGraph object
    g = dgl.add_self_loop(knn_adj)  # Add self-loop to the graph
    adj = g.adjacency_matrix()  # Get the adjacency matrix
    row, col = adj.coo()  # Convert the adjacency matrix to COO format
    row = row.numpy()
    col = col.numpy()
    coo_adj_data = adj.val.numpy()  # Get COO format adjacency matrix data

    # Create SciPy's COO sparse matrix
    sp_adj = sp.coo_matrix((coo_adj_data, (row, col)), shape=(g.number_of_nodes(), g.number_of_nodes()))

    adj_normalized = normalize_adj(sp_adj)  # Normalize the adjacency matrix
    adj = dgl.from_scipy(adj_normalized)  # Convert the normalized SciPy sparse matrix back to DGL graph

    labels = torch.LongTensor(label)

    # Return the number of features
    n_features = features.shape[1]

    # Check for NaN values in the features or labels
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
    # Extract edges from the adjacency matrix of the graph
    src, dst = graph.edges()
    # Create a SciPy sparse matrix
    A = sp.coo_matrix((np.ones(len(src)), (src.numpy(), dst.numpy())), shape=(graph.number_of_nodes(), graph.number_of_nodes()))
    adj_label = torch.FloatTensor(A.toarray())
    pos_weight = float(A.shape[0] * A.shape[0] - A.sum()) / A.sum()
    pos_weight = np.array(pos_weight).reshape(1, 1)
    pos_weight = torch.from_numpy(pos_weight)
    norm = A.shape[0] * A.shape[0] / float((A.shape[0] * A.shape[0] - A.sum()) * 2)
    return adj_label, pos_weight, norm


def compute_positional_encodings(graph, num_eigenvectors):
    # Use dgl.lap_pe to compute positional encoding
    pos_enc = dgl.lap_pe(graph, k=num_eigenvectors, padding=True)
    # Normalize positional encodings
    pos_enc_mean = pos_enc.mean(dim=0, keepdim=True)
    pos_enc_std = pos_enc.std(dim=0, keepdim=True)
    pos_enc_normalized = (pos_enc - pos_enc_mean) / (pos_enc_std + 1e-6)  # Add small constant to avoid division by zero
    return pos_enc_normalized
