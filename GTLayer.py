import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv

from preprocess import load_data_drug


class SparseMHA(nn.Module):
    """Sparse Multi-head Attention Module"""
    def __init__(self, hidden_dim, num_heads, bias=False):
        super(SparseMHA, self).__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scaling = hidden_dim ** -0.5
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)

    def forward(self, A, h):
        N = len(h)
        # # [N, dh, nh]
        q = self.q_proj(h).reshape(N, self.head_dim, self.num_heads) * self.scaling
        # # [N, dh, nh]
        k = self.k_proj(h).reshape(N, self.head_dim, self.num_heads)
        v = self.v_proj(h).reshape(N, self.head_dim, self.num_heads)
        # # bmm: (b, n, m) * (b, m, q) = (m, n, q)
        # # q.transpose(0, 2).transpose(1, 2) = (nh, N, dh)
        # # k.transpose(0, 2) = (nh, dh, N)
        # # attn = (nh, N, N)
        # attn = torch.bmm(q.transpose(0, 2).transpose(1, 2), k.transpose(0, 2))
        attn = torch.mul(A, torch.bmm(q.transpose(0, 2).transpose(1, 2), k.transpose(0, 2)))
        # attn = dglsp.bsddmm(A, q, k.transpose(1, 0))  # (sparse) [N, N, nh]
        # Sparse softmax by default applies on the last sparse dimension.
        # attn = attn.softmax()  # (sparse) [N, N, nh]
        attn = F.softmax(attn, dim=2)
        # v.transpose(0, 2).transpose(1, 2) = [nh,N,dh]
        # attn = [nh, N, N]
        # out = [nh, N, dh]
        out = torch.bmm(attn, v.transpose(0, 2).transpose(1, 2))
        # out = [N, dh, nh]
        out = out.transpose(0, 1).transpose(1, 2)
        # out = dglsp.bspmm(attn, v)  # [N, dh, nh]
        out = self.out_proj(out.reshape(N, -1))
        return out

class GTLayer(nn.Module):
    """Graph Transformer Layer"""
    def __init__(self, hidden_dim, num_heads, residual=True, bias=False):
        super(GTLayer, self).__init__()
        # 自动调整hidden_dim以确保能被num_heads整除
        if hidden_dim % num_heads != 0:
            corrected_dim = hidden_dim + (num_heads - hidden_dim % num_heads)
        else:
            corrected_dim = hidden_dim
        self.MHA = SparseMHA(corrected_dim, num_heads, bias)
        self.residual = residual
        self.batchnorm1 = nn.BatchNorm1d(corrected_dim)
        self.batchnorm2 = nn.BatchNorm1d(corrected_dim)
        self.FFN1 = nn.Linear(hidden_dim, hidden_dim * 2, bias=bias)
        self.FFN2 = nn.Linear(hidden_dim * 2, hidden_dim, bias=bias)


    def forward(self, A, h):
        if isinstance(A, dgl.DGLGraph):
            A = A.adj().to_dense()  
        if isinstance(A, torch.Tensor):
            A = A.to(h.device)  
        # attention-layer
        h1 = h
        h = self.MHA(A, h)

        if self.residual:
            h = h + h1

        h = self.batchnorm1(h)

        h2 = h
        # two-layer FNN
        h = self.FFN2(F.relu(self.FFN1(h)))

        if self.residual:
            h = h2 + h
        h = self.batchnorm2(h)
        return h

class GraphTransformer(nn.Module):
    def __init__(self, in_feat, hidden_dim, num_heads, num_layers, dropout=0.1):
        super(GraphTransformer, self).__init__()
        if hidden_dim % num_heads != 0:
            hidden_dim += (num_heads - hidden_dim % num_heads)
        self.input_proj = nn.Linear(in_feat, hidden_dim)  
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            GTLayer(hidden_dim, num_heads, bias=False)
            for _ in range(num_layers)
        ])
        self.residual = True
        # self.linear1 = nn.Linear(hidden_dim1, hidden_dim2)
        # self.linear2 = nn.Linear(hidden_dim2, hidden_dim2)

    def forward(self, A, features):
        h = self.input_proj(features)
        h1 = h
        # h = features
        for layer in self.layers:
            h = layer(A, h)
        # g.ndata['h'] = h
        # h = self.linear1(h)
        # h = F.relu(h)
        # h = self.linear2(h)
        h = self.dropout(h)

        if self.residual:
            h = h1 + h
        # print(h)
        return h





