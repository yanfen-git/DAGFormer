import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from GTLayer import GraphTransformer
from dgl.nn.pytorch import SAGEConv


class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelVAE, self).__init__()
        self.gc1 = GraphTransformer(input_feat_dim, hidden_dim1, num_heads=4, num_layers=3)
        self.gc2 = GraphTransformer(hidden_dim1, hidden_dim2, num_heads=4, num_layers=3)
        self.gc3 = GraphTransformer(hidden_dim1, hidden_dim2, num_heads=4, num_layers=3)
        self.dropout = dropout


    def encode(self, x, g):
        if not isinstance(g, dgl.DGLGraph):
            raise TypeError("Expected DGLGraph; got {}".format(type(g)))
        if g.device != x.device:
            g = g.to(x.device)
        # hidden1 = self.gc1(g, x, pos_enc)
        hidden1 = self.gc1(g, x)
        hidden1 = F.normalize(hidden1)
        hidden1 = F.relu(hidden1)
        hidden1 = F.dropout(hidden1, self.dropout, training=self.training)

        gc2_output = self.gc2(g, hidden1)
        gc2_output = F.normalize(gc2_output)
        gc3_output = self.gc3(g, hidden1)
        gc3_output = F.normalize(gc3_output)

        return gc2_output, gc3_output
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            # 添加小常数避免除以零
            std = std + 1e-12
            z = eps.mul(std).add_(mu)

            return z
        else:
            return mu

    def forward(self, x, g):
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Input x contains NaN or Inf values.")

        # mu, logvar = self.encode(x, g, pos_enc)
        mu, logvar = self.encode(x, g)
        z = self.reparameterize(mu, logvar)


        return z, mu, logvar




class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""
    # 解码器模块，它使用内积来预测邻接矩阵，通常用于图生成或重构任务。
    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj