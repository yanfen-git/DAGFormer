import torch.nn as nn
import dgl
from GTLayer import GraphTransformer
import torch.nn.functional as F
import torch

class GT(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GT, self).__init__()
        self.gc1 = GraphTransformer(nfeat, nhid, num_heads=4, num_layers=3)
        self.gc2 = GraphTransformer(nhid, nclass, num_heads=4, num_layers=3)
        self.gc3 = GraphTransformer(nhid, nclass, num_heads=4, num_layers=3)
        self.dropout = dropout

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            std = std + 1e-12
            z = eps.mul(std).add_(mu)
            if torch.isnan(z).any():
                print("NaN detected in reparameterize output")
            return z
        else:
            return mu

    def forward(self, g, features):
        with g.local_scope():
            x = self.gc1(g, features)
            x = F.normalize(x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

            x1 = self.gc2(g, x)
            x1 = F.normalize(x1)
            y1 = self.gc3(g, x)
            y1 = F.normalize(y1)

            z = self.reparameterize(x1, y1)

            return z, x1, y1


