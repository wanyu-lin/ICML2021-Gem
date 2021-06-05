import torch
import torch.nn as nn
import torch.nn.functional as F

from gae.layers import GraphConvolution


class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelVAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        # print('mu', mu.shape)
        # print('logvar', logvar.shape)
        z = self.reparameterize(mu, logvar)
        # print('x', z.shape)
        # print('decoded_adj', self.dc(z).shape)
        return self.dc(z), mu, logvar


class GCNModelVAE3(GCNModelVAE):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelVAE3, self).__init__(input_feat_dim, hidden_dim1, hidden_dim2, dropout)
        self.gc1_1 = GraphConvolution(hidden_dim1, hidden_dim1, dropout, act=F.relu)

    def encode(self, x, adj):
        # print('x', x.shape)
        # print('adj', adj.shape)
        hidden1 = self.gc1(x, adj)
        # print('hidden1', hidden1.shape)
        hidden2 = self.gc1_1(hidden1, adj)
        # print('hidden2', hidden2.shape)
        return self.gc2(hidden2, adj), self.gc3(hidden2, adj)


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.bmm(z, torch.transpose(z, 1, 2)))
        return adj