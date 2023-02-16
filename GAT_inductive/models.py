import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer_latest


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer_latest(nfeat, nhid[0], dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer_latest(nhid[0] * nheads, nhid[1], dropout=dropout, alpha=alpha, concat=False)
        self.fc = nn.Linear(nhid[1], nclass)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc(x)

        return x
