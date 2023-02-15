import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


class UniformNeighborSampler(object):
    def __init__(self, adj):
        self.adj = adj
    
    def __call__(self, ids, n_samples=-1):
        tmp = self.adj[ids]
        perm = torch.randperm(tmp.shape[1])
        if ids.is_cuda:
            perm = perm.cuda()
        
        tmp = tmp[:, perm]
        return tmp[:, :n_samples]


sampler_lookup = {
    "uniform_neighbor_sampler": UniformNeighborSampler,
}


# Preprocesser
class IdentityPrep(nn.Module):
    def __init__(self, input_dim, n_nodes=None):
        super(IdentityPrep, self).__init__()
        self.input_dim = input_dim

    @property
    def output_dim(self):
        return self.input_dim

    def forward(self, ids, feats, layer_idx=0):
        return feats


prep_lookup = {
    "identity": IdentityPrep
}


class AggregatorMixin(object):
    @property
    def output_dim(self):
        tmp = torch.zeros((1, self.output_dim_))
        return self.combine_fn([tmp, tmp]).size(1)


class AttentionAggregator(nn.Module, AggregatorMixin):
    def __init__(self, input_dim, output_dim, activation, gpu=False, dropout=0.5, hidden_dim=256, alpha=0.2, combine_fn=lambda x: torch.cat(x, dim=1)):
        super(AttentionAggregator, self).__init__()
        self.dropout = dropout
        
        self.att = nn.Sequential(*[
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.LeakyReLU()
        ])
        self.fc_x = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neib = nn.Linear(input_dim, output_dim, bias=False)
        self.a = nn.Parameter(torch.zeros(size=(2 * hidden_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.alpha = alpha
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.gpu = gpu
        if self.gpu:
            self.att = self.att.cuda()
            self.fc_x = self.fc_x.cuda()
            self.fc_neib = self.fc_neib.cuda()
        
        self.output_dim_ = output_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.combine_fn = combine_fn
    
    def forward(self, x, neibs):
        neib_att = self.att(neibs)
        x_att = self.att(x)
        N = int(neib_att.size()[0]/x_att.size()[0])
        a_input = torch.cat([x_att.repeat(1, N).view(x_att.size()[0] * N, -1), neib_att], dim=1).view(x_att.size()[0], -1,
                                                                                           2 * self.hidden_dim)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        attention = F.softmax(e, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        agg_neib = neibs.view(x.shape[0], -1, neibs.shape[1])
        agg_neib = torch.sum(agg_neib * attention.unsqueeze(-1), dim=1)
        out = self.combine_fn([self.fc_x(x), self.fc_neib(agg_neib)])
        out = F.dropout(out, self.dropout, training=self.training)
        if self.activation:
            out = self.activation(out)

        return out


aggregator_lookup = {
    "attention": AttentionAggregator,
}
