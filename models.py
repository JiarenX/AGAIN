from __future__ import division
from __future__ import print_function
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
import random
from problem import NodeProblem
import numpy as np
import pickle as cp
from nn_modules import aggregator_lookup, prep_lookup, sampler_lookup


class att_net(nn.Module):
    def __init__(self,
                 input_dim,
                 n_nodes,
                 n_classes,
                 layer_specs,
                 aggregator_class,
                 prep_class,
                 sampler_class, adj, train_adj,
                 gpu=False):
        
        super(att_net, self).__init__()

        self.train_sampler = sampler_class(adj=train_adj)
        self.val_sampler = sampler_class(adj=adj)
        self.train_sample_fns = [partial(self.train_sampler, n_samples=s['n_train_samples']) for s in layer_specs]
        self.val_sample_fns = [partial(self.val_sampler, n_samples=s['n_val_samples']) for s in layer_specs]

        self.prep = prep_class(input_dim=input_dim, n_nodes=n_nodes)
        input_dim = self.prep.output_dim
        self.gpu = gpu

        agg_layers = []
        for spec in layer_specs:
            agg = aggregator_class(
                input_dim=input_dim,
                output_dim=spec['output_dim'],
                activation=spec['activation'],
                gpu=gpu
            )
            agg_layers.append(agg)
            input_dim = agg.output_dim
        
        self.agg_layers = nn.Sequential(*agg_layers)
        self.fc = nn.Linear(input_dim, n_classes, bias=True)
        if self.gpu:
            self.fc = self.fc.cuda()

    def forward(self, ids, feats, train=True, encode_only=False):
        # Sample neighbors
        sample_fns = self.train_sample_fns if train else self.val_sample_fns
        has_feats = feats is not None
        tmp_feats = feats[ids] if has_feats else None
        all_feats = [self.prep(ids, tmp_feats, layer_idx=0)]
        for layer_idx, sampler_fn in enumerate(sample_fns):
            ids = sampler_fn(ids=ids).contiguous().view(-1)
            tmp_feats = feats[ids] if has_feats else None
            all_feats.append(self.prep(ids, tmp_feats, layer_idx=layer_idx + 1))

        for agg_layer in self.agg_layers.children():
            all_feats = [agg_layer(all_feats[k], all_feats[k + 1]) for k in range(len(all_feats) - 1)]

        assert len(all_feats) == 1, "len(all_feats) != 1"
        out = F.normalize(all_feats[0], dim=1)

        if encode_only:
            return out

        return self.fc(out)


class MLP_D(nn.Module):
    def __init__(self, ninput, noutput, layers, gpu=False):
        super(MLP_D, self).__init__()
        self.ninput = ninput
        self.noutput = noutput
        self.gpu = gpu

        """
            parse network structure
        """
        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]

        self.model = nn.Sequential(
            nn.Linear(layer_sizes[0], layer_sizes[1]),
            nn.LeakyReLU(0.2),
            nn.Linear(layer_sizes[1], layer_sizes[2]),
            nn.LeakyReLU(0.2),
            nn.Linear(layer_sizes[2], layer_sizes[3]),
            nn.LeakyReLU(0.2),
            nn.Linear(layer_sizes[3], noutput),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.model(x)

        return x


def init_setup(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        _ = torch.cuda.manual_seed(args.seed)

    assert args.saved_model is not None
    with open(args.save_dir + '/' + '%s_args.pkl' % args.saved_model, 'rb') as f:
        base_args = cp.load(f)
    problem = NodeProblem(args=args, problem_path=base_args.problem_path, cuda=args.cuda)
    # Build the model
    mod = att_net
    n_train_samples = base_args.n_train_samples.split(',')
    n_val_samples = base_args.n_val_samples.split(',')
    output_dims = base_args.output_dims.split(',')
    emb_model = mod(**{
        "sampler_class": sampler_lookup[base_args.sampler_class],
        "adj": problem.adj,
        "train_adj": problem.train_adj,

        "prep_class": prep_lookup[base_args.prep_class],
        "aggregator_class": aggregator_lookup[base_args.aggregator_class],

        "input_dim": problem.feats_dim,
        "n_nodes": problem.n_nodes,
        "n_classes": problem.n_classes,
        "layer_specs": [
            {
                "n_train_samples": int(n_train_samples[0]),
                "n_val_samples": int(n_val_samples[0]),
                "output_dim": int(output_dims[0]),
                "activation": F.relu,
            },
            {
                "n_train_samples": int(n_train_samples[1]),
                "n_val_samples": int(n_val_samples[1]),
                "output_dim": int(output_dims[1]),
                "activation": lambda x: x,
            },
        ],
        "gpu": args.cuda,
    })
    if args.cuda == True:
        emb_model = emb_model.cuda()
    emb_model.load_state_dict(torch.load(args.save_dir + '/' + args.saved_model + '.model'))
    emb_model.eval()

    return emb_model, problem
