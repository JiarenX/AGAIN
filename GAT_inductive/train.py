from __future__ import division
from __future__ import print_function

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import top_k_preds, cal_f1_score, load_citation
from models import GAT
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def compute_test(model, adj, features, labels, idx_test):
    model.eval()
    output = model(features, adj)
    preds_test = top_k_preds(labels[idx_test].cpu().numpy(), output[idx_test].detach().cpu().numpy())
    micro_f1_test, macro_f1_test = cal_f1_score(labels[idx_test].cpu().numpy().astype('int64'), preds_test)

    return micro_f1_test, macro_f1_test


def train(args):
    # set up seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    # Load data
    adj_s, adj_train_s, features_s, labels_s, idx_train_s, idx_val_s, idx_test_s, idx_tot_s = load_citation(dataset=args.input_dataset)
    # model and optimizer
    model = GAT(nfeat=features_s.shape[1],
                nhid=args.hidden,
                nclass=int(labels_s.shape[1]),
                dropout=args.dropout,
                nheads=args.nb_heads,
                alpha=args.alpha)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()
        features_s = features_s.cuda()
        adj_s = adj_s.cuda()
        adj_train_s = adj_train_s.cuda()
        labels_s = labels_s.cuda()
        idx_train_s = idx_train_s.cuda()
        idx_val_s = idx_val_s.cuda()
        idx_test_s = idx_test_s.cuda()
        idx_tot_s = idx_tot_s.cuda()

    best_test = None
    for epoch in range(args.epochs):
        # train on labeled nodes
        model.train()
        optimizer.zero_grad()
        output_s = model(features_s, adj_train_s)
        labels_idx_s = torch.argmax(labels_s[idx_train_s], dim=1)
        loss_train_s = F.cross_entropy(output_s[idx_train_s], labels_idx_s)
        preds_s = top_k_preds(labels_s[idx_train_s].cpu().numpy(), output_s[idx_train_s].detach().cpu().numpy())
        micro_f1_s, macro_f1_s = cal_f1_score(labels_s[idx_train_s].cpu().numpy().astype('int64'), preds_s)
        loss_train_s.backward()
        optimizer.step()
        # evaluate on test nodes
        model.eval()
        output_test = model(features_s, adj_s)
        labels_idx_t = torch.argmax(labels_s[idx_test_s], dim=1)
        loss_test = F.cross_entropy(output_test[idx_test_s], labels_idx_t)
        preds_test = top_k_preds(labels_s[idx_test_s].cpu().numpy(), output_test[idx_test_s].detach().cpu().numpy())
        micro_f1_t, macro_f1_t = cal_f1_score(labels_s[idx_test_s].cpu().numpy().astype('int64'), preds_test)
        print("epoch {:03d} | train loss {:.4f} | train micro-F1 {:.4f} | train macro-F1 {:.4f}".
              format(epoch, loss_train_s.item(), micro_f1_s, macro_f1_s))
        print("test loss {:.4f} | test micro-F1 {:.4f} | test macro-F1 {:.4f}".format(loss_test.item(), micro_f1_t, macro_f1_t))
        # save model with best test accuracy
        if (best_test is None or micro_f1_t > best_test) and epoch > 0:
            best_test = micro_f1_t
            print('---- saving as best model since this is the best test acc so far ----')
            torch.save(model.state_dict(), './planetoid_induc/best.model')

    # Restore best model
    print('Loading best model')
    model.load_state_dict(torch.load('./planetoid_induc/best.model'))
    micro_f1_test, macro_f1_test = compute_test(model, adj_s, features_s, labels_s, idx_test_s)
    print("test metrics:")
    print('---------- random seed: {:03d} ----------'.format(args.seed))
    print("micro-F1 {:.4f} | macro-F1 {:.4f}".format(micro_f1_test, macro_f1_test)+'\n')

    return micro_f1_test, macro_f1_test


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dataset', type=str, default='citeseer')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, nargs='+', default=[256, 256])
    parser.add_argument('--nb_heads', type=int, default=1, help='Number of head attentions.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    micro_f1, macro_f1 = train(args)


