import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as sio
from sklearn import metrics
import sys
import pickle as pkl
import networkx as nx


def label_node(nodes, targets, ins_num):
    train_tar = targets[nodes]
    n_classes = list(set(targets[:, -1]))
    class_idx = {}
    train_nodes = []
    for i in n_classes:
        class_idx[i] = np.where(np.array(train_tar) == i)[0][:ins_num]
        train_nodes.append(nodes[class_idx[i]])

    return np.hstack(train_nodes)


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))

    return index


def load_blogcatalog(path="./", dataset="blogcatalog", seed=123):
    data_mat = sio.loadmat(path + dataset + '.mat')
    adj = data_mat['adj'].tocoo()
    features = data_mat['feature'].tocsr()
    labels = data_mat['label']

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = np.array(range(1500, labels.shape[0]))
    idx_val = range(1000, 1500)
    idx_test = range(1000)
    idx_tot = np.arange(labels.shape[0])
    # select the labeled nodes
    np.random.seed(seed)
    np.random.shuffle(idx_train)
    idx_train = label_node(idx_train, np.argmax(labels, axis=1).reshape(-1, 1), ins_num=20)
    # convert to tensor
    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    idx_tot = torch.LongTensor(idx_tot)
    adj_train = adj.clone()
    adj_train[:, idx_test] = torch.zeros(labels.shape[0], idx_test.shape[0])

    return adj, adj_train, features, labels, idx_train, idx_val, idx_test, idx_tot


def load_citation(path="./planetoid_induc", dataset="cora"):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(path + "/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(path + "/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position.
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = np.array(features.todense())
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    adj = adj.tocoo().astype('float32')
    features = sp.csr_matrix(features, dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = range(len(y))
    idx_val = range(labels.shape[0]-1500, labels.shape[0]-1000)
    idx_test = range(labels.shape[0]-1000, labels.shape[0])
    idx_tot = np.arange(labels.shape[0])

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    idx_tot = torch.LongTensor(idx_tot)
    adj_train = adj.clone()
    adj_train[:, idx_test] = torch.zeros(labels.shape[0], idx_test.shape[0])

    return adj, adj_train, features, labels, idx_train, idx_val, idx_test, idx_tot


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)

    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def top_k_preds(y_true, y_pred):
    top_k_list = np.array(np.sum(y_true, 1), np.int32)
    predictions = []
    for i in range(y_true.shape[0]):
        pred_i = np.zeros(y_true.shape[1])
        pred_i[np.argsort(y_pred[i, :])[-top_k_list[i]:]] = 1
        predictions.append(np.reshape(pred_i, (1, -1)))
    predictions = np.concatenate(predictions, axis=0)
    top_k_array = np.array(predictions, np.int64)

    return top_k_array


def cal_f1_score(y_true, y_pred):
    micro_f1 = metrics.f1_score(y_true, y_pred, average='micro')
    macro_f1 = metrics.f1_score(y_true, y_pred, average='macro')

    return micro_f1, macro_f1
