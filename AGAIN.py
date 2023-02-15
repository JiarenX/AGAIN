from __future__ import division
from __future__ import print_function
import torch
import numpy as np
from time import time
from functools import partial
from torch.nn import functional as F

from main_AGAIN import printout
from models import att_net, MLP_D, init_setup
from problem import NodeProblem
from helpers import set_seeds, to_gpu
from nn_modules import aggregator_lookup, prep_lookup, sampler_lookup
from lr import LRSchedule
from parse_args import save_args


def evaluate(model, problem, mode='val'):
    assert mode in ['test', 'val']
    preds, acts = [], []
    for (ids, targets, _) in problem.iterate(mode=mode, shuffle=False):
        preds_per_batch = model(ids, problem.feats, train=False)
        preds.append(preds_per_batch.cpu().detach().numpy())
        acts.append(targets.cpu().detach().numpy())

    return problem.metric_fn(np.vstack(acts), np.vstack(preds))


def run_test(args, flog):
    t_test = time()
    emb_model, problem = init_setup(args)
    test_metric = evaluate(emb_model, problem, mode='test')
    printout(flog, "Test micro-F1 {:.4f} | Test macro-F1 {:.4f} | Test accuracy {:.4f} | Time(s) {:.4f}".
             format(test_metric['micro'], test_metric['macro'], test_metric['acc'], time()-t_test))

    return test_metric


def train_emb(ids, feats, targets, loss_fn, progress, emb_model, emb_lr_scheduler, optimizer_emb):
    emb_model.train()
    optimizer_emb.zero_grad()

    emb_lr = emb_lr_scheduler(progress)
    LRSchedule.set_lr(optimizer_emb, emb_lr)
    preds = emb_model(ids, feats, train=True)
    loss = loss_fn(preds, targets.squeeze())
    loss.backward()
    optimizer_emb.step()

    return preds, loss


def train_gan_g(ids, feats, args, emb_model, gan_disc, optimizer_emb):
    emb_model.train()
    optimizer_emb.zero_grad()
    one = to_gpu(args.cuda, torch.tensor(1, dtype=torch.float))
    adversarial_loss = torch.nn.BCELoss()   # loss function of discriminator
    valid = to_gpu(args.cuda, torch.FloatTensor(ids.shape[0], 1).fill_(1.0))
    fake_hidden = emb_model(ids, feats, train=True, encode_only=True)
    errG = gan_disc(fake_hidden)
    g_loss = adversarial_loss(errG, valid)
    g_loss.backward(one)
    optimizer_emb.step()

    return g_loss


def train_gan_d(ids, feats, args, emb_model, gan_disc, optimizer_gan_d):
    # clip weights for discriminator
    for p in gan_disc.parameters():
        p.data.clamp_(-args.disc_clip, args.disc_clip)
    gan_disc.train()
    optimizer_gan_d.zero_grad()
    adversarial_loss = torch.nn.BCELoss()
    valid = to_gpu(args.cuda, torch.FloatTensor(ids.shape[0], 1).fill_(1.0))
    fake = to_gpu(args.cuda, torch.FloatTensor(ids.shape[0], 1).fill_(0.0))
    one = to_gpu(args.cuda, torch.tensor(1, dtype=torch.float))
    # negative samples
    fake_hidden = emb_model(ids, feats, train=True, encode_only=True)
    errD_fake = gan_disc(fake_hidden.detach())
    fake_loss = adversarial_loss(errD_fake, fake)
    # positive samples
    real_hidden = to_gpu(args.cuda, torch.FloatTensor(np.random.normal(0, 0.01, size=fake_hidden.shape)))
    errD_real = gan_disc(real_hidden)
    real_loss = adversarial_loss(errD_real, valid)
    # total loss for discriminator
    d_loss = (real_loss + fake_loss)/2
    d_loss.backward(one)
    optimizer_gan_d.step()

    return d_loss, real_loss, fake_loss


def train(args, flog):
    set_seeds(args.seed)
    problem = NodeProblem(args=args, problem_path=args.problem_path, cuda=args.cuda)
    n_train_samples = args.n_train_samples.split(',')
    n_val_samples = args.n_val_samples.split(',')
    output_dims = args.output_dims.split(',')
    emb_model = att_net(**{
        "sampler_class": sampler_lookup[args.sampler_class],
        "adj": problem.adj,
        "train_adj": problem.train_adj,
        
        "prep_class": prep_lookup[args.prep_class],
        "aggregator_class": aggregator_lookup[args.aggregator_class],
        
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

    gan_disc = MLP_D(ninput=2*int(output_dims[1]), noutput=1, layers=args.arch_d, gpu=args.cuda)

    emb_lr_scheduler = partial(getattr(LRSchedule, args.lr_schedule), lr_init=args.lr_init)
    emb_lr = emb_lr_scheduler(0.0)
    optimizer_emb = torch.optim.Adam(emb_model.parameters(), lr=emb_lr, weight_decay=args.weight_decay)
    optimizer_gan_d = torch.optim.Adam(gan_disc.parameters(), lr=args.lr_gan_d, betas=(args.beta1, 0.999))
    if args.cuda:
        emb_model = emb_model.cuda()
        gan_disc = gan_disc.cuda()
    # ####################
    # Train
    # ####################
    best_val = None
    start_time = time()
    val_metric = None
    print('---------- Train start ----------')
    for epoch in range(args.epochs):
        train_data = problem.batch_div(mode='train', shuffle=True, batch_size=args.batch_size)
        for i in range(len(train_data)):
            ids, targets, epoch_progress = train_data[i]
            progress = (epoch + epoch_progress) / args.epochs
            preds, loss = train_emb(ids=ids, feats=problem.feats, targets=targets, loss_fn=problem.loss_fn, progress=progress,
                                    emb_model=emb_model, emb_lr_scheduler=emb_lr_scheduler, optimizer_emb=optimizer_emb)
            train_metric = problem.metric_fn(targets.cpu().detach().numpy(), preds.cpu().detach().numpy())
            # train GAN
            if epoch_progress == 1.0:
                # train discriminator
                for i in range(args.niters_gan_d):
                    ids, targets, _ = train_data[np.random.randint(0, len(train_data) - 1) if len(train_data)>1 else 0]
                    d_loss, real_loss, fake_loss = train_gan_d(ids=ids, feats=problem.feats, args=args, emb_model=emb_model,
                                                               gan_disc=gan_disc, optimizer_gan_d=optimizer_gan_d)
                # train generator
                for j in range(args.niters_gan_g):
                    ids, targets, _ = train_data[np.random.randint(0, len(train_data) - 1) if len(train_data)>1 else 0]
                    g_loss = train_gan_g(ids=ids, feats=problem.feats, args=args, emb_model=emb_model,
                                         gan_disc=gan_disc, optimizer_emb=optimizer_emb)
            # evaluate on validation set
            _ = emb_model.eval()
            val_metric = evaluate(emb_model, problem, mode='test')
            print("Epoch {:03d} | Training loss {:.4f} | Train micro-F1 {:.4f} | Train macro-F1 {:.4f} | Val micro-F1 {:.4f} | Val macro-F1 {:.4f}"
                  .format(epoch, loss.data, train_metric['micro'], train_metric['macro'], val_metric['micro'], val_metric['macro']))

        # save model with best valid accuracy
        if (best_val is None or val_metric['micro'] > best_val) and epoch > 0:
            best_val = val_metric['micro']
            print('---- save model ----')
            torch.save(emb_model.state_dict(), args.save_dir + '/' + args.saved_model + '.model')
            save_args(args.save_dir + '/'  + args.saved_model + '_args.pkl', args)

    print('---------- Train finished ----------')

    # ####################
    # Test
    # ####################
    print('---------- Testing ----------')
    test_metric = run_test(args, flog)

    return test_metric
