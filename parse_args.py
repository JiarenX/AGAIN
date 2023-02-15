import argparse
import pickle as pkl


def parse_args():
    """
    Parse the AGAIN arguments
    :return: args
    """
    parser = argparse.ArgumentParser()

    # train params
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--problem_path', type=str, default='./data/problem_citeseer.h5')
    parser.add_argument('--cuda', default=True)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr_init', type=float, default=0.001)
    parser.add_argument('--lr_schedule', type=str, default='constant')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--label_per_class', type=int, default=60)
    # architecture params
    parser.add_argument('--sampler_class', type=str, default='uniform_neighbor_sampler')
    parser.add_argument('--aggregator_class', type=str, default='attention')
    parser.add_argument('--prep_class', type=str, default='identity')
    parser.add_argument('--n_train_samples', type=str, default='25,10')
    parser.add_argument('--n_val_samples', type=str, default='25,10')
    parser.add_argument('--output_dims', type=str, default='128,128')
    parser.add_argument('--niters_gan_g', type=int, default=1,
                        help='number of generator iterations in training')  # in each epoch, number of iterations for training generator
    parser.add_argument('--arch_d', type=str, default='1024-1024-256',
                        help='critic/discriminator architecture (MLP)')  # specify the MLP structure of discriminator in GAN;
    parser.add_argument('--lr_gan_d', type=float, default=0.001,
                        help='critic/discriminator learning rate')  # learning rate for discriminator, because it is using ADM, by default it is a smaller one
    parser.add_argument('--niters_gan_d', type=int, default=1,
                        help='number of discriminator iterations in training')  # in each epoch, number of iterations for training discriminator
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')  # beta for adam
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--disc_clip', type=float, default=0.01, help='discriminator clip')
    # Logging
    parser.add_argument('--save_dir', type=str, default='./model_test')
    parser.add_argument('--saved_model', type=str, default=parser.parse_args().aggregator_class +
                                                           '_early_stop_model', help='saved model')
    args = parser.parse_args()

    return args


def save_args(fout, args):
    with open(fout, 'wb') as f:
        pkl.dump(args, f, pkl.HIGHEST_PROTOCOL)
