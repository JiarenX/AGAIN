import os
import AGAIN
from parse_args import parse_args
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def printout(flog, data):
    print(data)
    flog.write(data + '\n')


def main(args):
    printout(flog, '---------- number of labeled nodes per class: {:03d} ----------'.format(args.label_per_class))
    printout(flog, '---------- random seed: {:03d} ----------'.format(args.seed))
    test_metric = AGAIN.train(args, flog)


if __name__ == "__main__":
    args = parse_args()
    # write logs
    flog = open(args.save_dir + '/' + args.aggregator_class + '_log.txt', 'a')
    main(args)
    flog.close()
