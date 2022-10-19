'''
main.py

Welcome, this is the entrance to 3dgan
'''

import argparse
from trainer import trainer
import torch

from tester import tester
import params


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    # add arguments
    parser = argparse.ArgumentParser()

    # loggings parameters
    parser.add_argument('--logs', type=str, default='first_test', help='logs by tensorboardX')
    parser.add_argument('--local_test', type=str2bool, default=False, help='local test verbose')
    parser.add_argument('--model_name', type=str, default="dcgan", help='model name for saving')
    parser.add_argument('--test', type=str2bool, default=False, help='call tester.py')
    parser.add_argument('--use_visdom', type=str2bool, default=False, help='visualization by visdom')
    # OOB
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--precision', default="float32", type=str, help='precision')
    parser.add_argument('--channels_last', default=1, type=int, help='Use NHWC or not')
    parser.add_argument('--jit', action='store_true', default=False, help='enable JIT')
    parser.add_argument('--profile', action='store_true', default=False, help='collect timeline')
    parser.add_argument('--num_iter', default=200, type=int, help='test iterations')
    parser.add_argument('--num_warmup', default=20, type=int, help='test warmup')
    parser.add_argument('--device', default='cpu', type=str, help='cpu, cuda or xpu')
    parser.add_argument('--nv_fuser', action='store_true', default=False, help='enable nv fuser')
    args = parser.parse_args()
    print(args)

    # list params
    args = params.print_params(args)

    # run program
    if not args.test:
        trainer(args)
    else:
        tester(args)


if __name__ == '__main__':
    main()
