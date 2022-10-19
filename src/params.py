'''
params.py

Managers of all hyper-parameters

'''

import torch


def print_params(args):
    args.epochs = 500
    args.soft_label = False
    args.adv_weight = 0
    args.d_thresh = 0.8
    args.z_dim = 200
    args.z_dis = "norm"
    args.model_save_step = 1
    args.g_lr = 0.0025
    args.d_lr = 0.00001
    args.beta = (0.5, 0.999)
    args.cube_len = 32
    args.leak_value = 0.2
    args.bias = False
    
    args.data_dir = '../volumetric_data/'
    args.model_dir = 'chair/'  # change it to train on other data models
    args.output_dir = '../outputs'
    # images_dir = '../test_outputs'

    l = 16
    print(l * '*' + 'hyper-parameters' + l * '*')

    print('epochs =', args.epochs)
    print('batch_size =', args.batch_size)
    print('soft_labels =', args.soft_label)
    print('adv_weight =', args.adv_weight)
    print('d_thresh =', args.d_thresh)
    print('z_dim =', args.z_dim)
    print('z_dis =', args.z_dis)
    print('model_images_save_step =', args.model_save_step)
    print('data =', args.model_dir)
    print('device =', args.device)
    print('g_lr =', args.g_lr)
    print('d_lr =', args.d_lr)
    print('cube_len =', args.cube_len)
    print('leak_value =', args.leak_value)
    print('bias =', args.bias)

    print(l * '*' + 'hyper-parameters' + l * '*')

    return args
