'''
tester.py

Test the trained 3dgan models
'''

import torch
from torch import optim
from torch import nn
from collections import OrderedDict
from utils import *
import os
from model import net_G, net_D
# from lr_sh import  MultiStepLR

# added
import datetime
import time
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import params
import visdom


# def test_gen(args):
#     test_z = []
#     test_num = 1000
#     for i in range(test_num):
#         z = generateZ(args, 1)
#         z = z.numpy()
#         test_z.append(z)

#     test_z = np.array(test_z)
#     print (test_z.shape)
# np.save("test_z", test_z)

def tester(args):
    print('Evaluation Mode...')

    # image_saved_path = '../images'
    # image_saved_path = params.images_dir
    image_saved_path = args.output_dir + '/' + args.model_name + '/' + args.logs + '/test_outputs'
    if not os.path.exists(image_saved_path):
        os.makedirs(image_saved_path)

    if args.use_visdom:
        vis = visdom.Visdom()

    save_file_path = args.output_dir + '/' + args.model_name
    pretrained_file_path_G = save_file_path + '/' + args.logs + '/models/G.pth'
    pretrained_file_path_D = save_file_path + '/' + args.logs + '/models/D.pth'

    print(pretrained_file_path_G)

    D = net_D(args)
    G = net_G(args)

    if not torch.cuda.is_available():
        G.load_state_dict(torch.load(pretrained_file_path_G, map_location={'cuda:0': 'cpu'}))
        D.load_state_dict(torch.load(pretrained_file_path_D, map_location={'cuda:0': 'cpu'}))
    else:
        G.load_state_dict(torch.load(pretrained_file_path_G))
        D.load_state_dict(torch.load(pretrained_file_path_D, map_location={'cuda:0': 'cpu'}))

    print('visualizing model')

    # test generator
    # test_gen(args)
    G.to(args.device)
    D.to(args.device)
    G.eval()
    D.eval()
    datatype = torch.float16 if args.precision == "float16" else torch.bfloat16 if args.precision == "bfloat16" else torch.float
    G = torch.xpu.optimize(model=G, dtype=datatype)
    D = torch.xpu.optimize(model=D, dtype=datatype)

    if args.channels_last:
        try:
            G = G.to(memory_format=torch.channels_last)
            D = D.to(memory_format=torch.channels_last)
            print("---- use NHWC format")
        except RuntimeError as e:
            print("---- use normal format")
            print("failed to enable NHWC: ", e)

    if args.nv_fuser:
       fuser_mode = "fuser2"
    else:
       fuser_mode = "none"
    print("---- fuser mode:", fuser_mode)

    # test_z = np.load("test_z.npy")
    # print (test_z.shape)
    # N = test_z.shape[0]

    N = args.num_iter + args.num_warmup
    total_time = 0.0
    total_sample = 0

    if args.profile and args.device == "xpu":
        for i in range(N):
            z = generateZ(args, args.batch_size)
            if args.channels_last:
                z = z.to(memory_format=torch.channels_last) if len(z.shape) == 4 else z
            if args.jit and i == 0:
                try:
                    G = torch.jit.trace(G, z, check_trace=False, strict=False)
                    print("---- JIT trace enable.")
                    
                except (RuntimeError, TypeError) as e:
                    print("---- JIT trace disable.")
                    print("failed to use PyTorch jit mode due to: ", e)
            
            with torch.autograd.profiler_legacy.profile(enabled=args.profile, use_xpu=True, record_shapes=False) as prof:
                elapsed = time.time()
                z = z.to(args.device)
                fake = G(z)
                samples = fake.unsqueeze(dim=0).detach().cpu().numpy()
                if args.jit and i == 0:
                    try:
                        D = torch.jit.trace(D, fake, check_trace=False, strict=False)
                        print("---- JIT trace enable.")
                        
                    except (RuntimeError, TypeError) as e:
                        print("---- JIT trace disable.")
                        print("failed to use PyTorch jit mode due to: ", e)
                y_prob = D(fake)
                torch.xpu.synchronize()
                elapsed = time.time() - elapsed

            y_real = torch.ones_like(y_prob)

            print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
            if i >= args.num_warmup:
                    total_sample += args.batch_size
                    total_time += elapsed
            if args.profile and i == int((args.num_iter + args.num_warmup)/2):
                import pathlib
                timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
                if not os.path.exists(timeline_dir):
                    try:
                        os.makedirs(timeline_dir)
                    except:
                        pass
                torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"),
                    timeline_dir+'profile.pt')
                torch.save(prof.key_averages(group_by_input_shape=True).table(),
                    timeline_dir+'profile_detail.pt')
    elif args.profile and args.device == "cuda":
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            schedule=torch.profiler.schedule(
                wait=int((args.num_iter + args.num_warmup)/2),
                warmup=2,
                active=1,
            ),
            on_trace_ready=trace_handler,
        ) as p:
            for i in range(N):
                z = generateZ(args, args.batch_size)
                if args.channels_last:
                    z = z.to(memory_format=torch.channels_last) if len(z.shape) == 4 else z
                if args.jit and i == 0:
                    try:
                        G = torch.jit.trace(G, z, check_trace=False, strict=False)
                        print("---- JIT trace enable.")
                        
                    except (RuntimeError, TypeError) as e:
                        print("---- JIT trace disable.")
                        print("failed to use PyTorch jit mode due to: ", e)
                
                with torch.jit.fuser(fuser_mode):
                    elapsed = time.time()
                    z = z.to(args.device)
                    fake = G(z)
                    samples = fake.unsqueeze(dim=0).detach().cpu().numpy()
                    if args.jit and i == 0:
                        try:
                            D = torch.jit.trace(D, fake, check_trace=False, strict=False)
                            print("---- JIT trace enable.")
                            
                        except (RuntimeError, TypeError) as e:
                            print("---- JIT trace disable.")
                            print("failed to use PyTorch jit mode due to: ", e)
                    y_prob = D(fake)
                torch.cuda.synchronize()
                elapsed = time.time() - elapsed
                p.step()
                y_real = torch.ones_like(y_prob)

                print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
                if i >= args.num_warmup:
                        total_sample += args.batch_size
                        total_time += elapsed
    elif args.profile and args.device == "cpu":
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            schedule=torch.profiler.schedule(
                wait=int((args.num_iter + args.num_warmup)/2),
                warmup=2,
                active=1,
            ),
            on_trace_ready=trace_handler,
        ) as p:
            for i in range(N):
                z = generateZ(args, args.batch_size)
                if args.channels_last:
                    z = z.to(memory_format=torch.channels_last) if len(z.shape) == 4 else z
                if args.jit and i == 0:
                    try:
                        G = torch.jit.trace(G, z, check_trace=False, strict=False)
                        print("---- JIT trace enable.")
                        
                    except (RuntimeError, TypeError) as e:
                        print("---- JIT trace disable.")
                        print("failed to use PyTorch jit mode due to: ", e)
                
                elapsed = time.time()
                z = z.to(args.device)
                fake = G(z)
                samples = fake.unsqueeze(dim=0).detach().cpu().numpy()
                if args.jit and i == 0:
                    try:
                        D = torch.jit.trace(D, fake, check_trace=False, strict=False)
                        print("---- JIT trace enable.")
                        
                    except (RuntimeError, TypeError) as e:
                        print("---- JIT trace disable.")
                        print("failed to use PyTorch jit mode due to: ", e)
                y_prob = D(fake)
                elapsed = time.time() - elapsed
                p.step()
                y_real = torch.ones_like(y_prob)

                print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
                if i >= args.num_warmup:
                        total_sample += args.batch_size
                        total_time += elapsed
    elif not args.profile and args.device == "cuda":
        for i in range(N):
            z = generateZ(args, args.batch_size)
            if args.channels_last:
                z = z.to(memory_format=torch.channels_last) if len(z.shape) == 4 else z
            if args.jit and i == 0:
                try:
                    G = torch.jit.trace(G, z, check_trace=False, strict=False)
                    print("---- JIT trace enable.")
                    
                except (RuntimeError, TypeError) as e:
                    print("---- JIT trace disable.")
                    print("failed to use PyTorch jit mode due to: ", e)
            

            with torch.jit.fuser(fuser_mode):
                elapsed = time.time()
                z = z.to(args.device)
                fake = G(z)
                samples = fake.unsqueeze(dim=0).detach().cpu().numpy()
                if args.jit and i == 0:
                    try:
                        D = torch.jit.trace(D, fake, check_trace=False, strict=False)
                        print("---- JIT trace enable.")
                        
                    except (RuntimeError, TypeError) as e:
                        print("---- JIT trace disable.")
                        print("failed to use PyTorch jit mode due to: ", e)
                y_prob = D(fake)
            torch.cuda.synchronize()
            elapsed = time.time() - elapsed

            y_real = torch.ones_like(y_prob)

            print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
            if i >= args.num_warmup:
                    total_sample += args.batch_size
                    total_time += elapsed
    else:
        for i in range(N):
            z = generateZ(args, args.batch_size)
            if args.channels_last:
                z = z.to(memory_format=torch.channels_last) if len(z.shape) == 4 else z
            if args.jit and i == 0:
                try:
                    G = torch.jit.trace(G, z, check_trace=False, strict=False)
                    print("---- JIT trace enable.")
                    
                except (RuntimeError, TypeError) as e:
                    print("---- JIT trace disable.")
                    print("failed to use PyTorch jit mode due to: ", e)
            

            elapsed = time.time()
            z = z.to(args.device)
            fake = G(z)
            samples = fake.unsqueeze(dim=0).detach().cpu().numpy()
            if args.jit and i == 0:
                try:
                    D = torch.jit.trace(D, fake, check_trace=False, strict=False)
                    print("---- JIT trace enable.")
                    
                except (RuntimeError, TypeError) as e:
                    print("---- JIT trace disable.")
                    print("failed to use PyTorch jit mode due to: ", e)
            y_prob = D(fake)
            if args.device == "xpu":
                torch.xpu.synchronize()
            elapsed = time.time() - elapsed

            y_real = torch.ones_like(y_prob)

            print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
            if i >= args.num_warmup:
                    total_sample += args.batch_size
                    total_time += elapsed

    latency = total_time / total_sample * 1000
    throughput = total_sample / total_time
    print("inference Latency: {} ms".format(latency))
    print("inference Throughput: {} samples/s".format(throughput))

    # visualization
    if not args.use_visdom:
        SavePloat_Voxels(samples, image_saved_path, 'tester_' + str(i))  # norm_
    else:
        plotVoxelVisdom(samples[0, :], vis, "tester_" + str(i))

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cpu_time_total")
    print(output)
    import pathlib
    timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
    if not os.path.exists(timeline_dir):
        try:
            os.makedirs(timeline_dir)
        except:
            pass
    timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + \
            '-' + str(p.step_num) + '-' + str(os.getpid()) + '.json'
    p.export_chrome_trace(timeline_file)
