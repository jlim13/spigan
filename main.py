import os
import argparse

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
import torchvision.utils as vutils

import utils as utils
import trainer_helper

#networks
import patchGAN
import resnetGen
import fcn8
import perceptual_loss

#TODO: Reimplement my own networks for the GAN components.
#TODO: Use soft labels for gan discrimin [x]
#TODO: THere is an issue with fcn8 and weight init. Params are all zero if I keep it on
#TODO: Split the vgg feature extractor to two gpus

USE_CUDA = torch.cuda.is_available()

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def train(args, real_dataloader, syn_dataloader, D_net, G_net, T_net, P_net,
        optim_D, optim_G, optim_T, optim_P, perceptual_feat_net,
        adv_criterion, norm_l1_criterion, tnet_criterion):

    if args.resume_warmup:
        print ("Loading up the weights for the warmed up generator")
        generator_warmup_path = os.path.join(args.model_weight_dir,
                                args.warmup_resume_path)

        G_net, optim_G, resume_iter = utils.load_model(args,
                                    model_type = 'G_net',
                                    model = G_net,
                                    optimizer = optim_G,
                                    fname = generator_warmup_path,
                                    USE_CUDA = USE_CUDA)

    else:
        print ("Warming up the generator")
        G_net, optim_G = trainer_helper.warmup_generator(args, USE_CUDA,
                syn_dataloader, G_net, optim_G,
                perceptual_feat_net, norm_l1_criterion)

        #save the warmed up generator
        out_file_warmup = os.path.join(args.model_weight_dir, args.warmup_resume_path)
        utils.save_model(args, model_type = 'G_net', model = G_net,
            optimizer = optim_G, iter_num = args.gen_warmup_iters, fname = out_file_warmup)


    if args.resume_generator:

        g_net_weight_pth = os.path.join(args.model_weight_dir,
                                            args.warmup_resume_path)
        print ("Loading the generator weights from {0}".format(g_net_weight_pth))

        G_net, optim_G, resume_iter = utils.load_model(args,
                                    model_type = 'G_net',
                                    model = G_net,
                                    optimizer = optim_G,
                                    fname = g_net_weight_pth,
                                    USE_CUDA = USE_CUDA)

    if args.resume_discrim:
        d_net_weight_pth = os.path.join(args.model_weight_dir,
                                            args.discrim_resume_path)
        print ("Loading the discrim weights from {0}".format(d_net_weight_pth))
        D_net, optim_D, resume_iter = utils.load_model(args,
                                    model_type = 'D_net',
                                    model = D_net,
                                    optimizer = optim_D,
                                    fname = d_net_weight_pth,
                                    USE_CUDA = USE_CUDA)

    if args.resume_task:
        t_net_weight_pth = os.path.join(args.model_weight_dir,
                                            args.task_resume_path)
        print ("Loading the task weights from {0}".format(t_net_weight_pth))
        T_net, optim_T, resume_iter = utils.load_model(args,
                                    model_type = 'T_net',
                                    model = T_net,
                                    optimizer = optim_T,
                                    fname = t_net_weight_pth,
                                    USE_CUDA = USE_CUDA)

    if args.resume_priv:
        p_net_weight_pth = os.path.join(args.model_weight_dir,
                                            args.priv_resume_path)
        print ("Loading the privileged weights from {0}".format(p_net_weight_pth))
        P_net, optim_P, _ = utils.load_model(args,
                                    model_type = 'P_net',
                                    model = P_net,
                                    optimizer = optim_P,
                                    fname = p_net_weight_pth,
                                    USE_CUDA = USE_CUDA)


    #real training

    G_losses = []
    D_losses = []
    T_losses = []
    Pi_losses = []
    Perc_losses = []

    for iter_num in range(args.max_iters):

        if args.resume_task:
            iter_num += resume_iter
        print (iter_num)

        D_net, optim_D, running_d_loss = trainer_helper.train_discrim(args,
                USE_CUDA, syn_dataloader, real_dataloader,
                D_net, G_net, optim_D, adv_criterion)

        D_losses.append(running_d_loss)

        #Train generator
        running_g_loss = 0.0
        running_perception_loss = 0.0

        G_net, optim_G, running_g_loss, \
                running_perception_loss = trainer_helper.train_generator(args,
                    USE_CUDA, syn_dataloader, G_net, D_net, optim_G,
                    adv_criterion, norm_l1_criterion,
                    perceptual_feat_net, real_dataloader)

        G_losses.append(running_g_loss)
        Perc_losses.append(running_perception_loss)

        #train privileged info network, i.e. depth
        P_net, optim_P, running_priv_loss = \
            trainer_helper.train_pi_network(args, USE_CUDA,
                    syn_dataloader, G_net, P_net, optim_P, norm_l1_criterion)
        Pi_losses.append(running_priv_loss)

        # #Finally, train task network
        T_net, optim_T, running_t_loss = \
            trainer_helper.train_task_network(args, USE_CUDA, syn_dataloader,
                    G_net, T_net, optim_T, tnet_criterion)
        T_losses.append(running_t_loss)

        if (iter_num % args.save_every) == 0:
            utils.save_generator_outputs(USE_CUDA, G_net,
                            iter_num, real_dataloader, syn_dataloader)
            utils.save_depth_outputs(USE_CUDA, P_net, G_net,
                            iter_num, real_dataloader, syn_dataloader)
            if args.T_steps > 0:
                utils.save_segmentation_outputs(USE_CUDA, T_net, G_net,
                                iter_num, real_dataloader, syn_dataloader)

            g_net_weight_pth = os.path.join(args.model_weight_dir,
                                                args.warmup_resume_path)
            utils.save_model(args,model_type = 'G_net', model = G_net,
                                optimizer = optim_G, iter_num = iter_num,
                                fname = g_net_weight_pth )

            d_net_weight_pth = os.path.join(args.model_weight_dir,
                                                args.discrim_resume_path)
            utils.save_model(args,model_type = 'D_net', model = D_net,
                                optimizer = optim_D, iter_num = iter_num,
                                fname = d_net_weight_pth )

            if args.P_steps > 0:
                p_net_weight_pth = os.path.join(args.model_weight_dir,
                                                    args.priv_resume_path)
                utils.save_model(args,model_type = 'P_net', model = P_net,
                                    optimizer = optim_P, iter_num = iter_num,
                                    fname = p_net_weight_pth )
            if args.T_steps > 0:
                t_net_weight_pth = os.path.join(args.model_weight_dir,
                                                    args.task_resume_path)
                utils.save_model(args,model_type = 'T_net', model = T_net,
                                    optimizer = optim_T, iter_num = iter_num,
                                    fname = t_net_weight_pth )


            utils.save_loss_plots(loss_list = [G_losses, D_losses],
                            label_list = ["G", "D"],
                            title = "Generator and Discriminator Loss During Training",
                            out_fname = "loss_plots/Adversarial Loss Graph {0}".format(iter_num) )

            utils.save_loss_plots(loss_list = [T_losses, Pi_losses, Perc_losses],
                            label_list = ["T", "PI", "Perc"],
                            title = "Task, Privileged, and Perception Loss During Training",
                            out_fname = "loss_plots/Task, Priv, Perc Loss Graph {0}".format(iter_num) )



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    #general arguments
    parser.add_argument('--real_data', default = 'cityscapes')
    parser.add_argument('--model_weight_dir', default = 'model_weights')
    parser.add_argument('--warmup_resume_path', default = 'G_net_warmup.pth')
    parser.add_argument('--discrim_resume_path', default = 'D_net_weights.pth')
    parser.add_argument('--task_resume_path', default = 'T_net_weights.pth')
    parser.add_argument('--priv_resume_path', default = 'P_net_weights.pth')

    parser.add_argument('--resume_warmup', default = False, type = str2bool,
        help = 'Load the path of the warmed up generator')
    parser.add_argument('--resume_generator', default = False, type = str2bool)
    parser.add_argument('--resume_discrim', default = False, type = str2bool)
    parser.add_argument('--resume_task', default = False, type = str2bool)
    parser.add_argument('--resume_priv', default = False, type = str2bool)
    parser.add_argument('--save_every', default = 100, type = int)

    #experiment and hyperparameter arguments
    parser.add_argument('--debug_mode', default = False, type = str2bool,
        help = 'This checks that the weights of the models are being updated')
    parser.add_argument('--multi_gpu', default = False, type = str2bool)
    parser.add_argument('--batch_size', default = 1, type = int)
    parser.add_argument('--max_iters', default = 10000, type =int)
    parser.add_argument('--resize_input', default = (320,640))
    parser.add_argument('--num_workers', default = 0, type = int)
    parser.add_argument('--initial_lr', default = .0002, type = float)
    parser.add_argument('--gan_loss_weight', default = 1.0, type = float,
        help = 'the weighting for the adversarial loss.')
    parser.add_argument('--task_loss_weight', default = 0.5, type = float,
        help = 'the weighting for the task loss.')
    parser.add_argument('--priv_loss_weight', default = 0.1, type = float,
        help = 'the weighting for the privileged network loss.')
    parser.add_argument('--percep_loss_weight', default = 0.33, type = float,
        help = 'the weighting for the  perceptual regularization.')
    parser.add_argument('--warmup_lambda', default = 0.005, type=float,
        help= 'weight for the initial warmup')

    parser.add_argument('--G_steps', default = 1, type = int,
        help = 'Train the Generator for G_steps')
    parser.add_argument('--P_steps', default =1, type = int,
        help = 'Train the Privileged network for P_steps')
    parser.add_argument('--T_steps', default = 1, type = int,
        help = 'Train the Task network for T_steps')
    parser.add_argument('--D_steps', default = 5, type = int,
        help = 'Train the Discriminator for D_steps before the other networks')
    parser.add_argument('--gen_warmup_iters', default = 500, type = int,
        help = 'Warm up the generator for this many steps with regularization')

    parser.add_argument('--adversarial_loss', default = 'least_squares',
            type = str, help = 'loss function for the generator and discrim')
    parser.add_argument('--perceptual_loss', default = 'vgg_features',
            type = str, help = 'the features to use for the perceptual loss')
    parser.add_argument('--label_smoothing', type = str2bool, default = False,
            help = 'True will be a float between 0.7 - 1.2, False will be between (0, .3)')


    #### load data

    args = parser.parse_args()

    print (args)

    if args.resize_input == (320, 640):
        random_crop = (320,320)
    elif args.resize_input == (512,1024):
        random_crop = (400, 400)
    else:
        print ("Keep the resize inputs the same as the ones in the spigan paper.")
        exit()

    if args.real_data == 'cityscapes':

        real_dataloaders = utils.load_cityscape(args.batch_size,
                    args.resize_input, random_crop, args.num_workers)
        class_numbers = 16 #according to the paper, there are 16 common classes

    syn_dataloader = utils.load_synthia(args.batch_size, args.resize_input,
                    random_crop, args.num_workers)


    ### create networks

    D_net = patchGAN.NLayerDiscriminator(input_nc = 3) #RGB input
    G_net = resnetGen.ResnetGenerator(input_nc = 3,
                                    output_nc = 3) #RGB input and output
    T_net = fcn8.FCN8s(n_class = class_numbers)
    P_net = fcn8.FCN8s(n_class = 1) #depth is one class

    if args.perceptual_loss == 'vgg_features':
        if args.multi_gpu:
            vgg19 = torchvision.models.vgg19(pretrained=True)
            vgg19 = nn.DataParallel(vgg19, device_ids=[0,1])
            perceptual_feat_net = perceptual_loss.LossNetwork(vgg19, args.multi_gpu)
        else:
            vgg19 = torchvision.models.vgg19(pretrained=True)
            perceptual_feat_net = perceptual_loss.LossNetwork(vgg19, args.multi_gpu)

        if USE_CUDA:
            perceptual_feat_net = perceptual_feat_net.cuda()

    if USE_CUDA:
        D_net = D_net.cuda()
        G_net = G_net.cuda()
        T_net = T_net.cuda()
        P_net = P_net.cuda()

    if args.multi_gpu:
        T_net = nn.DataParallel(T_net, device_ids=[0,1])
        G_net = nn.DataParallel(G_net, device_ids=[0,1])
        P_net = nn.DataParallel(P_net, device_ids=[0,1])
        D_net = nn.DataParallel(D_net, device_ids=[0,1])

    G_net.apply(utils.weights_init)
    D_net.apply(utils.weights_init)


    optim_D = torch.optim.Adam(D_net.parameters(), lr = args.initial_lr)
    #optim_D = torch.optim.SGD(D_net.parameters(), lr = args.initial_lr)
    optim_G = torch.optim.Adam(G_net.parameters(), lr = args.initial_lr)
    optim_T = torch.optim.Adam(T_net.parameters(), lr = args.initial_lr)
    optim_P = torch.optim.Adam(P_net.parameters(), lr = args.initial_lr)


    if args.adversarial_loss == 'least_squares':
        adversarial_criterion = nn.MSELoss()
    else:
        adversarial_criterion = nn.BCEWithLogitsLoss()

    norm_l1_criterion = nn.L1Loss()
    tnet_criterion = nn.CrossEntropyLoss( ignore_index=255, reduction = 'mean')


    train(args, real_dataloaders['train'], syn_dataloader,
                D_net, G_net, T_net, P_net,
                optim_D, optim_G, optim_T, optim_P, perceptual_feat_net,
                adversarial_criterion, norm_l1_criterion, tnet_criterion)
