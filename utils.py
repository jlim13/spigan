import torchvision
import torch
import torchvision.utils as vutils
import torch.nn as nn
from torch.nn import init

from PIL import Image
import numpy as np
from data import Cityscapes, Synthia

import matplotlib.pyplot as plt

color_to_label = {
            0:[128,64,128],
            1:[244,35,232],
            2:[70,70,70],
            3:[102,102,156],
            4:[190,153,153],
            5:[153,153,153],
            6:[250,170,30],
            7:[220,220,0],
            8:[107,142,35],
            9:[70,130,180],
            10:[220,20,60],
            11:[255,0,0],
            12:[0,0,142],
            13:[0,60,100],
            14:[0,0,230],
            15:[119,11,32],
            16: [0,0,0]}

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def load_cityscape(batch_size, resize_input, random_crop, num_workers):


    transform_info = {'resize': resize_input, 'random_crop': random_crop}

    loaders = {}


    for dataset in ['train', 'test', 'val']:

        cityscapes_data = Cityscapes(
                            root = 'data/mapped_cityscapes',
                            split=dataset,
                            mode='fine',
                            target_type='semantic',
                            transforms = transform_info)

        cityscapes_data = torch.utils.data.DataLoader(cityscapes_data,
                              batch_size=batch_size,
                              shuffle=True, num_workers = num_workers)

        dataiter = iter(cycle(cityscapes_data))
        loaders[dataset] = dataiter

    return loaders

def load_synthia(batch_size, resize_input, random_crop, num_workers):

    synthia_data = Synthia(
                root = 'data/RAND_CITYSCAPES/',
                crop_size = resize_input,
                random_crop = random_crop)

    synthia_data = torch.utils.data.DataLoader(synthia_data,
                          batch_size=batch_size,
                          shuffle=True, num_workers = num_workers)


    dataiter = iter(cycle(synthia_data))

    return dataiter

def save_loss_plots(loss_list, label_list, title, out_fname):
    plt.figure(figsize=(10,5))
    plt.title(title)

    for loss, label in zip(loss_list, label_list):

        plt.plot(loss, label = label)

    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(out_fname)

def save_model(args, model_type, model, optimizer, iter_num, fname):

    if args.multi_gpu:
        torch.save({
            'model_type' : model_type,
            '{0}_params'.format(model_type): model.module.state_dict(),
            '{0}_optim_params'.format(model_type) : optimizer.state_dict(),
            'iter_num': iter_num}, fname)
    else:
        torch.save({
            'model_type' : model_type,
            '{0}_params'.format(model_type): model.state_dict(),
            '{0}_optim_params'.format(model_type) : optimizer.state_dict(),
            'iter_num': iter_num}, fname)

def load_model(args, model_type, model, optimizer,  fname, USE_CUDA):

    model.module.load_state_dict(torch.load(fname)['{0}_params'.format(model_type)])
    optimizer.load_state_dict(torch.load(fname)['{0}_optim_params'.format(model_type)])
    if USE_CUDA:
        model = model.cuda()
    iter_num = torch.load(fname)['iter_num']
    print ("The weights were loaded from {0}".format(iter_num))

    return model, optimizer, iter_num

def save_outputs(network, network_input, out_file, additional_images = None):

    with torch.no_grad():
        network_output = network(network_input)

    to_output = [network_output, network_input]

    if additional_images:
        for im in additional_images:
            to_output.append(im)
    out_im = torch.cat(to_output, dim = 0).detach().cpu()
    grid = vutils.make_grid(out_im, padding=2, nrow= 5, normalize=True)

    torchvision.utils.save_image(grid, out_file, nrow=5, padding=2,
        normalize=False, range=None, scale_each=False, pad_value=0)

def weights_init(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def color_segmentation_outputs(segmentation_ims, color_to_label):

    '''
    Expecting a tensor of size of B x C x H x W

    '''

    color_ims = []

    for sim in segmentation_ims:

        sim = sim.numpy().squeeze()
        color_im = np.zeros(( sim.shape[0], sim.shape[1],3))
        #color_im is H x W x C

        for label in np.unique(sim):
            color = color_to_label[label]
            idxs = sim == label

            color_im[idxs] = color

        color_im = torch.from_numpy(np.transpose(color_im, (2, 0, 1))).float()
        color_ims.append(color_im)

    return color_ims

def save_generator_outputs(USE_CUDA, G_net, iter_num, real_dataloader, syn_dataloader ):
    out_file = 'debug_outputs/generator_training_outputs/{0}_image.png'.format(iter_num)
    print ("Saving to file {0}".format(out_file))

    real_images, _, _ = next(real_dataloader) # B, C, H, W
    syn_images, _, _, _ = next(syn_dataloader)
    if USE_CUDA:
        real_images = real_images.cuda()
        syn_images = syn_images.cuda()

    save_outputs(G_net, syn_images, out_file, additional_images = [real_images])

def save_depth_outputs(USE_CUDA, P_net, G_net, iter_num, real_dataloader, syn_dataloader):
    out_file = 'debug_outputs/depth_training_outputs/{0}_image.png'.format(iter_num)
    print ("Saving to file {0}".format(out_file))

    real_images, _, _ = next(real_dataloader) # B, C, H, W
    syn_images, _, _, _ = next(syn_dataloader)

    if USE_CUDA:
        real_images = real_images.cuda()
        syn_images = syn_images.cuda()

    with torch.no_grad():
        refined_images = G_net(syn_images)
        refined_output = P_net(refined_images)
        syn_output = P_net(syn_images)
        # real_output = P_net(real_images)


    outputs = [refined_output, syn_output]
    inputs = [refined_images, syn_images]

    out_im = torch.cat(outputs, dim = 0).detach().cpu()
    in_im = torch.cat(inputs, dim = 0).detach().cpu()

    out_grid = vutils.make_grid(out_im, padding=2, nrow= 2, normalize=True)
    in_grid = vutils.make_grid(in_im, padding=2, nrow= 2, normalize=True)

    torchvision.utils.save_image([out_grid, in_grid], out_file, nrow=2, padding=2,
        normalize=False, range=None, scale_each=False, pad_value=0)

def save_segmentation_outputs(USE_CUDA, P_net, G_net, iter_num, real_dataloader, syn_dataloader):
    out_file = 'debug_outputs/segmentation_training_outputs/{0}_image.png'.format(iter_num)
    print ("Saving to file {0}".format(out_file))

    real_images, _, _ = next(real_dataloader) # B, C, H, W
    syn_images, _, _, _ = next(syn_dataloader)

    if USE_CUDA:
        real_images = real_images.cuda()
        syn_images = syn_images.cuda()

    with torch.no_grad():
        refined_images = G_net(syn_images)
        refined_output = P_net(refined_images)
        syn_output = P_net(syn_images)
        syn_preds = torch.argmax(syn_output, dim=1).int().unsqueeze(1)
        refined_preds = torch.argmax(refined_output, dim =1).int().unsqueeze(1)


    outputs = [refined_preds, syn_preds]
    inputs = [refined_images, syn_images]

    out_im = torch.cat(outputs, dim = 0).detach().cpu()
    in_im = torch.cat(inputs, dim = 0).detach().cpu()

    out_im = color_segmentation_outputs(out_im, color_to_label)

    out_grid = vutils.make_grid(out_im, padding=2, nrow= 2, normalize=True)
    in_grid = vutils.make_grid(in_im, padding=2, nrow= 2, normalize=True)

    torchvision.utils.save_image([out_grid, in_grid], out_file, nrow=2, padding=2,
        normalize=False, range=None, scale_each=False, pad_value=0)

def check_grads(model, model_name):
    grads = []
    for p in model.parameters():
        if not p.grad is None:
            grads.append(float(p.grad.mean()))

    grads = np.array(grads)
    print (grads, model_name)
    if grads.any() and grads.mean() > 100:
        print(f"WARNING! gradients mean is over 100 ({model_name})")
    if grads.any() and grads.max() > 100:
        print(f"WARNING! gradients max is over 100 ({model_name})")
