import torch
import torch.nn as nn
import utils
import torchvision.utils as vutils
import numpy as np

def warmup_generator(args, USE_CUDA, syn_dataloader, G_net, optim_G, perceptual_feat_net, norm_l1_criterion):

    for warmup_step in range(args.gen_warmup_iters):
        #warm up the generator with perception loss. Use a different weighting for
        #the warm up process
        #self-regulariation loss ?
        #https://arxiv.org/pdf/1612.07828.pdf
        #could be a good idea

        #mse vs perceptual loss. Let the user decide.

        optim_G.zero_grad()

        fake_images, _, _,  _ = next(syn_dataloader)

        if args.debug_mode:
            a = list(G_net.parameters())[0].clone()
        if USE_CUDA:
            fake_images = fake_images.cuda()

        refined_images = G_net(fake_images)

        fake_features = perceptual_feat_net(fake_images)
        refined_features = perceptual_feat_net(refined_images)

        perceptual_feature_loss = 0.0

        for f, r in zip(fake_features, refined_features):
            perceptual_feature_loss += norm_l1_criterion(f,r)

        perceptual_feature_loss *= args.warmup_lambda
        perceptual_feature_loss.backward()
        optim_G.step()

        if args.debug_mode:

            b = list(G_net.parameters())[0].clone()

            #if_not_learning == True means that we are not learning
            #that means the network is not updating its parameters
            if_not_learning = torch.equal(a.data,b.data)
            assert (if_not_learning == False)

        if (warmup_step % 50) == 0:
            out_file_ims = 'debug_outputs/generator_warmup_outputs/{0}_image.png'.format(warmup_step)
            print ("Saving to file {0}".format(out_file_ims))
            utils.save_outputs(G_net, fake_images, out_file_ims)

    return G_net, optim_G

def train_discrim(args, USE_CUDA, syn_dataloader, real_dataloader, D_net, G_net, optim_D, adv_criterion):

    running_d_loss = 0.0
    for d_step in range(args.D_steps):

        #update the discriminator parameters

        utils.check_grads(D_net, "discriminator")

        optim_D.zero_grad()

        if args.debug_mode:
            a = list(D_net.parameters())[0].clone()

        #discriminator should be good at deciding between which images are real
        # and which images are refined.
        #refined images are those that are output from the G_net

        real_images, _, _ = next(real_dataloader) # B, C, H, W
        fake_images, _, _, _ = next(syn_dataloader) # B, C, H, W

        
        if USE_CUDA:
            real_images = real_images.cuda()
            fake_images = fake_images.cuda()

        #we generate refined images here
        #the discrimin is supposed to be good at predicting these as fake
        #images. That means the labels are zero

        refined_images_output = G_net(fake_images).detach()
        refined_images_output +=  torch.randn_like(refined_images_output)

        refined_decision = D_net(refined_images_output)


        fake_labels = torch.zeros(refined_decision.shape)

        if USE_CUDA:
            fake_labels = fake_labels.cuda()

        loss_D_refined = adv_criterion(refined_decision, fake_labels)
        loss_D_refined.backward()


        #we sample from the real images here
        #the discrim is supposed to be good at predicting these as real images
        #that means the labels are one

        real_images +=  torch.randn_like(real_images)
        real_decision = D_net(real_images)
        print (real_decision.requires_grad)

        if args.label_smoothing:
            #soft labels for discriminator on the real points
            real_labels = np.random.uniform(low=0.75,
                    high=1.2, size=real_decision.shape)
            real_labels = torch.from_numpy(real_labels).float()
        else:
            real_labels = torch.ones(real_decision.shape)

        if USE_CUDA:
            real_labels = real_labels.cuda()

        loss_D_real = adv_criterion(real_decision, real_labels)
        loss_D_real.backward()

        discrim_loss =  args.gan_loss_weight * (loss_D_real + loss_D_refined)
        optim_D.step()

        running_d_loss += discrim_loss.item()

        total_norm = 0.0
        for p in D_net.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)

        print (total_norm, 'Discrim norm')

        if args.debug_mode:
            b = list(D_net.parameters())[0].clone()
            #if_not_learning == True means that we are not learning
            #that means the network is not updating its parameters
            if_not_learning = torch.equal(a.data,b.data)
            assert (if_not_learning == False)

    running_d_loss /= float(args.D_steps)

    print (running_d_loss, 'd_losss')

    return D_net, optim_D, running_d_loss

def train_generator(args, USE_CUDA, syn_dataloader, G_net, D_net, optim_G, adv_criterion, norm_l1_criterion, perceptual_feat_net, real_dataloader):

    running_g_loss = 0.0
    running_perception_loss = 0.0
    for g_step in range(args.G_steps):
        #optimize generator

        #the generator wants to fool the discriminator
        #that means we want the refined outputs to trick the discriminiator
        #such that the labels from the discriminator are 1 (real)
        #note that the labels while training the discrim on the generator inputs
        #were 0 (fake)
        utils.check_grads(G_net, "generator")

        optim_G.zero_grad()

        if args.debug_mode:
            a = list(G_net.parameters())[0].clone()

        fake_images, _, _, _ = next(syn_dataloader) # B, C, H, W

        if USE_CUDA:
            fake_images = fake_images.cuda()

        refined_images = G_net(fake_images)
        refined_outputs = D_net(refined_images)

        labels_to_fool = torch.ones(refined_outputs.shape)

        if USE_CUDA:
            labels_to_fool = labels_to_fool.cuda()

        loss_G = adv_criterion(refined_outputs, labels_to_fool)

        #perceptual regularization term
        fake_features = perceptual_feat_net(fake_images)
        refined_features = perceptual_feat_net(refined_images)

        perceptual_feature_loss = 0.0

        for f, r in zip(fake_features, refined_features):
            perceptual_feature_loss += norm_l1_criterion(f,r)

        total_g_loss = ( args.gan_loss_weight * loss_G) + (args.percep_loss_weight * perceptual_feature_loss)

        total_g_loss.backward()
        optim_G.step()

        print (total_g_loss, 'generator loss')
        total_norm = 0.0
        for p in G_net.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)

        print (total_norm, 'gen norm')

        running_g_loss += (0.5 * loss_G).item()
        running_perception_loss += perceptual_feature_loss.item()

        if args.debug_mode:

            b = list(G_net.parameters())[0].clone()

            #if_not_learning == True means that we are not learning
            #that means the network is not updating its parameters
            if_not_learning = torch.equal(a.data,b.data)
            assert (if_not_learning == False)


    running_g_loss /= float(args.G_steps)
    running_perception_loss /= float(args.G_steps)

    return G_net, optim_G, running_g_loss, running_perception_loss

def train_pi_network(args, USE_CUDA, syn_dataloader, G_net, P_net, optim_P, norm_l1_criterion):

    running_priv_loss = 0.0
    for p_step in range(args.P_steps):
        #train the P network to predict depth
        #both x_s (fake_image) and G(x_s) (generated fake image) learn the depth
        #for y_s (fake_depth)
        optim_P.zero_grad()

        if args.debug_mode:
            a = list(P_net.parameters())[0].clone()

        fake_images, _, fake_depth, _ = next(syn_dataloader) # B, C, H, W

        if USE_CUDA:
            fake_images = fake_images.cuda()
            fake_depth = fake_depth.cuda()

        refined_images = G_net(fake_images).detach()
        refined_depth_output = P_net(refined_images) #change to disparity?
        loss_P_refined = norm_l1_criterion(refined_depth_output, fake_depth)

        syn_depth_output = P_net(fake_images)
        loss_P_syn = norm_l1_criterion(syn_depth_output, fake_depth)

        loss_P = args.priv_loss_weight * (loss_P_refined + loss_P_syn )
        loss_P.backward()

        optim_P.step()

        running_priv_loss += (loss_P.item() / args.priv_loss_weight)

        total_norm = 0.0
        for p in P_net.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)

        print (total_norm, 'P norm')

        if args.debug_mode:

            b = list(P_net.parameters())[0].clone()
            #if_not_learning == True means that we are not learning
            #that means the network is not updating its parameters
            if_not_learning = torch.equal(a.data,b.data)
            assert (if_not_learning == False)

    if args.P_steps == 0:
        running_priv_loss = 0
    else:
        running_priv_loss /= float(args.P_steps)

    return P_net, optim_P, running_priv_loss

def train_task_network(args, USE_CUDA, syn_dataloader, G_net, T_net, optim_T, tnet_criterion):

    running_t_loss = 0.0
    for t_step in range(args.T_steps):
        #Train T_network for semantic segmentation
        ##both x_s (fake_image) and G(x_s) (generated fake image) learn semseg
        #using y_s (the fake labels)

        optim_T.zero_grad()

        if args.debug_mode:
            a = list(T_net.parameters())[0].clone()

        fake_images, fake_labels, _, _ = next(syn_dataloader) # B, C, H, W

        if USE_CUDA:
            fake_images = fake_images.cuda()
            fake_labels = fake_labels.cuda()

        refined_images = G_net(fake_images).detach()

        refined_seg_output = T_net(refined_images)
        loss_T_refined = tnet_criterion(refined_seg_output, fake_labels)

        syn_seg_output = T_net(fake_images)
        loss_T_syn = tnet_criterion(syn_seg_output, fake_labels)

        loss_T = args.task_loss_weight * (loss_T_refined + loss_T_syn)
        loss_T.backward()

        optim_T.step()

        running_t_loss += (loss_T.item()/args.task_loss_weight)

        if args.debug_mode:

            b = list(T_net.parameters())[0].clone()
            #if_not_learning == True means that we are not learning
            #that means the network is not updating its parameters
            if_not_learning = torch.equal(a.data,b.data)
            assert (if_not_learning == False)

    if args.T_steps == 0:
        running_t_loss = 0
    else:
        running_t_loss /= float(args.T_steps)

    return T_net, optim_T, running_t_loss
