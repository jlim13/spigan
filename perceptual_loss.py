import torch
import torch.nn as nn
import collections

# https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model, multi_gpu):
        super(LossNetwork, self).__init__()
        if multi_gpu:
            self.vgg_layers = vgg_model.module.features
        else:
            self.vgg_layers = vgg_model.features

        self.loss_output = collections.namedtuple("loss_output",
            ["conv1_2", "conv2_2", "conv3_2", "conv4_2", "conv5_2"])
        self.layer_name_mapping = {
            '2': "conv1_2",
            '7': "conv2_2",
            '14': "conv3_2",
            '21': "conv4_2",
            '23': "conv5_2"
        }


    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            
            x = module(x)
            if name in self.layer_name_mapping:

                output[self.layer_name_mapping[name]] = x


        return self.loss_output(**output)
