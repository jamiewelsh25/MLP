import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms

import mlp.data_providers as data_providers
from pytorch_mlp_framework.arg_extractor import get_args
from pytorch_mlp_framework.experiment_builder import ExperimentBuilder
from pytorch_mlp_framework.model_architectures import *

import os 
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

class ConvolutionalProcessingBlockBN(nn.Module):
    def __init__(self, input_shape, num_filters, kernel_size, padding, bias, dilation):
        super(ConvolutionalProcessingBlockBN, self).__init__()

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        self.padding = padding
        self.bias = bias
        self.dilation = dilation
        self.bn = nn.BatchNorm2d(input_shape[1])
        self.build_module()

    def build_module(self):
    	
        
        self.layer_dict = nn.ModuleDict()
        x = torch.zeros(self.input_shape)
        out = x

        self.layer_dict['conv_0'] = nn.Conv2d(in_channels=out.shape[1], out_channels=self.num_filters, bias=self.bias,
                                              kernel_size=self.kernel_size, dilation=self.dilation,
                                              padding=self.padding, stride=1)

        out = self.layer_dict['conv_0'].forward(out)
        out = self.bn(out)
        out = F.leaky_relu(out)

        self.layer_dict['conv_1'] = nn.Conv2d(in_channels=out.shape[1], out_channels=self.num_filters, bias=self.bias,
                                              kernel_size=self.kernel_size, dilation=self.dilation,
                                              padding=self.padding, stride=1)

        out = self.layer_dict['conv_1'].forward(out)
        out = self.bn(out)
        out = F.leaky_relu(out)

        print(out.shape)

    def forward(self, x):
        m = nn.BatchNorm2d(32)
        out = x

        out = self.layer_dict['conv_0'].forward(out)
        out = self.bn(out)
        out = F.leaky_relu(out)

        out = self.layer_dict['conv_1'].forward(out)
        out = self.bn(out)
        out = F.leaky_relu(out)

        return out

class ConvolutionalDimensionalityReductionBlockBN(nn.Module):
    def __init__(self, input_shape, num_filters, kernel_size, padding, bias, dilation, reduction_factor):
        super(ConvolutionalDimensionalityReductionBlockBN, self).__init__()

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        self.padding = padding
        self.bias = bias
        self.dilation = dilation
        self.reduction_factor = reduction_factor
        self.build_module()
        

    def build_module(self):
        m = nn.BatchNorm2d(32)
        self.layer_dict = nn.ModuleDict()
        x = torch.zeros(self.input_shape)
        out = x

        self.layer_dict['conv_0'] = nn.Conv2d(in_channels=out.shape[1], out_channels=self.num_filters, bias=self.bias,
                                              kernel_size=self.kernel_size, dilation=self.dilation,
                                              padding=self.padding, stride=1)
        
        out = self.layer_dict['conv_0'].forward(out)
        self.bn1 = nn.BatchNorm2d(out.shape[1])
        out = self.bn1(out)
        out = F.leaky_relu(out)
       

        out = F.avg_pool2d(out, self.reduction_factor)

        self.layer_dict['conv_1'] = nn.Conv2d(in_channels=out.shape[1], out_channels=self.num_filters, bias=self.bias,
                                              kernel_size=self.kernel_size, dilation=self.dilation,
                                              padding=self.padding, stride=1)

        out = self.layer_dict['conv_1'].forward(out)
        self.bn2 = nn.BatchNorm2d(out.shape[1])
        out = self.bn2(out)
        out = F.leaky_relu(out)

        print(out.shape)

    def forward(self, x):
        out = x

        out = self.layer_dict['conv_0'].forward(out)
        out = self.bn1(out)
        out = F.leaky_relu(out)

        out = F.avg_pool2d(out, self.reduction_factor)

        out = self.layer_dict['conv_1'].forward(out)
        out = self.bn2(out)
        out = F.leaky_relu(out)

        return out

args = get_args()  # get arguments from command line
rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment
torch.manual_seed(seed=args.seed)  # sets pytorch's seed

# set up data augmentation transforms for training and testing
transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_data = data_providers.CIFAR100(root='data', set_name='train',
                 transform=transform_train,
                 download=True)  # initialize our rngs using the argument set seed
val_data = data_providers.CIFAR100(root='data', set_name='val',
                 transform=transform_test,
                 download=True)  # initialize our rngs using the argument set seed
test_data = data_providers.CIFAR100(root='data', set_name='test',
                 transform=transform_test,
                 download=True)  # initialize our rngs using the argument set seed

train_data_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
val_data_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_data_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

if args.block_type == 'conv_block':
    processing_block_type = ConvolutionalProcessingBlock
    dim_reduction_block_type = ConvolutionalDimensionalityReductionBlock
elif args.block_type == 'empty_block':
    processing_block_type = EmptyBlock
    dim_reduction_block_type = EmptyBlock
elif args.block_type == 'conv_bn_block':
    processing_block_type = ConvolutionalProcessingBlockBN
    dim_reduction_block_type = ConvolutionalDimensionalityReductionBlockBN
elif args.block_type == 'conv_bn_rc_block':
    processing_block_type = ConvolutionalProcessingBlockBNRC
    dim_reduction_block_type = ConvolutionalDimensionalityReductionBlockBNRC
else:
    raise ModuleNotFoundError

custom_conv_net = ConvolutionalNetwork(  # initialize our network object, in this case a ConvNet
    input_shape=(args.batch_size, args.image_num_channels, args.image_height, args.image_width),
    num_output_classes=args.num_classes, num_filters=args.num_filters, use_bias=False,
    num_blocks_per_stage=args.num_blocks_per_stage, num_stages=args.num_stages,
    processing_block_type=processing_block_type,
    dimensionality_reduction_block_type=dim_reduction_block_type)

conv_experiment = ExperimentBuilder(network_model=custom_conv_net,
                                    experiment_name=args.experiment_name,
                                    num_epochs=args.num_epochs,
                                    weight_decay_coefficient=args.weight_decay_coefficient,
                                    use_gpu=args.use_gpu,
                                    continue_from_epoch=args.continue_from_epoch,
                                    train_data=train_data_loader, val_data=val_data_loader,
                                    test_data=test_data_loader)  # build an experiment object
experiment_metrics, test_metrics = conv_experiment.run_experiment()  # run experiment and return experiment metrics
