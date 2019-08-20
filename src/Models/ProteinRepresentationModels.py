import torch
from torch import nn
from torch.nn.modules.module import Module
import torch.nn.functional as F
import numpy as np

from se3cnn import SE3Convolution
from se3cnn.non_linearities import ScalarActivation
from se3cnn.blocks import GatedBlock
from .MultiplyVolumes import MultiplyVolumes
from se3cnn.filter import low_pass_filter

from TorchProteinLibrary.Volume import VolumeConvolution, VolumeRotation

import _Volume

def init_weights(m):
	if type(m) == nn.Conv3d:
		torch.nn.init.xavier_uniform_(m.weight)
	if type(m) == nn.Linear:
		torch.nn.init.xavier_uniform_(m.weight)

class SE3MultiResReprScalar(Module):
	def __init__(self, num_input_channels=11, multiplier=16):
		super(SE3MultiResReprScalar, self).__init__()

		common_block_params = {
			'size': 5,
			'padding': 2,
			'dyn_iso': True,
			#'smooth_stride': True,
			'bias': None
		}

		self.num_outputs_res0 = multiplier*2
		self.num_outputs_res1 = multiplier*4
		
		self.sequence_res0 = torch.nn.Sequential(
			SE3Convolution([(11,0)], [(multiplier*2,0)], **common_block_params),
			ScalarActivation([(multiplier*2, F.relu)], bias=False),

			SE3Convolution([(multiplier*2,0)], [(multiplier*2,0)], **common_block_params),
			ScalarActivation([(multiplier*2, F.relu)], bias=False),

			SE3Convolution([(multiplier*2,0)], [(multiplier*2,0)], **common_block_params),
			ScalarActivation([(multiplier*2, F.relu)], bias=False),

			SE3Convolution( [(multiplier*2,0)], [(self.num_outputs_res0,0)], **common_block_params)
		)
		self.sequence_res1 = torch.nn.Sequential(
			SE3Convolution([(multiplier*2,0)], [(multiplier*4,0)], **common_block_params, stride=2),
			ScalarActivation([(multiplier*4, F.relu)], bias=False),

			SE3Convolution([(multiplier*4,0)], [(multiplier*4,0)], **common_block_params),
			ScalarActivation([(multiplier*4, F.relu)], bias=False),

			SE3Convolution([(multiplier*4,0)], [(multiplier*4,0)], **common_block_params),
			ScalarActivation([(multiplier*4, F.relu)], bias=False),

			SE3Convolution( [(multiplier*4,0)], [(self.num_outputs_res1,0)], **common_block_params)
		)
		
		self.last_grad = torch.zeros(2, device='cpu', dtype=torch.float)
		def grad_hook(module, grad_input, grad_output):
			self.last_grad[0] = torch.min(grad_output[0])
			self.last_grad[1] = torch.max(grad_output[0])
		self.sequence_res0.register_backward_hook(grad_hook)

	def get_num_outputs(self):
		return [self.num_outputs_res0, self.num_outputs_res1]

	def forward(self, volume):
		# volume = low_pass_filter(volume, 2)
		vol1 = self.sequence_res0(volume)
		vol2 = self.sequence_res1(vol1)
		return [vol1, vol2]

class E3MultiResRepr4x4(Module):
	def __init__(self, num_input_channels=11, multiplier=16):
		super(E3MultiResRepr4x4, self).__init__()
		
		self.num_outputs_res0 = multiplier*2
		self.num_outputs_res1 = multiplier*4

		self.conv1 = nn.Sequential(
			nn.Conv3d(num_input_channels, multiplier*2, kernel_size=5, padding=2, bias=False),
			nn.ReLU(),
			
			nn.Conv3d(multiplier*2, multiplier*2, kernel_size=3, padding=1, bias=False), 
			nn.ReLU(),

			nn.Conv3d(multiplier*2, multiplier*2, kernel_size=3, padding=1, bias=False), 
			nn.ReLU(),

			nn.Conv3d(multiplier*2, multiplier*2, kernel_size=3, padding=1, bias=False), 
			nn.ReLU(),

			nn.Conv3d(multiplier*2, multiplier*2, kernel_size=5, padding=2, bias=False)
		)

		self.conv2 = nn.Sequential(
			torch.nn.MaxPool3d(kernel_size=5, stride=2, padding=2),

			nn.Conv3d(multiplier*2, multiplier*4, kernel_size=5, padding=2, bias=False),
			nn.ReLU(),
			
			nn.Conv3d(multiplier*4, multiplier*4, kernel_size=3, padding=1, bias=False), 
			nn.ReLU(),

			nn.Conv3d(multiplier*4, multiplier*4, kernel_size=3, padding=1, bias=False), 
			nn.ReLU(),

			nn.Conv3d(multiplier*4, multiplier*4, kernel_size=3, padding=1, bias=False)
		)
		
		
		self.last_grad = torch.zeros(2, device='cpu', dtype=torch.float)
		def grad_hook(module, grad_input, grad_output):
			self.last_grad[0] = torch.min(grad_output[0])
			self.last_grad[1] = torch.max(grad_output[0])
		self.conv1.register_backward_hook(grad_hook)

	def get_num_outputs(self):
		return [self.num_outputs_res0, self.num_outputs_res1]

	def forward(self, volume):
		vol1 = self.conv1(volume)
		vol2 = self.conv2(vol1)
		return [vol1, vol2]