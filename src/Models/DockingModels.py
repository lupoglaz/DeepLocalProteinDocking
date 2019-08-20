import os
import sys
import torch
from torch import nn
from torch.nn.modules.module import Module
import torch.nn.functional as F
import numpy as np

from se3cnn import SE3Convolution
from se3cnn.non_linearities import ScalarActivation
from .MultiplyVolumes import MultiplyVolumes

from TorchProteinLibrary.Volume import VolumeConvolution, VolumeRotation

import _Volume

def init_weights(m):
	if type(m) == nn.Conv3d:
		torch.nn.init.xavier_uniform_(m.weight)
	if type(m) == nn.Linear:
		torch.nn.init.xavier_uniform_(m.weight)

class SimpleFilter(Module):
	def __init__(self, inputs_sizes):
		super(SimpleFilter, self).__init__()

		self.fc_input_size = np.sum(inputs_sizes)
		self.fc = nn.Sequential(
			nn.Linear(self.fc_input_size, int(self.fc_input_size/2), bias=True),
			nn.ReLU(),
			nn.Linear(int(self.fc_input_size/2), 1, bias=True),	
		)

		self.fc.apply(init_weights)
	
	def forward(self, input):
		return self.fc(input)

class GlobalDockingModel(Module):
	def __init__(self, representation, filter, threshold_clash=300, normalize=False, rotate_ligand=False, exclude_clashes=True):
		super(GlobalDockingModel, self).__init__()
		
		self.threshold_clash = threshold_clash
		self.representation = representation
		self.filter = filter

		self.mult = MultiplyVolumes()
		self.convolve = VolumeConvolution(clip=5.0)
		self.vol_rotate = VolumeRotation()

		self.rotate_ligand = rotate_ligand
		self.normalize = normalize
		self.exclude = exclude_clashes

	def save(self, directory, epoch, model_name="DPD_Model"):
		torch.save(self.representation.state_dict(), os.path.join(directory, '%s_repr_epoch%d.th'%(model_name,epoch)))
		torch.save(self.filter.state_dict(), os.path.join(directory, '%s_filter_epoch%d.th'%(model_name,epoch)))
	
	def load(self, directory, epoch, model_name="DPD_Model"):
		self.representation.load_state_dict(torch.load(os.path.join(directory, '%s_repr_epoch%d.th'%(model_name,epoch))))
		self.filter.load_state_dict(torch.load(os.path.join(directory, '%s_filter_epoch%d.th'%(model_name,epoch))))	
	
	def forward(self, receptor_volumes, ligand_volumes):
		batch_size = receptor_volumes[0].size(0)
		prot_size = receptor_volumes[0].size(2)
		conv_size = prot_size*2
		
		#Convolutions
		convolved_volumes = []
		for i in range(len(receptor_volumes)):
			convolved_volumes.append(self.convolve(receptor_volumes[i], ligand_volumes[i]))
		
		#Scaling
		for i in range(len(convolved_volumes)):
			if convolved_volumes[i].size(2) < conv_size:
			   convolved_volumes[i] = torch.nn.functional.interpolate(convolved_volumes[i], size=(conv_size, conv_size, conv_size) ) 

		# Selection of translations
		V = torch.cat(convolved_volumes, dim=1)
		V = V.transpose(1,2).transpose(2,3).transpose(3,4).contiguous()
		V = V.resize(batch_size*conv_size*conv_size*conv_size, V.size(4)).contiguous()
		V = self.filter(V).squeeze()
		V = V.resize(batch_size, conv_size, conv_size, conv_size).contiguous()
		return V
		
class LocalDockingModel(Module):
	def __init__(self, representation, filter):
		super(LocalDockingModel, self).__init__()
		
		self.representation = representation
		self.filter = filter
		self.mult = MultiplyVolumes()
	
	def save(self, directory, epoch, model_name="DPD_Model"):
		torch.save(self.representation.state_dict(), os.path.join(directory, '%s_repr_epoch%d.th'%(model_name,epoch)))
		torch.save(self.filter.state_dict(), os.path.join(directory, '%s_filter_epoch%d.th'%(model_name,epoch)))
	
	def load(self, directory, epoch, model_name="DPD_Model"):
		self.representation.load_state_dict(torch.load(os.path.join(directory, '%s_repr_epoch%d.th'%(model_name,epoch))))
		self.filter.load_state_dict(torch.load(os.path.join(directory, '%s_filter_epoch%d.th'%(model_name,epoch))))
	
	def forward(self, receptor, ligand, T):
		batch_size = receptor.size(0)
		prot_size = receptor.size(2)
		
		receptor_volumes = self.representation(receptor)
		ligand_volumes = self.representation(ligand)

		w_res = []
		for receptor_vol, ligand_vol in zip(receptor_volumes, ligand_volumes):
			# Rescaling translations to the resolution of each volume
			vol_size = receptor_vol.size(2)
			T_res = T*float(vol_size)/float(prot_size)
			#Multiplying volumes with rescaled translations
			w_res.append(self.mult(receptor_vol, ligand_vol, T_res))
		
		# Selection of conformations
		w = torch.cat(w_res, dim=1)
		y = self.filter(w)
		return y
