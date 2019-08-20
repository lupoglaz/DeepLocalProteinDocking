import torch
from torch import nn
from torch.nn.modules.module import Module
import numpy as np

def mabs(dx, dy, dz):		
	return np.abs(dx), np.abs(dy), np.abs(dz)

class MultiplyVolumes(Module):
	def __init__(self):
		super(MultiplyVolumes, self).__init__()
	
	def multiply(self, v1, v2, dx, dy, dz, L):
		
		#all positive
		if dx>=0 and dy>=0 and dz>=0:
			dx, dy, dz = mabs(dx, dy, dz)
			result = v1[:,dx:L, dy:L, dz:L] * v2[:,0:L-dx, 0:L-dy, 0:L-dz]
		
		#one negative
		elif dx<0 and dy>=0 and dz>=0:
			dx, dy, dz = mabs(dx, dy, dz)
			result = v1[:,0:L-dx, dy:L, dz:L] * v2[:,dx:L, 0:L-dy, 0:L-dz]
		elif dx>=0 and dy<0 and dz>=0:
			dx, dy, dz = mabs(dx, dy, dz)
			result = v1[:,dx:L, 0:L-dy, dz:L] * v2[:,0:L-dx, dy:L, 0:L-dz]
		elif dx>=0 and dy>=0 and dz<0:
			dx, dy, dz = mabs(dx, dy, dz)
			result = v1[:,dx:L, dy:L, 0:L-dz] * v2[:,0:L-dx, 0:L-dy, dz:L]
		
		#one positive
		elif dx<0 and dy<0 and dz>=0:
			dx, dy, dz = mabs(dx, dy, dz)
			result = v1[:,0:L-dx, 0:L-dy, dz:L] * v2[:,dx:L, dy:L, 0:L-dz]
		elif dx>=0 and dy<0 and dz<0:
			dx, dy, dz = mabs(dx, dy, dz)
			result = v1[:,dx:L, 0:L-dy, 0:L-dz] * v2[:,0:L-dx, dy:L, dz:L]
		elif dx<0 and dy>=0 and dz<0:
			dx, dy, dz = mabs(dx, dy, dz)
			result = v1[:,0:L-dx, dy:L, 0:L-dz] * v2[:,dx:L, 0:L-dy, dz:L]
		
		#all negative
		elif dx<0 and dy<0 and dz<0:
			dx, dy, dz = mabs(dx, dy, dz)
			result = v1[:,0:L-dx, 0:L-dy, 0:L-dz] * v2[:,dx:L, dy:L, dz:L]

		return result.sum(dim=3).sum(dim=2).sum(dim=1).squeeze()

	def forward(self, receptor, ligand, T):
		batch_size = receptor.size(0)
		L = receptor.size(2)
		mults = []
		for i in range(batch_size):
			v1 = receptor[i,:,:,:,:].squeeze()
			v2 = ligand[i,:,:,:,:].squeeze()
			dx = int(T[i,0])
			dy = int(T[i,1])
			dz = int(T[i,2])
			mults.append(self.multiply(v1,v2,dx,dy,dz,L))
		return torch.stack(mults, dim=0)
		