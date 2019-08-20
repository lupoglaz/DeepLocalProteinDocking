import os
import sys
import torch

class VolumeViz(object):
	def __init__(self, resolution=1.25):
		self.resolution = resolution

	def plot_circ_volume(self, circ_volume, name='volume'):
		import _Volume
		L = circ_volume.size(1)
		L2 = int(L/2)
		output_volume = torch.zeros(L, L, L, device='cpu', dtype=torch.float)
		
		# quadrants 1 <--> 8
		output_volume[:L2, :L2, :L2] = circ_volume[L2:L, L2:L, L2:L]
		output_volume[L2:L, L2:L, L2:L] = circ_volume[:L2, :L2, :L2]

		# quadrants 2 <--> 9
		output_volume[L2:L, :L2, :L2] = circ_volume[:L2, L2:L, L2:L]
		output_volume[:L2, L2:L, L2:L] = circ_volume[L2:L, :L2, :L2]

		# quadrants 3 <--> 5
		output_volume[L2:L, L2:L, :L2] = circ_volume[:L2, :L2, L2:L]
		output_volume[:L2, :L2, L2:L] = circ_volume[L2:L, L2:L, :L2]

		# quadrants 4 <--> 6
		output_volume[:L2, L2:L, :L2] = circ_volume[L2:L, :L2, L2:L]
		output_volume[L2:L, :L2, L2:L] = circ_volume[:L2, L2:L, :L2]

		_Volume.Volume2Xplor(output_volume, "%s.xplor"%name, self.resolution)

	def plot_double_volume(self, volume, name='volume'):
		import _Volume
		L = volume.size(1)
		halfL = int(L/2)
		doubleL = L*2
		
		output_volume = torch.zeros(doubleL, doubleL, doubleL, device='cpu', dtype=torch.float)
		output_volume[L-halfL:L+halfL, L-halfL:L+halfL, L-halfL:L+halfL ] = volume[0:L, 0:L, 0:L]
		
		_Volume.Volume2Xplor(output_volume, "%s.xplor"%name, self.resolution)


	def overlap_plot(self, receptor, ligand, T, name='overlap'):
		import _Volume
		vt1 = receptor.sum(dim=0)
		vt2 = ligand.sum(dim=0)
		_Volume.Volume2Xplor(vt1.cpu(), "receptor_%s.xplor"%name, self.resolution)
		_Volume.Volume2Xplor(vt2.cpu(), "ligand_%s.xplor"%name, self.resolution)

		dx = int(T[0])
		dy = int(T[1])
		dz = int(T[2])

		L = self.box_size
		def mabs(dx, dy, dz):
			import numpy as np
			return np.abs(dx), np.abs(dy), np.abs(dz)

		#all positive
		if dx>0 and dy>0 and dz>0:
			dx, dy, dz = mabs(dx,dy,dz)
			vt1[dx:L, dy:L, dz:L] += vt2[0:L-dx, 0:L-dy, 0:L-dz]
		
		#one negative
		elif dx<0 and dy>0 and dz>0:
			dx, dy, dz = mabs(dx,dy,dz)
			vt1[0:L-dx, dy:L, dz:L] += vt2[dx:L, 0:L-dy, 0:L-dz]
		elif dx>0 and dy<0 and dz>0:
			dx, dy, dz = mabs(dx,dy,dz)
			vt1[dx:L, 0:L-dy, dz:L] += vt2[0:L-dx, dy:L, 0:L-dz]
		elif dx>0 and dy>0 and dz<0:
			dx, dy, dz = mabs(dx,dy,dz)
			vt1[dx:L, dy:L, 0:L-dz] += vt2[0:L-dx, 0:L-dy, dz:L]
		
		#one positive
		elif dx<0 and dy<0 and dz>0:
			dx, dy, dz = mabs(dx,dy,dz)
			vt1[0:L-dx, 0:L-dy, dz:L] += vt2[dx:L, dy:L, 0:L-dz]
		elif dx>0 and dy<0 and dz<0:
			dx, dy, dz = mabs(dx,dy,dz)
			vt1[dx:L, 0:L-dy, 0:L-dz] += vt2[0:L-dx, dy:L, dz:L]
		elif dx<0 and dy>0 and dz<0:
			dx, dy, dz = mabs(dx,dy,dz)
			vt1[0:L-dx, dy:L, 0:L-dz] += vt2[dx:L, 0:L-dy, dz:L]
		
		#all negative
		elif dx<0 and dy<0 and dz<0:
			dx, dy, dz = mabs(dx,dy,dz)
			vt1[0:L-dx, 0:L-dy, 0:L-dz] += vt2[dx:L, dy:L, dz:L]

		_Volume.Volume2Xplor(vt1.cpu(), "%s.xplor"%name, self.resolution)