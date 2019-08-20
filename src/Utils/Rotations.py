import os
import sys
import torch
from math import *
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src import REPOSITORY_DIR

class Rotations(object):
	def __init__(self, angle_inc=12):
		self.loadSOI(angle_inc)
		print("Angle increment:", angle_inc)
		print("Number of rotations:", self.R.size(0))

	def writeMatrix(self, i, phi, theta, psi):
		with torch.no_grad():
			cpsi = cos(psi)
			spsi = sin(psi)
			ctheta = cos(theta)
			stheta = sin(theta)
			cphi = cos(phi)
			sphi = sin(phi)
			self.R[i,0,0] = cpsi*cphi  -  spsi*ctheta*sphi
			self.R[i,0,1] = -cpsi*sphi  -  spsi*ctheta*cphi
			self.R[i,0,2] = spsi*stheta

			self.R[i,1,0] = spsi*cphi  +  cpsi*ctheta*sphi
			self.R[i,1,1] = -spsi*sphi  +  cpsi*ctheta*cphi
			self.R[i,1,2] = -cpsi*stheta

			self.R[i,2,0] = stheta*sphi
			self.R[i,2,1] = stheta*cphi
			self.R[i,2,2] = ctheta

	def loadSOI(self, angle_inc):
		"""
		Loading SOI samples from SO(3) group:
		https://mitchell-lab.biochem.wisc.edu/SOI/index.php
		"""
		filename = os.path.join(REPOSITORY_DIR, 'data', 'oim%d.eul'%angle_inc)
		if not os.path.exists(filename):
			raise Exception("Can't find rotation angles:", filename)
		if angle_inc == 20:
			N = 1854
		elif angle_inc == 15:
			N = 4392
		elif angle_inc == 12:
			N = 8580
		elif angle_inc == 10:
			N = 14868
		elif angle_inc == 8:
			N = 29025
		elif angle_inc == 6:
			N = 68760
		else:
			raise Exception("Can't find rotation angles:", angle_inc)
		
		self.R = torch.zeros(N, 3, 3, device='cpu', dtype=torch.double)
		index = 0
		with open(filename, 'r') as fin:
			for line in fin:
				sline = line.split()
				phi = float(sline[0])
				theta = float(sline[1])
				psi = float(sline[2])
				self.writeMatrix(index, phi, theta, psi)
				index += 1


	def select_rotation(self, coords, num_atoms, threshold_lrmsd):
		
		batch_size = coords.size(0)
		rotations = torch.zeros(batch_size, 3, 3, device='cpu', dtype=torch.double)
		with torch.no_grad():
			for batch_idx in range(batch_size):
				Nat = num_atoms[batch_idx].item()
				Nrot = self.R.size(0)
				rcoords = (coords[batch_idx,:]).resize(int(coords.size(1)/3), 3)
				rcoords = rcoords[:Nat, :].to(dtype=torch.double)
				X = torch.zeros(Nrot, 3, 3, dtype=torch.double)
				for i in range(0,3):
					for j in range(0,3):
						X[:,i,j] = torch.sum(rcoords[:,i]*rcoords[:,j])/Nat
				
				R0 = torch.zeros(Nrot, 3, 3, dtype=torch.double, device='cpu')
				R0[:,0,0]=1.0
				R0[:,1,1]=1.0
				R0[:,2,2]=1.0
				rot_part = 2.0*(R0 - self.R)
				rmsd = torch.sqrt((rot_part * X).sum(dim=2).sum(dim=1))
				mask = torch.le(rmsd, threshold_lrmsd).unsqueeze(dim=1).unsqueeze(dim=2)
				num_selected = mask.sum().item()
				if num_selected == 0:
					raise Exception("No appropriate rotations")
				
				selected_rot = torch.masked_select(self.R, mask).contiguous()
				selected_rot = selected_rot.resize(num_selected, 3, 3).contiguous()

				selected_index = torch.randint(0, num_selected, (1,)).item()
				rotations[batch_idx, :, :] = selected_rot[selected_index, :, :]

		return rotations