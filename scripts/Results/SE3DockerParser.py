import torch
import torch.nn as nn
import atexit
import numpy as np
import math
import os
import sys
from TorchProteinLibrary.FullAtomModel import CoordsTransform
from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered, getBBox, writePDB
from TorchProteinLibrary.RMSD import Coords2RMSD
from tqdm import tqdm
from math import *

import _Volume


sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src import REPOSITORY_DIR
from src.Utils import Rotations

class SE3DockerParser:
	def __init__(self, angle_inc=15, box_size=80, resolution=1.5):
		
		self.box_size = box_size
		self.resolution = resolution
		self.box_length = box_size*resolution
						
		self.rot = Rotations(angle_inc=angle_inc)
		self.rotate = CoordsTransform.CoordsRotate()
		self.translate = CoordsTransform.CoordsTranslate()
		
		self.box_center = torch.zeros(1, 3, dtype=torch.double, device='cpu')
		self.box_center.fill_(self.box_length/2.0)

		self.pdb2coords = PDB2CoordsUnordered()
				
	def load_list(self, filename):
		self.top_list = []
		self.cluster = []
		with open(filename) as fin:
			for line in fin:
				sline = line.split()
				ind = int(sline[0])
				ix = int(float(sline[1]))
				iy = int(float(sline[2]))
				iz = int(float(sline[3]))
				score = float(sline[4])
				self.top_list.append( (ind, ix, iy, iz, score) )
				self.cluster.append(-1)
		
		print("Filename: ", filename)
		print("Number of conformations: ", len(self.top_list))

	def cluster_decoys(self, ligand_path, num_clusters=10, cluster_threshold=15.0):

		pdb2coords = PDB2CoordsUnordered()
		rmsd = Coords2RMSD()
		num_conf = len(self.top_list)
		
		lcoords, lchainnames, lresnames, lresnums, latomnames, lnum_atoms = pdb2coords([ligand_path])
		a,b = getBBox(lcoords, lnum_atoms)
		lcoords = self.translate(lcoords, -(a+b)*0.5, lnum_atoms)
		t = torch.zeros(1,3,dtype=torch.double, device='cpu')
		
		N = lnum_atoms[0].item()
		is0C = torch.eq(latomnames[:,:,0], 67).squeeze()
		is1A = torch.eq(latomnames[:,:,1], 65).squeeze()
		is20 = torch.eq(latomnames[:,:,2], 0).squeeze()
		isCA = is0C*is1A*is20
		num_ca_atoms = isCA.sum().item()
		num_atoms_single = torch.zeros(1, dtype=torch.int, device='cpu').fill_(num_ca_atoms)
		
		lcoords.resize_(1, N, 3)
		ca_x = torch.masked_select(lcoords[:,:,0], isCA)[:num_ca_atoms]
		ca_y = torch.masked_select(lcoords[:,:,1], isCA)[:num_ca_atoms]
		ca_z = torch.masked_select(lcoords[:,:,2], isCA)[:num_ca_atoms]
		ca_coords = torch.stack([ca_x, ca_y, ca_z], dim=1).resize_(1, num_ca_atoms*3).contiguous()
						
		lrmsd = np.zeros((num_conf, num_conf))
		cluster_num = 0
		for i in range(num_conf):
			if self.cluster[i]>-1:
				continue
			else:
				self.cluster[i] = cluster_num
				print("Found %d cluster focus %d"%(cluster_num, i))

			ind, ix, iy, iz, score = self.top_list[i]
			r = self.rot.R[ind,:,:].unsqueeze(dim=0)
			t[0,0] = ix
			t[0,1] = iy
			t[0,2] = iz
			if ix >= self.box_size:
				t[0,0] = -(2*self.box_size - ix)
			if iy >= self.box_size:
				t[0,1] = -(2*self.box_size - iy)
			if iz >= self.box_size:
				t[0,2] = -(2*self.box_size - iz)
			
			ca_rot = self.rotate(ca_coords, r, num_atoms_single)
			ca_rot_trans_i = self.translate(ca_rot, t*self.resolution, num_atoms_single)

			for j in range(num_conf):
				if self.cluster[j]>-1:
					continue

				ind, ix, iy, iz, score = self.top_list[j]
				r = self.R[ind,:,:].unsqueeze(dim=0)
				t[0,0] = ix
				t[0,1] = iy
				t[0,2] = iz
				if ix >= self.box_size:
					t[0,0] = -(2*self.box_size - ix)
				if iy >= self.box_size:
					t[0,1] = -(2*self.box_size - iy)
				if iz >= self.box_size:
					t[0,2] = -(2*self.box_size - iz)
				
				ca_rot = self.rotate(ca_coords, r, num_atoms_single)
				ca_rot_trans_j = self.translate(ca_rot, t*self.resolution, num_atoms_single)
				rmsd2 = ((ca_rot_trans_i - ca_rot_trans_j)*(ca_rot_trans_i - ca_rot_trans_j)).sum()
				lrmsd = torch.sqrt(rmsd2/num_ca_atoms).item()
				if lrmsd < cluster_threshold:
					self.cluster[j] = cluster_num
			
			cluster_num += 1
			if cluster_num > num_clusters:
				break

	def save_clusters(self, complex_filename, receptor_path, ligand_path, num_clusters=10):
		pdb2coords = PDB2CoordsUnordered()
		num_conf = len(self.top_list)
		
		lcoords, lchainnames, lresnames, lresnums, latomnames, lnum_atoms = pdb2coords([ligand_path])
		a,b = getBBox(lcoords, lnum_atoms)
		lcoords = self.translate(lcoords, -(a+b)*0.5, lnum_atoms)
		t = torch.zeros(1,3,dtype=torch.double, device='cpu')

		rcoords, rchainnames, rresnames, rresnums, ratomnames, rnum_atoms = pdb2coords([receptor_path])
		a,b = getBBox(rcoords, rnum_atoms)
		rcoords = self.translate(rcoords, -(a+b)*0.5, rnum_atoms)
		
		cluster_num = 0
		for i in range(num_conf):
			if self.cluster[i]<cluster_num:
				continue

			elif self.cluster[i]==cluster_num:
				ind, ix, iy, iz, score = self.top_list[i]
				r = self.R[ind,:,:].unsqueeze(dim=0)
				lcoords_rot = self.rotate(lcoords, r, lnum_atoms)
				t[0,0] = ix
				t[0,1] = iy
				t[0,2] = iz
				if ix >= self.box_size:
					t[0,0] = -(2*self.box_size - ix)
				if iy >= self.box_size:
					t[0,1] = -(2*self.box_size - iy)
				if iz >= self.box_size:
					t[0,2] = -(2*self.box_size - iz)
				print(ind, t*self.resolution, score)
				lcoords_rot_trans = self.translate(lcoords_rot, t*self.resolution, lnum_atoms)
				otput_filename = complex_filename + '_%d.pdb'%cluster_num
				writePDB(otput_filename, rcoords, rchainnames, rresnames, rresnums, ratomnames, rnum_atoms, add_model=False, rewrite=True)
				writePDB(otput_filename, lcoords_rot_trans, lchainnames, lresnames, lresnums, latomnames, lnum_atoms, add_model=False, rewrite=False)
				
				cluster_num += 1
			
			else:
				raise(Exception("Wrong cluster number"))