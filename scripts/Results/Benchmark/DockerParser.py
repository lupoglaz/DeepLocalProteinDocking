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

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pylab as plt
import mpl_toolkits.mplot3d.axes3d as p3
import seaborn as sea
sea.set_style("whitegrid")


sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src import REPOSITORY_DIR
from src.Utils import Rotations

class DockerParser:
	def __init__(self, box_size=80, resolution=1.5):
		
		self.box_size = box_size
		self.resolution = resolution
		self.box_length = box_size*resolution
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
				rot = torch.zeros(3,3, dtype=torch.double, device='cpu')
				t = torch.zeros(3, dtype=torch.double, device='cpu')
				rot[0,0] = float(sline[0])
				rot[0,1] = float(sline[1])
				rot[0,2] = float(sline[2])
				rot[1,0] = float(sline[3])
				rot[1,1] = float(sline[4])
				rot[1,2] = float(sline[5])
				rot[2,0] = float(sline[6])
				rot[2,1] = float(sline[7])
				rot[2,2] = float(sline[8])

				t[0] = int(float(sline[9]))
				t[1] = int(float(sline[10]))
				t[2] = int(float(sline[11]))

				score = float(sline[12])
				self.top_list.append( (rot.unsqueeze(dim=0), t.unsqueeze(dim=0), score) )
				self.cluster.append(-1)

		self.num_conf = len(self.top_list)
		print("Filename: ", filename)
		print("Number of conformations: ", self.num_conf)
	
	def load_protein(self, path_list, return_all=False, center=True):
		with torch.no_grad():
			coords, chainnames, resnames, resnums, atomnames, num_atoms = self.pdb2coords(path_list)
			if center:
				a,b = getBBox(coords, num_atoms)
				coords = self.translate(coords, -(a+b)*0.5, num_atoms)
		
		if return_all:
			return coords, chainnames, resnames, resnums, atomnames, num_atoms

		return coords, atomnames, num_atoms
	
	def cluster_decoys(self, ligand_path, num_clusters=10, cluster_threshold=15.0):
		lcoords, latomnames, lnum_atoms = self.load_protein([ligand_path])
				
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
						
		cluster_num = 0
		for i in range(self.num_conf):
			if self.cluster[i]>-1:
				continue
			else:
				self.cluster[i] = cluster_num
				print("Found %d cluster focus %d"%(cluster_num, i))

			ri, ti, scorei = self.top_list[i]
			ca_rot_i = self.rotate(ca_coords, ri, num_atoms_single)
			ca_rot_trans_i = self.translate(ca_rot_i, ti, num_atoms_single)

			for j in range(self.num_conf):
				if self.cluster[j]>-1:
					continue

				rj, tj, score = self.top_list[j]
				ca_rot_j = self.rotate(ca_coords, rj, num_atoms_single)
				ca_rot_trans_j = self.translate(ca_rot_j, tj, num_atoms_single)
				rmsd2 = ((ca_rot_trans_i - ca_rot_trans_j)*(ca_rot_trans_i - ca_rot_trans_j)).sum()
				lrmsd = torch.sqrt(rmsd2/num_ca_atoms).item()
				if lrmsd < cluster_threshold:
					self.cluster[j] = cluster_num
			
			cluster_num += 1
			if cluster_num > num_clusters:
				break

	def select_CA(self, coords, atomnames, num_atoms):
		N = num_atoms[0].item()
		is0C = torch.eq(atomnames[:,:,0], 67).squeeze()
		is1A = torch.eq(atomnames[:,:,1], 65).squeeze()
		is20 = torch.eq(atomnames[:,:,2], 0).squeeze()
		isCA = is0C*is1A*is20
		
		num_ca_atoms =  isCA.sum().item()
		num_atoms_single = torch.zeros(1, dtype=torch.int, device='cpu').fill_(num_ca_atoms)
		
		coords.resize_(1, N, 3)
		ca_x = torch.masked_select(coords[:,:,0], isCA)[:num_ca_atoms]
		ca_y = torch.masked_select(coords[:,:,1], isCA)[:num_ca_atoms]
		ca_z = torch.masked_select(coords[:,:,2], isCA)[:num_ca_atoms]
		ca_coords = torch.stack([ca_x, ca_y, ca_z], dim=1).resize_(1, num_ca_atoms*3).contiguous()
		
		return ca_coords

	def select_CA_resnums(self, coords, atomnames, resnums, select_resnums, num_atoms):
		N = num_atoms[0].item()
		is0C = torch.eq(atomnames[:,:,0], 67).squeeze()
		is1A = torch.eq(atomnames[:,:,1], 65).squeeze()
		is20 = torch.eq(atomnames[:,:,2], 0).squeeze()
		isCA = is0C*is1A*is20

		isSelectedResnums = torch.zeros(resnums.size(1), dtype=torch.uint8, device='cpu')
		for i in range(resnums.size(1)):
			if resnums[0, i].item() in select_resnums:
				isSelectedResnums[i] = 1
				
		isCA = isCA * isSelectedResnums
		
		num_ca_atoms =  isCA.sum().item()
		num_atoms_single = torch.zeros(1, dtype=torch.int, device='cpu').fill_(num_ca_atoms)
		
		coords = coords.reshape(1, N, 3)
		ca_x = torch.masked_select(coords[:,:,0], isCA)[:num_ca_atoms]
		ca_y = torch.masked_select(coords[:,:,1], isCA)[:num_ca_atoms]
		ca_z = torch.masked_select(coords[:,:,2], isCA)[:num_ca_atoms]
		ca_coords = torch.stack([ca_x, ca_y, ca_z], dim=1).reshape(1, num_ca_atoms*3).contiguous()
		
		return ca_coords

	def plot_3d(self, axis, coords, line_type, label):
		coords = coords.resize_(1, int(coords.size(1)/3), 3)
		sx, sy, sz = coords[0,:,0].numpy(), coords[0,:,1].numpy(), coords[0,:,2].numpy()
		axis.plot(sx,sy,sz, line_type, label = label)
		


	def get_hits(self, ureceptor_path, uligand_path, breceptor_path, bligand_path, receptor_int_nums, ligand_int_nums, irmsd_threshold=15.0):
		rmsd = Coords2RMSD()
		# fig = plt.figure()
		# ax = p3.Axes3D(fig)
		# plt.title("Interface")
		
		rcoords, rchainnames, rresnames, rresnums, ratomnames, rnum_atoms = self.load_protein([breceptor_path], return_all=True, center=False)
		lcoords, lchainnames, lresnames, lresnums, latomnames, lnum_atoms = self.load_protein([bligand_path], return_all=True, center=False)
		
		# self.plot_3d(ax, self.select_CA(rcoords, ratomnames, rnum_atoms), 'g--', label='bound_receptor')
		# self.plot_3d(ax, self.select_CA(lcoords, latomnames, lnum_atoms), 'r--', label='bound_ligand')

		brint_coords = self.select_CA_resnums(rcoords, ratomnames, rresnums, receptor_int_nums, rnum_atoms)
		blint_coords = self.select_CA_resnums(lcoords, latomnames, lresnums, ligand_int_nums, lnum_atoms)
		bound_interface = torch.cat([brint_coords, blint_coords], dim=1).contiguous()
				
		# self.plot_3d(ax, brint_coords, 'rx', label='bound_receptor')
		# self.plot_3d(ax, blint_coords, 'bx', label='bound_ligand')
		
		
		rcoords, rchainnames, rresnames, rresnums, ratomnames, rnum_atoms = self.load_protein([ureceptor_path], return_all=True, center=True)
		lcoords, lchainnames, lresnames, lresnums, latomnames, lnum_atoms = self.load_protein([uligand_path], return_all=True, center=True)
		urint_coords = self.select_CA_resnums(rcoords, ratomnames, rresnums, receptor_int_nums, rnum_atoms)
		ulint_coords = self.select_CA_resnums(lcoords, latomnames, lresnums, ligand_int_nums, lnum_atoms)
		lint_num_atoms = torch.zeros(1, dtype=torch.int, device='cpu').fill_(int(ulint_coords.size(1)/3))
		rint_num_atoms = torch.zeros(1, dtype=torch.int, device='cpu').fill_(int(urint_coords.size(1)/3))
		# unbound_interface = torch.cat([urint_coords, ulint_coords], dim=1)

		int_num_atoms = torch.zeros(1, dtype=torch.int, device='cpu').fill_(int(bound_interface.size(1)/3))

		# irmsd = rmsd(bound_interface, unbound_interface, int_num_atoms)
		# print(torch.sqrt(irmsd))
		hits = []

		for i in range(self.num_conf):
			ri, ti, scorei = self.top_list[i]
			ulint_coords_rot = self.rotate(ulint_coords, ri, lint_num_atoms)
			ulint_coords_rot_trans = self.translate(ulint_coords_rot, ti, lint_num_atoms)
			# self.plot_3d(ax, urint_coords, 'rx', label='unbound_receptor')
			# self.plot_3d(ax, ulint_coords, 'bx', label='unbound_ligand')
			
			lcoords_rot = self.rotate(lcoords, ri, lnum_atoms)
			lcoords_rot_trans = self.translate(lcoords_rot, ti, lnum_atoms)
			
			# self.plot_3d(ax, self.select_CA(rcoords, ratomnames, rnum_atoms), 'g--', label='unbound_receptor')
			# self.plot_3d(ax, self.select_CA(lcoords_rot_trans, latomnames, lnum_atoms), 'r--', label='unbound_ligand')
			
			
			# ax.legend()
			# plt.savefig("Test_result_interface.png")
			# return

			unbound_interface = torch.cat([urint_coords, ulint_coords_rot_trans], dim=1).contiguous()
			irmsd = rmsd(unbound_interface, bound_interface, int_num_atoms).item()
			hits.append((i, irmsd))

		return hits


			

	def save_clusters(self, complex_filename, receptor_path, ligand_path, num_clusters=10):
		pdb2coords = PDB2CoordsUnordered()
		num_conf = len(self.top_list)
		
		lcoords, lchainnames, lresnames, lresnums, latomnames, lnum_atoms = self.load_protein([ligand_path], True)
		rcoords, rchainnames, rresnames, rresnums, ratomnames, rnum_atoms = self.load_protein([receptor_path], True)
		
		cluster_num = 0
		for i in range(self.num_conf):
			if self.cluster[i]<cluster_num:
				continue

			elif self.cluster[i]==cluster_num:
				r, t, score = self.top_list[i]
				lcoords_rot = self.rotate(lcoords, r, lnum_atoms)
				lcoords_rot_trans = self.translate(lcoords_rot, t, lnum_atoms)
				otput_filename = complex_filename + '_%d.pdb'%cluster_num
				writePDB(otput_filename, rcoords, rchainnames, rresnames, rresnums, ratomnames, rnum_atoms, add_model=False, rewrite=True)
				writePDB(otput_filename, lcoords_rot_trans, lchainnames, lresnames, lresnums, latomnames, lnum_atoms, add_model=False, rewrite=False)
				
				cluster_num += 1
			
			else:
				raise(Exception("Wrong cluster number"))