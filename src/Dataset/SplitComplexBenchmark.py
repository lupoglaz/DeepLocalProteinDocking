# -*- coding: utf-8 -*-
"""

"""

import itertools
import os
import sys
import torch
from torch.utils.data import Dataset

import random
random.seed(42)

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src import DATA_DIR, REPOSITORY_DIR



def read_pdb_list(benchmark_dir, pdb_list_file, struct_folder="structures"):
	targets = []
	cplx_type = 0
	with open(pdb_list_file) as fin:
		for line in fin:
			if line.find("Rigid-body")!=-1:
				cplx_type = 1
				continue
			elif line.find("Medium Difficulty")!=-1:
				cplx_type = 2
				continue
			elif line.find("Difficult")!=-1:
				cplx_type = 3
				continue

			if cplx_type>0:
				sline = line.split('\t')
				complex_name = sline[0]
				pdb_name = complex_name.split('_')[0]
				ureceptor_path = os.path.join(benchmark_dir, struct_folder, pdb_name+'_r_u.pdb')
				uligand_path = os.path.join(benchmark_dir, struct_folder, pdb_name+'_l_u.pdb')
				breceptor_path = os.path.join(benchmark_dir, struct_folder, pdb_name+'_r_b.pdb')
				bligand_path = os.path.join(benchmark_dir, struct_folder, pdb_name+'_l_b.pdb')
				native_path = os.path.join(benchmark_dir, 'Matched', pdb_name+'_b.pdb')
				targets.append((pdb_name, native_path, ureceptor_path, uligand_path, breceptor_path, bligand_path, cplx_type))	

	return targets

def read_dataset_list(benchmark_dir, pdb_list_file):
	targets = []
	cplx_type = 0
	with open(pdb_list_file) as fin1:
		for line in fin1:
			sline = line.split()
			pdb_name = sline[0]
			decoys_list_file = os.path.join(benchmark_dir, 'Description', pdb_name + '.dat')
			with open(decoys_list_file) as fin2:
				header = fin2.readline()
				sline = fin2.readline().split()
				receptor = sline[0]
				ligand = sline[1]
			native_path = os.path.join(benchmark_dir, 'Structures', pdb_name+'.pdb')
			targets.append((pdb_name, native_path, receptor, ligand, receptor, ligand, cplx_type))
	
	return targets


class SplitComplexBenchmark(Dataset):
	"""
	The dataset that loads protein complexes
	"""
	def __init__(self, dataset_dir, struct_folder="structures", description_set='Table_BM5.csv', debug=False):
		"""
		Loads dataset description
		@Arguments:
		dataset_dir: path to the dataset folder
		description_dir: description folder name
		description_set: the subset file
		"""
		self.dataset_dir = dataset_dir
		
		
		if debug:
			self.targets = read_dataset_list(dataset_dir, os.path.join(dataset_dir, 'Description', description_set))
		else:
			self.targets = read_pdb_list(dataset_dir, os.path.join(dataset_dir, description_set), struct_folder=struct_folder)

		self.dataset_size = len(self.targets)

		print ("Dataset file: ", self.dataset_dir)
		print ("Dataset size: ", self.dataset_size)

		
			
	
	# @profile
	
	
	def __getitem__(self, index):
		"""
		Returns coordinates, relative translation vector, relative rotation matrix
		"""
		pdb_name, native_path, ureceptor, uligand, breceptor, bligand, cplx = self.targets[index]
		
		return pdb_name, native_path, ureceptor, uligand, breceptor, bligand, cplx

		
	def __len__(self):
		"""
		Returns length of the dataset
		"""
		return self.dataset_size
		

def get_benchmark_stream(data_dir, struct_folder="structures", subset='Table_BM5.csv', debug=False):
	dataset = SplitComplexBenchmark(data_dir, struct_folder=struct_folder, description_set=subset, debug=debug)
	trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
	return trainloader


if __name__=='__main__':
	"""
	Testing data load procedure
	"""
	sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
	from src import DATA_DIR

	from matplotlib import pylab as plt
	local_data_dir = '/media/lupoglaz/ProteinsDataset/Docking/benchmark5'
	dataiter = iter(get_dataset_stream(local_data_dir, box_size=80, resolution=1.5))
	pdb_name, rcoords, rnat, roff, lcoords, lnat, loff, T, cplx = dataiter.next()
	print(pdb_name)
	print(rcoords.size())
	print(rnat.size())
	print(roff.size())
	print(T)
	print(cplx)
	# import _Volume

	# volume1, volume2, dT, labels, receptor, ligand = dataiter.next()
	# print(receptor)
	# print(ligand)
	# dT = dT.squeeze()
	# volume1 = volume1.squeeze()
	# volume2 = volume2.squeeze()
	# labels = labels.squeeze()
	
	# vt1 = volume1[0,:,:,:,:].squeeze()
	# vt2 = volume2[0,:,:,:,:].squeeze()
	# vt1 = vt1.sum(dim=0)
	# vt2 = vt2.sum(dim=0)
	
	# dx = int(dT[0,0])
	# dy = int(dT[0,1])
	# dz = int(dT[0,2])
	
	# _Volume.Volume2Xplor(vt1.cpu(), "receptor.xplor")
	# _Volume.Volume2Xplor(vt2.cpu(), "ligand.xplor")
	
	# print(dx,dy,dz)
	# L = 80
	# def mabs(dx, dy, dz):
	# 	import numpy as np
	# 	return np.abs(dx), np.abs(dy), np.abs(dz)

	# #all positive
	# if dx>0 and dy>0 and dz>0:
	# 	dx, dy, dz = mabs(dx,dy,dz)
	# 	vt1[dx:L, dy:L, dz:L] += vt2[0:L-dx, 0:L-dy, 0:L-dz]
	
	# #one negative
	# elif dx<0 and dy>0 and dz>0:
	# 	dx, dy, dz = mabs(dx,dy,dz)
	# 	vt1[0:L-dx, dy:L, dz:L] += vt2[dx:L, 0:L-dy, 0:L-dz]
	# elif dx>0 and dy<0 and dz>0:
	# 	dx, dy, dz = mabs(dx,dy,dz)
	# 	vt1[dx:L, 0:L-dy, dz:L] += vt2[0:L-dx, dy:L, 0:L-dz]
	# elif dx>0 and dy>0 and dz<0:
	# 	dx, dy, dz = mabs(dx,dy,dz)
	# 	vt1[dx:L, dy:L, 0:L-dz] += vt2[0:L-dx, 0:L-dy, dz:L]
	
	# #one positive
	# elif dx<0 and dy<0 and dz>0:
	# 	dx, dy, dz = mabs(dx,dy,dz)
	# 	vt1[0:L-dx, 0:L-dy, dz:L] += vt2[dx:L, dy:L, 0:L-dz]
	# elif dx>0 and dy<0 and dz<0:
	# 	dx, dy, dz = mabs(dx,dy,dz)
	# 	vt1[dx:L, 0:L-dy, 0:L-dz] += vt2[0:L-dx, dy:L, dz:L]
	# elif dx<0 and dy>0 and dz<0:
	# 	dx, dy, dz = mabs(dx,dy,dz)
	# 	vt1[0:L-dx, dy:L, 0:L-dz] += vt2[dx:L, 0:L-dy, dz:L]
	
	# #all negative
	# elif dx<0 and dy<0 and dz<0:
	# 	dx, dy, dz = mabs(dx,dy,dz)
	# 	vt1[0:L-dx, 0:L-dy, 0:L-dz] += vt2[dx:L, dy:L, dz:L]

	# _Volume.Volume2Xplor(vt1.cpu(), "complex.xplor")

	

	



	
	
	
	

	