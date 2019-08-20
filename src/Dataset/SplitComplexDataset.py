# -*- coding: utf-8 -*-
"""

"""

import itertools
import os
import sys
import torch
from torch.utils.data import Dataset
import numpy as np
import random
random.seed(42)

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src import DATA_DIR, REPOSITORY_DIR

from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered, Coords2TypedCoords, CoordsTranslate, getBBox
from TorchProteinLibrary.Volume import TypedCoords2Volume

class SplitComplexDataset(Dataset):
	"""
	The dataset that loads protein complexes
	"""
	def __init__(self, field, threshold, dataset_dir, description_dir='Description', description_set='datasetDescription.dat'):
		"""
		Loads dataset description
		@Arguments:
		dataset_dir: path to the dataset folder
		description_dir: description folder name
		description_set: the subset file
		batch_size: num examples in one output
		threshold: l-rmsd threshold between positive and negative examples
		"""
		self.dataset_dir = dataset_dir
		self.description_path = os.path.join(dataset_dir, description_dir, description_set)
		self.targets = []
		self.field = field
		self.threshold = threshold

		with open(self.description_path, 'r') as fin:
			for line in fin:
				target_name = line.split()[0]
				self.targets.append(target_name)

		self.decoys = {}
		self.indexed = []

		for target in self.targets:
			self.decoys[target] = []
			target_file = os.path.join(dataset_dir, description_dir, '%s.dat'%target)
			with open(target_file, 'r') as fin:
				fields = fin.readline()
				fields = fields.split()
				for line in fin:
					sline = line.split()
					decoy_description = {}
					for n, field in enumerate(fields):
						try:
							decoy_description[field] = float(sline[n])
						except:
							aline = sline[n].split('/')
							decoys_dir = aline[-2]
							decoy_file = aline[-1]
							decoy_description[field] = os.path.join(self.dataset_dir, decoys_dir, decoy_file)
								
					self.decoys[target].append(decoy_description)
					self.indexed.append(decoy_description)
					

		self.dataset_size = len(list(self.decoys.keys()))

		print ("Dataset file: ", self.dataset_dir)
		print ("Dataset size: ", self.dataset_size)
		
	def __getitem__(self, index):
		"""
		Returns volumes, relative translation and gdt-ts
		"""
		target_index, decoy_index = index
		target = self.targets[target_index]
		decoy = self.decoys[target][decoy_index]
		# print(target, decoy["receptor"], decoy[self.field])

		receptor = decoy["receptor"]
		ligand = decoy["ligand"]
		label = torch.zeros(1, dtype=torch.float, device='cuda')
		label[0] = decoy[self.field]
		# if decoy[self.field] >= self.threshold:
		# 	label[0] = 1.0
		# else:
		# 	label[0] = 0.0
				
		return receptor, ligand, label

		
	def __len__(self):
		"""
		Returns length of the dataset
		"""
		return self.dataset_size


class BatchSamplerQuality(object):

	def __init__(self, targets, decoys, field, threshold, batch_size, shuffle=False):
		self.targets = targets
		self.decoys = decoys
		self.batch_size = batch_size
		self.field = field
		self.threshold = threshold
		self.shuffle = shuffle

	def shuffle_data(self):
		for target in self.targets:
			for decoy in self.decoys[target]:
				random.shuffle(self.decoys[target])

	def select_batch(self, decoys_list):
		"""
		Selects examples for the batch
		"""
		positive = []
		negative = []
		
		num_added = 0
		for idx, decoy in enumerate(decoys_list):
			if decoy[self.field] >= self.threshold and num_added<(self.batch_size/2.0):
				positive.append(idx)
				num_added+=1
		
		for idx, decoy in enumerate(decoys_list):
			if decoy[self.field] < self.threshold and num_added<self.batch_size:
				negative.append(idx)
				num_added+=1

		return positive + negative

	def __iter__(self):
		if self.shuffle:
			self.shuffle_data()

		for idx, target in enumerate(self.targets):
			targets_idx = [idx for i in range(self.batch_size)]
			decoys_idx = self.select_batch(self.decoys[target])
			out = list(zip(targets_idx, decoys_idx))
			yield out

	def __len__(self):
		return len(self.targets)

def get_dataset_stream(data_dir, subset, batch_size = 10, field='quality', threshold=1, shuffle = False):
	
	dataset = SplitComplexDataset( field, threshold, data_dir, description_set=subset)

	batch_sampler = BatchSamplerQuality(dataset.targets, dataset.decoys, field, threshold, batch_size, shuffle)

	trainloader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=0)
	
	return trainloader








# def load_batch(self, filenames):
# 	try:
# 		coords, resnames, resnums, atomnames, num_atoms = self.pdb2coords(filenames)
# 		coords, num_atoms_of_type, offsets = self.assignTypes(coords, resnames, atomnames, num_atoms)
# 		batch_size = coords.size(0)
# 		a,b = getBBox(coords, num_atoms)
# 		translation = -(a+b)*0.5 + self.box_length/2.0
# 		coords = self.translate(coords, translation, num_atoms)
# 		volume = self.project(coords.cuda(), num_atoms_of_type.cuda(), offsets.cuda())
# 	except:
# 		print("Dataset:load_batch exception")
# 		with open("Debug/data_problems.dat", 'w') as fout:
# 			for name in filenames:
# 				fout.write("%s\n"%(name))
# 		sys.exit(1)

# 	return volume.detach(), translation.detach()/self.resolution

if __name__=='__main__':
	"""
	Testing data load procedure
	"""
	sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
	from src import DATA_DIR

	from matplotlib import pylab as plt
	local_data_dir = '/media/lupoglaz/ProteinsDataset/Docking/SplitComplexes'
	dataiter = iter(get_dataset_stream(local_data_dir, 'validation_set.dat', shuffle=True))

	receptor, ligand, labels = dataiter.next()
	print(receptor)
	print(ligand)
	print(labels.size())
	print(labels)
	sys.exit()

	import _Volume

	volume1, volume2, dT, labels, weights, receptor, ligand = dataiter.next()
	dT = dT.squeeze()
	volume1 = volume1.squeeze()
	volume2 = volume2.squeeze()
	labels = labels.squeeze()
	print(receptor[0])
	print(ligand[0])
	print(labels[0])
	print(labels)
	
	vt1 = volume1[0,:,:,:,:].squeeze()
	vt2 = volume2[0,:,:,:,:].squeeze()
	vt1 = vt1.sum(dim=0)
	vt2 = vt2.sum(dim=0)
	
	dx = int(dT[0,0])
	dy = int(dT[0,1])
	dz = int(dT[0,2])
	
	_Volume.Volume2Xplor(vt1.cpu(), "receptor.xplor", 1.5)
	_Volume.Volume2Xplor(vt2.cpu(), "ligand.xplor", 1.5)
	
	print(dx,dy,dz)
	L = 80
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

	_Volume.Volume2Xplor(vt1.cpu(), "complex.xplor", 1.5)

	

	



	
	
	
	

	