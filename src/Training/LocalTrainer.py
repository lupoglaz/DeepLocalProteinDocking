import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import atexit
import numpy as np
import math
import os
import sys
from torch.optim.lr_scheduler import LambdaLR
from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered, Coords2TypedCoords, CoordsTranslate, CoordsRotate, getBBox, getRandomRotation
from TorchProteinLibrary.Volume import TypedCoords2Volume


sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src import REPOSITORY_DIR
from src.Utils import Rotations, VolumeViz

class LocalTrainer:
	def __init__(self, model, loss, lr=0.001, lr_decay=0.0001, 
						box_size=120, resolution=1.0,
						add_neg=False, neg_weight=0.5, 
						add_zero=False, zero_weight=1.0,
						randomize_rot = True):
		self.lr = lr
		self.lr_decay = lr_decay
		self.model = model
		self.loss = loss
		self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
		self.log = None
		self.lr_scheduler = LambdaLR(self.optimizer, lambda epoch: 1.0/(1.0+epoch*self.lr_decay))
		
		#zero-score condition
		self.add_zero = add_zero
		self.zero_weight = zero_weight

		#negative-score condition
		self.add_neg = add_neg
		self.neg_weight = neg_weight

		self.box_length = box_size * resolution
		self.box_size = box_size
		self.resolution = resolution
		
		self.pdb2coords = PDB2CoordsUnordered()
		self.assignTypes = Coords2TypedCoords()
		self.translate = CoordsTranslate()
		self.rotate = CoordsRotate()
		self.project = TypedCoords2Volume(self.box_size, self.resolution)

		self.relu = nn.ReLU()
		self.randomize_rot = randomize_rot	

		atexit.register(self.cleanup)
	
	def new_log(self, log_file_name):
		if not self.log is None:
			self.log.close()
		self.log = open(log_file_name, "w")

	def cleanup(self):
		if not self.log is None:
			self.log.close()

	def load_batch(self, filenames, random_rotations=None):
		with torch.no_grad():
			coords, _, resnames, resnums, atomnames, num_atoms = self.pdb2coords(filenames)
			if not random_rotations is None:
				coords = self.rotate(coords, random_rotations, num_atoms)
			
			coords, num_atoms_of_type, offsets = self.assignTypes(coords, resnames, atomnames, num_atoms)
			batch_size = coords.size(0)
			a,b = getBBox(coords, num_atoms)
			translation = -(a+b)*0.5 + self.box_length/2.0
			coords = self.translate(coords, translation, num_atoms)
			volume = self.project(coords.to(dtype=torch.float, device='cuda'), num_atoms_of_type.cuda(), offsets.cuda())
		
		return volume, translation/self.resolution, a, b

	def optimize(self, data):
		"""
		Optimization step. 
		Input: data
		Output: loss
		"""
		self.model.train()
		self.optimizer.zero_grad()
		
		receptor_list, ligand_list, labels = data
		receptor_list = list(receptor_list)
		ligand_list = list(ligand_list)
		labels = torch.squeeze(labels)
		
		batch_size = len(receptor_list)

		#Main batch
		with torch.no_grad():
			if self.randomize_rot:
				random_rotations = getRandomRotation(batch_size)
				receptor, T1, reca, recb = self.load_batch(receptor_list, random_rotations)
				ligand, T2, liga, ligb = self.load_batch(ligand_list, random_rotations)
			else:
				receptor, T1, reca, recb = self.load_batch(receptor_list)
				ligand, T2, liga, ligb = self.load_batch(ligand_list)

			T = T1 - T2
			

		#Ranking interactions
		model_out = self.model(receptor, ligand, T).squeeze()
		L_decoys = self.loss(model_out, labels)
		L = L_decoys

		#Interacting decoys should be score<0
		if self.add_neg:
			L_neg = torch.mean(torch.relu(model_out))
			L = L + self.neg_weight * L_neg

		#Non-interaction
		if self.add_zero:
			input_zero = torch.zeros(1, self.model.filter.fc_input_size, device='cuda', dtype=torch.float)
			output_zero = self.model.filter.fc(input_zero)
			L_zero = torch.abs(output_zero)
			L = L + self.zero_weight * L_zero
		
		L.backward()
		
		if not self.log is None:
			if self.add_neg and self.add_zero:
				self.log.write("Loss\t%f\t%f\t%f\t%f\t%f\n"%(L.item(), L_decoys.item(), 0.0, L_neg.item(), L_zero.item()))
			elif (not self.add_neg) and self.add_zero:
				self.log.write("Loss\t%f\t%f\t%f\t%f\t%f\n"%(L.item(), L_decoys.item(), 0.0, 0.0, L_zero.item()))
			elif (not self.add_neg) and (not self.add_zero):
				self.log.write("Loss\t%f\t%f\t%f\t%f\t%f\n"%(L.item(), L_decoys.item(), 0.0, 0.0, 0.0))
			else:
				raise Exception("Stupid choice of the loss function(neg/zero): ", self.add_neg, self.add_zero)

			for i in range(len(receptor_list)):
				self.log.write("%s\t%s\t%f\t%f\n"%(receptor_list[i], ligand_list[i], model_out[i].item(), labels[i].item()))

		self.optimizer.step()
		self.lr_scheduler.step()
		return L.item()

	def score(self, data):
		"""
		Scoring of the data. 
		Input: data
		Output: None
		"""
		self.model.eval()
		receptor_list, ligand_list, labels = data
		receptor_list = list(receptor_list)
		ligand_list = list(ligand_list)
		labels = torch.squeeze(labels)

		batch_size = len(receptor_list)
		if self.randomize_rot:
			random_rotations = getRandomRotation(batch_size)
			receptor, T1, _, _ = self.load_batch(receptor_list, random_rotations)
			ligand, T2, _, _ = self.load_batch(ligand_list, random_rotations)
		else:
			receptor, T1, _, _ = self.load_batch(receptor_list)
			ligand, T2, _, _ = self.load_batch(ligand_list)
		
		T = T1 - T2
		
		model_out = self.model(receptor, ligand, T).squeeze()
		L = self.loss(model_out, labels)
		
		if not self.log is None:
			self.log.write("Loss\t%f\t%f\t%f\t%f\t%f\n"%(L.item(), L.item(), 0.0, 0.0, 0.0))
			for i in range(len(receptor_list)):
				self.log.write("%s\t%s\t%f\t%f\n"%(receptor_list[i], ligand_list[i], model_out[i].item(), labels[i].item()))

		return L.item()
