import torch
import torch.nn as nn
import atexit
import numpy as np
import math
import os
import sys
from TorchProteinLibrary.FullAtomModel import CoordsTransform
from TorchProteinLibrary.Volume import TypedCoords2Volume, VolumeConvolution, VolumeRotation
from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered, Coords2TypedCoords, CoordsTranslate, getBBox, getRandomRotation
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src import REPOSITORY_DIR
from src.Utils import Rotations, VolumeViz

class Docker:
	def __init__(self, docking_model, angle_inc=15.0, box_size=80, resolution=1.25, max_conf=1000, randomize_rot=False):
		self.docking_model = docking_model
		self.log = None
		
		self.box_size = box_size
		self.resolution = resolution
		self.box_length = box_size*resolution
		
		self.max_conf = max_conf
		self.rot = Rotations(angle_inc=angle_inc)
		
		self.rotate = CoordsTransform.CoordsRotate()
		self.translate = CoordsTransform.CoordsTranslate()
		self.project = TypedCoords2Volume(self.box_size, self.resolution)
		self.convolve = VolumeConvolution()
				
		self.box_center = torch.zeros(1, 3, dtype=torch.double, device='cpu')
		self.box_center.fill_(self.box_length/2.0)

		self.pdb2coords = PDB2CoordsUnordered()
		self.assignTypes = Coords2TypedCoords()
		self.translation = CoordsTranslate()
		self.vol_rotate = VolumeRotation()

		self.randomize_rot = randomize_rot
		if self.randomize_rot:
			self.randR = getRandomRotation(1)
			print("Adding random rotation to the receptor:", self.randR)
		
		atexit.register(self.cleanup)

	def load_batch(self, filenames, bbox_center=True):
		with torch.no_grad():
			coords, chains, resnames, resnums, atomnames, num_atoms = self.pdb2coords(filenames)
			coords, num_atoms_of_type, offsets = self.assignTypes(coords, resnames, atomnames, num_atoms)
			
			a,b = getBBox(coords, num_atoms)
			if bbox_center:
				translation = -(a+b)*0.5 + self.box_length/2.0
			else:
				translation = -(a+b)*0.5
			coords = self.translation(coords, translation, num_atoms)

		return coords, num_atoms_of_type, offsets, translation, num_atoms
	
	def new_log(self, log_file_name, rewrite=True):
		if not self.log is None:
			self.log.close()

		if os.path.exists(log_file_name) and (not rewrite):
			lines = []
			with open(log_file_name, 'r') as fin:
				for line in fin:
					if len(line.split())>0:
						lines.append(line)
			if len(lines)>1:
				return False
			else:
				self.log = open(log_file_name, "w")
				return True
		else:
			self.log = open(log_file_name, "w")
			return True

	def cleanup(self):
		if not self.log is None:
			self.log.close()
		
	def update_top(self, V, rotation_index):
		
		#Getting top scoring translations
		top = []
		for i in range(self.max_conf):
			maxval_z, ind_z = torch.min(V, dim=2, keepdim=False)
			maxval_y, ind_y = torch.min(maxval_z, dim=1)
			maxval_x, ind_x = torch.min(maxval_y, dim=0)
			x = ind_x.item()
			y = ind_y[x].item()
			z = ind_z[x, y].item()
			top.append((x,y,z, V[x,y,z].item()))
			V[x,y,z] = 0.0
		
		for x,y,z,score in top:
			self.top_list.append((rotation_index, x,y,z, score))
		
		#Resorting the top conformations and cutting the max number
		self.top_list.sort(key = lambda t: t[4])
		self.top_list = self.top_list[:self.max_conf]

	def write_conformations(self):
		if not self.log is None:
			for i, x,y,z, score in self.top_list:
				r = self.rot.R[i,:,:]
				t = torch.zeros(3, device='cpu', dtype=torch.double)
				t[0] = x
				t[1] = y
				t[2] = z
				if x >= self.box_size:
					t[0] = -(2*self.box_size - x)
				if y >= self.box_size:
					t[1] = -(2*self.box_size - y)
				if z >= self.box_size:
					t[2] = -(2*self.box_size - z)

				t = t*self.resolution

				if self.randomize_rot:
					randRT = torch.transpose(self.randR.squeeze(), 0, 1)
					t = torch.matmul(randRT, t)
					r = torch.matmul(randRT, r)
									
				self.log.write("%f\t%f\t%f\t"%(r[0,0].item(),r[0,1].item(),r[0,2].item()))
				self.log.write("%f\t%f\t%f\t"%(r[1,0].item(),r[1,1].item(),r[1,2].item()))
				self.log.write("%f\t%f\t%f\t"%(r[2,0].item(),r[2,1].item(),r[2,2].item()))
				self.log.write("%f\t%f\t%f\t"%(t[0], t[1], t[2]))
				self.log.write("%f\n"%(score))
	
	def dockE3(self, ureceptor, uligand, batch_size):
		"""
		Docking two proteins.
		Input: receptor and ligand data
		Output: None
		"""
		self.top_list = []
		self.docking_model.eval()
		rcoords, rnat, roff, rT, rnatoms = self.load_batch([ureceptor for i in range(batch_size)], bbox_center=False)
		lcoords, lnat, loff, lT, lnatoms = self.load_batch([uligand for i in range(batch_size)], bbox_center=False)
				
		if self.randomize_rot:
			randR_batch = self.randR.repeat(batch_size, 1, 1)
			rcoords = self.rotate(rcoords, randR_batch, rnatoms)
		box_center_batch = self.box_center.squeeze().repeat(batch_size, 1)
		rcoords = self.translate(rcoords, box_center_batch, rnatoms)
		
		#Projecting receptor to the grid
		receptor = self.project(rcoords.to(dtype=torch.float, device='cuda'), rnat.cuda(), roff.cuda())
		receptor_volumes = self.docking_model.representation(receptor)
		receptor_forbidden = receptor.sum(dim=1).unsqueeze(dim=1).contiguous() #receptor forbidden volume

		for i in tqdm(range(0, self.rot.R.size(0), batch_size)):
			beg_rot = i
			end_rot = min(i+batch_size, self.rot.R.size(0))
			r_batch = self.rot.R[beg_rot:end_rot,:,:]
			
			#Projecting rotated ligand
			lcoords_rot = self.rotate(lcoords, r_batch, lnatoms)
			lcoords_rot_trans = self.translate(lcoords_rot, box_center_batch, lnatoms)
			ligand = self.project(lcoords_rot_trans.to(dtype=torch.float, device='cuda'), lnat.cuda(), loff.cuda())
			ligand_volumes = self.docking_model.representation(ligand)
			ligand_forbidden = ligand.sum(dim=1).unsqueeze(dim=1).contiguous() #ligand forbidden volume

			normalization = self.convolve(receptor_forbidden, ligand_forbidden)
			forbidden_V = torch.lt(normalization, self.docking_model.threshold_clash).to(dtype=torch.float).squeeze()

			#Computing results for all translations
			V = self.docking_model(receptor_volumes, ligand_volumes).squeeze()
			
			#Excluding forbidden translations
			V = forbidden_V * V

			#updating top conformations list
			for rot_idx in range(beg_rot, end_rot):
				self.update_top(V[rot_idx-beg_rot,:,:,:], rot_idx)
			
		self.write_conformations()

	def dockSE3(self, ureceptor, uligand, batch_size):
		"""
		Docking two proteins.
		Input: receptor and ligand data
		Output: None
		"""
		self.top_list = []
		self.docking_model.eval()
		rcoords, rnat, roff, rT, rnatoms = self.load_batch([ureceptor for i in range(batch_size)], bbox_center=False)
		lcoords, lnat, loff, lT, lnatoms = self.load_batch([uligand for i in range(batch_size)], bbox_center=False)
				
		if self.randomize_rot:
			randR_batch = self.randR.repeat(batch_size, 1, 1)
			rcoords = self.rotate(rcoords, randR_batch, rnatoms)

		box_center_batch = self.box_center.squeeze().repeat(batch_size, 1)
		rcoords = self.translate(rcoords, box_center_batch, rnatoms)
		lcoords_trans = self.translate(lcoords, box_center_batch, lnatoms)
		
		#Projecting receptor to the grid
		receptor = self.project(rcoords.to(dtype=torch.float, device='cuda'), rnat.cuda(), roff.cuda())
		receptor_volumes = self.docking_model.representation(receptor)
		receptor_forbidden = receptor.sum(dim=1).unsqueeze(dim=1).contiguous() #receptor forbidden volume

		ligand = self.project(lcoords_trans.to(dtype=torch.float, device='cuda'), lnat.cuda(), loff.cuda())
		ligand_volumes = self.docking_model.representation(ligand)
		
		for i in tqdm(range(0, self.rot.R.size(0), batch_size)):
			beg_rot = i
			end_rot = min(i+batch_size, self.rot.R.size(0))
			r_batch_cpu = self.rot.R[beg_rot:end_rot,:,:]
			r_batch_gpu = r_batch_cpu.to(dtype=torch.float, device='cuda')
			
			#Rotating ligand representations
			ligand_volumes_rotated = [self.vol_rotate(volume, r_batch_gpu) for volume in ligand_volumes]
			
			#Projecting rotated ligand to get forbidden volume
			lcoords_rot = self.rotate(lcoords, r_batch_cpu, lnatoms)
			lcoords_rot_trans = self.translate(lcoords_rot, box_center_batch, lnatoms)
			ligand = self.project(lcoords_rot_trans.to(dtype=torch.float, device='cuda'), lnat.cuda(), loff.cuda())
			ligand_forbidden_rotated = ligand.sum(dim=1).unsqueeze(dim=1).contiguous() #ligand forbidden volume
			normalization = self.convolve(receptor_forbidden, ligand_forbidden_rotated).squeeze()
			forbidden_V = torch.lt(normalization, self.docking_model.threshold_clash).to(dtype=torch.float)

			#Computing results for all translations
			V = self.docking_model(receptor_volumes, ligand_volumes_rotated).squeeze()
			
			#Excluding forbidden translations
			V = forbidden_V * V

			#updating top conformations list
			for rot_idx in range(beg_rot, end_rot):
				self.update_top(V[rot_idx-beg_rot,:,:,:], rot_idx)
			
		self.write_conformations()