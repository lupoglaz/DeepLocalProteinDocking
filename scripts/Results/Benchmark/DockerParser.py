import os
import sys
import torch
import argparse
from math import *
from collections import OrderedDict
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from src import LOG_DIR, DATA_DIR

from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered, CoordsTranslate, CoordsRotate, writePDB, getBBox
from TorchProteinLibrary.RMSD import Coords2RMSD

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from Dataset.processing_utils import _get_contacts, _get_fnat, _get_capri_quality
from DockingBenchmark import DockingBenchmark
from VisualizeBenchmark import ProteinStructure, unite_proteins

import _pickle as pkl

from ZDOCKParser import ZDOCKParser


class DockerParser(ZDOCKParser):
	def __init__(self, decoys_dir):
		self.p2c = PDB2CoordsUnordered()
		self.rotate = CoordsRotate()
		self.translate = CoordsTranslate()
		self.decoys_dir = decoys_dir

					
	def __str__(self):
		out += "Num conformations: " + str(len(self.target_dict["conformations"]))
		return out

	def parse_output(self, target_name, header_only=True):
		filename = os.path.join(self.decoys_dir, target_name+'.dat')
		
		if not os.path.exists(filename):
			return None
		
		self.target_dict = OrderedDict({
						"target_name": target_name,
						"filename": filename,
						"conformations": []
						})

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
				self.target_dict["conformations"].append( (rot.unsqueeze(dim=0), t.unsqueeze(dim=0), score) )
		
		return self.target_dict
	
	def load_protein(self, path):
		coords, chainnames, resnames, resnums, atomnames, num_atoms = self.p2c(path)
		a,b = getBBox(coords, num_atoms)
		coords = self.translate(coords, -(a+b)*0.5, num_atoms)
		return coords, chainnames, resnames, resnums, atomnames, num_atoms

	def transform_ligand(self, ligand, conf_num):
		ligand_coords = ligand[0]
		ligand_numatoms = ligand[-1]
		
		R, T, score = self.target_dict["conformations"][conf_num]
		
		ligand_coords = self.rotate(ligand_coords, R, ligand_numatoms)
		coords_out = self.translate(ligand_coords, T, ligand_numatoms)
				
		old_coords, lchains, lres_names, lres_nums, latom_names, lnum_atoms = ligand
		return coords_out, lchains, lres_names, lres_nums, latom_names, lnum_atoms	
