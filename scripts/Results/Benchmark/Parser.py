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


def get_irmsd(benchmark, parser, num_conf=1000):
	result = {}
	p2c = PDB2CoordsUnordered()
	problematic_alignments = benchmark.get_prob_alignments()
	problematic_superposition = benchmark.get_prob_superpos()
	
	skip = ['2VIS','1AZS', '2J7P', '2O3B', '1FAK', '1FC2', '1GCQ', 
			'2VDB', '1ZM4', '2H7V', '3CPH', '1E4K', '2HMI', '1ZLI',
			'2I9B', '1EER', '1NW9']
	
	for n, target_name in enumerate(benchmark.get_target_names()):
		if target_name in skip:
			print("Skipping prediction for", target_name, n)	
			continue
		print("Processing prediction for", target_name, n)
		target = benchmark.get_target(target_name)
		
		bound_target, unbound_target = benchmark.parse_structures(target)
		interfaces = benchmark.get_unbound_interfaces(bound_target, unbound_target)
		
		res = parser.parse_output(target_name, header_only=False)
		if res is None:
			continue
		
		unbound_receptor = parser.load_protein([unbound_target["receptor"]["path"]])
		unbound_receptor = ProteinStructure(*unbound_receptor)
		unbound_receptor.set(*unbound_receptor.select_CA())
		
		unbound_ligand = parser.load_protein([unbound_target["ligand"]["path"]])
		unbound_ligand = ProteinStructure(*unbound_ligand)
		unbound_ligand.set(*unbound_ligand.select_CA())
		
		#This interface will be rotated later
		unbound_interfaces = []
		for urec_sel, ulig_sel, brec_sel, blig_sel in interfaces:
			rec = ProteinStructure(*unbound_receptor.select_residues_list(urec_sel))
			lig = ProteinStructure(*unbound_ligand.select_residues_list(ulig_sel))
			unbound_interfaces.append( (rec, lig) )
		
		bound_receptor = ProteinStructure(*p2c([bound_target["receptor"]["path"]]))
		bound_receptor.set(*bound_receptor.select_CA())
		bound_ligand = ProteinStructure(*p2c([bound_target["ligand"]["path"]]))
		bound_ligand.set(*bound_ligand.select_CA())
		
		#This interface is static
		bound_interfaces = []
		for urec_sel, ulig_sel, brec_sel, blig_sel in interfaces:
			rec = ProteinStructure(*bound_receptor.select_residues_list(brec_sel))
			lig = ProteinStructure(*bound_ligand.select_residues_list(blig_sel))
			cplx = unite_proteins(rec, lig)
			bound_interfaces.append(cplx)
		
		c2rmsd = Coords2RMSD()	
		result[target_name] = []
		for i in range(num_conf):
			all_rmsd = []
			for rec, lig in unbound_interfaces:
				new_lig = ProteinStructure(*parser.transform_ligand(lig.get(), i))
				mobile_cplx = unite_proteins(rec, new_lig)
				for static_cplx in bound_interfaces:
					all_rmsd.append(c2rmsd(mobile_cplx.coords, static_cplx.coords, static_cplx.numatoms).item())
			
			min_rmsd = min(all_rmsd)						
			result[target_name].append( min_rmsd )	
		
	return result

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')
	parser.add_argument('-experiment', default='LocalE3MultiResRepr4x4', help='Experiment name')
	# parser.add_argument('-experiment', default='LocalSE3MultiResReprScalar', help='Experiment name')
	
	parser.add_argument('-dataset', default='DockingBenchmarkV4', help='Dataset name')
	parser.add_argument('-table', default='TableS1.csv', help='Targets table')
	parser.add_argument('-threshold_clash', default=300, help='Clash theshold for excluding conformations', type=float)
	parser.add_argument('-angle_inc', default=15, help='Angle increment, int', type=int)
		
	args = parser.parse_args()
	
	experiment_dir = os.path.join(LOG_DIR, args.experiment)
	decoys_dir = os.path.join(experiment_dir, args.dataset + '_%d'%args.angle_inc + "%.1f"%args.threshold_clash)

	benchmark_dir = os.path.join(DATA_DIR, args.dataset)
	benchmark_table = os.path.join(benchmark_dir, "TableCorrect.csv")
	natives_dir = os.path.join(benchmark_dir, 'structures')
	benchmark = DockingBenchmark(benchmark_dir, benchmark_table, natives_dir)

	parser = DockerParser(decoys_dir)
	get_irmsd(benchmark, parser)