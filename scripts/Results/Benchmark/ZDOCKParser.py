import os
import sys
import torch
from math import *
from collections import OrderedDict
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from src import LOG_DIR, DATA_DIR

from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered, CoordsTranslate, CoordsRotate, writePDB
from TorchProteinLibrary.RMSD import Coords2RMSD

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from Dataset.processing_utils import _get_contacts, _get_fnat, _get_capri_quality
from DockingBenchmark import DockingBenchmark
from VisualizeBenchmark import ProteinStructure, unite_proteins

import _pickle as pkl


class ZDOCKParser:
	def __init__(self):
		self.p2c = PDB2CoordsUnordered()
		self.rotate = CoordsRotate()
		self.translate = CoordsTranslate()

		self.ligand = None
		self.receptor = None
			
	def __str__(self):
		keys = list(self.target_dict.keys())
		out = ''
		for key in keys[0:3]:
			out += key + ': ' + str(self.target_dict[key]) + '\n'
		
		if self.target_dict["rec"] is None:
			return out
		
		out += 'Receptor: ' + str(self.target_dict["rec"]) + '\n'
		out += 'Ligand: ' + str(self.target_dict["lig"]) + '\n'

		out += 'Random receptor rotation:'
		for key in keys[3:6]:
			out += ' ' + str(self.target_dict[key])
		out += '\n'
		out += 'Receptor translation:'
		for key in keys[10:13]:
			out += ' ' + str(self.target_dict[key])
		out += '\n'
		out += 'Random ligand rotation:'
		for key in keys[6:9]:
			out += ' ' + str(self.target_dict[key])
		out += '\n'
		out += 'Ligand translation:'
		for key in keys[14:17]:
			out += ' ' + str(self.target_dict[key])
		out += '\n'
		out += "Num conformations: " + str(len(self.target_dict["conformations"]))

		return out

	def parse_output(self, output_file, irmsd_output_file=None, header_only=True):
		self.target_dict = OrderedDict({"box_size": None,
						"spacing": None,
						"switch_num": None,
						"rec_rand1": None, "rec_rand2": None, "rec_rand3": None,
						"lig_rand1": None, "lig_rand2": None, "lig_rand3": None,
						"rec": None, "r1": None, "r2": None, "r3":None,
						"lig": None, "l1": None, "l2": None, "l3":None,
						"conformations": [],
						"irmsd_reported": []
						})

		with open(output_file) as fin:
			header = fin.readline().split()
			self.target_dict["box_size"] = int(header[0])
			self.target_dict["spacing"] = float(header[1])
			if len(header)==3:
				self.target_dict["switch_num"] = int(header[2])
			else:
				self.target_dict["switch_num"] = None
			
			if header_only:
				return

			if not self.target_dict["switch_num"] is None:
				self.target_dict["rec_rand1"], self.target_dict["rec_rand2"], self.target_dict["rec_rand3"] = \
					map(lambda x: float(x), fin.readline().split())

			self.target_dict["lig_rand1"], self.target_dict["lig_rand2"], self.target_dict["lig_rand3"] = \
				map(lambda x: float(x), fin.readline().split())
			
			rec, r1, r2, r3 = fin.readline().split()
			r1, r2, r3 = float(r1), float(r2), float(r3)
			
			lig, l1, l2, l3 = fin.readline().split()
			l1, l2, l3 = float(l1), float(l2), float(l3)

			if self.target_dict["switch_num"] == 1:
				tmp = rec
				rec = lig
				lig = tmp
			
			self.target_dict["rec"], self.target_dict["r1"], self.target_dict["r2"], self.target_dict["r3"] = rec, r1, r2, r3
			self.target_dict["lig"], self.target_dict["l1"], self.target_dict["l2"], self.target_dict["l3"] = lig, l1, l2, l3

			for line in fin:
				aline = line.split()
				an_x, an_y, an_z = map(lambda x: float(x), aline[:3])
				tr_x, tr_y, tr_z = map(lambda x: int(x), aline[3:6])
				score = float(aline[6])

				self.target_dict["conformations"].append((an_x, an_y, an_z, tr_x, tr_y, tr_z, score))

		if not(irmsd_output_file is None):
			with open(irmsd_output_file) as fin:
				for line in fin:
					self.target_dict["irmsd_reported"].append(float(line.split()[1]))
						
	def getRotationMatrix(self, psi, theta, phi, rev=0):
		if rev == 0:
			r11 = cos(psi)*cos(phi)  -  sin(psi)*cos(theta)*sin(phi)
			r21 = sin(psi)*cos(phi)  +  cos(psi)*cos(theta)*sin(phi)
			r31 = sin(theta)*sin(phi)

			r12 = -cos(psi)*sin(phi)  -  sin(psi)*cos(theta)*cos(phi)
			r22 = -sin(psi)*sin(phi)  +  cos(psi)*cos(theta)*cos(phi)
			r32 = sin(theta)*cos(phi)

			r13 = sin(psi)*sin(theta)
			r23 = -cos(psi)*sin(theta)
			r33 = cos(theta)
		else:
			r11 = cos(psi)*cos(phi)  -  sin(psi)*cos(theta)*sin(phi)
			r12 = sin(psi)*cos(phi)  +  cos(psi)*cos(theta)*sin(phi)
			r13 = sin(theta)*sin(phi)

			r21 = -cos(psi)*sin(phi)  -  sin(psi)*cos(theta)*cos(phi)
			r22 = -sin(psi)*sin(phi)  +  cos(psi)*cos(theta)*cos(phi)
			r23 = sin(theta)*cos(phi)

			r31 = sin(psi)*sin(theta)
			r32 = -cos(psi)*sin(theta)
			r33 = cos(theta)

		R = torch.tensor([	[r11, r12, r13],
							[r21, r22, r23],
							[r31, r32, r33]], 
							dtype=torch.double, device='cpu')
		return R.unsqueeze(dim=0)

	def transform_ligand(self, ligand, conf_num):
		box_size = self.target_dict["box_size"]
		spacing = self.target_dict["spacing"]
		l1, l2, l3 = self.target_dict["l1"], self.target_dict["l2"], self.target_dict["l3"]
		r1, r2, r3 = self.target_dict["r1"], self.target_dict["r2"], self.target_dict["r3"]
		lig_rand1, lig_rand2, lig_rand3 = self.target_dict["lig_rand1"], self.target_dict["lig_rand2"], self.target_dict["lig_rand3"]
		rec_rand1, rec_rand2, rec_rand3 = self.target_dict["rec_rand1"], self.target_dict["rec_rand2"], self.target_dict["rec_rand3"]
		
		an_x, an_y, an_z, tr_x, tr_y, tr_z, score = self.target_dict["conformations"][conf_num]

		# print(self.target_dict["switch_num"], rec_rand1, rec_rand2, rec_rand3, lig_rand1, lig_rand2, lig_rand3, r1, r2, r3, l1, l2, l3, 
		# 		an_x, an_y, an_z, tr_x, tr_y, tr_z, box_size, spacing)

		if tr_x >= box_size/2:
			tr_x -= box_size;
		if tr_y >= box_size/2:
			tr_y -= box_size
		if tr_z >= box_size/2:
			tr_z -= box_size

		num_atoms = ligand[-1]
		coords = ligand[0]
		if self.target_dict["switch_num"] is None:
			
			Tcenter = torch.tensor([-l1, -l2, -l3], dtype=torch.double, device='cpu').unsqueeze(dim=0)
			coords_ce = self.translate(coords, Tcenter, num_atoms)
			# print(coords_ce[0, :3])
			
			Rrand = self.getRotationMatrix(lig_rand1, lig_rand2, lig_rand3)
			coords_rand = self.rotate(coords_ce, Rrand, num_atoms)
			# print(coords_rand[0, :3])

			Ra = self.getRotationMatrix(an_x, an_y, an_z)
			coords_rand_a = self.rotate(coords_rand, Ra, num_atoms)
			# print(coords_rand_a[0, :3])

			T = torch.tensor([-tr_x*spacing + r1, -tr_y*spacing + r2, -tr_z*spacing + r3], dtype=torch.double, device='cpu').unsqueeze(dim=0)
			coords_out = self.translate(coords_rand_a, T, num_atoms)
			# print(coords_out[0, :3])
		
		else:

			Rrand = self.getRotationMatrix(rec_rand1, rec_rand2, rec_rand3)
			coords_rand = self.rotate(coords, Rrand, num_atoms)

			Tcenter = torch.tensor([-r1 + tr_x*spacing, -r2 + tr_y*spacing, -r3 + tr_z*spacing], dtype=torch.double, device='cpu').unsqueeze(dim=0)
			coords_ce = self.translate(coords_rand, Tcenter, num_atoms)

			Ra = self.getRotationMatrix(an_x, an_y, an_z, rev=1)
			coords_a = self.rotate(coords_ce, Ra, num_atoms)

			Rlig_rand = self.getRotationMatrix(lig_rand1, lig_rand2, lig_rand3, rev=1)
			coords_rand = self.rotate(coords_a, Rlig_rand, num_atoms)

			T = torch.tensor([l1, l2, l3], dtype=torch.double, device='cpu').unsqueeze(dim=0)
			coords_out = self.translate(coords_rand_a, T, num_atoms)

		old_coords, lchains, lres_names, lres_nums, latom_names, lnum_atoms = ligand
		return (coords_out, lchains, lres_names, lres_nums, latom_names, lnum_atoms)



def get_irmsd(benchmark_dir, benchmark_table, natives_dir, decoys_dir, num_conf=1000):
	benchmark = DockingBenchmark(benchmark_dir, benchmark_table, natives_dir)
	parser = ZDOCKParser()
	parser_native = ZDOCKParser()
	p2c = PDB2CoordsUnordered()
	
	result = {}

	problematic_alignments = benchmark.get_prob_alignemnts()
	problematic_superposition = benchmark.get_prob_superpos()
	skip = ['1N8O', '1AZS', '1GP2', '1K5D', '2J7P', '2O3B', '1FAK', '1A2K', '1HCF', #double free corruption
			'2VIS', '1FC2', '1GCQ', '2VDB', '1ZM4', '2H7V', '3CPH', '1E4K', '2HMI', '1ZLI', '2I9B', '1NW9', #interface is measured between one of three subunits
			'1XU1', #segmentation fault
			'1EER' #chains problem
			]
	
	for n, target_name in enumerate(benchmark.get_target_names()):
		# target_name = '1OYV'
		if target_name in skip:
			print("Skipping prediction for", target_name, n)	
			continue
		print("Processing prediction for", target_name, n)
		target = benchmark.get_target(target_name)
		bound_target, unbound_target = benchmark.parse_structures(target)
		interfaces = benchmark.get_unbound_interfaces(bound_target, unbound_target)
		N = len(interfaces)

		zdock_output_name = "%s.zd3.0.2.cg.fixed.out"%target_name
		zdock_irmsd_name = "%s.zd3.0.2.cg.fixed.out.rmsds"%target_name
		parser.parse_output(os.path.join(decoys_dir, zdock_output_name), os.path.join(decoys_dir, zdock_irmsd_name),
							header_only=False)

		unbound_receptor = ProteinStructure(*p2c([unbound_target["receptor"]["path"]]))
		unbound_receptor.set(*unbound_receptor.select_CA())
		unbound_ligand = ProteinStructure(*p2c([unbound_target["ligand"]["path"]]))
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
			
			if min_rmsd > (parser.target_dict["irmsd_reported"][i] + 5.0):
				raise(Exception("Conformation wrong", i, min_rmsd, parser.target_dict["irmsd_reported"][i]))
						
			result[target_name].append( min_rmsd )

		# break

	return result


def plot_protein(protein):
	pass

if __name__=='__main__':
	benchmark_dir = os.path.join(DATA_DIR, "DockingBenchmarkV4")
	benchmark_table = os.path.join(benchmark_dir, "TableCorrect.csv")
	zdock_dir = os.path.join(LOG_DIR, "ZDOCK", "decoys_bm4_zd3.0.2_15deg_fixed")
	natives_dir = os.path.join(zdock_dir, 'input_pdbs')
	natives_dir = os.path.join(benchmark_dir, 'structures')
	decoys_dir = os.path.join(zdock_dir, 'results')
	overwrite = True
		
	zdock_results_filename = os.path.join(zdock_dir, 'DockingBenchmarkV4_15deg_irmsd.pkl')
	
	if (not os.path.exists(zdock_results_filename)) or overwrite:
		results = get_irmsd(benchmark_dir, benchmark_table, natives_dir, decoys_dir, num_conf=1000)
		with open(zdock_results_filename, 'wb') as fout:
			pkl.dump(results, fout)
	else:
		with open(zdock_results_filename, 'rb') as fin:
			results = pkl.load(fin)

	for target_name in results.keys():
		print(target_name, "Min irmsd = ", min(results[target_name]))
	
	

