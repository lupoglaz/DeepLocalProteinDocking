import os
import sys
import torch
from math import *
from collections import OrderedDict
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from src import LOG_DIR, DATA_DIR

from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered, CoordsTranslate, CoordsRotate, writePDB


class ZDOCKParser:
	def __init__(self):
		self.p2c = PDB2CoordsUnordered()
		self.rotate = CoordsRotate()
		self.translate = CoordsTranslate()

		self.target_dict = OrderedDict({"box_size": None,
						"spacing": None,
						"switch_num": None,
						"rec_rand1": None, "rec_rand2": None, "rec_rand3": None,
						"lig_rand1": None, "lig_rand2": None, "lig_rand3": None,
						"rec": None, "r1": None, "r2": None, "r3":None,
						"lig": None, "l1": None, "l2": None, "l3":None,
						"conformations": []
						})

		self.ligand = None
		self.receptor = None
		self.complex = None
	
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

	def parse_ouput(self, output_file, header_only=True):

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

	def read_targets(self, pdb_dir):
		p2c = PDB2CoordsUnordered()
		self.ligand = p2c([os.path.join(pdb_dir, self.target_dict["lig"])])
		self.receptor = p2c([os.path.join(pdb_dir, self.target_dict["rec"])])

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

	def write_conformation(self, output_filename, conf_num):
		box_size = self.target_dict["box_size"]
		spacing = self.target_dict["spacing"]
		l1, l2, l3 = self.target_dict["l1"], self.target_dict["l2"], self.target_dict["l3"]
		r1, r2, r3 = self.target_dict["r1"], self.target_dict["r2"], self.target_dict["r3"]
		lig_rand1, lig_rand2, lig_rand3 = self.target_dict["lig_rand1"], self.target_dict["lig_rand2"], self.target_dict["lig_rand3"]
		rec_rand1, rec_rand2, rec_rand3 = self.target_dict["rec_rand1"], self.target_dict["rec_rand2"], self.target_dict["rec_rand3"]

		an_x, an_y, an_z, tr_x, tr_y, tr_z, score = self.target_dict["conformations"][conf_num]

		if tr_x >= box_size/2:
			tr_x -= box_size;
		if tr_y >= box_size/2:
			tr_y -= box_size
		if tr_z >= box_size/2:
			tr_z -= box_size

		num_atoms = self.ligand[-1]
		coords = self.ligand[0]
		if self.target_dict["switch_num"] is None:
			
			Tcenter = torch.tensor([-l1, -l2, -l3], dtype=torch.double, device='cpu').unsqueeze(dim=0)
			coords_ce = self.translate(coords, Tcenter, num_atoms)
			
			Rrand = self.getRotationMatrix(lig_rand1, lig_rand2, lig_rand3)
			coords_rand = self.rotate(coords_ce, Rrand, num_atoms)

			Ra = self.getRotationMatrix(an_x, an_y, an_z)
			coords_rand_a = self.rotate(coords_rand, Ra, num_atoms)

			T = torch.tensor([-tr_x*spacing + r1, -tr_y*spacing + r2, -tr_z*spacing + r3], dtype=torch.double, device='cpu').unsqueeze(dim=0)
			coords_out = self.translate(coords_rand_a, T, num_atoms)
		
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


		lcoords, lchains, lres_names, lres_nums, latom_names, lnum_atoms = self.ligand
		rcoords, rchains, rres_names, rres_nums, ratom_names, rnum_atoms = self.receptor
		cchains = torch.cat([lchains, rchains], dim=1)
		cres_names = torch.cat([lres_names, rres_names], dim=1)
		cres_nums = torch.cat([lres_nums, rres_nums], dim=1)
		catom_names = torch.cat([latom_names, ratom_names], dim=1)
		cnum_atoms = lnum_atoms + rnum_atoms

		self.complex = (torch.cat([coords_out, rcoords], dim=1), 
						cchains, cres_names, cres_nums, catom_names, cnum_atoms)
		writePDB(output_filename, *(self.complex))


def read_pdb_list(pdb_list_file, benchmark_dir):
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
				bound_complex_name = sline[0]
				native_name = bound_complex_name.split('_')[0]
				targets.append((native_name, cplx_type))	
	
	return targets

if __name__=='__main__':
	print(os.listdir(DATA_DIR))
	print(os.listdir(LOG_DIR))

	benchmark_dir = os.path.join(DATA_DIR, "DockingBenchmarkV4")
	benchmark_list = os.path.join(benchmark_dir, "TableS2.csv")
	# print(read_pdb_list(benchmark_list, benchmark_dir))
	
	parser = ZDOCKParser()
	parser.parse_ouput(os.path.join(LOG_DIR, "ZDOCK", "DockingBenchmarkV4_15deg", "1A2K.zd3.0.2.cg.fixed.out"), False)
	print(parser)
	parser.read_targets(os.path.join(LOG_DIR, "ZDOCK", "decoys_bm4_zd3.0.2_15deg_fixed","input_pdbs"))
	parser.write_conformation("tmp.pdb", 0)
	
