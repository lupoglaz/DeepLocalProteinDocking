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

	def select_atoms(self, protein, atomic_mask):
		coords, chains, res_names, res_nums, atom_names, num_atoms = protein
		N = num_atoms[0].item()
		
		isSel = atomic_mask
		isSel_coords = torch.stack([atomic_mask for i in range(3)], dim=1).unsqueeze(dim=0)
		isSel_names = torch.stack([atomic_mask for i in range(4)], dim=1).unsqueeze(dim=0)	
		num_sel_atoms =  atomic_mask.sum().item()
		sel_num_atoms = torch.tensor([num_sel_atoms], dtype=torch.int, device='cpu')	
		
		coords = coords.view(1, N, 3)
		sel_coords = torch.masked_select(coords, isSel_coords).view(1, num_sel_atoms*3).contiguous()
		sel_chains = torch.masked_select(chains, isSel_names).view(1, num_sel_atoms, 4).contiguous()
		sel_resnames = torch.masked_select(res_names, isSel_names).view(1, num_sel_atoms, 4).contiguous()
		sel_resnums = torch.masked_select(res_nums, isSel).view(1, num_sel_atoms).contiguous()
		sel_atom_names = torch.masked_select(atom_names, isSel_names).view(1, num_sel_atoms, 4).contiguous()
				
		return sel_coords, sel_chains, sel_resnames, sel_resnums, sel_atom_names, sel_num_atoms

	def select_atoms_seq(self, protein, atom_list):
		coords, chains, res_names, res_nums, atom_names, num_atoms = protein
		N = num_atoms[0].item()
		
		coords = coords.view(1, N, 3)
		sel_coords, sel_chains, sel_resnames, sel_resnums, sel_atom_names = [], [], [], [], []
		sel_num_atoms = 0
		for chain, res_num, res_name in atom_list:
			for i in range(N):
				if (chain == str(chr(chains[0, i, 0].item()))) and (res_num == res_nums[0, i].item()):
					sel_coords.append(coords[:, i, :])
					sel_chains.append(chains[:, i, :])
					sel_resnames.append(res_names[:, i, :])
					sel_resnums.append(res_nums[:, i])
					sel_atom_names.append(atom_names[:, i, :])
					sel_num_atoms += 1
		
		sel_coords = torch.stack(sel_coords, dim=1).view(1, sel_num_atoms*3).contiguous()
		sel_chains = torch.stack(sel_chains, dim=1).contiguous()
		sel_resnames = torch.stack(sel_resnames, dim=1).contiguous()
		sel_resnums = torch.stack(sel_resnums, dim=1).contiguous()
		sel_atom_names = torch.stack(sel_atom_names, dim=1).contiguous()
		sel_num_atoms = torch.tensor([sel_num_atoms], dtype=torch.int, device='cpu').contiguous()
				
		return sel_coords, sel_chains, sel_resnames, sel_resnums, sel_atom_names, sel_num_atoms
	
	def make_CA_mask(self, protein):
		atom_names = protein[4]
		is0C = torch.eq(atom_names[:,:,0], 67).squeeze()
		is1A = torch.eq(atom_names[:,:,1], 65).squeeze()
		is20 = torch.eq(atom_names[:,:,2], 0).squeeze()
		isCA = is0C*is1A*is20
		
		return isCA

	def read_targets(self, pdb_dir):
		p2c = PDB2CoordsUnordered()
		self.ligand = p2c([os.path.join(pdb_dir, self.target_dict["lig"])])
		self.receptor = p2c([os.path.join(pdb_dir, self.target_dict["rec"])])

	def read_native(self, receptor_path, ligand_path):
		p2c = PDB2CoordsUnordered()
		self.ligand = p2c([ligand_path])
		self.receptor = p2c([receptor_path])

	def select_target_CA(self):
		self.receptor = self.select_atoms(self.receptor, self.make_CA_mask(self.receptor))
		self.ligand = self.select_atoms(self.ligand, self.make_CA_mask(self.ligand))

	def select_target_residues(self, rec_residues, lig_residues):
		self.receptor = self.select_atoms_seq(self.receptor, rec_residues)
		self.ligand = self.select_atoms_seq(self.ligand, lig_residues)
		
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


	def transform_ligand(self, conf_num):
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

		old_coords, lchains, lres_names, lres_nums, latom_names, lnum_atoms = self.ligand
		return (coords_out, lchains, lres_names, lres_nums, latom_names, lnum_atoms)


	def get_complex(self, receptor, ligand):
		lcoords, lchains, lres_names, lres_nums, latom_names, lnum_atoms = ligand
		rcoords, rchains, rres_names, rres_nums, ratom_names, rnum_atoms = receptor
		ccoords = torch.cat([lcoords, rcoords], dim=1).contiguous()
		cchains = torch.cat([lchains, rchains], dim=1).contiguous()
		cres_names = torch.cat([lres_names, rres_names], dim=1).contiguous()
		cres_nums = torch.cat([lres_nums, rres_nums], dim=1).contiguous()
		catom_names = torch.cat([latom_names, ratom_names], dim=1).contiguous()
		cnum_atoms = lnum_atoms + rnum_atoms

		complex = (ccoords, cchains, cres_names, cres_nums, catom_names, cnum_atoms)
		return complex


def get_irmsd(benchmark_dir, benchmark_table, zdock_dir, num_conf=1000):
	benchmark = DockingBenchmark(benchmark_dir, benchmark_table)
	parser = ZDOCKParser()
	parser_native = ZDOCKParser()
	p2c = PDB2CoordsUnordered()
	
	result = {}

	problematic_alignments = ['1E6J', '2VIS', '1OYV', '1YVB', '1FC2', '1FQJ', '1GCQ', '1H9D', '1HCF', '1HE1', '1JWH', '1K74', 
								'1KLU', '1KTZ', '1ML0', '1QA9', '1US7', '1XD3', '1Z0K', '1ZHI', '2B4J', '2OOB', '1NW9', '1HE8',
								'1I2M', '1XQS', '2CFH', '2OZA']
	problematic_superposition = ['2VIS', '2FD6', '1HIA', '1YVB', '1E96', '1GCQ', '1HE1', '1I4D', '1K74', '2FJU', '3BP8', '1KKL',
								'1NW9', '2H7V', '2NZ8']
	skip = ['1N8O', '1AZS', '1GP2', '1K5D', '2J7P']
	
	for n, target_name in enumerate(benchmark.get_target_names()[145:]):
		# target_name = '1OYV'
		if target_name in skip:
			print("Skipping prediction for", target_name, n)	
			continue
		print("Processing prediction for", target_name, n)
		target = benchmark.get_target(target_name)
		bound_target, unbound_target = benchmark.parse_structures(target)
		try:
			urec_sel, ulig_sel, brec_sel, blig_sel = benchmark.get_unbound_contacts(bound_target, unbound_target, 
				align_tol=bool(target_name in problematic_alignments),
				dist_tol=bool(target_name in problematic_superposition)
				)
			
		except Exception as inst:
			if inst.args[0] == "Alignment is bad":
				print(target_name, inst.args[0])
				print(inst.args[1], inst.args[2])
				print(inst.args[3][0])
				print(inst.args[3][1])
				if target_name in problematic_alignments:
					pass
				else:
					raise inst
			
			elif inst.args[0] == "Best match is bad":
				print(target_name, inst.args[0])
				print(inst.args[1], inst.args[2])
				print(inst.args[3][0])
				print(inst.args[3][1])
				print('Best rmsd = ', inst.args[4])
				if target_name in problematic_superposition:
					pass
				else:
					raise inst
			else:
				raise inst

		
		zdock_output_name = "%s.zd3.0.2.cg.fixed.out"%target_name
		parser.parse_ouput(os.path.join(zdock_dir, zdock_output_name), False)
		parser.receptor = p2c([unbound_target["receptor"]["path"]])
		parser.ligand = p2c([unbound_target["ligand"]["path"]])
		parser.select_target_CA()
		parser.select_target_residues(urec_sel, ulig_sel)
		unbound_interface = parser.get_complex(parser.receptor, parser.ligand)

		parser_native.receptor = p2c([bound_target["receptor"]["path"]])
		parser_native.ligand = p2c([bound_target["ligand"]["path"]])
		parser_native.select_target_CA()
		parser_native.select_target_residues(brec_sel, blig_sel)
		bound_interface = parser_native.get_complex(parser_native.receptor, parser_native.ligand)

		result[target_name] = []

		for i in range(num_conf):
			new_ligand = parser.transform_ligand(i)
			decoy_interface = parser.get_complex(parser.receptor, new_ligand)
					
			c2rmsd = Coords2RMSD()
			conf_bound_rmsd = c2rmsd(decoy_interface[0], bound_interface[0], bound_interface[-1])
			
			result[target_name].append( conf_bound_rmsd.item() )

		# break
	
	return result
			


if __name__=='__main__':
	benchmark_dir = os.path.join(DATA_DIR, "DockingBenchmarkV4")
	benchmark_table = os.path.join(benchmark_dir, "TableS1.csv")
	zdock_dir = os.path.join(LOG_DIR, "ZDOCK", "DockingBenchmarkV4_15deg")

	result = get_irmsd(benchmark_dir, benchmark_table, zdock_dir, num_conf=10)
	for target_name in result.keys():
		print(target_name, "Min irmsd = ", min(result[target_name]))
	
	

