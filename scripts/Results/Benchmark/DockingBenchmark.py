import os
import sys
import torch
from math import *
from collections import OrderedDict
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from src import LOG_DIR, DATA_DIR

from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered, CoordsTranslate, CoordsRotate, writePDB

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from scripts.Dataset.processing_utils import _get_contacts, _get_fnat, _get_capri_quality
from scripts.Dataset.global_alignment import get_alignment

import Bio.PDB
from Bio.PDB.Polypeptide import d3_to_index, dindex_to_1, standard_aa_names
from Bio.PDB import PDBParser, NeighborSearch, Selection
pdb_parser = PDBParser(QUIET=True)

def get_chain_seq(chain):
	seq = ""
	resnums = []
	for residue in chain:
		if residue.get_resname() in standard_aa_names:
			seq += dindex_to_1[d3_to_index[residue.get_resname()]]
			resnum = residue.get_id()[1]
			resnums.append(resnum)
	return seq, resnums
	

def match_chains(chains1, chains2):
	chain_match_bu = {}
	
	for bchain in chains1.keys():
		bseq, bresnum = chains1[bchain]
		if len(bseq) == 0:
			raise Exception("Chain is empty", bchain)
		max_id = 0.0
		matching_chain_id = None
		max_alignment = None
		for uchain in chains2.keys():
			useq, uresnums = chains2[uchain]
			if len(useq) == 0:
				raise Exception("Chain is empty", uchain)
			
			alignment, global_id = get_alignment( (bseq, useq) )
			if global_id > max_id:
				matching_chain_id = uchain
				max_id = global_id
				max_alignment = alignment

		if matching_chain_id is None:
			raise Exception("Can't align chains", bchain, uchain)
					
		chain_match_bu[bchain] = (matching_chain_id, max_id, dict(max_alignment))
		
	return chain_match_bu

class DockingBenchmark:
	def __init__(self, benchmark_dir, table):
		self.benchmark_dir = benchmark_dir
		table_path = os.path.join(benchmark_dir, table)
		self.targets = []
		difficulty = 0
		with open(table) as fin:
			for line in fin:
				if line.find("Rigid-body")!=-1:
					difficulty = 1
					continue
				elif line.find("Medium Difficulty")!=-1:
					difficulty = 2
					continue
				elif line.find("Difficult")!=-1:
					difficulty = 3
					continue
				
				if difficulty>0:
					target = OrderedDict({
						"difficulty": None,
						"category": None,
						"IRMSD": None,
						"complex": {
							"name": None,
							"chain_rec": None,
							"chain_lig": None
						},
						"receptor": {
							"name": None,
							"chain": None,
							"description": None
						},
						"ligand": {
							"name": None,
							"chain": None,
							"description": None
						}
					})

					sline = line.split('\t')
					target["difficulty"] = difficulty
					target["category"] = sline[1]
					target["complex"] = self.parse_strucutre_name(sline[0])
					target["receptor"] = self.parse_strucutre_name(sline[2], sline[3])
					target["ligand"] = self.parse_strucutre_name(sline[4], sline[5])
					self.targets.append(target)
	
	def parse_strucutre_name(self, name, description=None):
		sname = name.split('_')
		struct_name = sname[0]
		if len(sname) == 2:
			schains = sname[1].split(':')
			if len(schains)==1:
				return {
					"name":struct_name,
					"chain": schains[0],
					"description": description
				}
			else:
				return {
					"name":struct_name,
					"chain_rec": schains[0],
					"chain_lig": schains[1]
				}
				return struct_name, chain_rec, chain_lig
		else:
			return {
					"name":struct_name,
					"chain": "",
					"description": description
				}

	def get_target_names(self):
		return [target["complex"]["name"] for target in self.targets]

	def get_contacts(self, complex, contact_dist=5.0):
		receptor_atoms = Selection.unfold_entities(complex["receptor"]["structure"], 'A')
		ns = NeighborSearch(receptor_atoms)
	
		ligand_residues = Selection.unfold_entities(complex["ligand"]["structure"], 'R')

		contacts_lig = set([])
		contacts_rec = set([])
		for ligand_res in ligand_residues:
			if not ligand_res.get_resname() in standard_aa_names:
				continue
			lig_resname = dindex_to_1[d3_to_index[ligand_res.get_resname()]]
			lig_resnum = ligand_res.get_id()[1]
			lig_chname = ligand_res.get_parent().get_id()
			res_contacts = []
			
			for lig_atom in ligand_res:
				neighbors = ns.search(lig_atom.get_coord(), contact_dist)
				res_contacts += Selection.unfold_entities(neighbors, 'R')
			
			for receptor_res in res_contacts:
				if not receptor_res.get_resname() in standard_aa_names:
					continue
				rec_resname = dindex_to_1[d3_to_index[receptor_res.get_resname()]]
				rec_resnum = receptor_res.get_id()[1]
				rec_chname = receptor_res.get_parent().get_id()
				
				contacts_lig.add( (lig_chname, lig_resnum, lig_resname))
				contacts_rec.add( (rec_chname, rec_resnum, rec_resname))
		
		return contacts_rec, contacts_lig

	def parse_structures(self, target):
		
		bound_complex = OrderedDict({
			"receptor": {
				"path": os.path.join(self.benchmark_dir, "structures", target["complex"]["name"]+'_r_b.pdb'),
				"structure": None,
				"chains": {}
			},
			"ligand": {
				"path": os.path.join(self.benchmark_dir, "structures", target["complex"]["name"]+'_l_b.pdb'),
				"structure": None,
				"chains": {}
			}
		})
		
		struct1 = pdb_parser.get_structure('X', bound_complex["receptor"]["path"])[0]
		struct1_chains = set([chain.get_id() for chain in struct1.get_chains()])
		struct2 = pdb_parser.get_structure('X', bound_complex["ligand"]["path"])[0]
		struct2_chains = set([chain.get_id() for chain in struct2.get_chains()])
		rec_chains = set(list(target["complex"]["chain_rec"]))
		lig_chains = set(list(target["complex"]["chain_lig"]))

		if rec_chains == struct1_chains and lig_chains == struct2_chains:
			pass
		elif rec_chains == struct2_chains and lig_chains == struct1_chains:
			tmp = target["complex"]["chain_rec"]
			target["complex"]["chain_rec"] = target["complex"]["chain_lig"]
			target["complex"]["chain_lig"] = tmp
		else:
			raise(Exception("Wrong chains"))

		bound_complex["receptor"]["structure"] = struct1
		bound_complex["ligand"]["structure"] = struct2

		r_chains = set([chain.get_id() for chain in bound_complex["receptor"]["structure"].get_chains()])
		l_chains = set([chain.get_id() for chain in bound_complex["ligand"]["structure"].get_chains()])
		print(r_chains, l_chains)
		print(target["complex"]["chain_rec"], target["complex"]["chain_lig"])

		#complex receptor chains sequences
		for chain in target["complex"]["chain_rec"]:
			seq, resnums = get_chain_seq(bound_complex["receptor"]["structure"][chain])
			bound_complex["receptor"]["chains"][chain] = (seq, resnums)
		
		#complex ligand chains sequences
		complex_lig_seq = []
		for chain in target["complex"]["chain_lig"]:
			seq, resnums = get_chain_seq(bound_complex["ligand"]["structure"][chain])
			bound_complex["ligand"]["chains"][chain] = (seq, resnums)


		unbound_complex = OrderedDict({
			"receptor": {
				"path": os.path.join(self.benchmark_dir, "structures", target["complex"]["name"]+'_r_u.pdb'),
				"structure": None,
				"chains": {}
			},
			"ligand": {
				"path": os.path.join(self.benchmark_dir, "structures", target["complex"]["name"]+'_l_u.pdb'),
				"structure": None,
				"chains": {}
			}
		})

		struct3 = pdb_parser.get_structure('X', unbound_complex["receptor"]["path"])[0]
		struct3_chains = set([chain.get_id() for chain in struct3.get_chains()])
		struct4 = pdb_parser.get_structure('X', unbound_complex["ligand"]["path"])[0]
		struct4_chains = set([chain.get_id() for chain in struct4.get_chains()])
		rec_chains = set(list(target["receptor"]["chain"]))
		lig_chains = set(list(target["ligand"]["chain"]))

		if rec_chains == struct3_chains and lig_chains == struct4_chains:
			pass
		elif rec_chains == struct4_chains and lig_chains == struct3_chains:
			tmp = target["ligand"]["chain"]
			target["ligand"]["chain"] = target["receptor"]["chain"]
			target["receptor"]["chain"] = tmp
		else:
			raise(Exception("Wrong chains"))

		unbound_complex["receptor"]["structure"] = struct3
		unbound_complex["ligand"]["structure"] = struct4

		#unbound receptor chains sequences
		for chain in target["receptor"]["chain"]:
			seq, resnums = get_chain_seq(unbound_complex["receptor"]["structure"][chain])
			unbound_complex["receptor"]["chains"][chain] = (seq, resnums)
		
		#unbound ligand chains sequences
		for chain in target["ligand"]["chain"]:
			seq, resnums = get_chain_seq(unbound_complex["ligand"]["structure"][chain])
			unbound_complex["ligand"]["chains"][chain] = (seq, resnums)

		return bound_complex, unbound_complex

	def match_complexes(self, complex1, complex2, cross_check=False):
		c1r_c2r = match_chains( complex1["receptor"]["chains"], complex2["receptor"]["chains"] )
		c1l_c2l = match_chains( complex1["ligand"]["chains"], complex2["ligand"]["chains"] )

		if cross_check:
			av_id = 0.0
			print("Receptor <-> receptor:")
			for chain_id in c1r_c2r.keys():
				match, identity, mapping = c1r_c2r[chain_id]
				av_id += identity
				print(match, identity)

			print("Ligand <-> ligand:")
			for chain_id in c1l_c2l.keys():
				match, identity, mapping = c1l_c2l[chain_id]
				av_id += identity
				print(match, identity)

			av_id /= float(len(c1r_c2r.keys())+ len(c1l_c2l.keys()))

			print("Ligand <-> receptor:")
			av_cross_id = 0.0
			c11_c2r = match_chains( complex1["ligand"]["chains"], complex2["receptor"]["chains"] )
			for chain_id in c11_c2r.keys():
				match, identity, mapping = c11_c2r[chain_id]
				av_cross_id += identity
				print(match, identity)
			
			print("Receptor <-> ligand:")
			c1r_c2l = match_chains( complex1["receptor"]["chains"], complex2["ligand"]["chains"] )
			for chain_id in c1r_c2l.keys():
				match, identity, mapping = c1r_c2l[chain_id]
				av_cross_id += identity
				print(match, identity)

			av_cross_id /= float(len(c11_c2r.keys())+ len(c1r_c2l.keys()))
			
			print("Norm: %.3f Cross: %.3f"%(av_id, av_cross_id))
			if av_id < av_cross_id:
				raise Exception("Receptor and ligand are swapped")
		
		return c1r_c2r, c1l_c2l

	def get_unbound_contacts(self, target, contact_dist=5.0):
		bound_complex, unbound_complex = self.parse_structures(target)
		brec_cont, blig_cont = self.get_contacts(bound_complex, contact_dist)
		receptor_match, ligand_match = self.match_complexes(bound_complex, unbound_complex, True)
				
		urec_cont = set([])
		for chain_id, res_num, res in brec_cont:
			matching_chain, id, alignment = receptor_match[chain_id]
			res_idx = bound_complex["receptor"]["chains"][chain_id][1].index(res_num)
			if not res_idx in alignment.keys():
				print("Skipping residue", chain_id, res_num, res)
				continue
			matching_res_idx = alignment[res_idx]
			matching_res_num = unbound_complex["receptor"]["chains"][matching_chain][1][matching_res_idx]
			matching_res = unbound_complex["receptor"]["chains"][matching_chain][0][matching_res_idx]
			urec_cont.add((matching_chain, matching_res_num, matching_res))
			if matching_res != res:
				raise(Exception("Residues are not matching", chain_id, res_num, res, ':', matching_chain, matching_res_num, matching_res))

		ulig_cont = set([])
		for chain_id, res_num, res in blig_cont:
			matching_chain, id, alignment = ligand_match[chain_id]		
			res_idx = bound_complex["ligand"]["chains"][chain_id][1].index(res_num)
			if not res_idx in alignment.keys():
				print("Skipping residue", chain_id, res_num, res)
				continue
			matching_res_idx = alignment[res_idx]
			matching_res_num = unbound_complex["ligand"]["chains"][matching_chain][1][matching_res_idx]
			matching_res = unbound_complex["ligand"]["chains"][matching_chain][0][matching_res_idx]
			ulig_cont.add((matching_chain, matching_res_num, matching_res))
			if matching_res != res:
				print("Residues are not matching", chain_id, res_num, res, ':', matching_chain, matching_res_num, matching_res)
		
		return urec_cont, ulig_cont		

	def get_target(self, target_name):
		return self.targets[self.get_target_names().index(target_name)]

if __name__=='__main__':
	print(os.listdir(DATA_DIR))
	print(os.listdir(LOG_DIR))

	benchmark_dir = os.path.join(DATA_DIR, "DockingBenchmarkV4")
	benchmark_list = os.path.join(benchmark_dir, "TableS1.csv")
	b = DockingBenchmark(benchmark_dir, benchmark_list)
	
	bc, uc = b.parse_structures(b.targets[0])
	print(bc["receptor"]["chains"])

	rmatch, lmatch = b.match_complexes(bc, uc, True)
	
	cr, cl = b.get_contacts(bc)

	rec, lig = b.get_unbound_contacts(b.targets[0])
	print(rec)
	print(lig)
	
