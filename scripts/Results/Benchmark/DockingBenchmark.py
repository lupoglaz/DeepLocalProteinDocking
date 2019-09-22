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
		if residue.get_id()[2] != ' ':
			print('Skipping:', residue.get_id())
			continue

		if residue.get_resname() in standard_aa_names:
			seq += dindex_to_1[d3_to_index[residue.get_resname()]]
			resnum = residue.get_id()[1]
			resnums.append(resnum)
	return seq, resnums


def get_res_dist(struct1, struct2, alignment):
	struct1_residues = Selection.unfold_entities(struct1, 'R')
	struct2_residues = Selection.unfold_entities(struct2, 'R')
	av_dist = 0.0	
	for res1, res2 in alignment:
		if ('CA' in struct1_residues[res1]) and ('CA' in struct2_residues[res2]):
			ca1 = struct1_residues[res1]['CA']
			ca2 = struct2_residues[res2]['CA']
			av_dist += ca1-ca2
		else:
			continue

	return av_dist/float(len(alignment))

def match_chains(structure1, structure2, align_tol=False, dist_tol=False):
	chain_match = {}
	
	chains1 = structure1["chains"]
	chains2 = structure2["chains"]
	
	best_match = None
	best_alignment = None
	for chain1 in chains1.keys():
		seq1, resnum1 = chains1[chain1]
		min_dist = float('+Inf')
		for chain2 in chains2.keys():
			seq2, resnum2 = chains2[chain2]
		
			if len(seq1) == 0 or len(seq2) == 0:
				raise Exception("Chain is empty")

			alignment, global_id, seq_aligned = get_alignment( (seq1, seq2) )
			av_dist = get_res_dist(structure1["structure"][chain1], structure2["structure"][chain2], alignment)
			
			if av_dist < min_dist:
				min_dist = av_dist
				best_alignment = seq_aligned
				best_match = (chain2, global_id, dict(alignment))

		if best_match[1] < 0.9 and (not align_tol):
			raise(Exception("Alignment is bad", chain1, best_match[0], best_alignment))
		
		if min_dist > 5.0 and (not dist_tol):
			raise(Exception("Best match is bad", chain1, best_match[0], best_alignment, min_dist))
		
		chain_match[chain1] = best_match
				
	return chain_match


class DockingBenchmark:
	def __init__(self, benchmark_dir, table):
		#1QFW_HL deleted
		#2I25_ added 3LZT_A chain
		#1OYV_B:I deleted
		#1XD3 added 1UCH_A
		#1Z5Y added 1L6P_A
		#2H7V added 1MH1_A
		#2NZ8 added 1MH1_A

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
					target["complex"] = self.parse_strucutre_name(sline[0].strip())
					target["receptor"] = self.parse_strucutre_name(sline[2].strip(), sline[3])
					target["ligand"] = self.parse_strucutre_name(sline[4].strip(), sline[5])
					self.targets.append(target)
	
	def parse_strucutre_name(self, name, description=None):
		sname = name.split('_')
		struct_name = sname[0]
		if len(sname) == 2:
			schains = sname[1].split(':')
			if len(schains)==1:
				chain = schains[0].split('(')[0]
				if len(chain) == 0:
					chain = " "
				return {
					"name":struct_name,
					"chain": chain,
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
					"chain": " ",
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
			if ligand_res.get_id()[2] != ' ':
				print('Skipping:', ligand_res.get_id())
				continue
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
				if receptor_res.get_id()[2] != ' ':
					print('Skipping:', receptor_res.get_id())
					continue
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
		struct1_chains = set([chain.get_id() for chain in struct1.get_chains() if len(get_chain_seq(chain)[0])>0 ])
		struct2 = pdb_parser.get_structure('X', bound_complex["ligand"]["path"])[0]
		struct2_chains = set([chain.get_id() for chain in struct2.get_chains() if len(get_chain_seq(chain)[0])>0])
		rec_chains = set(list(target["complex"]["chain_rec"]))
		lig_chains = set(list(target["complex"]["chain_lig"]))

		if rec_chains == struct1_chains and lig_chains == struct2_chains:
			pass
		elif rec_chains == struct2_chains and lig_chains == struct1_chains:
			tmp = target["complex"]["chain_rec"]
			target["complex"]["chain_rec"] = target["complex"]["chain_lig"]
			target["complex"]["chain_lig"] = tmp
		else:
			raise(Exception("Wrong chains", rec_chains, struct1_chains, lig_chains, struct2_chains))

		bound_complex["receptor"]["structure"] = struct1
		bound_complex["ligand"]["structure"] = struct2

				
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
		struct3_chains = set([chain.get_id() for chain in struct3.get_chains() if len(get_chain_seq(chain)[0])>0])
		struct4 = pdb_parser.get_structure('X', unbound_complex["ligand"]["path"])[0]
		struct4_chains = set([chain.get_id() for chain in struct4.get_chains() if len(get_chain_seq(chain)[0])>0])
		rec_chains = set(list(target["receptor"]["chain"]))
		lig_chains = set(list(target["ligand"]["chain"]))

		if rec_chains == struct3_chains and lig_chains == struct4_chains:
			pass
		elif rec_chains == struct4_chains and lig_chains == struct3_chains:
			tmp = target["ligand"]["chain"]
			target["ligand"]["chain"] = target["receptor"]["chain"]
			target["receptor"]["chain"] = tmp
		else:
			raise(Exception("Wrong chains", rec_chains, struct1_chains, lig_chains, struct2_chains))

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


	def get_unbound_contacts(self, bound_complex, unbound_complex, contact_dist=5.0, align_tol=False, dist_tol=False):
		brec_cont, blig_cont = self.get_contacts(bound_complex, contact_dist)
		
		receptor_match = match_chains( bound_complex["receptor"], unbound_complex["receptor"], align_tol, dist_tol)
		ligand_match = match_chains( bound_complex["ligand"], unbound_complex["ligand"], align_tol, dist_tol)
				
		urec_cont = []
		new_brec_cont = []
		for chain_id, res_num, res in brec_cont:
			matching_chain, id, alignment = receptor_match[chain_id]
			res_idx = bound_complex["receptor"]["chains"][chain_id][1].index(res_num)
			if not res_idx in alignment.keys():
				print("Skipping residue", chain_id, res_num, res)
				continue
			matching_res_idx = alignment[res_idx]
			matching_res_num = unbound_complex["receptor"]["chains"][matching_chain][1][matching_res_idx]
			matching_res = unbound_complex["receptor"]["chains"][matching_chain][0][matching_res_idx]
			urec_cont.append((matching_chain, matching_res_num, matching_res))
			new_brec_cont.append((chain_id, res_num, res))
			if matching_res != res:
				print(chain_id, res_num, res)
				print(matching_chain, matching_res_num, matching_res)
				s1 = bound_complex["receptor"]["chains"][chain_id][0]
				print(bound_complex["receptor"]["chains"][chain_id])
				print(s1[res_idx])
				s2 = unbound_complex["receptor"]["chains"][matching_chain][0]
				print(unbound_complex["receptor"]["chains"][matching_chain])
				print(s2[matching_res_idx])

				raise(Exception("Residues are not matching", chain_id, res_num, res, ':', matching_chain, matching_res_num, matching_res))

		ulig_cont = []
		new_blig_cont = []
		for chain_id, res_num, res in blig_cont:
			matching_chain, id, alignment = ligand_match[chain_id]		
			res_idx = bound_complex["ligand"]["chains"][chain_id][1].index(res_num)
			if not res_idx in alignment.keys():
				print("Skipping residue", chain_id, res_num, res)
				continue
			matching_res_idx = alignment[res_idx]
			matching_res_num = unbound_complex["ligand"]["chains"][matching_chain][1][matching_res_idx]
			matching_res = unbound_complex["ligand"]["chains"][matching_chain][0][matching_res_idx]
			ulig_cont.append((matching_chain, matching_res_num, matching_res))
			new_blig_cont.append((chain_id, res_num, res))
			if matching_res != res:
				raise(Exception("Residues are not matching", chain_id, res_num, res, ':', matching_chain, matching_res_num, matching_res))
		
		return urec_cont, ulig_cont, new_brec_cont, new_blig_cont

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

	rec, lig = b.get_unbound_contacts(bc, uc)
	print(rec)
	print(lig)
	
