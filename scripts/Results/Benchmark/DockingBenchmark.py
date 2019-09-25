import os
import sys
import torch
from math import *
from collections import OrderedDict
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from src import LOG_DIR, DATA_DIR

from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered, CoordsTranslate, CoordsRotate, writePDB

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from scripts.Dataset.global_alignment import get_alignment

import Bio.PDB
from Bio.PDB.Polypeptide import d3_to_index, dindex_to_1, standard_aa_names
from Bio.PDB import PDBParser, NeighborSearch, Selection
pdb_parser = PDBParser(QUIET=True)

import subprocess
import json
from LinkedSelection import LinkedSelection, double_link_selections

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

def get_self_match(structure, align_tol=0.8):
	chains = structure["chains"]
	matches = {}
	for chain1 in chains.keys():
		seq1, resnum1 = chains[chain1]
		matches[chain1] = []
		for chain2 in chains.keys():
			if chain1 == chain2 : continue
			seq2, resnum2 = chains[chain2]
			alignment, global_id, seq_aligned = get_alignment( (seq1, seq2) )
			if global_id >= align_tol:
				matches[chain1].append( (chain2, global_id, dict(alignment)) )
	return matches

def get_symmetry(protein_path, align_tol=0.8, rmsd_tol=5.0, anans_path="./tools/ananas"):
	exec_path = os.path.abspath(anans_path)
	prot_abs_path = os.path.abspath(protein_path)
	tmp_output_path = os.path.abspath('tmp.json')
	process = subprocess.check_output([exec_path, prot_abs_path, "-C", str(rmsd_tol), "-p", "-a", "-j", tmp_output_path])
	
	with open(tmp_output_path, "r") as fin:
		data = json.load(fin)
	
	if data is None:
		return None
		print(str(process))

	permutations = []

	for symmetry in data:
		for transform in symmetry['transforms']:
			permutations.append(transform['permutation'])
			for i in range(transform['ORDER']-1):
				permutations.append([permutations[-1][i] for i in transform['permutation']])
				
	return permutations

def get_best_match(structure1, structure2, align_tol=False, dist_tol=False):
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

def get_match(structure1, structure2, chain_match, align_tol=0.6):
	match = {}
	chains1 = structure1["chains"]
	chains2 = structure2["chains"]
	for chain1 in chains1.keys():
		seq1, resnum1 = chains1[chain1]
		chain2 = chain_match[chain1]
		seq2, resnum2 = chains2[chain2]
		alignment, global_id, seq_aligned = get_alignment( (seq1, seq2) )
		if global_id < align_tol:
			return None
		match[chain1] = (chain2, global_id, dict(alignment))
	return match

class DockingBenchmark:
	def __init__(self, benchmark_dir, table, structures_dir):
		#1QFW_HL deleted
		#2I25_ added 3LZT_A chain
		#1OYV_B:I deleted
		#1XD3 added 1UCH_A
		#1Z5Y added 1L6P_A
		#2H7V added 1MH1_A
		#2NZ8 added 1MH1_A
		#3CPH wrong chain in unbound ligand 1G16_C
		#1JMO_l_b.pdb swapped H and L chains in the structure

		self.benchmark_dir = benchmark_dir
		self.structures_dir = structures_dir
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
						"irmsd": None,
						"dasa": None,
						"version": None,
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
					target["irmsd"] = float(sline[6])
					target["dasa"] = float(sline[7])
					target["version"] = float(sline[8])
					self.targets.append(target)

	def save_table(self, table_path):
		with open(table_path, 'w') as fout:
			for difficulty in [1, 2, 3]:
				target_names = self.get_target_names(difficulty=difficulty)
				if difficulty == 1:
					fout.write("Rigid-body(%d)\n"%(len(target_names)))
				elif difficulty == 2:
					fout.write("Medium Difficulty(%d)\n"%(len(target_names)))
				elif difficulty == 3:
					fout.write("Difficult(%d)\n"%(len(target_names)))

				for target_name in target_names:
					target = self.get_target(target_name)
					fout.write("%s_%s:%s"%(target["complex"]["name"], target["complex"]["chain_rec"], target["complex"]["chain_lig"]))
					fout.write('\t')
					fout.write(target["category"])
					fout.write('\t')
					fout.write("%s_%s"%(target["receptor"]["name"], target["receptor"]["chain"]))
					fout.write('\t')
					fout.write("%s"%(target["receptor"]["description"]))
					fout.write('\t')
					fout.write("%s_%s"%(target["ligand"]["name"], target["ligand"]["chain"]))
					fout.write('\t')
					fout.write("%s"%(target["ligand"]["description"]))
					fout.write('\t')
					fout.write("%.2f\t%.0f\t%d\n"%(target["irmsd"], target["dasa"], target["version"]))
	
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

	def get_difficulies(self):
		return set([target["difficulty"] for target in self.targets])
	
	def get_categories(self):
		return set([target["category"] for target in self.targets])

	def get_target_names(self, difficulty=None, category=None):
		if (difficulty is None) and (category is None):
			return [target["complex"]["name"] for target in self.targets]
		elif (not difficulty is None) and (category is None):
			return [target["complex"]["name"] for target in self.targets if target["difficulty"] in difficulty]
		elif (difficulty is None) and (not category is None):
			return [target["complex"]["name"] for target in self.targets if target["category"] in category]
		else:
			return [target["complex"]["name"] for target in self.targets if (target["difficulty"] in difficulty) and (target["category"] in category)]

	def get_target(self, target_name):
		return self.targets[self.get_target_names().index(target_name)]

	def restrict_to_set(self, target_set):
		all_targets = self.get_target_names()
		for index in range(len(all_targets)-1, -1, -1):
			target = all_targets[index]
			if not target in target_set:
				del self.targets[index]

	def get_prob_alignments(self):
		return ['1E6J', '2VIS', '1OYV', '1YVB', '1FC2', '1FQJ', '1GCQ', '1H9D', '1HCF', '1HE1', '1JWH', '1K74', 
				'1KLU', '1KTZ', '1ML0', '1QA9', '1US7', '1XD3', '1Z0K', '1ZHI', '2B4J', '2OOB', '1NW9', '1HE8',
				'1I2M', '1XQS', '2CFH', '2OZA', '1H1V', '1JK9', '1R8S', '2IDO']

	def get_prob_superpos(self):
		return	['2VIS', '2FD6', '1HIA', '1YVB', '1E96', '1GCQ', '1HE1', '1I4D', '1K74', '2FJU', '3BP8', '1KKL',
				'1NW9', '2H7V', '2NZ8', '3CPH', '1F6M', '1ZLI', '1H1V', '1IRA', '1JK9', '1JMO', '1Y64']

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
		
		return LinkedSelection(list(contacts_rec)), LinkedSelection(list(contacts_lig))

	def parse_protein(self, structure_name, target_chains=None):
		protein = {
			"path": os.path.join(self.structures_dir, structure_name),
			"structure": None,
			"chains": OrderedDict({})
		}
		if not os.path.exists(protein["path"]):
			raise(Exception("File not found", protein["path"]))

		protein["structure"] = pdb_parser.get_structure('X', protein["path"])[0]
		struct_chains = [chain.get_id() for chain in protein["structure"].get_chains() if len(get_chain_seq(chain)[0])>0 ]
		
		if not target_chains is None:
			table_chains = list(target_chains)
			if not (struct_chains == table_chains):
				if set(struct_chains) == set(table_chains):
					print(structure_name, ":Wrong order of chains in the table")
				else:
					raise(Exception(structure_name, "Wrong chains", table_chains, struct_chains))
		
		#complex receptor chains sequences
		for chain in struct_chains:
			seq, resnums = get_chain_seq(protein["structure"][chain])
			protein["chains"][chain] = (seq, resnums)

		return protein

	def parse_structures(self, target, suffix=""):
		
		bound_complex = OrderedDict({
			"receptor": None,
			"ligand": None
		})
		try:
			bound_complex["receptor"] = self.parse_protein(target["complex"]["name"]+'_r_b.pdb%s'%(suffix), target["complex"]["chain_rec"])
			bound_complex["ligand"] = self.parse_protein(target["complex"]["name"]+'_l_b.pdb%s'%(suffix), target["complex"]["chain_lig"])
		except Exception as inst:
			if inst.args[1] == 'Wrong chains':
				tmp = target["complex"]["chain_rec"]
				target["complex"]["chain_rec"] = target["complex"]["chain_lig"]
				target["complex"]["chain_lig"] = tmp
				
				bound_complex["receptor"] = self.parse_protein(target["complex"]["name"]+'_r_b.pdb%s'%(suffix), target["complex"]["chain_rec"])
				bound_complex["ligand"] = self.parse_protein(target["complex"]["name"]+'_l_b.pdb%s'%(suffix), target["complex"]["chain_lig"])

				print('Swapped chains in the table')
		
				
		unbound_complex = OrderedDict({
			"receptor": None,
			"ligand": None
		})
		try:
			unbound_complex["receptor"] = self.parse_protein(target["complex"]["name"]+'_r_u.pdb', target["receptor"]["chain"])
			unbound_complex["ligand"] = self.parse_protein(target["complex"]["name"]+'_l_u.pdb', target["ligand"]["chain"])
		except Exception as inst:
			if inst.args[1] == 'Wrong chains':
				tmp = target["receptor"]["chain"]
				target["receptor"]["chain"] = target["ligand"]["chain"]
				target["ligand"]["chain"] = tmp
				
				unbound_complex["receptor"] = self.parse_protein(target["complex"]["name"]+'_r_u.pdb', target["receptor"]["chain"])
				unbound_complex["ligand"] = self.parse_protein(target["complex"]["name"]+'_l_u.pdb', target["ligand"]["chain"])

				print('Swapped chains in the table')

		return bound_complex, unbound_complex

	def transfer_selection(self, selection_src, structure_src, structure_dst, match):
		selection_dst = []
		match_src2dst = {}
		to_remove = []
		for chain_id, res_num, res in selection_src:
			matching_chain, id, alignment = match[chain_id]
			res_idx = structure_src["chains"][chain_id][1].index(res_num)
			if not res_idx in alignment.keys():
				to_remove.append((chain_id, res_num, res))
				continue
			matching_res_idx = alignment[res_idx]
			matching_res_num = structure_dst["chains"][matching_chain][1][matching_res_idx]
			matching_res = structure_dst["chains"][matching_chain][0][matching_res_idx]
			selection_dst.append((matching_chain, matching_res_num, matching_res))
			match_src2dst[(chain_id, res_num, res)] = (matching_chain, matching_res_num, matching_res)
			if matching_res != res:
				raise(Exception("Residues are not matching", chain_id, res_num, res, ':', matching_chain, matching_res_num, matching_res))
		
		for element in to_remove:
			selection_src.remove(element)
		
		selection_dst = LinkedSelection(selection_dst)
		double_link_selections(selection_src, selection_dst, match_src2dst)
		return selection_dst

	def intersect_selections(self, selection_a, selection_b, match):
		new_selection_a = []
		new_selection_b = []
		for chain_id, res_num, res in selection_a:
			matching_chain, id, alignment = match[chain_id]
			matching_res_idx = alignment[res_idx]
	
	def get_symmetric_selections(self, protein, selection):
		permutations = get_symmetry(protein["path"])
		if permutations is None:
			permutations = [[i for i, chain in enumerate(protein["chains"].keys())]]
		permuted_selections = []
		for permutation in permutations:
			chain_list = list(protein["chains"].keys())
			permuted_chain_list = [chain_list[i] for i in permutation]
			chain_match = dict(zip(chain_list, permuted_chain_list))
			self_match = get_match(protein, protein, chain_match, align_tol=0.5)
			if self_match is None:
				continue
			permuted_selections.append( self.transfer_selection(selection, protein, protein, self_match) )
		return permuted_selections

	def get_unbound_interfaces(self, bound_complex, unbound_complex, contact_dist=5.0, align_tol=False, dist_tol=False):
		brec_cont, blig_cont = self.get_contacts(bound_complex, contact_dist)
		
		brec_sym_cont = self.get_symmetric_selections(bound_complex["receptor"], brec_cont)
		blig_sym_cont = self.get_symmetric_selections(bound_complex["ligand"], blig_cont)
				
		receptor_match = get_best_match( bound_complex["receptor"], unbound_complex["receptor"], True, True)
		ligand_match = get_best_match( bound_complex["ligand"], unbound_complex["ligand"], True, True)
		
		int_urec = []
		int_ulig = []
		for receptor_interface in brec_sym_cont:
			for ligand_interface in blig_sym_cont:
				a = self.transfer_selection(receptor_interface, bound_complex["receptor"], unbound_complex["receptor"], receptor_match)
				b = self.transfer_selection(ligand_interface, bound_complex["ligand"], unbound_complex["ligand"], ligand_match)
				int_urec.append(a)
				int_ulig.append(b)
		
		assembly = []
		i = 0 
		for receptor_interface in brec_sym_cont:
			for ligand_interface in blig_sym_cont:
				urec_cont, ulig_cont = int_urec[i], int_ulig[i]
				assembly.append( (urec_cont.selection, ulig_cont.selection, receptor_interface.selection, ligand_interface.selection) )
				i+=1

		return assembly

	def check_match_permutation(self, protein1, protein2, align_tol=True, dist_tol=True):
		proteins_match = get_best_match( protein1, protein2, align_tol, dist_tol)
		match = []
		for chain_idx, chain_name in enumerate(protein1["chains"].keys()):
			matching_chain_name = proteins_match[chain_name][0]
			matching_chain_idx = list(protein2["chains"].keys()).index(matching_chain_name)
			print(chain_idx, chain_name, matching_chain_idx, matching_chain_name)
			match.append(matching_chain_idx)
		
		return sorted(match) != match

	def check_symmetry(self, complex, align_tol=0.8):
		receptor_sym = False		
		receptor_self_match = get_self_match(complex["receptor"], align_tol=align_tol)
		for chain1 in receptor_self_match.keys():
			for chain2, global_id, alignment in receptor_self_match[chain1]:
				if chain1 != chain2:
					receptor_sym = True
		
		ligand_sym = False
		ligand_self_match = get_self_match(complex["ligand"], align_tol=align_tol)
		for chain1 in ligand_self_match.keys():	
			for chain2, global_id, alignment in ligand_self_match[chain1]:
				if chain1 != chain2:
					ligand_sym = True
						
		return receptor_sym, ligand_sym

	

if __name__=='__main__':
	print(os.listdir(DATA_DIR))
	print(os.listdir(LOG_DIR))

	benchmark_dir = os.path.join(DATA_DIR, "DockingBenchmarkV4")
	benchmark_list = os.path.join(benchmark_dir, "TableS1.csv")
	benchmark_structures = os.path.join(benchmark_dir, "structures")
	benchmark = DockingBenchmark(benchmark_dir, benchmark_list, benchmark_structures)
	for target_idx, target_name in enumerate(benchmark.get_target_names()[11:]):
		print('Processing target', target_name, target_idx)
		target = benchmark.get_target(target_name)
		bound_complex, unbound_complex = benchmark.parse_structures(target)
		
		print('Bound receptor:', target["complex"]["chain_rec"], ''.join(list(bound_complex["receptor"]["chains"].keys())))
		print('Bound ligand:', target["complex"]["chain_lig"], ''.join(list(bound_complex["ligand"]["chains"].keys())))
		print('Unbound receptor:', target["receptor"]["chain"], ''.join(list(unbound_complex["receptor"]["chains"].keys())))
		print('Unbound ligand:', target["ligand"]["chain"], ''.join(list(unbound_complex["ligand"]["chains"].keys())))
		
		perm_rec = benchmark.check_match_permutation(bound_complex["receptor"], unbound_complex["receptor"])
		if perm_rec:
			raise Exception("Permutation in receptor")
		perm_lig = benchmark.check_match_permutation(bound_complex["ligand"], unbound_complex["ligand"])
		if perm_lig:
			raise Exception("Permutation in ligand")

		print('Symmetry:')
		rec_sym, lig_sym = benchmark.check_symmetry(bound_complex)
		print(rec_sym, lig_sym)
		if rec_sym:
			get_symmetry(bound_complex["receptor"]["path"])
			raise Exception("Symmetry detected")
		if lig_sym:
			get_symmetry(bound_complex["ligand"]["path"])
			raise Exception("Symmetry detected")

		break

	benchmark.save_table(os.path.join(benchmark_dir, "TableCorrect.csv"))
	
