import os
import sys
import random
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src import DATA_DIR, REPOSITORY_DIR
from global_alignment import get_complex_pdb_sequences
import _pickle as pkl

from Bio import PDB
from Bio.PDB.Polypeptide import standard_aa_names, dindex_to_1, d3_to_index

from global_alignment import get_alignment

def read_pdb_list(pdb_list_file, benchmark_dir):
	targets = [[], [], []]
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
				complex_name = sline[0]
				pdb_name = complex_name.split('_')[0]
				ureceptor_path = os.path.join(benchmark_dir, 'structures', pdb_name+'_r_u.pdb')
				uligand_path = os.path.join(benchmark_dir, 'structures', pdb_name+'_l_u.pdb')
				breceptor_path = os.path.join(benchmark_dir, 'structures', pdb_name+'_r_b.pdb')
				bligand_path = os.path.join(benchmark_dir, 'structures', pdb_name+'_l_b.pdb')
				targets[cplx_type-1].append((ureceptor_path, uligand_path, breceptor_path, bligand_path))	

	return targets

def get_chain_seq(chain):
	seq = ""
	resnums = []
	for residue in chain:
		if residue.get_resname() in standard_aa_names:
			seq += dindex_to_1[d3_to_index[residue.get_resname()]]
			resnum = residue.get_id()[1]
			resnums.append(resnum)
	return seq, resnums

def match_chains(chains_bound, chains_unbound):
	chain_match_bu = {}
	chain_match_ub = {}

	for bchain in chains_bound:
		bseq, bresnums = get_chain_seq(bchain)
		if len(bseq) == 0:
			continue
		max_id = 0.0
		matching_chain_id = None
		max_alignment = None
		for uchain in chains_unbound:
			useq, uresnums = get_chain_seq(uchain)
			if len(useq) == 0:
				continue
			alignment, global_id = get_alignment( (bseq, useq) )  

			if global_id > max_id:
				matching_chain_id = uchain.get_id()
				max_id = global_id
				max_alignment = (alignment, bresnums, uresnums)

		if matching_chain_id is None:
			print(bchain)
			print(uchain)
			print(bseq)
			print(useq)
			sys.exit()
		
		chain_match_bu[bchain.get_id()] = (matching_chain_id, max_id, max_alignment)
		chain_match_ub[matching_chain_id] = (bchain.get_id(), max_id, max_alignment)

	return chain_match_bu, chain_match_ub

def align_chains(targets):
	parser = PDB.PDBParser(QUIET=True)
		
	chain_match = {}

	for ureceptor_path, uligand_path, breceptor_path, bligand_path in (targets[0]+targets[1]+targets[2]):
		complex_name = breceptor_path.split('_')[0].split('/')[-1]
		print(complex_name)
		breceptor = parser.get_structure('R', breceptor_path)[0]
		ureceptor = parser.get_structure('R', ureceptor_path)[0]

		bligand = parser.get_structure('L', bligand_path)[0]
		uligand = parser.get_structure('L', uligand_path)[0]

		chain_match[complex_name] = {}
		chain_match_r, _ = match_chains(breceptor, ureceptor)
		# print(chain_match_r)
		chain_match_l, _ = match_chains(bligand, uligand)
		# print(chain_match_l)

		chain_match[complex_name] = (chain_match_r, chain_match_l)

	return chain_match


def merge_bound_chains(targets, chain_match, output_dir):
	
	parser = PDB.PDBParser(QUIET=True)
	io = PDB.PDBIO()
	bound = PDB.StructureBuilder.StructureBuilder()
	ubound = PDB.StructureBuilder.StructureBuilder()
	
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)

	form_pdb_list = []
	for ureceptor_path, uligand_path, breceptor_path, bligand_path in (targets[0]+targets[1]+targets[2]):
		complex_name = breceptor_path.split('_')[0].split('/')[-1]
		print(complex_name)
		chain_match_r, chain_match_l = chain_match[complex_name]
		
		bound.init_structure("bound")
		bound.init_model(0)
		bound.init_chain('R')
		bound.init_seg("R")
		
		ubound.init_structure("ubound")
		ubound.init_model(0)
		ubound.init_chain('R')
		ubound.init_seg("R")
		
		new_res_num = 1
		new_atom_num = 1
				
		breceptor = parser.get_structure('R', breceptor_path)
		ureceptor = parser.get_structure('R', ureceptor_path)
		for bchain in chain_match_r.keys():
			uchain, _, aligment = chain_match_r[bchain]
			mapping, bresnums, uresnums = aligment
			print('receptor:', bchain, uchain)

			for bresidue_idx, uresidue_idx in mapping:
				try:
					bresidue = breceptor[0][bchain][bresnums[bresidue_idx]]
					uresidue = ureceptor[0][uchain][uresnums[uresidue_idx]]
				except:
					continue

				batoms = set([atom.get_name() for atom in bresidue])
				uatoms = set([atom.get_name() for atom in uresidue])
				if batoms != uatoms:
					continue
				
				bound.init_residue(bresidue.get_resname(), " ", new_res_num, " ")
				ubound.init_residue(uresidue.get_resname(), " ", new_res_num, " ")

				for batom, uatom in zip(bresidue, uresidue):
					bound.init_atom(batom.get_name(), batom.get_coord(), batom.get_bfactor(), batom.get_occupancy(), batom.get_altloc(), batom.get_fullname(), new_atom_num)
					ubound.init_atom(uatom.get_name(), uatom.get_coord(), uatom.get_bfactor(), uatom.get_occupancy(), uatom.get_altloc(), uatom.get_fullname(), new_atom_num)
					new_atom_num += 1
				new_res_num += 1
		
		bound.init_chain('L')
		bound.init_seg("L")
		
		ubound.init_chain('L')
		ubound.init_seg("L")
		
		new_res_num = 1
		new_atom_num = 1

		bligand = parser.get_structure('R', bligand_path)
		uligand = parser.get_structure('R', uligand_path)
		for bchain in chain_match_l.keys():
			uchain, _, aligment = chain_match_l[bchain]
			mapping, bresnums, uresnums = aligment
			print('ligand:', bchain, uchain)

			for bresidue_idx, uresidue_idx in mapping:
				try:
					bresidue = bligand[0][bchain][bresnums[bresidue_idx]]
					uresidue = uligand[0][uchain][uresnums[uresidue_idx]]
				except:
					continue

				batoms = set([atom.get_name() for atom in bresidue])
				uatoms = set([atom.get_name() for atom in uresidue])
				if batoms != uatoms:
					continue
				
				bound.init_residue(bresidue.get_resname(), " ", new_res_num, " ")
				ubound.init_residue(uresidue.get_resname(), " ", new_res_num, " ")
				
				for batom, uatom in zip(bresidue, uresidue):
					bound.init_atom(batom.get_name(), batom.get_coord(), batom.get_bfactor(), batom.get_occupancy(), batom.get_altloc(), batom.get_fullname(), new_atom_num)
					ubound.init_atom(uatom.get_name(), uatom.get_coord(), uatom.get_bfactor(), uatom.get_occupancy(), uatom.get_altloc(), uatom.get_fullname(), new_atom_num)
					new_atom_num += 1
				new_res_num += 1
		
		new_bound = bound.get_structure()
		io.set_structure(new_bound)
		io.save(os.path.join(output_dir, complex_name+'_b.pdb'))
		
		new_bound_r = bound.get_structure()[0]["R"]
		io.set_structure(new_bound_r)
		io.save(os.path.join(output_dir, complex_name+'_r_b.pdb'))

		new_bound_l = bound.get_structure()[0]["L"]
		io.set_structure(new_bound_l)
		io.save(os.path.join(output_dir, complex_name+'_l_b.pdb'))
		
		new_ubound = ubound.get_structure()
		io.set_structure(new_ubound)
		io.save(os.path.join(output_dir, complex_name+'_u.pdb'))

		new_ubound_r = ubound.get_structure()[0]["R"]
		io.set_structure(new_ubound_r)
		io.save(os.path.join(output_dir, complex_name+'_r_u.pdb'))

		new_ubound_l = ubound.get_structure()[0]["L"]
		io.set_structure(new_ubound_l)
		io.save(os.path.join(output_dir, complex_name+'_l_u.pdb'))


if __name__=='__main__':
	targets = read_pdb_list("/media/lupoglaz/ProteinsDataset/DockingBenchmarkV5/Table_BM5.csv",
							benchmark_dir="/media/lupoglaz/ProteinsDataset/DockingBenchmarkV5/")
	# save_sequences(targets, 'benchmark5_sequences.pkl')
	# targets = read_pdb_list("/media/lupoglaz/ProteinsDataset/DockingBenchmarkV4/TableS1.csv",
	# 						benchmark_dir="/media/lupoglaz/ProteinsDataset/DockingBenchmarkV4/")
	
	
	if os.path.exists("benchmark5_chain_match.pkl"):
		with open("benchmark5_chain_match.pkl", 'rb') as fin:
			match = pkl.load(fin)
	else:
		match = align_chains(targets)
		with open("benchmark5_chain_match.pkl", 'wb') as fout:
			pkl.dump(match, fout)
	
	merge_bound_chains(targets, match, "/media/lupoglaz/ProteinsDataset/DockingBenchmarkV5/Matched")
