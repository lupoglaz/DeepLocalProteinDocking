import os
import sys
import random
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src import DATA_DIR, REPOSITORY_DIR
from global_alignment import get_complex_pdb_sequences
import _pickle as pkl

from Bio import PDB
from Bio.PDB.Polypeptide import standard_aa_names

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

def save_sequences(targets, filename):
	form_pdb_list = []
	for ureceptor_path, uligand_path, breceptor_path, bligand_path in (targets[0]+targets[1]+targets[2]):
		form_pdb_list.append( (ureceptor_path, uligand_path) )

	sequences = get_complex_pdb_sequences(form_pdb_list)
	with open(filename, 'wb') as fout:
		pkl.dump(sequences, fout)

if __name__=='__main__':
	# targets = read_pdb_list("/media/lupoglaz/ProteinsDataset/DockingBenchmarkV5/Table_BM5.csv",
	# 						benchmark_dir="/media/lupoglaz/ProteinsDataset/DockingBenchmarkV5/")
	# save_sequences(targets, 'benchmark_sequences.pkl')
	targets = read_pdb_list("/media/lupoglaz/ProteinsDataset/DockingBenchmarkV4/TableS1.csv",
							benchmark_dir="/media/lupoglaz/ProteinsDataset/DockingBenchmarkV4/")
	merge_bound_chains(targets, "/media/lupoglaz/ProteinsDataset/DockingBenchmarkV4/Natives")
