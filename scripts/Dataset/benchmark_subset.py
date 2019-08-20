import os
import sys
import random
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src import DATA_DIR, REPOSITORY_DIR
from global_alignment import get_complex_pdb_sequences
import _pickle as pkl
from benchmark_cleanup import read_pdb_list

from processing_utils import _get_bbox
from prody import *
import numpy as np

def _get_center(structure):
	cm = np.array([0.0, 0.0, 0.0])
	for atom in structure:
		v = np.array(atom.getCoords())
		cm += v
	return cm / len(structure)

def get_bbox_subset(targets, bbox_threshold=80, separation_threshold=30, rewrite=False):
	pdb_sizes = {}
	if (not os.path.exists('benchmark_chains_boxes.pkl')) or rewrite:
		for clpx_type in targets:
			for ureceptor_path, uligand_path, breceptor_path, bligand_path in clpx_type:
				pdb_name = ureceptor_path.split('/')[-1].split('_')[0]

				receptor = parsePDB(ureceptor_path)
				receptor_chain = receptor.select("protein")

				ligand = parsePDB(uligand_path)
				ligand_chain = ligand.select("protein")
				
				ligand_a, ligand_b = _get_bbox(ligand_chain)
				receptor_a, receptor_b = _get_bbox(receptor_chain)
				
				receptor_center = _get_center(receptor_chain)
				ligand_center = _get_center(ligand_chain)
				separation = np.linalg.norm(receptor_center - ligand_center)
				print(receptor_center - ligand_center)

				ligand_size = np.max(np.abs(ligand_b - ligand_a))
				receptor_size = np.max(np.abs(receptor_b - receptor_a))
				
				pdb_sizes[pdb_name] = (receptor_size, ligand_size, separation)
			
			with open('benchmark_chains_boxes.pkl', 'wb') as fout:
				pkl.dump(pdb_sizes, fout)
	else:
		with open('benchmark_chains_boxes.pkl', 'rb') as fin:
			pdb_sizes = pkl.load(fin)
		
	print(pdb_sizes)		
	exclusion_list_bbox = []
	exclusion_list_sep = []
	for pdb_name in pdb_sizes.keys():	
		if max(pdb_sizes[pdb_name][0], pdb_sizes[pdb_name][1]) > bbox_threshold:
			exclusion_list_bbox.append(pdb_name)
		if pdb_sizes[pdb_name][2] > separation_threshold:
			exclusion_list_sep.append(pdb_name)
	
	return exclusion_list_bbox, exclusion_list_sep

def write_target_subset(targets, exclusion_list, filename):
	cplx_type = 0
	type_string = ["Rigid-body", "Medium Difficulty", "Difficult"]
	with open(filename, 'w') as fout:
		for cplx_type, targets_type in enumerate(targets):
			fout.write(type_string[cplx_type] + '\n')
			for ureceptor_path, uligand_path, breceptor_path, bligand_path in targets_type:
				pdb_name = ureceptor_path.split('/')[-1].split('_')[0]
				if not pdb_name in exclusion_list:
					fout.write('%s_\t\n'%pdb_name)
				else:
					print("Excluded ", pdb_name)

if __name__=='__main__':
	targets = read_pdb_list("/media/lupoglaz/ProteinsDataset/DockingBenchmarkV4/TableS1.csv",
							benchmark_dir="/media/lupoglaz/ProteinsDataset/DockingBenchmarkV4/")
	# targets = read_pdb_list("/media/lupoglaz/ProteinsDataset/DockingBenchmarkV5/Table_BM5.csv",
	# 						benchmark_dir="/media/lupoglaz/ProteinsDataset/DockingBenchmarkV5/")
	
	elist_bbox, elist_sep = get_bbox_subset(targets, separation_threshold=30.01, rewrite=False)
	print('Dataset size:', len(targets[0]+targets[1]+targets[2]))
	print('Total exclusion size:', len(set(elist_bbox)&set(elist_sep)))
	print('Bbox exclusion size:', len(elist_bbox))
	print('Separation exclusion size:', len(elist_sep))
	

	write_target_subset(targets, elist_sep + elist_bbox, "/media/lupoglaz/ProteinsDataset/DockingBenchmarkV4/Table_PEPSI.csv")
	# write_target_subset(targets, elist_sep, "/media/lupoglaz/ProteinsDataset/DockingBenchmarkV4/Table_PEPSI.csv")