import os
import sys
from prody import *
import _pickle as pkl
from tqdm import tqdm
import numpy as np
import multiprocessing
from processing import DataProcessing
from directories import DataDirectories
from processing_utils import _get_chains, _get_chain, _get_bbox

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pylab as plt

import seaborn as sea
sea.set_style("whitegrid")

from global_alignment import get_pdb_seq, get_alignment

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src import DATA_DIR

class DataAnalysis:
	def __init__(self, prefixes, num_models=10, num_solutions=10):
		self.num_models = num_models
		self.num_solutions = num_solutions
		self.data = {}
		self.data_path = {}
		self.prefixes = prefixes
		data_dir = os.path.join(DATA_DIR, 'Docking')
		for prefix in prefixes:
			dataset = DataProcessing(data_dir=data_dir, prefix=prefix)
			for pdb_name in dataset.pdb_chains.keys():
				decoys = {}
				decoys_path = {}
				receptor_chain = dataset.pdb_chains[pdb_name][0]
				ligand_chain = dataset.pdb_chains[pdb_name][1]
				for model_num in range(num_models):
					for solution_num in range(num_solutions):
						decoy_name = dataset.dirs.get_complex_decoy_prefix(pdb_name, receptor_chain, ligand_chain, model_num, solution_num)
						if decoy_name in dataset.decoy_data.keys():
							receptor_file = dataset.dirs.get_complex_chain_file(pdb_name, receptor_chain, model_num, solution_num)
							ligand_file = dataset.dirs.get_complex_chain_file(pdb_name, ligand_chain, model_num, solution_num)
							decoys[decoy_name] = dataset.decoy_data[decoy_name]
							decoys_path[decoy_name] = (receptor_file, ligand_file)
				if not pdb_name in self.data.keys():
					self.data[pdb_name] = {}
				if not pdb_name in self.data_path.keys():
					self.data_path[pdb_name] = {}
				self.data[pdb_name] = {**self.data[pdb_name], **decoys}
				self.data_path[pdb_name] = {**self.data_path[pdb_name], **decoys_path}

		self.data_chains = {}
		for prefix in prefixes:
			dataset = DataProcessing(data_dir=data_dir, prefix=prefix)
			for pdb_name in dataset.pdb_chains.keys():
				decoys = []
				receptor_chain = dataset.pdb_chains[pdb_name][0]
				ligand_chain = dataset.pdb_chains[pdb_name][1]
				for model_num in range(1, num_models+1):
					receptor_file = dataset.dirs.get_protein_chains_file(pdb_name, receptor_chain, model_num)
					ligand_file = dataset.dirs.get_protein_chains_file(pdb_name, ligand_chain, model_num)
					decoys.append((receptor_file, ligand_file))
				if not pdb_name in self.data_chains.keys():
					self.data_chains[pdb_name] = []
				self.data_chains[pdb_name] += decoys


	def compute_alignments(self, axis, seq_list, threshold=0.9, prefix='_nearnative', rewrite=False, num_processes=10):
		dataset = DataProcessing(prefix)

		all_alignments = {}
		for n, pdb_name in enumerate(dataset.pdb_chains.keys()):
			if (not os.path.exists('Alignments/%s.pkl'%pdb_name)) or rewrite:
				print('Processing ', pdb_name, ' %d/%d'%(n, len(dataset.pdb_chains.keys())))
				receptor_chain = dataset.pdb_chains[pdb_name][0]
				ligand_chain = dataset.pdb_chains[pdb_name][1]
				model_num = None
				solution_num = None

				for model in range(1,11):
					if not model_num is None:
						break
					for solution in range(1,11):
						decoy_name = dataset.dirs.get_complex_decoy_prefix(pdb_name, receptor_chain, ligand_chain, model, solution)
						if decoy_name in dataset.decoy_data:
							model_num=model
							solution_num=solution
							break
				
				if model_num is None:
					continue

				receptor_file = dataset.dirs.get_complex_chain_file(pdb_name, receptor_chain, model_num, solution_num)
				ligand_file = dataset.dirs.get_complex_chain_file(pdb_name, ligand_chain, model_num, solution_num)
				receptor_sequence = get_pdb_seq(receptor_file)
				ligand_sequence = get_pdb_seq(ligand_file)

				jobs = []
				for target_receptor_path, target_receptor_seq, target_ligand_path, target_ligand_seq in seq_list:
					jobs.append((receptor_sequence, target_receptor_seq))
					jobs.append((receptor_sequence, target_ligand_seq))
					jobs.append((ligand_sequence, target_receptor_seq))
					jobs.append((ligand_sequence, target_ligand_seq))
					
				pool = multiprocessing.Pool(num_processes)
				results = pool.map(get_alignment, jobs)
				pool.close()
				
				alignments = []
				for n, target in enumerate(seq_list):
					max_id = max(results[4*n][1], results[4*n+1][1], results[4*n+2][1], results[4*n+3][1])
					alignments.append( (target[0], target[2], max_id) )
				
				with open('Alignments/%s.pkl'%pdb_name, 'wb') as fout:
					pkl.dump(alignments, fout)
			else:
				with open('Alignments/%s.pkl'%pdb_name, 'rb') as fin:
					alignments = pkl.load(fin)
			
			all_alignments[pdb_name] = alignments
		
		exclusion_list = set([])

		N = len(all_alignments.keys())
		M = len(seq_list)
		
		mat = np.zeros( (N, M) )
		for i, key in enumerate(all_alignments.keys()):
			for j, al in enumerate(all_alignments[key]):
				mat[i, j] = al[2]
				if al[2]>threshold:
					exclusion_list.add(key)
					print(key, al[0])
		
		axis.imshow(mat)
		return exclusion_list

			


	def get_chain_bounding_boxes(self, axis, threshold=100.0, rewrite=False):
		pdb_sizes = {}
		if (not os.path.exists('chains_boxes.pkl')) or rewrite:
			for prefix in self.prefixes:
				dataset = DataProcessing(prefix)
				for pdb_name in tqdm(dataset.pdb_chains.keys()):
					pdb_file = dataset.dirs.get_structure_file(pdb_name)
					
					receptor_chain = _get_chain(pdb_file, dataset.pdb_chains[pdb_name][0], do_center=False)
					ligand_chain = _get_chain(pdb_file, dataset.pdb_chains[pdb_name][1], do_center=False)
					
					ligand_a, ligand_b = _get_bbox(ligand_chain)
					receptor_a, receptor_b = _get_bbox(receptor_chain)
					
					ligand_size = np.max(np.abs(ligand_b - ligand_a))
					receptor_size = np.max(np.abs(receptor_b - receptor_a))
					
					pdb_sizes[pdb_name] = (receptor_size, ligand_size)

				break
			
			with open('chains_boxes.pkl', 'wb') as fout:
				pkl.dump(pdb_sizes, fout)
		else:
			with open('chains_boxes.pkl', 'rb') as fin:
				pdb_sizes = pkl.load(fin)
		
		sizes = []
		exclusion_list = []
		for pdb_name in pdb_sizes.keys():
			sizes += list(pdb_sizes[pdb_name])
			if max(pdb_sizes[pdb_name][0], pdb_sizes[pdb_name][1]) > threshold:
				exclusion_list.append(pdb_name)

		axis.hist(sizes, bins=40, alpha=0.7)
		axis.set_xlabel('BBox size')
		axis.set_ylabel('Num targets')
		return exclusion_list

			
	def total_distribution(self, axis_lrmsd, axis_irmsd, axis_fnats, axis_q, label=None):
		lrmsds = []
		irmsds = []
		fnats = []
		qual = []
		for pdb_name in self.data.keys():
			decoys = self.data[pdb_name]
			for decoy_name in decoys.keys():
				lrmsd, irmsd, fnat, fnonnat, quality = decoys[decoy_name]
				lrmsds.append(float(lrmsd))
				irmsds.append(float(irmsd))
				fnats.append(float(fnat))
				qual.append(float(quality))

		if label is None:
			label = prefix
		
		axis_lrmsd.set_title("1A", loc='right')
		axis_lrmsd.hist(lrmsds, bins=100, alpha=0.7)
		axis_lrmsd.set_xlabel('LRMSD')
		axis_lrmsd.set_ylabel('Num models')

		axis_irmsd.set_title("2A", loc='right')
		axis_irmsd.hist(irmsds, bins=100, alpha=0.7)
		axis_irmsd.set_xlabel('IRMSD')
		axis_irmsd.set_ylabel('Num models')

		axis_fnats.set_title("1B", loc='right')
		axis_fnats.hist(fnats, bins=100, alpha=0.7)
		axis_fnats.set_xlabel('Fnat')
		axis_fnats.set_ylabel('Num models')

		axis_q.set_title("2B", loc='right')
		axis_q.hist(qual, bins=[-0.5,0.5,1.5,2.5], alpha=0.7)
		axis_q.set_xlabel('Quality')
		axis_q.set_xticks([0, 1, 2, 3])
		axis_q.set_ylabel('Num models')
		axis_q.set_xlim([-1, 4])

	def pdb_distribution(self, axis2d, axis1d, threshold=10, label=None):
		pdb_accept = []
		pdb_incorrect = []
		pdb_accept_dict = {}
		
		for pdb_name in self.data.keys():
			num_accept = 0
			num_incorrect = 0
			decoys = self.data[pdb_name]
			for decoy_name in decoys.keys():
				lrmsd, irmsd, fnat, fnonnat, quality = decoys[decoy_name]
				if quality>0:
					num_accept += 1
				else:
					num_incorrect += 1
			pdb_accept.append(num_accept)
			pdb_incorrect.append(num_incorrect)
			pdb_accept_dict[pdb_name] = num_accept

		if label is None:
			label = prefix
		
		axis2d.set_title("1C", loc='right')
		axis2d.scatter(pdb_accept, pdb_incorrect, label='target complex')
		axis2d.set_xlabel("Num acceptable")
		axis2d.set_ylabel("Num incorrect")
		
		axis1d.set_title("2C", loc='right')
		axis1d.hist(pdb_accept, bins=40, alpha=0.7)
		axis1d.set_xlabel('Num acceptable')
		axis1d.set_ylabel('Num targets')

		exclusion_list = []
		for pdb_name in pdb_accept_dict.keys():
			if pdb_accept_dict[pdb_name] < threshold:
				exclusion_list.append(pdb_name)
		
		return exclusion_list

def plot_pdb_bboxes(output_file=None, prefixes=[''], threshold=100.0):
	d = DataAnalysis(prefixes=prefixes)
	
	# f, axes = plt.subplots(1,1, figsize=(6,6))
	f = plt.figure(figsize=(10,6))
	ax = plt.subplot(111)
	exclusion_list = d.get_chain_bounding_boxes(ax, threshold=threshold, rewrite=False)

	plt.legend()
	
	if output_file is None:
		plt.show()
	else:
		if not os.path.exists('Analysis'):
			os.mkdir('Analysis')
		plt.savefig(os.path.join('Analysis', output_file))
	return exclusion_list
		
def plot_quality_distributions(output_file=None, prefixes=[''], threshold=10):
	d = DataAnalysis(prefixes=prefixes)
	
	f, axes = plt.subplots(3,2, figsize=(10,10))
	d.total_distribution(axes[0,0], axes[0,1], axes[1,0], axes[1,1], label="L-RMSD")

	exclusion_list = d.pdb_distribution(axes[2,0], axes[2,1], label="Quality", threshold=threshold)

	# plt.legend()
	f.tight_layout()
	
	if output_file is None:
		plt.show()
	else:
		if not os.path.exists('Analysis'):
			os.mkdir('Analysis')
		plt.savefig(os.path.join('Analysis', output_file))
	
	return exclusion_list

def plot_homology(output_file=None, prefixes=[''], threshold=0.9):
	d = DataAnalysis(prefixes=['_nearnative'])
	with open('benchmark_sequences.pkl', 'rb') as fin:
		sequences = pkl.load(fin)

	f = plt.figure(figsize=(10,10))
	ax = plt.subplot(111)
	exclution_list = d.compute_alignments(ax, sequences, threshold=threshold, rewrite=False)
	
	if output_file is None:
		plt.show()
	else:
		if not os.path.exists('Analysis'):
			os.mkdir('Analysis')
		plt.savefig(os.path.join('Analysis', output_file))
	
	return exclution_list


if __name__=='__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Train deep protein folder')
	parser.add_argument('-quality', default=None, help='Plot decoys RMSD distribution')
	args = parser.parse_args()

	# if not args.quality is None:
	q_exclusion_list = plot_quality_distributions( prefixes=['_nearnative', '_nonnative'], threshold=5,
								output_file='lrmsd_distribution.png')

	b_exclusion_list = plot_pdb_bboxes(prefixes=['_nearnative', '_nonnative'], threshold=80.0,
									output_file='bbox_distribution.png')

	# h_exclusion_list = plot_homology(prefixes=['_nearnative'], output_file='homology.png', threshold=0.95)
	# print(len(h_exclusion_list))
	
	exclusion_set = set(q_exclusion_list + b_exclusion_list)

	print(len(exclusion_set))
	with open("exclusion_set.pkl", "wb") as fout:
		pkl.dump(exclusion_set, fout)
	
	


