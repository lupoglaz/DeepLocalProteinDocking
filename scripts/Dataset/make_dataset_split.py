import os
import sys
import random
import _pickle as pkl
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src import DATA_DIR, REPOSITORY_DIR

from analysis import DataAnalysis
from processing import DataProcessing

class DataSplit(DataAnalysis):
	def __init__(self, prefixes, num_models=10, num_solutions=10):
		super(DataSplit, self).__init__(prefixes, num_models, num_solutions)

	def exclude_set(self, pdb_exclude_list):
		for pdb_name in pdb_exclude_list:
			if pdb_name in self.data.keys():
				del self.data[pdb_name]

	def train_val_split(self, percentage=0.8):
		pdb_list = list(self.data.keys())
		random.shuffle(pdb_list)
		N_training = int(0.8*len(pdb_list))
		training_set = pdb_list[:N_training]
		validation_set = pdb_list[N_training:]
		return training_set, validation_set
	
	def make_description(	self, pdb_list,
							data_dir = '',
							dataset_dir = '',
							description_dirname='Description', 
							description_filename='description.dat'):
		
		description_dir = os.path.join(dataset_dir, description_dirname)
		if not os.path.exists(description_dir):
			os.mkdir(description_dir)
		
		with open(os.path.join(description_dir, description_filename), 'w') as fout_global:
			for pdb_name in pdb_list:
				fout_global.write("%s\n"%pdb_name)
				with open(os.path.join(description_dir, pdb_name + '.dat'), 'w') as fout_target:
					fout_target.write("receptor\tligand\tlrmsd\tirmsd\tfnat\tfnonnat\tquality\n")
					for decoy_name in self.data[pdb_name].keys():
						
						receptor_file, ligand_file = self.data_path[pdb_name][decoy_name]
						fout_target.write("%s\t%s\t"%(receptor_file, ligand_file) + "%f\t%f\t%f\t%f\t%d\n"%self.data[pdb_name][decoy_name])

		return

	def make_description_chains(self, pdb_list,
							data_dir = '',
							dataset_dir = '',
							description_dirname='Description', 
							description_filename='description.dat'):
		description_dir = os.path.join(dataset_dir, description_dirname)
		if not os.path.exists(description_dir):
			os.mkdir(description_dir)
		
		with open(os.path.join(description_dir, description_filename), 'w') as fout_global:
			for pdb_name in pdb_list:
				fout_global.write("%s\n"%pdb_name)
				with open(os.path.join(description_dir, pdb_name + '.dat'), 'w') as fout_target:
					fout_target.write("receptor\tligand\n")
					for receptor_file, ligand_file in self.data_chains[pdb_name]:
						fout_target.write("%s\t%s\n"%(receptor_file, ligand_file))

if __name__=='__main__':
	# pdb_list = read_pdb_list([  os.path.join(REPOSITORY_DIR, 'scripts', 'Dataset', 'homo.txt'),
	# 							os.path.join(REPOSITORY_DIR, 'scripts', 'Dataset', 'hetero.txt')
	# 						],
	# 						os.path.join(DATA_DIR, 'Complexes'))

	# random.shuffle(pdb_list)
	# N_training = int(0.8*len(pdb_list))
	# training_set = pdb_list[:N_training]
	# validation_set = pdb_list[N_training:]
	d = DataSplit(prefixes=['_nearnative', '_nonnative'],num_models=10, num_solutions=10)
	
	with open("exclusion_set.pkl", "rb") as fin:
		exclusion_set = pkl.load(fin)

	d.exclude_set(exclusion_set)
	
	train_set, val_set = d.train_val_split(percentage=0.8)

	# d.make_description( train_set,
	# 					data_dir = os.path.join(DATA_DIR, 'Docking'),
	# 					dataset_dir = os.path.join(DATA_DIR, 'Docking', 'SplitComplexes'),
	# 				 	description_filename='training_set.dat'
	# 					)

	# d.make_description( val_set,
	# 					data_dir = os.path.join(DATA_DIR, 'Docking'),
	# 					dataset_dir = os.path.join(DATA_DIR, 'Docking', 'SplitComplexes'),
	# 				 	description_filename='validation_set.dat'
	# 					)

	d.make_description_chains( train_set,
						data_dir = os.path.join(DATA_DIR, 'Docking'),
						dataset_dir = os.path.join(DATA_DIR, 'Docking', 'SplitChains'),
					 	description_filename='training_set.dat'
						)

	d.make_description_chains( val_set,
						data_dir = os.path.join(DATA_DIR, 'Docking'),
						dataset_dir = os.path.join(DATA_DIR, 'Docking', 'SplitChains'),
					 	description_filename='validation_set.dat'
						)
	
	# make_description(	training_set, ['_nearnative'], 
	# 					data_dir = os.path.join(DATA_DIR, 'SplitComplexes'),
	# 				 	description_filename='training_set.dat')
	
	# make_description(	validation_set, ['_nearnative'], 
	# 					data_dir = os.path.join(DATA_DIR, 'SplitComplexes'),
	# 					description_filename='validation_set.dat')