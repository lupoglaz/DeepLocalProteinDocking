import os
import sys
from prody import *

class DataDirectories:
	def __init__(self, data_dir, prefix=''):
		self.data_dir = data_dir
		self.prefix = prefix
		self.targets_list = os.path.join(self.data_dir,'targets.dat')
		self.raw_structures_dir = os.path.join(self.data_dir,'Structures')

		if not os.path.exists(self.raw_structures_dir):
			raise Exception("No input directory")

		self.chains_dir = os.path.join(self.data_dir,'Chains')
		if not os.path.exists(self.chains_dir):
			os.mkdir(self.chains_dir)

		self.decoys_dir = os.path.join(self.data_dir, 'Decoys')
		if not os.path.exists(self.decoys_dir):
			os.mkdir(self.decoys_dir)

		self.complex_decoys_dir = os.path.join(self.data_dir, 'Complexes')
		if not os.path.exists(self.complex_decoys_dir):
			os.mkdir(self.complex_decoys_dir)

		self.split_decoys_dir = os.path.join(self.data_dir, 'SplitComplexes')
		if not os.path.exists(self.split_decoys_dir):
			os.mkdir(self.split_decoys_dir)

		self.split_complex_chains_dir = os.path.join(self.data_dir, 'SplitChains')
		if not os.path.exists(self.split_complex_chains_dir):
			os.mkdir(self.split_complex_chains_dir)

	def get_description_dir(self, description_dirname):
		return os.path.join(self.data_dir, description_dirname)
	
	def get_structure_file(self, pdb_name, check=False):
		filename = os.path.join(self.raw_structures_dir, pdb_name.lower()+'.pdb')
		if (not os.path.exists(filename)) and check:
			raise Exception("File not found %s"%filename)
		return filename

	def get_decoy_file(self, pdb_name, chain_name, check=True):
		filename = os.path.join(self.decoys_dir, pdb_name.lower()+'_%s%s_nlb_decoys.pdb'%(chain_name, self.prefix))
		if (not os.path.exists(filename)) and check:
			raise Exception("File not found %s"%filename)
		return filename

	def get_decoy_output_param(self, pdb_name, chain_name, check=False):
		filename = os.path.join(self.decoys_dir, pdb_name.lower()+'_%s%s.pdb'%(chain_name, self.prefix))
		if (not os.path.exists(filename)) and check:
			raise Exception("File not found %s"%filename)
		return filename

	def get_chain_file(self, pdb_name, chain_name, check=False):
		filename = os.path.join(self.chains_dir, pdb_name.lower()+'_%s.pdb'%chain_name)
		if (not os.path.exists(filename)) and check:
			raise Exception("File not found %s"%filename)
		return filename
	
	def get_complex_decoys_dir(self, pdb_name, create_dir=True):
		file_dir = os.path.join(self.complex_decoys_dir, pdb_name.upper())
		if (not os.path.exists(file_dir)) and create_dir:
			os.mkdir(file_dir)
		return file_dir

	def get_complex_decoy_prefix(self, pdb_name, receptor_chain, ligand_chain, model_num, solution_num=None, prefix=None):
		if prefix is None:
			prefix = self.prefix
		if solution_num is None:
			return pdb_name.lower()+'_r'+receptor_chain+'_l'+ligand_chain+'_m%d'%model_num + prefix + '_s'
		else:
			return pdb_name.lower()+'_r'+receptor_chain+'_l'+ligand_chain+'_m%d'%model_num + prefix + '_s%04d'%solution_num+'.pdb'
	
	def get_complex_decoy_path(self, pdb_name, receptor_chain, ligand_chain, model_num, solution_num, prefix=None):
		return os.path.join(self.get_complex_decoys_dir(pdb_name, False), 
							self.get_complex_decoy_prefix(pdb_name, receptor_chain, ligand_chain, model_num, solution_num, prefix=prefix))
	
	def get_complex_list_path(self, pdb_name, prefix=None):
		if prefix is None:
			prefix = self.prefix
		return os.path.join(self.get_complex_decoys_dir(pdb_name, False), 'list%s.dat'%(prefix))


	def get_complex_chain_dir(self, pdb_name, check=False):
		dirname = os.path.join(self.split_decoys_dir, pdb_name.upper())
		return dirname

	def get_complex_chain_file(self, pdb_name, chain_name, model_num, solution_num, check=False, prefix=None):
		if prefix is None:
			prefix=self.prefix
		filename = os.path.join(self.split_decoys_dir, pdb_name.upper(), pdb_name.lower() + '_%s'%chain_name +
		'_m%d'%model_num + prefix + '_s%04d'%solution_num + '.pdb')
		if (not os.path.exists(filename)) and check:
			raise Exception("File not found %s"%filename)
		return filename

	def get_complex_chain_files(self, complex_decoy_prefix):

		s_decoy_prefix = complex_decoy_prefix.split('.')[0].split('_')
		if len(s_decoy_prefix)<6:
			raise Exception("Wrong complex_decoy_prefix %s"%complex_decoy_prefix)

		pdb_name = s_decoy_prefix[0]
		receptor_chain = s_decoy_prefix[1][1:]
		ligand_chain = s_decoy_prefix[2][1:]
		model_num = int(s_decoy_prefix[3][1:])
		prefix = '_'+s_decoy_prefix[4]
		solution_num = int(s_decoy_prefix[5][1:])
		
		out_receptor_file = self.get_complex_chain_file(pdb_name, receptor_chain, model_num, solution_num, prefix=prefix)
		out_ligand_file = self.get_complex_chain_file(pdb_name, ligand_chain, model_num, solution_num, prefix=prefix)

		return out_receptor_file, out_ligand_file

	def get_complex_chain_list_path(self, pdb_name, check=False, prefix=None):
		if prefix is None:
			prefix=self.prefix
		return os.path.join(self.get_complex_chain_dir(pdb_name), 'list%s.dat'%(prefix))

	def get_protein_chains_dir(self, pdb_name, check=False):
		dirname = os.path.join(self.split_complex_chains_dir, pdb_name.upper())
		if (not os.path.exists(dirname)) and check:
			raise Exception("File not found %s"%dirname)
		return dirname

	def get_protein_chains_file(self, pdb_name, chain_name, model_num, check=False):
		filename = os.path.join(self.get_protein_chains_dir(pdb_name), pdb_name.lower() + '_%s'%chain_name + '_m%d'%model_num + '%s.pdb'%self.prefix)
		if (not os.path.exists(filename)) and check:
			raise Exception("File not found %s"%filename)
		return filename

	def get_protein_chains_list_path(self, pdb_name, prefix=None):
		if prefix is None:
			prefix=self.prefix
		return os.path.join(self.get_protein_chains_dir(pdb_name), 'list%s.dat'%(prefix))
