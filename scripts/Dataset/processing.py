import os
import sys
import numpy as np
import multiprocessing
from directories import DataDirectories
from processing_utils import _get_chains, _get_chain, _separate_chain, _run_nolb, _run_hex, run_TMScore, _get_contacts, _get_fnat, _get_capri_quality, _separate_model
from profit import ligand_rmsd, get_irmsd
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src import DATA_DIR

def load_raw_pdblist(lists):
	"""
	Loads lists of pdb entries for downloading and cleaning
	"""
	pdb_names = []
	for pdb_list_file in lists:
		with open(pdb_list_file) as fin:
			for line in fin:
				pdb_name = line.split()[0]
				pdb_names.append(pdb_name)		
	print('Number of raw proteins', len(pdb_names))
	return pdb_names

class DataProcessing:
	def __init__(self, data_dir, prefix='', num_models=10, num_solutions=10):

		self.dirs = DataDirectories(data_dir=data_dir, prefix=prefix)
		self.num_models = num_models
		self.num_solutions = num_solutions
		self.decoy_data = {}
		self.pdb_chains = {}

		if os.path.exists(self.dirs.targets_list):
			with open(self.dirs.targets_list) as fin:
				header = fin.readline()
				for line in fin:
					sline = line.split()
					self.pdb_chains[sline[0]] = (sline[1], sline[2])
			try:
				for pdb_name in list(self.pdb_chains.keys()):		
					if os.path.exists(self.dirs.get_complex_list_path(pdb_name)):
						with open(self.dirs.get_complex_list_path(pdb_name),'r') as fin:
							header = fin.readline()
							for line in fin:
								decoy_name, lrmsd, irmsd, fnat, fnonnat, quality = line.split()
								lrmsd = float(lrmsd)
								irmsd = float(irmsd)
								fnat = float(fnat)
								fnonnat = float(fnonnat)
								quality = int(quality)
								self.decoy_data[decoy_name] = (lrmsd, irmsd, fnat, fnonnat, quality)
					else:
						print("No list for", pdb_name)
			except:
				print("Reading problem in lists")
		else:
			print("No target_list file found. Separate chains first")
		
		print('Number of proteins with correct chains', len(self.pdb_chains.keys()))

	def _check_docking_job_complete(self, pdb_name, receptor_chain, ligand_chain, model_num):
		for solution_num in range(1, self.num_solutions+1):
			if not os.path.exists(self.dirs.get_complex_decoy_path(pdb_name, receptor_chain, ligand_chain, model_num, solution_num)):
				return False
		return True

	def break_chains(self, pdb_names):
		"""
		Breaks pdb into two chains: receptor and ligand. Writes target list with chains.
		Input directory: Structures
		Output directory: Chains
		"""
		with open(self.dirs.targets_list, 'w') as fout:
			fout.write('target_name\treceptor_chain\tligand_chain\n')
			for pdb_name in pdb_names:
				if pdb_name == "1B9N": #incorrect numbering of residues in the source structure
					continue
				print(pdb_name)
				try:
					chains = _get_chains(self.dirs.get_structure_file(pdb_name))
				except:
					continue
				for chain_name in chains:
					print('Chain', chain_name)
					_separate_chain(self.dirs.get_structure_file(pdb_name), chain_name, self.dirs.get_chain_file(pdb_name, chain_name))
				fout.write("%s\t%s\t%s\n"%(pdb_name, chains[0], chains[1]))

	def generate_decoys(self, rewrite=False, max_rmsd=5.0):
		"""
		Generates wobbled number of structures for each chain.
		Input directory: Chains
		Output directory: Decoys
		"""
		for pdb_name in list(self.pdb_chains.keys()):
			for chain_name in self.pdb_chains[pdb_name]:
				input_pdb_path = self.dirs.get_chain_file(pdb_name, chain_name)
				output_pdb_path = self.dirs.get_decoy_output_param(pdb_name, chain_name)
				if (not os.path.exists(output_pdb_path)) or rewrite:
					_run_nolb(input_pdb_path, output_pdb_path, max_rmsd=max_rmsd)
	
	def dock_decoys(self, rewrite=False, 
					r_angle=180, r_step=7.5, l_angle=180, l_step=7.5, alpha_angle=360, alpha_step=5.5,
					grid_size=0.6, r12_range=40.0, r12_step=0.8, h_main=18, h_scan=25):
		"""
		Docks decoys.
		Input directory: Decoys
		Output directory: Complexes
		"""
		for pdb_name in list(self.pdb_chains.keys()):	
			receptor_chain = self.pdb_chains[pdb_name][0]
			ligand_chain = self.pdb_chains[pdb_name][1]
			receptor_file = self.dirs.get_decoy_file(pdb_name, receptor_chain)
			ligand_file = self.dirs.get_decoy_file(pdb_name, ligand_chain)
			
			for model_num in range(1, self.num_models+1):
				if self._check_docking_job_complete(pdb_name, receptor_chain, ligand_chain, model_num) and (not rewrite):
					print('Skipping:', pdb_name, model_num)
					continue
				output_dir = self.dirs.get_complex_decoys_dir(pdb_name, create_dir=True)
				output_prefix = self.dirs.get_complex_decoy_prefix(pdb_name, receptor_chain, ligand_chain, model_num)
				_run_hex(receptor_file, ligand_file, model_num, output_dir, output_prefix,
						r_angle, r_step, l_angle, l_step, alpha_angle, alpha_step,
						grid_size, r12_range, r12_step, h_main, h_scan)
				
	
	def measure_quality(self, contact_dist=5.0, interface_dist=10.0, num_processes=10, rewrite=False):
		"""
		Measures ligand RMSD of decoys wrt native structures, Interface-RMSD, F-NAT and F-nonNat and computes the quality of the decoy.
		Input directory: Complexes
		Output directory: Complexes, list.dat
		"""
		problematic_pdbs = []
		for i, pdb_name in enumerate(list(self.pdb_chains.keys())):
			print(i, pdb_name)
			if os.path.exists(self.dirs.get_complex_list_path(pdb_name)) and (not rewrite):
				continue
			job_schedule_lrmsd = []
			job_schedule_irmsd = []
			job_schedule_fnat = []
			job_schedule_decoy_names = []
			receptor_chain = self.pdb_chains[pdb_name][0]
			ligand_chain = self.pdb_chains[pdb_name][1]
			
			native_path = self.dirs.get_structure_file(pdb_name)
			nat_contacts, _, _ = _get_contacts( native_path, receptor_chain, ligand_chain, contact_dist )
			_, nat_interface_receptor, nat_interface_ligand = _get_contacts( native_path, receptor_chain, ligand_chain, interface_dist )

			for model_num in range(1, self.num_models+1):
				for solution_num in range(1, self.num_solutions+1):
					decoy_path = self.dirs.get_complex_decoy_path(pdb_name, receptor_chain, ligand_chain, model_num, solution_num)				
					if not os.path.exists(decoy_path):
						continue
					decoy_contacts, decoy_contact_receptor, decoy_contact_ligand = _get_contacts( decoy_path, receptor_chain, ligand_chain, contact_dist )
					decoy_name = self.dirs.get_complex_decoy_prefix(pdb_name, receptor_chain, ligand_chain, model_num, solution_num)

					job_schedule_lrmsd.append((native_path, decoy_path, receptor_chain+"*", ligand_chain+"*"))				
					job_schedule_fnat.append(_get_fnat(nat_contacts, decoy_contacts))
					job_schedule_irmsd.append((native_path, decoy_path, receptor_chain, ligand_chain, nat_interface_receptor, nat_interface_ligand))
					job_schedule_decoy_names.append(decoy_name)
			try:			
				pool = multiprocessing.Pool(num_processes)
				results_lrmsd = pool.map(ligand_rmsd, job_schedule_lrmsd)
				pool.close()

				pool = multiprocessing.Pool(num_processes)
				results_irmsd = pool.map(get_irmsd, job_schedule_irmsd)
				pool.close()
				
				with open(self.dirs.get_complex_list_path(pdb_name),'w') as fout:
					fout.write('decoy\tlrmsd\tirmsd\tfnat\tfnonnat\tquality\n')
					for lrmsd, irmsd, fnat, decoy_name in zip(results_lrmsd, results_irmsd, job_schedule_fnat, job_schedule_decoy_names):
						quality = _get_capri_quality(lrmsd, irmsd, fnat[0])
						fout.write('%s\t%f\t%f\t%f\t%f\t%d\n'%(decoy_name, lrmsd, irmsd, fnat[0], fnat[1], quality))
			except:
				problematic_pdbs.append(pdb_name)

		with open('problematic_pdbs'+self.dirs.prefix, 'w') as fout:
			for pdb_name in problematic_pdbs:
				fout.write(pdb_name+'\n')
	
	def split_decoy_chains(self):
		"""
		Splits chains in docked decoys complexes into two separate files
		Input directory: Complexes
		Output directory: SplitComplexes
		"""
		for i, pdb_name in enumerate(list(self.pdb_chains.keys())):
			print(i, pdb_name)
			if not os.path.exists(self.dirs.get_complex_chain_dir(pdb_name)):
				os.mkdir(self.dirs.get_complex_chain_dir(pdb_name))

			for model_num in range(1, self.num_models+1):
				for solution_num in range(1, self.num_solutions+1):
											
					receptor_chain = self.pdb_chains[pdb_name][0]
					ligand_chain = self.pdb_chains[pdb_name][1]
					decoy_name = self.dirs.get_complex_decoy_prefix(pdb_name, receptor_chain, ligand_chain, model_num, solution_num)
					decoy_path = self.dirs.get_complex_decoy_path(pdb_name, receptor_chain, ligand_chain, model_num, solution_num)
					if not decoy_name in self.decoy_data:
						continue
					
					receptor_file = self.dirs.get_complex_chain_file(pdb_name, receptor_chain, model_num, solution_num)
					ligand_file = self.dirs.get_complex_chain_file(pdb_name, ligand_chain, model_num, solution_num)
					
					_separate_chain(decoy_path, receptor_chain, receptor_file)
					_separate_chain(decoy_path, ligand_chain, ligand_file)
						
	def split_list_chains(self):
		"""
		Writes lists.dat for split complexes
		Input directory: SplitComplexes
		Output directory: SplitComplexes, list.dat
		"""
		for pdb_name in list(self.pdb_chains.keys()):
			with open(self.dirs.get_complex_chain_list_path(pdb_name), 'w') as fout:
				for model_num in range(1, self.num_models+1):
					for solution_num in range(1, self.num_solutions+1):
						receptor_chain = self.pdb_chains[pdb_name][0]
						ligand_chain = self.pdb_chains[pdb_name][1]
						receptor_file = self.dirs.get_complex_chain_file(pdb_name, receptor_chain, model_num, solution_num)
						ligand_file = self.dirs.get_complex_chain_file(pdb_name, ligand_chain, model_num, solution_num)
						decoy_name = self.dirs.get_complex_decoy_prefix(pdb_name, receptor_chain, ligand_chain, model_num, solution_num)
						if decoy_name in self.decoy_data.keys():
							fout.write("%s\t%s\t"%(receptor_file, ligand_file)+"%f\t%f\t%f\t%f\t%d\n"%self.decoy_data[decoy_name])

	def split_chain_models(self):
		"""
		Splits models in chains decoys into separate files
		Input directory: Decoys
		Output directory: SplitChains
		"""
		for i, pdb_name in enumerate(list(self.pdb_chains.keys())):
			print(i, pdb_name)
			if not os.path.exists(self.dirs.get_protein_chains_dir(pdb_name)):
				os.mkdir(self.dirs.get_protein_chains_dir(pdb_name))


			receptor_chain = self.pdb_chains[pdb_name][0]
			ligand_chain = self.pdb_chains[pdb_name][1]
			intput_receptor_file = self.dirs.get_decoy_file(pdb_name, receptor_chain)
			input_ligand_file = self.dirs.get_decoy_file(pdb_name, ligand_chain)
			with open(self.dirs.get_protein_chains_list_path(pdb_name), 'w') as fout:
				fout.write('%s\t%s'%(receptor_chain, ligand_chain))

			for model_num in range(1, self.num_models+1):			
				output_receptor_file = self.dirs.get_protein_chains_file(pdb_name, receptor_chain, model_num)
				output_ligand_file = self.dirs.get_protein_chains_file(pdb_name, ligand_chain, model_num)
				_separate_model(intput_receptor_file, model_num, output_receptor_file)
				_separate_model(input_ligand_file, model_num, output_ligand_file)
	
if __name__ == '__main__':
	#Workflow:
	# 1. Break chains
	# 2. Generate decoys
	# 3. Dock decoys
	# 4. Measure l-rmsd
	# 5. Split docked complexes

	#NEAR NATIVE DOCKING PARAMETERS
	data_dir = os.path.join(DATA_DIR, 'Docking')
	
	processing = DataProcessing(data_dir=data_dir, prefix='_nearnative')
	processing.break_chains(load_raw_pdblist(['homo.txt', 'hetero.txt']))
	processing.generate_decoys(rewrite=True, max_rmsd=1.5)
	processing.dock_decoys(	rewrite=False, 
							r_angle=15, r_step=2.0, l_angle=15, l_step=2.0, alpha_angle=15, alpha_step=2.0,
							grid_size=0.6, r12_range=15.0, r12_step=0.5, h_main=20, h_scan=25)
	processing.measure_quality(rewrite=True)
	processing.split_list_chains()
	
	#FAR FROM NATIVE DOCKING PARAMETERS
	processing = DataProcessing(data_dir=data_dir, prefix='_nonnative')
	processing.break_chains(load_raw_pdblist(['homo.txt', 'hetero.txt']))
	processing.generate_decoys(rewrite=True, max_rmsd=3.0)
	processing.dock_decoys(	rewrite=True, 
							r_angle=30, r_step=2.8, l_angle=30, l_step=2.8, alpha_angle=30, alpha_step=2.8,
							grid_size=0.6, r12_range=15.0, r12_step=0.5, h_main=20, h_scan=25)
	processing.measure_quality(rewrite=True)
	processing.split_list_chains()
	
	#DEFAULT DOCKING PARAMETERS
	# processing = DataProcessing(['homo.txt', 'hetero.txt'], prefix='')
	# processing.generate_decoys(rewrite=True)
	# processing.dock_decoys(	rewrite=True, 
	# 						r_angle=180, r_step=7.5, l_angle=180, l_step=7.5, alpha_angle=360, alpha_step=5.5,
	# 						grid_size=0.6, r12_range=40.0, r12_step=0.8, h_main=18, h_scan=25)
	# processing.measure_lrmsd()
	# processing.split_list_chains()
	# processing.split_decoy_chains(data_type='lrmsd')


	#SPLIT CHAINS MODELS DATASET
	# processing = DataProcessing(data_dir=data_dir, prefix='_nearnative')
	# processing.split_chain_models()

	# processing = DataProcessing(data_dir=data_dir, prefix='_nonnative')
	# processing.split_chain_models()