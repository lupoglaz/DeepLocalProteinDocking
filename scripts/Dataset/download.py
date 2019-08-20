import os
import sys
from prody import *
import _pickle as pkl
from tqdm import tqdm
from directories import DataDirectories


def download_structures(pdb_list, pdb_folder=None):
	"""
	Downloads list of pdbs into PDB_FOLDER
	Input:
		list of pdb identifiers
	"""
	if not os.path.exists(pdb_folder):
		os.mkdir(pdb_folder)
	pathPDBFolder(pdb_folder)
	for pdb in pdb_list:
		if (not os.path.exists(os.path.join(pdb_folder,pdb.lower()+'.pdb.gz'))) and (not os.path.exists(os.path.join(pdb_folder,pdb.lower()+'.pdb'))):
			fetchPDB(pdb)

	this_path = os.path.abspath(os.path.dirname(__file__))
	os.chdir(pdb_folder)
	os.system('gunzip -f *.gz')
	os.system('rm *.gz')
	os.chdir(this_path)

def clean_sctructures(pdb_list, pdb_folder=None):
	if not os.path.exists(pdb_folder):
		os.mkdir(pdb_folder)
	pathPDBFolder(pdb_folder)
	for pdb in pdb_list:
		path = os.path.join(pdb_folder,pdb.lower()+'.pdb')
		if os.path.exists(path):
			structure = parsePDB(path)
			protein_only = structure.select("protein and stdaa")
			writePDB(path, protein_only)

if __name__=='__main__':
	pdb_list = []
	
	with open('homo.txt') as fin:
		for line in fin:
			pdb_name = line.split()[0]
			pdb_list.append(pdb_name)
	
	with open('hetero.txt') as fin:
		for line in fin:
			pdb_name = line.split()[0]
			pdb_list.append(pdb_name)
	
	dirs = DataDirectories()
	structures_path = dirs.raw_structures_dir
	
	download_structures(pdb_list, pdb_folder=structures_path)
	clean_sctructures(pdb_list, pdb_folder=structures_path)