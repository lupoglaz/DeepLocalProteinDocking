import os
import sys
from prody import *
import numpy as np

import Bio.PDB
from Bio.PDB.Polypeptide import d3_to_index, dindex_to_1
from Bio.PDB import PDBParser, NeighborSearch, Selection
pdb_parser = PDBParser(QUIET=True)

import subprocess
import shutil


def run_TMScore( tpl ):
	import subprocess
	path1, path2 = tpl
	rmsd=-1
	tmscore=-1
	gdt_ts_score=-1
	gdt_ha_score=-1

	try:
		output = subprocess.check_output(['./utils/TMscore', '-c', path1, path2])
	except:
		print('Error in TM-score', path1, path2)
		return rmsd,tmscore, gdt_ts_score, gdt_ha_score
	
	for line in output.split('\n'):
		if not(line.find('RMSD')==-1) and not(line.find('common')==-1) and not(line.find('=')==-1):
			rmsd = float(line.split()[-1])
		elif not(line.find('TM-score')==-1) and not(line.find('d0')==-1) and not(line.find('=')==-1):
			tmscore = float(line[line.find('=')+1:line.rfind('(')])
		elif not(line.find('GDT-TS-score')==-1) and not(line.find('d')==-1) and not(line.find('=')==-1):
			gdt_ts_score = float(line[line.find('=')+1:line.find('%')])
		elif not(line.find('GDT-HA-score')==-1) and not(line.find('d')==-1) and not(line.find('=')==-1):
			gdt_ha_score = float(line[line.find('=')+1:line.find('%')])
		else:
			continue

	return rmsd,tmscore, gdt_ts_score, gdt_ha_score

def _get_chains(input_pdb_path):
	pdb = parsePDB(input_pdb_path)
	chains = list(set(pdb.getChids()))
	
	if len(chains)!=2:
		raise Exception("Incorrect number of chains")
	chain_0_len = len(pdb.select("chain %s"%chains[0]))
	chain_1_len = len(pdb.select("chain %s"%chains[1]))
	
	if chain_0_len<chain_1_len:
		receptor_chain = chains[1]
		lingand_chain = chains[0]
	else:
		receptor_chain = chains[0]
		lingand_chain = chains[1]
	
	return (receptor_chain, lingand_chain)

def _get_chain(input_filename, chain_id, do_center=False):
	pdb = parsePDB(input_filename)
	pdb_ch = pdb.select("protein and chid %s"%chain_id)
	
	if do_center:
		center = np.array([0.,0.,0.])
		for atom in pdb_ch:
			center += atom.getCoords()
		center /= float(len(pdb_ch))

		for atom in pdb_ch:
			pd_coords = atom.getCoords() - center
			atom.setCoords(pd_coords)

	return pdb_ch

def _get_bbox(structure):
	min_vector = np.array([float('+inf'), float('+inf'), float('+inf')])
	max_vector = np.array([float('-inf'), float('-inf'), float('-inf')])
	for atom in structure:
		v = np.array(atom.getCoords())
		for i in range(0,3):
			if v[i] > max_vector[i]:
				max_vector[i] = v[i]
			if v[i] < min_vector[i]:
				min_vector[i] = v[i]
	return (min_vector, max_vector)


def _separate_chain(input_pdb_path, chain_id, output_pdb_path):
	pdb_ch = _get_chain(input_pdb_path, chain_id, do_center=False)
	writePDB(output_pdb_path, pdb_ch)

def _separate_model(input_pdb_path, model_id, output_pdb_path):
	pdb_mdl = parsePDB(input_pdb_path, model=model_id)
	writePDB(output_pdb_path, pdb_mdl)

def _run_nolb(input_filename, output_filename, max_rmsd=5.0):
	os.system("'utils/NOLB' %s -o %s -r %f --dist 1 -m"%(input_filename, output_filename, max_rmsd))

def _run_hex(	input_receptor_filename, input_ligand_filename, decoy_num, output_dir, output_prefix, 
				r_angle=180, r_step=7.5, l_angle=180, l_step=7.5, alpha_angle=360, alpha_step=5.5,
				grid_size=0.6, r12_range=40.0, r12_step=0.8, h_main=18, h_scan=25):
	files = """open_receptor %s
open_ligand %s
dock_receptor_model %d
dock_ligand_model %d
docking_fft_device 1
docking_fft_type 1
"""%(input_receptor_filename, input_ligand_filename, decoy_num, decoy_num)

	grid_settings = """
docking_grid_size %f
docking_r12_range %f
docking_r12_step %f
docking_box_size 10
docking_main_search %d
docking_main_scan %d
"""%(grid_size, r12_range, r12_step, h_main, h_scan)

	angle_settings="""
receptor_range_angle %f
docking_receptor_stepsize %f
ligand_range_angle %f
docking_ligand_stepsize %f
alpha_range_angle %f
docking_alpha_stepsize %f
"""%(r_angle, r_step, l_angle, l_step, alpha_angle, alpha_step)


	docking="""
activate_docking
max_docking_clusters 10
save_range 1 10 %s %s pdb
"""%(output_dir, output_prefix)

	script = files + grid_settings + angle_settings + docking

	with open('tmp.mac', 'w') as fout:
		fout.write(script)
		fout.write("exit")
	os.system("hex -e tmp.mac")

def _read_transform(input_filename, max_num_solutions=10):
	transforms = []
	with open(input_filename) as fin:
		for line in fin:
			if line.find('#')!=-1:
				continue
			al = line.split()
			if len(al) == 0:
				continue
			cluster_num = int(al[0])
			solution_num = int(al[1])
			x = float(al[4])
			y = float(al[5])
			z = float(al[6])
			tx = float(al[7])
			ty = float(al[8])
			tz = float(al[9])
			transforms.append(( np.array([x,y,z]), np.array([tx, ty, tz]) ))
			if solution_num>max_num_solutions:
				break
	return transforms

def _transform_group(atomic_group, translation, euler_angles):
	tx,ty,tz = tuple(euler_angles) 

	Rx = np.array([[1,0,0], [0, np.cos(tx), -np.sin(tx)], [0, np.sin(tx), np.cos(tx)]]) 
	Ry = np.array([[np.cos(ty), 0, -np.sin(ty)], [0, 1, 0], [np.sin(ty), 0, np.cos(ty)]]) 
	Rz = np.array([[np.cos(tz), -np.sin(tz), 0], [np.sin(tz), np.cos(tz), 0], [0,0,1]]) 
	R = np.dot(Rx, np.dot(Ry, Rz))
	
		
	for atom in atomic_group:
		pd_coords = atom.getCoords()
		r = np.array(pd_coords)
		r_t = np.dot(R, r) + translation
		atom.setCoords(r_t)

	return atomic_group

def _get_contacts( pdb_path, chain_rec, chain_lig, contact_dist ):
	structure = pdb_parser.get_structure('X', pdb_path)[0]
	

	receptor = [structure[chain_rec_id] for chain_rec_id in chain_rec]
	ligand = [structure[chain_lig_id] for chain_lig_id in chain_lig]

	receptor_atoms = Selection.unfold_entities(receptor, 'A')
	ns = NeighborSearch(receptor_atoms)
	
	ligand_residues = Selection.unfold_entities(ligand, 'R')
	contacts = set([])
	contacts_lig = set([])
	contacts_rec = set([])
	for ligand_res in ligand_residues:
		lig_resname = dindex_to_1[d3_to_index[ligand_res.get_resname()]]
		lig_resnum = ligand_res.get_id()[1]
		lig_chname = ligand_res.get_parent().get_id()
		res_contacts = []
		
		for lig_atom in ligand_res:
			neighbors = ns.search(lig_atom.get_coord(), contact_dist)
			res_contacts += Selection.unfold_entities(neighbors, 'R')
		
		for receptor_res in res_contacts:
			rec_resname = dindex_to_1[d3_to_index[receptor_res.get_resname()]]
			rec_resnum = receptor_res.get_id()[1]
			rec_chname = receptor_res.get_parent().get_id()
			
			contacts.add( (rec_resname, rec_resnum, rec_chname, lig_resname, lig_resnum, lig_chname))
			contacts_lig.add( (lig_resname, lig_resnum, lig_chname))
			contacts_rec.add( (rec_resname, rec_resnum, rec_chname))
			
	return contacts, contacts_rec, contacts_lig


def _get_fnat( nat_contacts, dec_contacts ):
	correct_contacts = nat_contacts & dec_contacts
	incorrect_contacts = dec_contacts - correct_contacts
	fnat = float(len(correct_contacts)) / (float(len(nat_contacts)) + 0.001)
	fnonnat = float(len(incorrect_contacts)) / (float(len(dec_contacts)) + 0.001)
	return fnat, fnonnat

def _get_capri_quality(lrmsd, irmsd, fnat):
	rank = 0
	if fnat<0.1:
		rank = 0
	if (fnat>=0.1 and lrmsd<=10.0) or irmsd<=4.0:
		rank = 1
	if (fnat>=0.3 and lrmsd<=5.0) or irmsd<=2.0:
		rank = 2
	if (fnat>=0.5 and lrmsd<=1.0) or irmsd<=1.0:
		rank = 3
	return rank