import os
import sys
import numpy as np

import subprocess
import shutil

PROFIT_PATH = '/home/lupoglaz/Programs/ProFitV3.1/profit'
#PROFIT_PATH = '/home1/gd387/Programs/ProFitV3.1/profit'

def ligand_rmsd( param ):
	native_structure, decoy, receptor, ligand = param
	
	script = "REFERENCE %s\nMOBILE %s\nATOMS N,CA,C,O\nZONE %s\nFIT\nZONE CLEAR\nRZONE %s\n"%(native_structure, decoy, receptor, ligand)
	cmd = [PROFIT_PATH]
	ps_script = subprocess.Popen(["echo", script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	ps_align = subprocess.Popen(cmd, stdout=subprocess.PIPE, stdin=ps_script.stdout, stderr=subprocess.STDOUT)
	ps_script.stdout.close()
	output = ps_align.communicate()[0]
	
	rmsd = []
	for line in output.split(b'\n'):
		if line.find(b'RMS:') != -1:
			rms_line = line.split(b'RMS:')[1]
			rmsd.append(float(rms_line))
	if len(rmsd) < 2:
		print(native_structure, decoy, receptor, ligand)
		print(output)
		raise(Exception("Error in script"))

	return rmsd[1]

def get_irmsd( param ):
	native_structure, decoy_structure, rec_chain, lig_chain, contact_receptor, contact_ligand = param

	zones = ''
	for contact in contact_receptor:
		zones += 'ZONE ' + rec_chain+str(contact[1])+ '-'+ rec_chain+str(contact[1]) + '\n'
	
	for contact in contact_ligand:
		zones += 'ZONE ' + lig_chain+str(contact[1])+ '-'+ lig_chain+str(contact[1]) + '\n'

	script = "REFERENCE %s\nMOBILE %s\nATOMS N,CA,C,O\n%sFIT\n"%(native_structure, decoy_structure, zones)
	
	cmd = [PROFIT_PATH]
	ps_script = subprocess.Popen(["echo", script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	ps_align = subprocess.Popen(cmd, stdout=subprocess.PIPE, stdin=ps_script.stdout, stderr=subprocess.STDOUT)
	ps_script.stdout.close()
	output = ps_align.communicate()[0]
	
	rmsd = None
	for line in output.split(b'\n'):
		if line.find(b'RMS:') != -1:
			rms_line = line.split(b'RMS:')[1]
			rmsd = float(rms_line)
	if rmsd is None:
		print(native_structure, decoy_structure, rec_chain, lig_chain)
		print(output)
		raise(Exception("Error in script"))

	return rmsd



if __name__=='__main__':
	# decoy = '/media/lupoglaz/ProteinsDataset/Docking/Complexes/1AA7/1aa7_rA_lB_m1_s0003.pdb'
	# native = '/media/lupoglaz/ProteinsDataset/Docking/Structures/1aa7.pdb'
	native = "/media/lupoglaz/ProteinsDataset/Docking/Structures/1bdm.pdb"
	decoy = "/media/lupoglaz/ProteinsDataset/Docking/Complexes/1BDM/1bdm_rA_lB_m10_nearnative_s0010.pdb"
	receptor = "A*"
	ligand = "B*"
	# receptor = 'A*'
	# ligand = 'B*'
	lrmsd = ligand_rmsd((native, decoy, receptor, ligand))
	print(lrmsd)