import os
import sys
import torch
from math import *
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from src import LOG_DIR, DATA_DIR

from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered, CoordsTranslate, CoordsRotate, writePDB



def read_pdb_list(pdb_list_file, benchmark_dir):
	targets = []
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
				bound_complex_name = sline[0]
				native_name = bound_complex_name.split('_')[0]
				targets.append((native_name, cplx_type))	
	
	return targets

def parse_zdock_output(filename, pdb_dir):
	with open(filename) as fin:
		header = fin.readline().split()
		box_size = int(header[0])
		spacing = float(header[1])
		if len(header)==3:
			switch_num = int(header[2])
		else:
			switch_num = None
		print("Header:", box_size, spacing, switch_num)

		rec_rand1, rec_rand2, rec_rand3 = None, None, None
		if not switch_num is None:
			rec_rand1, rec_rand2, rec_rand3 = map(lambda x: float(x), fin.readline().split())
			print("Random receptor rotation:", rec_rand1, rec_rand2, rec_rand3)

		lig_rand1, lig_rand2, lig_rand3 = map(lambda x: float(x), fin.readline().split())
		print("Random ligand rotation:", lig_rand1, lig_rand2, lig_rand3)

		rec, r1, r2, r3 = fin.readline().split()
		r1, r2, r3 = float(r1), float(r2), float(r3)
		rec = os.path.join(pdb_dir, rec)
		print("Receptor:", rec)
		print("Receptor translation:", r1, r2, r3)

		lig, l1, l2, l3 = fin.readline().split()
		l1, l2, l3 = float(l1), float(l2), float(l3)
		lig = os.path.join(pdb_dir, lig)
		print("Ligand:", lig)
		print("Ligand translation:", l1, l2, l3)

		if switch_num == 1:
			print("Switching receptor and ligand files")
			tmp = rec
			rec = lig 
			lig = tmp
		
		p2c = PDB2CoordsUnordered()
		lcoords, lchains, lres_names, lres_nums, latom_names, lnum_atoms = p2c([lig])
		rcoords, rchains, rres_names, rres_nums, ratom_names, rnum_atoms = p2c([rec])
		
		cchains = torch.cat([lchains, rchains], dim=1)
		cres_names = torch.cat([lres_names, rres_names], dim=1)
		cres_nums = torch.cat([lres_nums, rres_nums], dim=1)
		catom_names = torch.cat([latom_names, ratom_names], dim=1)
		cnum_atoms = lnum_atoms+rnum_atoms

		for line in fin:
			aline = line.split()
			an_x, an_y, an_z = map(lambda x: float(x), aline[:3])
			tr_x, tr_y, tr_z = map(lambda x: int(x), aline[3:6])
			score = float(aline[6])

			lcoords_new = create_lig(lcoords, lnum_atoms, 
						rec_rand1, rec_rand2, rec_rand3,
						lig_rand1, lig_rand2, lig_rand3,
						r1, r2, r3,
						l1, l2, l3,
						an_x, an_y, an_z,
						tr_x, tr_y, tr_z,
						box_size, spacing,
						switch_num
						)
			ccoords = torch.cat([lcoords_new, rcoords], dim=1)
			writePDB("tmp.pdb", ccoords, cchains, cres_names, cres_nums, catom_names, cnum_atoms)
			break

def getRotationMatrix(psi, theta, phi, rev=0):
	if rev == 0:
		r11 = cos(psi)*cos(phi)  -  sin(psi)*cos(theta)*sin(phi)
		r21 = sin(psi)*cos(phi)  +  cos(psi)*cos(theta)*sin(phi)
		r31 = sin(theta)*sin(phi)

		r12 = -cos(psi)*sin(phi)  -  sin(psi)*cos(theta)*cos(phi)
		r22 = -sin(psi)*sin(phi)  +  cos(psi)*cos(theta)*cos(phi)
		r32 = sin(theta)*cos(phi)

		r13 = sin(psi)*sin(theta)
		r23 = -cos(psi)*sin(theta)
		r33 = cos(theta)
	else:
		r11 = cos(psi)*cos(phi)  -  sin(psi)*cos(theta)*sin(phi)
		r12 = sin(psi)*cos(phi)  +  cos(psi)*cos(theta)*sin(phi)
		r13 = sin(theta)*sin(phi)

		r21 = -cos(psi)*sin(phi)  -  sin(psi)*cos(theta)*cos(phi)
		r22 = -sin(psi)*sin(phi)  +  cos(psi)*cos(theta)*cos(phi)
		r23 = sin(theta)*cos(phi)

		r31 = sin(psi)*sin(theta)
		r32 = -cos(psi)*sin(theta)
		r33 = cos(theta)
	R = torch.tensor([	[r11, r12, r13],
						[r21, r22, r23],
						[r31, r32, r33]], 
						dtype=torch.double, device='cpu')
	return R.unsqueeze(dim=0)

def create_lig(	coords, num_atoms, 
				rec_rand1, rec_rand2, rec_rand3,
				lig_rand1, lig_rand2, lig_rand3,
				r1, r2, r3,
				l1, l2, l3,
				an_x, an_y, an_z,
				tr_x, tr_y, tr_z,
				box_size, spacing, rot_rec=0):
	
	rotate = CoordsRotate()
	translate = CoordsTranslate()

	if tr_x >= box_size/2:
		tr_x -= box_size;
	if tr_y >= box_size/2:
		tr_y -= box_size
	if tr_z >= box_size/2:
		tr_z -= box_size

	if rot_rec is None:
		Tcenter = torch.tensor([-l1, -l2, -l3], dtype=torch.double, device='cpu').unsqueeze(dim=0)
		coords_ce = translate(coords, Tcenter, num_atoms)
		
		Rrand = getRotationMatrix(lig_rand1, lig_rand2, lig_rand3)
		coords_rand = rotate(coords_ce, Rrand, num_atoms)

		Ra = getRotationMatrix(an_x, an_y, an_z)
		coords_rand_a = rotate(coords_rand, Ra, num_atoms)

		T = torch.tensor([-tr_x*spacing + r1, -tr_y*spacing + r2, -tr_z*spacing + r3], dtype=torch.double, device='cpu').unsqueeze(dim=0)
		coords_out = translate(coords_rand_a, T, num_atoms)
	
	else:
		Rrand = getRotationMatrix(rec_rand1, rec_rand2, rec_rand3)
		coords_rand = rotate(coords, Rrand, num_atoms)

		Tcenter = torch.tensor([-r1 + tr_x*spacing, -r2 + tr_y*spacing, -r3 + tr_z*spacing], dtype=torch.double, device='cpu').unsqueeze(dim=0)
		coords_ce = translate(coords_rand, Tcenter, num_atoms)

		Ra = getRotationMatrix(an_x, an_y, an_z, rev=1)
		coords_a = rotate(coords_ce, Ra, num_atoms)

		Rlig_rand = getRotationMatrix(lig_rand1, lig_rand2, lig_rand3, rev=1)
		coords_rand = rotate(coords_a, Rlig_rand, num_atoms)

		T = torch.tensor([l1, l2, l3], dtype=torch.double, device='cpu').unsqueeze(dim=0)
		coords_out = translate(coords_rand_a, T, num_atoms)
	
	return coords_out


if __name__=='__main__':
	print(os.listdir(DATA_DIR))
	print(os.listdir(LOG_DIR))

	benchmark_dir = os.path.join(DATA_DIR, "DockingBenchmarkV4")
	benchmark_list = os.path.join(benchmark_dir, "TableS2.csv")
	# print(read_pdb_list(benchmark_list, benchmark_dir))
	parse_zdock_output(
						os.path.join(LOG_DIR, "ZDOCK", "DockingBenchmarkV4_15deg", "1A2K.zd3.0.2.cg.fixed.out"),
						os.path.join(LOG_DIR, "ZDOCK", "decoys_bm4_zd3.0.2_15deg_fixed","input_pdbs")
						)
	# print(os.listdir())