import os
import sys
import torch
from math import *
from collections import OrderedDict
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from src import LOG_DIR, DATA_DIR

from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered, CoordsTranslate, CoordsRotate, writePDB
from TorchProteinLibrary.RMSD import Coords2RMSD

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from Dataset.processing_utils import _get_contacts, _get_fnat, _get_capri_quality
from DockingBenchmark import DockingBenchmark

import matplotlib 
import matplotlib.pylab as plt
import mpl_toolkits.mplot3d.axes3d as p3


class ProteinStructure:
	def __init__(self, coords, chains, resnames, resnums, atomnames, numatoms):
		self.set(coords, chains, resnames, resnums, atomnames, numatoms)

	def set(self, coords, chains, resnames, resnums, atomnames, numatoms):
		self.coords, self.chains, self.resnames, self.resnums, self.atomnames, self.numatoms = coords, \
		chains, resnames, resnums, atomnames, numatoms

	def get(self):
		return self.coords, self.chains, self.resnames, self.resnums, self.atomnames, self.numatoms

	def select_atoms_mask(self, atomic_mask):
		N = self.numatoms[0].item()
		
		isSel = atomic_mask
		isSel_coords = torch.stack([atomic_mask for i in range(3)], dim=1).unsqueeze(dim=0)
		isSel_names = torch.stack([atomic_mask for i in range(4)], dim=1).unsqueeze(dim=0)	
		num_sel_atoms =  atomic_mask.sum().item()
		sel_numatoms = torch.tensor([num_sel_atoms], dtype=torch.int, device='cpu')	
		
		coords = self.coords.view(1, N, 3)
		sel_coords = torch.masked_select(coords, isSel_coords).view(1, num_sel_atoms*3).contiguous()
		sel_chains = torch.masked_select(self.chains, isSel_names).view(1, num_sel_atoms, 4).contiguous()
		sel_resnames = torch.masked_select(self.resnames, isSel_names).view(1, num_sel_atoms, 4).contiguous()
		sel_resnums = torch.masked_select(self.resnums, isSel).view(1, num_sel_atoms).contiguous()
		sel_atomnames = torch.masked_select(self.atomnames, isSel_names).view(1, num_sel_atoms, 4).contiguous()

		return sel_coords, sel_chains, sel_resnames, sel_resnums, sel_atomnames, sel_numatoms

		# self.coords, self.chains, self.resnames, self.resnums, self.atomnames, self.numatoms = sel_coords, \
		# sel_chains, sel_resnames, sel_resnums, sel_atomnames, sel_numatoms 

	def select_CA(self):
		is0C = torch.eq(self.atomnames[:,:,0], 67).squeeze()
		is1A = torch.eq(self.atomnames[:,:,1], 65).squeeze()
		is20 = torch.eq(self.atomnames[:,:,2], 0).squeeze()
		isCA = is0C*is1A*is20

		return self.select_atoms_mask(isCA)

	def select_chain(self, chain_name):
		is0C = torch.eq(self.chains[:,:,0], ord(chain_name)).squeeze()
		is10 = torch.eq(self.chains[:,:,1], 0).squeeze()
		isChain = is0C*is10

		return self.select_atoms_mask(isChain)

	def select_residues_list(self, atom_list):
		N = self.numatoms[0].item()
		
		coords = self.coords.view(1, N, 3)
		sel_coords, sel_chains, sel_resnames, sel_resnums, sel_atomnames = [], [], [], [], []
		sel_numatoms = 0
		for chain, resnum, resname in atom_list:
			for i in range(N):
				if (chain == str(chr(self.chains[0, i, 0].item()))) and (resnum == self.resnums[0, i].item()):
					sel_coords.append(coords[:, i, :])
					sel_chains.append(self.chains[:, i, :])
					sel_resnames.append(self.resnames[:, i, :])
					sel_resnums.append(self.resnums[:, i])
					sel_atomnames.append(self.atomnames[:, i, :])
					sel_numatoms += 1
		
		sel_coords = torch.stack(sel_coords, dim=1).view(1, sel_numatoms*3).contiguous()
		sel_chains = torch.stack(sel_chains, dim=1).contiguous()
		sel_resnames = torch.stack(sel_resnames, dim=1).contiguous()
		sel_resnums = torch.stack(sel_resnums, dim=1).contiguous()
		sel_atomnames = torch.stack(sel_atomnames, dim=1).contiguous()
		sel_numatoms = torch.tensor([sel_numatoms], dtype=torch.int, device='cpu').contiguous()

		return sel_coords, sel_chains, sel_resnames, sel_resnums, sel_atomnames, sel_numatoms
		# self.coords, self.chains, self.resnames, self.resnums, self.atomnames, self.numatoms = sel_coords, \
		# sel_chains, sel_resnames, sel_resnums, sel_atomnames, sel_numatoms

	def plot_coords(self, axis = None, type='line', args = {}):
		if axis is None:
			fig = plt.figure()
			axis = p3.Axes3D(fig)
		
		N = self.numatoms[0].item()
		chains = set([])
		for i in range(N):
			chains.add(str(chr(self.chains[0,i,0])))

		for chain in chains:
			prot_chain = self.select_chain(chain)
			print(chain, prot_chain[-1].item())
			coords = prot_chain[0].view(1, prot_chain[-1].item(), 3)
			sx, sy, sz = coords[0,:,0].numpy(), coords[0,:,1].numpy(), coords[0,:,2].numpy()
			if type=='line':
				axis.plot(sx, sy, sz, label = chain, **args)
			elif type=='scatter':
				axis.scatter(sx, sy, sz, label = chain, **args)
			else:
				raise(Exception("Unknown plot type", type))
		
		coords = self.coords.view(1, N, 3)
		x, y, z = coords[0,:,0], coords[0,:,1], coords[0,:,2]
		
		ax_min_x, ax_max_x = axis.get_xlim()
		ax_min_y, ax_max_y = axis.get_ylim()
		ax_min_z, ax_max_z = axis.get_zlim()

		#Preserving aspect ratio
		min_x = min(torch.min(x).item(), ax_min_x)
		max_x = max(torch.max(x).item(), ax_max_x)
		min_y = min(torch.min(y).item(), ax_min_y)
		max_y = max(torch.max(y).item(), ax_max_y)
		min_z = min(torch.min(z).item(), ax_min_z)
		max_z = max(torch.max(z).item(), ax_max_z)
		max_L = max([max_x - min_x, max_y - min_y, max_z - min_z])
		axis.set_xlim(min_x, min_x+max_L)
		axis.set_ylim(min_y, min_y+max_L)
		axis.set_zlim(min_z, min_z+max_L)
		
		if axis is None:
			axis.legend()
			plt.show()


def unite_proteins(receptor, ligand):
	lcoords, lchains, lres_names, lres_nums, latom_names, lnum_atoms = ligand.get()
	rcoords, rchains, rres_names, rres_nums, ratom_names, rnum_atoms = receptor.get()
	ccoords = torch.cat([lcoords, rcoords], dim=1).contiguous()
	cchains = torch.cat([lchains, rchains], dim=1).contiguous()
	cres_names = torch.cat([lres_names, rres_names], dim=1).contiguous()
	cres_nums = torch.cat([lres_nums, rres_nums], dim=1).contiguous()
	catom_names = torch.cat([latom_names, ratom_names], dim=1).contiguous()
	cnum_atoms = lnum_atoms + rnum_atoms

	complex = ProteinStructure(ccoords, cchains, cres_names, cres_nums, catom_names, cnum_atoms)
	return complex


if __name__=='__main__':
	benchmark_dir = os.path.join(DATA_DIR, "DockingBenchmarkV4")
	benchmark_list = os.path.join(benchmark_dir, "TableS1.csv")
	benchmark_structures = os.path.join(benchmark_dir, "structures")
	benchmark = DockingBenchmark(benchmark_dir, benchmark_list, benchmark_structures)

	p2c = PDB2CoordsUnordered()
	target_name = '2VIS'
	target = benchmark.get_target(target_name)
	bound_complex, unbound_complex = benchmark.parse_structures(target)
		
	print('Bound receptor:', target["complex"]["chain_rec"], ''.join(list(bound_complex["receptor"]["chains"].keys())))
	print('Bound ligand:', target["complex"]["chain_lig"], ''.join(list(bound_complex["ligand"]["chains"].keys())))
	print('Unbound receptor:', target["receptor"]["chain"], ''.join(list(unbound_complex["receptor"]["chains"].keys())))
	print('Unbound ligand:', target["ligand"]["chain"], ''.join(list(unbound_complex["ligand"]["chains"].keys())))
	
	bound_receptor = ProteinStructure(*p2c([bound_complex["receptor"]["path"]]))
	bound_receptor.set(*bound_receptor.select_CA())
	bound_ligand = ProteinStructure(*p2c([bound_complex["ligand"]["path"]]))
	bound_ligand.set(*bound_ligand.select_CA())

	# fig = plt.figure()
	# axis = p3.Axes3D(fig)
	# cmap = matplotlib.cm.get_cmap('Set1')
	# bound_ligand.plot_coords(axis, args={"color":"grey"})

	# brec_cont, blig_cont = benchmark.get_contacts(bound_complex, 5.0)
	# selections = benchmark.get_symmetric_selections(bound_complex["ligand"], blig_cont)
	# i = 0
	# colors = ["red", "blue"]
	# for selection_src, selection_dst in selections:
	# 	print(selection_dst)
	# 	bound_lig_interface = ProteinStructure(*bound_ligand.select_residues_list(selection_dst))
	# 	bound_lig_interface.plot_coords(axis, type='scatter', args={"color":colors[i] ,"s":30})
	# 	i+=1	

	# axis.legend()
	# plt.show()

	# sys.exit()

	# selections = benchmark.get_symmetric_selections(bound_complex["ligand"], blig_cont)
	# for selection_src, selection_dst in selections:
	# 	bound_lig_interface = ProteinStructure(*bound_ligand.select_residues_list(selection_dst))
	# 	bound_lig_interface.plot_coords(axis, type='scatter', args={"color":"red", "s":30})


	fig = plt.figure(figsize=(10, 10))
	# axis = p3.Axes3D(fig)

	interfaces = benchmark.get_unbound_interfaces(bound_complex, unbound_complex)
	N = floor(sqrt(len(interfaces)))
	M = N
	if len(interfaces) > N*M:
		M = N+1
	
	for i, interface in enumerate(interfaces):
		axis = fig.add_subplot(N, M, i+1, projection='3d')
		bound_receptor.plot_coords(axis, args={"color":"grey"})
		bound_ligand.plot_coords(axis, args={"color":"grey"})

		urec_cont, ulig_cont, brec_cont, blig_cont = interface
		bound_rec_interface = ProteinStructure(*bound_receptor.select_residues_list(brec_cont))
		bound_rec_interface.plot_coords(axis, type='scatter', args={"color":"blue", "s":30})
		bound_lig_interface = ProteinStructure(*bound_ligand.select_residues_list(blig_cont))
		bound_lig_interface.plot_coords(axis, type='scatter', args={"color":"red", "s":30})	

	axis.legend()
	plt.show()
