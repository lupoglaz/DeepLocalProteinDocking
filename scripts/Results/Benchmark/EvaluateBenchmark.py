import os
import sys
import torch
import argparse
from math import *
from collections import OrderedDict
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from src import LOG_DIR, DATA_DIR

from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered, CoordsTranslate, CoordsRotate, writePDB
from TorchProteinLibrary.RMSD import Coords2RMSD

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from Dataset.processing_utils import _get_contacts, _get_fnat, _get_capri_quality
from DockingBenchmark import DockingBenchmark
from VisualizeBenchmark import ProteinStructure, unite_proteins

import _pickle as pkl

from ZDOCKParser import ZDOCKParser
from DockerParser import DockerParser

import matplotlib 
import matplotlib.pylab as plt
import mpl_toolkits.mplot3d.axes3d as p3


def get_irmsd(	benchmark, parser, num_conf=1000, 
				debug_plot={"targets":[],
							"num_decoys":5,
							"dir": "Fig"}):
	result = {}
	p2c = PDB2CoordsUnordered()
	skip = parser.get_problematic_targets()
	
	for n, target_name in enumerate(benchmark.get_target_names()):
		if target_name in skip:
			print("Skipping prediction for", target_name, n)	
			continue
		print("Processing prediction for", target_name, n)
		target = benchmark.get_target(target_name)
		
		bound_target, unbound_target = benchmark.parse_structures(target)
		interfaces = benchmark.get_unbound_interfaces(bound_target, unbound_target)
		
		res = parser.parse_output(target_name, header_only=False)
		if res is None:
			print("Skipping prediction for", target_name, n)	
			continue
		
		unbound_receptor = parser.load_protein([unbound_target["receptor"]["path"]])
		unbound_receptor_ = ProteinStructure(*unbound_receptor) 
		unbound_receptor = ProteinStructure(*unbound_receptor)
		unbound_receptor.set(*unbound_receptor_.select_CA())
		
		unbound_ligand = parser.load_protein([unbound_target["ligand"]["path"]])
		unbound_ligand_ = ProteinStructure(*unbound_ligand)
		unbound_ligand = ProteinStructure(*unbound_ligand)
		unbound_ligand.set(*unbound_ligand.select_CA())
		
		#This interface will be rotated later
		unbound_interfaces = []
		for urec_sel, ulig_sel, brec_sel, blig_sel in interfaces:
			rec = ProteinStructure(*unbound_receptor.select_residues_list(urec_sel))
			lig = ProteinStructure(*unbound_ligand.select_residues_list(ulig_sel))
			unbound_interfaces.append( (rec, lig) )
		
		bound_receptor = ProteinStructure(*p2c([bound_target["receptor"]["path"]]))
		bound_receptor.set(*bound_receptor.select_CA())
		bound_ligand = ProteinStructure(*p2c([bound_target["ligand"]["path"]]))
		bound_ligand.set(*bound_ligand.select_CA())
		
		#This interface is static
		bound_interfaces = []
		for urec_sel, ulig_sel, brec_sel, blig_sel in interfaces:
			rec = ProteinStructure(*bound_receptor.select_residues_list(brec_sel))
			lig = ProteinStructure(*bound_ligand.select_residues_list(blig_sel))
			cplx = unite_proteins(rec, lig)
			bound_interfaces.append(cplx)
		
		c2rmsd = Coords2RMSD()	
		result[target_name] = []
		Nplotted = 0
		for i in range(num_conf):
			
			#Plotting transformed unbound structures
			if (target_name in debug_plot["targets"]) and (Nplotted < debug_plot["num_decoys"]): 
				fig = plt.figure()
				axis = p3.Axes3D(fig)
				cmap = matplotlib.cm.get_cmap('Set1')
				new_unbound_ligand_ = ProteinStructure(*parser.transform_ligand(unbound_ligand_.get(), i))
				unbound_receptor_.plot_coords(axis,type='line', args={"color":"blue"})
				new_unbound_ligand_.plot_coords(axis,type='line', args={"color":"red"})

			all_rmsd = []
			for rec, lig in unbound_interfaces:
				new_lig = ProteinStructure(*parser.transform_ligand(lig.get(), i))
				mobile_cplx = unite_proteins(rec, new_lig)
				for static_cplx in bound_interfaces:
					all_rmsd.append(c2rmsd(mobile_cplx.coords, static_cplx.coords, static_cplx.numatoms).item())
					
					#Plotting modile interface
					if (target_name in debug_plot["targets"]) and (Nplotted < debug_plot["num_decoys"]):
						mobile_cplx.plot_coords(axis, type='scatter', args={"color":"yellow"})
			
			#Plotting unbount decoy and all unbound interfaces
			if (target_name in debug_plot["targets"]) and (Nplotted < debug_plot["num_decoys"]):
				output_filename = os.path.join(debug_plot["dir"], target_name+'%d.png'%Nplotted)
				plt.savefig(output_filename)
				Nplotted += 1
			
			min_rmsd = min(all_rmsd)						
			result[target_name].append( min_rmsd )	
		
	return result


def BenchmarkTable(results, benchmark):
	print(benchmark.get_difficulies())
	print(benchmark.get_categories())
	table_cat = ['E', 'O']
	table_cat_nodiff = ['A', 'AB']
	for category in table_cat:
		print(category)
		for difficulty in benchmark.get_difficulies():
			targets = benchmark.get_target_names(category=[category], difficulty=[difficulty])
						
			av_min_rmsd = 0.0
			num_targets = 0
			for target in targets:
				if target in results.keys():
					av_min_rmsd += min(results[target])
					num_targets += 1
			
			if num_targets>0:
				av_min_rmsd /= float(num_targets)
				print(difficulty, '(%d/%d)'%(num_targets, len(targets)), av_min_rmsd)
			else:
				print(difficulty, '(%d/%d)'%(num_targets, len(targets)), '--')
	
	print('AB')
	targets = benchmark.get_target_names(category=['A', 'AB'])
	av_min_rmsd = 0.0
	num_targets = 0
	for target in targets:
		if target in results.keys():
			av_min_rmsd += min(results[target])
			num_targets += 1
	if num_targets>0:
		av_min_rmsd /= float(num_targets)
		print('(%d/%d)'%(num_targets, len(targets)), av_min_rmsd)
	else:
		print('--', '(%d/%d)'%(num_targets, len(targets)))


def CompareExperiments(	arguments, experiments = ['ZDOCK']):
	common_keys = None
	for experiment in experiments:
		results_filename = os.path.join(LOG_DIR, experiment, '%s_%ddeg_irmsd.pkl'%(arguments.dataset, arguments.angle_inc))
		with open(results_filename, 'rb') as fin:
			results = pkl.load(fin)
		key_set = set(list(results.keys()))
		if common_keys is None:
			common_keys = key_set
		else:
			common_keys = common_keys & key_set

	print('Number of common targets:', len(common_keys))
	benchmark.restrict_to_set(common_keys)

	for experiment in experiments:
		results_filename = os.path.join(LOG_DIR, experiment, '%s_%ddeg_irmsd.pkl'%(arguments.dataset, arguments.angle_inc))
		with open(results_filename, 'rb') as fin:
			results = pkl.load(fin)
		print(experiment)
		BenchmarkTable(results, benchmark)

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')
	# parser.add_argument('-experiment', default='LocalE3MultiResRepr4x4', help='Experiment name')
	# parser.add_argument('-experiment', default='LocalSE3MultiResReprScalar', help='Experiment name')
	parser.add_argument('-experiment', default='ZDOCK', help='Experiment name')
	
	parser.add_argument('-dataset', default='DockingBenchmarkV4', help='Dataset name')
	parser.add_argument('-table', default='TableCorrect.csv', help='Targets table')
	parser.add_argument('-threshold_clash', default=300, help='Clash theshold for excluding conformations', type=float)
	parser.add_argument('-angle_inc', default=15, help='Angle increment, int', type=int)
	parser.add_argument('-overwrite', default=0, help='Overwrite results, 1/0', type=int)
	parser.add_argument('-plot', default=0, help='Plot decoys, 1/0', type=int)
		
	args = parser.parse_args()
	
	experiment_dir = os.path.join(LOG_DIR, args.experiment)
	decoys_dir = os.path.join(experiment_dir, args.dataset + '_%d'%args.angle_inc + "%.1f"%args.threshold_clash)

	benchmark_dir = os.path.join(DATA_DIR, args.dataset)
	benchmark_table = os.path.join(benchmark_dir, args.table)
	natives_dir = os.path.join(benchmark_dir, 'structures')
	benchmark = DockingBenchmark(benchmark_dir, benchmark_table, natives_dir)

	if args.experiment == 'ZDOCK':
		parser = ZDOCKParser(decoys_dir)
	else:
		parser = DockerParser(decoys_dir)

	overwrite = bool(args.overwrite)
	plot = bool(args.plot)
	
	if args.plot:
		if not os.path.exists(args.experiment):
			os.mkdir(args.experiment)
		debug_plot={"targets":['1BVK', '1AHW'],
					"num_decoys":5,
					"dir": args.experiment}	
	else:
		debug_plot={"targets":[''],
					"num_decoys":5,
					"dir": args.experiment}	
	

	results_filename = os.path.join(experiment_dir, '%s_%ddeg_irmsd.pkl'%(args.dataset, args.angle_inc))
		
	if (not os.path.exists(results_filename)) or overwrite:
		#MAIN EVALUATION
		results = get_irmsd(benchmark, parser, num_conf=1000, debug_plot=debug_plot)

		with open(results_filename, 'wb') as fout:
			pkl.dump(results, fout)
	else:
		with open(results_filename, 'rb') as fin:
			results = pkl.load(fin)	

	BenchmarkTable(results, benchmark)

	# CompareExperiments(	args, experiments = ['LocalE3MultiResRepr4x4', 'LocalSE3MultiResReprScalar', 'ZDOCK'])

	