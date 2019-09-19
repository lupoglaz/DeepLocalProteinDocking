import os
import sys
import torch
import argparse
from DockerParser import DockerParser
from SE3DockerParser import SE3DockerParser
import multiprocessing

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from src import LOG_DIR, MODELS_DIR, DATA_DIR
from src.Dataset import get_benchmark_stream

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from Dataset.processing_utils import _get_chains, _get_chain, _separate_chain, _run_nolb, _run_hex, run_TMScore, _get_contacts, _get_fnat, _get_capri_quality
from Dataset.profit import ligand_rmsd, get_irmsd

import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pylab as plt

import seaborn as sea
sea.set_style("whitegrid")

import _pickle as pkl

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


def cluster_complexes(data_stream, COMPLEXES_DIR, TEST_DIR):
	# data_dir = os.path.join("/media/lupoglaz/ProteinsDataset", dataset)
	# stream_test = get_benchmark_stream(data_dir, struct_folder="Matched", subset='TableS2.csv')
	
	dp = DockerParser(box_size=80, resolution=1.25)
	
	if not os.path.exists(COMPLEXES_DIR):
		os.mkdir(COMPLEXES_DIR)
	
	for data in stream_test:
		pdb_name, native_path, ureceptor, uligand, breceptor, bligand, cplx = data

		pdb_name = pdb_name[0]
		rec_path = ureceptor[0]
		lig_path = uligand[0]
		
		COMPLEX_DIR = os.path.join(COMPLEXES_DIR, pdb_name)
		if not os.path.exists(COMPLEX_DIR):
			os.mkdir(COMPLEX_DIR)
		
		conf_list_path = os.path.join(TEST_DIR, pdb_name+'.dat')
		if not os.path.exists(conf_list_path):
			continue

		print(pdb_name)

		dp.load_list(conf_list_path)
		dp.cluster_decoys(lig_path, num_clusters=10, cluster_threshold=15.0)
		dp.save_clusters(os.path.join(COMPLEX_DIR, pdb_name), rec_path, lig_path, num_clusters=10)
		
def get_hits(data_stream, COMPLEXES_DIR, TEST_DIR):
	contact_dist = 5.0
	interface_dist = 10.0
	results = {}
	dp = DockerParser(box_size=80, resolution=1.25)
	for data in stream_test:
		pdb_name, native_path, ureceptor, uligand, breceptor, bligand, cplx = data
		native_path = native_path[0]
		pdb_name = pdb_name[0]
		print(pdb_name)
		_, nat_interface_receptor, nat_interface_ligand = _get_contacts( native_path, "R", "L", contact_dist=interface_dist )
		nat_interface_receptor = [res[1] for res in nat_interface_receptor]
		nat_interface_ligand = [res[1] for res in nat_interface_ligand]

		urec_path = ureceptor[0]
		ulig_path = uligand[0]

		brec_path = breceptor[0]
		blig_path = bligand[0]

		conf_list_path = os.path.join(TEST_DIR, pdb_name+'.dat')
		if not os.path.exists(conf_list_path):
			continue

		dp.load_list(conf_list_path)
		results[pdb_name] = dp.get_hits(urec_path, ulig_path, brec_path, blig_path, nat_interface_receptor, nat_interface_ligand)
		print("Hits:", len(results[pdb_name]))
		if len(results[pdb_name])>0:
			print("First hit:", results[pdb_name][0])

	return results


def measure_quality(stream_test, COMPLEXES_DIR, num_conf=1, num_processes=10):
	contact_dist = 5.0
	interface_dist = 10.0
	results = {}
	for data in stream_test:
		pdb_name, native_path, ureceptor, uligand, breceptor, bligand, cplx = data
		native_path = native_path[0]
		pdb_name = pdb_name[0]

		nat_contacts, _, _ = _get_contacts( native_path, "R", "L", contact_dist=contact_dist )
		_, nat_interface_receptor, nat_interface_ligand = _get_contacts( native_path, "R", "L", contact_dist=interface_dist )
		
		results[pdb_name] = []
		job_schedule_lrmsd = []
		job_schedule_irmsd = []
		results_contacts = []
		
		for i in range(0, num_conf):
			decoy_path = os.path.join(os.path.join(COMPLEXES_DIR, pdb_name), pdb_name+'_%d.pdb'%i)
			if not os.path.exists(decoy_path):
				continue
			
			#_get_contacts is not thread-safe
			results_contacts.append(_get_contacts( decoy_path, "R", "L", contact_dist ))
			#irmsd and l-rmsd are thread-safe
			job_schedule_lrmsd.append((native_path, decoy_path, "R*", "L*"))	
			job_schedule_irmsd.append((native_path, decoy_path, "R", "L", nat_interface_receptor, nat_interface_ligand))
								
		pool = multiprocessing.Pool(num_processes)
		results_lrmsd = pool.map(ligand_rmsd, job_schedule_lrmsd)
		pool.close()

		pool = multiprocessing.Pool(num_processes)
		results_irmsd = pool.map(get_irmsd, job_schedule_irmsd)
		pool.close()
		
		for i in range(0, len(results_contacts)):
			decoy_contacts, decoy_contact_receptor, decoy_contact_ligand = results_contacts[i]
			
			fnat = _get_fnat(nat_contacts, decoy_contacts)
			lrmsd = results_lrmsd[i]
			irmsd = results_irmsd[i]
			quality = _get_capri_quality(lrmsd, irmsd, fnat[0])

			results[pdb_name].append((fnat, lrmsd, irmsd, quality))
			
			print(pdb_name, i, fnat, lrmsd, irmsd, quality)

	return results

def analyse_results(targets, results, hits):
	acc_diff = [0, 0, 0]
	acc_irmsd_2_5 = [0, 0, 0]
	acc_irmsd_10 = [0, 0, 0]
	for pdb_name, type in targets:
		acc = False
		acc_pepsi = False
		acc_cluspro = False
		for result in results[pdb_name]:
			if result[3]>0:
				acc = True
			if result[2]<2.5:
				acc_pepsi = True
			if result[2]<10.0:
				acc_cluspro = True

		if acc:
			acc_diff[type-1] += 1
		if acc_pepsi:
			acc_irmsd_2_5[type-1] += 1
		if acc_cluspro:
			acc_irmsd_10[type-1] += 1
			
	print('Number of targets:', len(targets))
	print('Number of acceptable(CAPRI):', acc_diff[0] + acc_diff[1] + acc_diff[2])
	print('Easy:', acc_diff[0])
	print('Medium:', acc_diff[1])
	print('Hard:', acc_diff[2])

	print('Number of acceptable(PEPSI):', acc_irmsd_2_5[0] + acc_irmsd_2_5[1] + acc_irmsd_2_5[2])
	print('Easy:', acc_irmsd_2_5[0])
	print('Medium:', acc_irmsd_2_5[1])
	print('Hard:', acc_irmsd_2_5[2])

	print('Number of acceptable(ClusPro):', acc_irmsd_10[0] + acc_irmsd_10[1] + acc_irmsd_10[2])
	print('Easy:', acc_irmsd_10[0])
	print('Medium:', acc_irmsd_10[1])
	print('Hard:', acc_irmsd_10[2])

	zdock_hits = {}
	with open(os.path.join(LOG_DIR, 'ZDOCK', 'DockingBenchmarkV4', 'rmsd_stats_zdock_decoysets.csv')) as fin:
		header = fin.readline()
		header = fin.readline()
		for line in fin:
			sline = line.split()
			if len(sline) == 0:
				break

			target = sline[0]
			difficulty = sline[1]
			category = sline[2]
			hits2k = int(sline[3])
			if hits2k>0:
				rank = int(sline[4])
			else:
				rank = None
			zdock_hits[target] = (difficulty, category, hits2k, rank)

	num_zdock = 0
	num_nn = 0
	total_zdock = 0
	total_nn = 0
	top_zdock = [0,0,0] 
	top_nn = [0,0,0]
	category_zdock = {}
	category_nn = {}
	difficulty_zdock = {}
	difficulty_nn = {}
	for target in hits.keys():
		print(target, zdock_hits[target][2], len(hits[target]))
		category = zdock_hits[target][1]
		difficulty = zdock_hits[target][0]
		if zdock_hits[target][2]>0:
			num_zdock += 1
			total_zdock += zdock_hits[target][2]
			if zdock_hits[target][3]==1:
				top_zdock[0] += 1
			elif zdock_hits[target][3]<=5:
				top_zdock[1] += 1
			elif zdock_hits[target][3]<=10:
				top_zdock[2] += 1
			
			if not category in category_zdock:
				category_zdock[category] = 0
			category_zdock[category] += zdock_hits[target][2]

			if not difficulty in difficulty_zdock:
				difficulty_zdock[difficulty] = 0
			difficulty_zdock[difficulty] += zdock_hits[target][2]

		if len(hits[target])>0:
			num_nn += 1
			total_nn += len(hits[target])
			if hits[target][0][1]==0:
				top_nn[0] += 1
			elif hits[target][0][1]<=5:
				top_nn[1] += 1
			elif hits[target][0][1]<=10:
				top_nn[2] += 1

			if not category in category_nn:
				category_nn[category] = 0
			category_nn[category] += len(hits[target])

			if not difficulty in difficulty_nn:
				difficulty_nn[difficulty] = 0
			difficulty_nn[difficulty] += len(hits[target])

	print("Number of targets with hits: ZDOCK:", num_zdock, "NN:", num_nn)
	print("Total number of hits: ZDOCK:", total_zdock, "NN:", total_nn)
	print("ZDOCK Top1/5/10:", top_zdock[0], top_zdock[1], top_zdock[2])
	print("NN Top1/5/10:", top_nn[0], top_nn[1], top_nn[2])

	print('Difficulty:')
	print('NN:')
	for difficulty in difficulty_nn:
		print(difficulty, difficulty_nn[difficulty])
	print('ZDOCK:')
	for difficulty in difficulty_zdock:
		print(difficulty, difficulty_zdock[difficulty])

	print('Category:')
	print('NN:')
	for category in category_nn:
		print(category, category_nn[category])
	print('ZDOCK:')
	for category in category_zdock:
		print(category, category_zdock[category])


def analyze_ClusPro(targets, results, hits):
	zdock_hits = {}
	with open(os.path.join(LOG_DIR, 'ZDOCK', 'DockingBenchmarkV4', 'rmsd_stats_zdock_decoysets.csv')) as fin:
		header = fin.readline()
		header = fin.readline()
		for line in fin:
			sline = line.split()
			if len(sline) == 0:
				break

			target = sline[0]
			difficulty = sline[1]
			category = sline[2]
			hits2k = int(sline[3])
			if hits2k>0:
				rank = int(sline[4])
			else:
				rank = None
			zdock_hits[target] = (difficulty, category, hits2k, rank)

	#Getting minimum irmsd, number of hits
	res = {}
	for target in hits.keys():
		min_irmsd = 1000.0
		num_hits = 0
		for n, irmsd in hits[target]:
			if irmsd < min_irmsd:
				min_irmsd = irmsd

			if irmsd<10.0:
				num_hits += 1
			
			if n==1000:
				break
		res[target] = (min_irmsd, num_hits)
	
	#Sorting into category and difficulty
	results = {}
	for target in hits.keys():
		category = zdock_hits[target][1]
		if category == 'A':
			category = 'AB'
		difficulty = zdock_hits[target][0]
		if not category in results.keys():
			results[category] = {}
		if not difficulty in results[category].keys():
			results[category][difficulty] = []
		results[category][difficulty].append(res[target])
	
	cat_results = {}
	for category in results.keys():
		cat_results[category] = {}
		if category!='AB': # classified into difficulties
			for difficulty in results[category].keys():
				av_irmsd = 0.0
				av_num_hits = 0.0
				n_targets = len(results[category][difficulty])
				for res in results[category][difficulty]:
					av_irmsd += res[0]
					av_num_hits += res[1]
				av_irmsd /= float(n_targets)
				av_num_hits /= float(n_targets)
				cat_results[category][difficulty] = (av_irmsd, av_num_hits)
		else: # no difficulties, all together
			av_irmsd = 0.0
			av_num_hits = 0.0
			n_targets = 0
			for difficulty in results[category].keys():
				n_targets += len(results[category][difficulty])
				for res in results[category][difficulty]:
					av_irmsd += res[0]
					av_num_hits += res[1]
				
			av_irmsd /= float(n_targets)
			av_num_hits /= float(n_targets)
			cat_results[category][difficulty] = (av_irmsd, av_num_hits)

	str = ''
	for difficulty in cat_results['O'].keys():
		str += '%s           \t'%difficulty
	print(str)
	for category in cat_results.keys():
		str = ''
		for difficulty in cat_results[category].keys():
			str += '%f, %f\t'%cat_results[category][difficulty]
		print(category)
		print(str)

		
def analyze_PEPSI(targets, results, hits):
	num_targets = [0, 0, 0]
	acc_irmsd_2_5 = [0, 0, 0]
	for pdb_name, type in targets:
		acc_pepsi = False
		if not pdb_name in results.keys():
			continue
		num_targets[type-1] += 1

		for result in results[pdb_name]:
			if result[2]<2.5:
				acc_pepsi = True

		if acc_pepsi:
			acc_irmsd_2_5[type-1] += 1
			
	print('Number of targets:', len(targets))
	print('Number of acceptable(CAPRI):', acc_irmsd_2_5[0] + acc_irmsd_2_5[1] + acc_irmsd_2_5[2])
	print('Easy:', acc_irmsd_2_5[0], '/', num_targets[0])
	print('Medium:', acc_irmsd_2_5[1], '/', num_targets[1])
	print('Hard:', acc_irmsd_2_5[2], '/', num_targets[2])


if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')
	parser.add_argument('-experiment', default='LocalE3MultiResRepr4x4', help='Experiment name')
	# parser.add_argument('-experiment', default='LocalSE3MultiResReprScalar', help='Experiment name')
	
	parser.add_argument('-dataset', default='DockingBenchmarkV4', help='Dataset name')
	parser.add_argument('-table', default='TableS2.csv', help='Targets table')
	parser.add_argument('-threshold_clash', default=300, help='Clash theshold for excluding conformations', type=float)
	parser.add_argument('-angle_inc', default=15, help='Angle increment, int', type=int)
		
	args = parser.parse_args()
	EXP_DIR = os.path.join(LOG_DIR, args.experiment)
	MDL_DIR = os.path.join(MODELS_DIR, args.experiment)
	TEST_DIR = os.path.join(EXP_DIR, args.dataset + '_%d'%args.angle_inc + "%.1f"%args.threshold_clash)
	DATASET_DIR = os.path.join(DATA_DIR, args.dataset)

	targets = read_pdb_list(os.path.join(DATASET_DIR, "TableS2.csv"),
							benchmark_dir=DATASET_DIR)

	COMPLEXES_DIR = "Complexes_%s"%(args.experiment)
	results_file = "Results_%s_%s_%s_%d"%(args.experiment, args.dataset, args.table, args.angle_inc)

	# data_dir = os.path.join(DATA_DIR, 'Docking', 'SplitComplexes')
	# stream_test = get_benchmark_stream(data_dir, subset='debug_set.dat', debug=True)
	
	stream_test = get_benchmark_stream(DATASET_DIR, struct_folder='Matched', subset=args.table)

	if not os.path.exists(results_file):
		cluster_complexes(stream_test, COMPLEXES_DIR, TEST_DIR)
		results = measure_quality(stream_test, COMPLEXES_DIR, num_conf=10)
		hits = get_hits(stream_test, COMPLEXES_DIR, TEST_DIR)

		with open(results_file, "wb") as fout:
			pkl.dump( (hits, results), fout)

	with open(results_file, "rb") as fin:
		hits, results = pkl.load(fin)

	analyze_ClusPro(targets, results, hits)


	targets = read_pdb_list(os.path.join(DATASET_DIR, "Table_PEPSI.csv"),
							benchmark_dir=DATASET_DIR)
	analyze_PEPSI(targets, results, hits)
	
	
	

	



	
		

