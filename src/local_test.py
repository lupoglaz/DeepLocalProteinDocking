import os
import sys
import torch
import argparse
from Docker import Docker
from Dataset import get_benchmark_stream
from Models import GlobalDockingModel, SimpleFilter, E3MultiResRepr4x4, SE3MultiResReprScalar

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src import LOG_DIR, MODELS_DIR, DATA_DIR

from tqdm import tqdm

from local_train import select_model

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-experiment', default='LocalDebugSE3', help='Experiment name')
	parser.add_argument('-dataset', default='DebugDockingBenchmark:Table.csv', help='Dataset name')
	parser.add_argument('-angle_inc', default=15, help='Angle increment, int', type=int)
	parser.add_argument('-threshold_clash', default=300.0, help='Clash theshold for excluding conformations', type=float)
	parser.add_argument('-group', default='SE3', help='Equivariance group of the algorithm', type=str)
	parser.add_argument('-model', default='SE3MultiResRepr', help='Name of the model to train', type=str)
	parser.add_argument('-filter', default='SimpleFilter', help='Name of the model to train', type=str)
	parser.add_argument('-load_epoch', default=299, help='Max epoch', type=int)
	parser.add_argument('-start', default=0, help='Starting id', type=int)
	parser.add_argument('-end', default=1, help='Ending id', type=int)
	parser.add_argument('-rewrite', default=0, help='Rewrite previous output', type=int)
	
	args = parser.parse_args()
	dataset_name = args.dataset.split(':')[0]
	subset_name = args.dataset.split(':')[1]

	EXP_DIR = os.path.join(LOG_DIR, args.experiment)
	MDL_DIR = os.path.join(MODELS_DIR, args.experiment)
	TEST_DIR = os.path.join(EXP_DIR, dataset_name+'_%d'%args.angle_inc + "%.1f"%args.threshold_clash)
	data_dir = os.path.join(DATA_DIR, dataset_name)
	try:
		os.mkdir(TEST_DIR)
	except:
		pass
	

	torch.cuda.set_device(0)
	
	stream_test = get_benchmark_stream(data_dir, struct_folder='Matched', subset=subset_name, debug=False)
	
	protein_model, conformations_filter = select_model(args)

	docking_model = GlobalDockingModel(	representation=protein_model, filter=conformations_filter, 
										normalize=False, rotate_ligand=False, exclude_clashes=True,
										threshold_clash=args.threshold_clash).cuda()
	docking_model.load(MDL_DIR, epoch=args.load_epoch)

	docker = Docker(docking_model=docking_model, angle_inc=args.angle_inc, box_size=80, resolution=1.25, max_conf=2000, randomize_rot=True)
		
	for n, data in enumerate(stream_test):
		if not(n>=args.start and n<args.end):
			continue
		pdb_name, native_path, ureceptor, uligand, breceptor, bligand, cplx = data
		pdb_name = pdb_name[0]
		rec_path = ureceptor[0]
		lig_path = uligand[0]
				
		if docker.new_log(os.path.join(TEST_DIR, "%s.dat"%pdb_name), rewrite=bool(args.rewrite)):
			print('Processing', pdb_name)
			with torch.no_grad():
				if args.group == 'E3':
					docker.dockE3(rec_path, lig_path, batch_size=2)
				elif args.group == 'SE3':
					docker.dockSE3(rec_path, lig_path, batch_size=2)
				else:
					raise Exception("Unknown equivariance group", args.group)
		else:
			print('Skipping', pdb_name)
