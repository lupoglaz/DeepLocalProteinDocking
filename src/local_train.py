import os
import sys
import torch
import argparse
from Training import LocalTrainer
from Dataset import get_dataset_stream
from Models import E3MultiResRepr4x4, SimpleFilter
from Models import SE3MultiResReprScalar
from Models import BatchRankingLoss, LocalDockingModel

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src import LOG_DIR, MODELS_DIR, DATA_DIR

from tqdm import tqdm
import numpy as np

torch.manual_seed(42)

def select_model(args):
	if args.group == 'E3':
		
		if args.model == "E3MultiResRepr4x4":
			protein_model = E3MultiResRepr4x4(multiplier=8).cuda()
		else:
			raise Exception("Unknown model name", args.model)
		
	elif args.group == 'SE3':
		
		if args.model == "SE3MultiResReprScalar":
			protein_model = SE3MultiResReprScalar(multiplier=8).cuda()
		else:
			raise Exception("Unknown model name", args.model)
	else:
		raise Exception("Unknown equivariance group", args.group)


	if args.filter == "SimpleFilter":	
		conformations_filter = SimpleFilter(protein_model.get_num_outputs()).cuda()
	else:
		raise Exception("Unknown filter name", args.filter)
	

	return protein_model, conformations_filter


if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-experiment', default='LocalDebugE3', help='Experiment name')
	parser.add_argument('-group', default='E3', help='Equivariance group of the algorithm', type=str)
	parser.add_argument('-model', default='E3MultiResRepr4x4', help='Name of the model to train', type=str)
	parser.add_argument('-filter', default='SimpleFilter', help='Name of the model to train', type=str)
	parser.add_argument('-dataset', default='Docking/SplitComplexes:debug_set.dat:debug_set.dat', help='Dataset name')
		
	parser.add_argument('-lr', default=0.0001 , help='Learning rate', type=float)
	parser.add_argument('-lrd', default=0.00001 , help='Learning rate decay', type=float)
	parser.add_argument('-load_epoch', default=-1, help='Max epoch', type=int)
	parser.add_argument('-max_epoch', default=100, help='Max epoch', type=int)
	
	args = parser.parse_args()

	torch.cuda.set_device(0)
	dataset_name = args.dataset.split(':')[0]
	dataset_subset_train = args.dataset.split(':')[1]
	dataset_subset_val = args.dataset.split(':')[2]
	data_dir = os.path.join(DATA_DIR, dataset_name)

	stream_train = get_dataset_stream(data_dir, subset=dataset_subset_train, field='quality', threshold=1, batch_size=10, shuffle=True)
	stream_valid = get_dataset_stream(data_dir, subset=dataset_subset_val, field='quality', threshold=1, batch_size=10, shuffle=True)

	protein_model, conformations_filter = select_model(args)
	
	docking_model = LocalDockingModel(representation=protein_model, filter=conformations_filter).cuda()
	loss = BatchRankingLoss().cuda()

	N_params = sum(p.numel() for p in docking_model.parameters() if p.requires_grad)
	print("Num model %s parameters:"%(args.model), N_params)
	
	trainer = LocalTrainer(	model=docking_model, loss=loss, lr = args.lr, lr_decay=args.lrd, 
							box_size=80, resolution=1.25,
							add_zero=True, zero_weight=1.0,
							add_neg=False, neg_weight=1.0,
							randomize_rot=True)

	EXP_DIR = os.path.join(LOG_DIR, args.experiment)
	MDL_DIR = os.path.join(MODELS_DIR, args.experiment)
	try:
		os.mkdir(EXP_DIR)
	except:
		pass
	try:
		os.mkdir(MDL_DIR)
	except:
		pass


	if args.load_epoch > -1:
		docking_model.load(MDL_DIR, epoch=args.load_epoch)
	
	for epoch in range(args.load_epoch+1, args.max_epoch):
		print("\nEpoch", epoch)
		trainer.new_log(os.path.join(EXP_DIR,"training_epoch%d.dat"%epoch))
		
		av_loss = []
		for data in tqdm(stream_train):
			loss = trainer.optimize(data)
			av_loss.append(loss)
		print('Loss training = ', np.mean(av_loss))
		
		docking_model.save(MDL_DIR, epoch=epoch)
		
		trainer.new_log(os.path.join(EXP_DIR,"validation_epoch%d.dat"%epoch))
		av_loss = 0.0
		for data in tqdm(stream_valid):
			loss = trainer.score(data)
			av_loss += loss
		
		av_loss/=len(stream_valid)
		print('Loss validation = ', av_loss)
