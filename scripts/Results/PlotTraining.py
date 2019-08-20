import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src import LOG_DIR, MODELS_DIR, DATA_DIR
from src.Models import BatchRankingLoss
import numpy as np
import torch
import argparse
import _pickle as pkl

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pylab as plt

import seaborn as sea
sea.set_style("whitegrid")

from tqdm import tqdm

def read_epoch_data(filename):
	data = {}
	proteins = []
	losses = []
	with open(filename, 'r') as f:
		for line in f:
			a = line.split()
			if a[0]!='Loss':
				receptor = a[0]
				ligand = a[1]
				protein_name = receptor.split('/')[-2]
				if not protein_name in proteins:
					proteins.append(protein_name)
				model_output = float(a[2])
				gdt = float(a[3])
				if not protein_name in data.keys():
					data[protein_name] = []	
				data[protein_name].append((model_output, gdt))
			else:
				loss = []
				for i in range(1, len(a)):
					loss.append(float(a[i]))
				losses.append(loss)

	return proteins, data, losses

def read_epoch_output(filename, compute_loss=False):
	proteins, data, losses = read_epoch_data(filename)

	if compute_loss:
		batch_size = len(data[proteins[0]])
		model_out = torch.zeros(batch_size, dtype=torch.float, device='cpu')
		quality = torch.zeros(batch_size, dtype=torch.float, device='cpu')
		L = BatchRankingLoss()
		ranking_losses = []
		
		for protein_name in proteins:
			for i in range(batch_size):
				model_out[i] = data[protein_name][i][0]
				quality[i] = data[protein_name][i][1]
				
			loss = L(model_out, quality)
			ranking_losses.append([loss.item()])
		return ranking_losses

	return losses

def get_training_losses(experiment, min_epoch=0, max_epoch=50, compute_loss=False):
	loss_tr = [[], [], [], [], []]
	loss_val = []
	for epoch in range(min_epoch, max_epoch):
		filename = os.path.join(LOG_DIR, experiment, 'training_epoch%d.dat'%epoch)
		losses = np.array(read_epoch_output(filename, compute_loss))
		if losses.shape[0] == 0:
			for i in range(len(loss_tr)):
				loss_tr[i].append(None)
			continue
		for i in range(losses.shape[1]):
			if len(loss_tr)<(i+1):
				loss_tr.append([])
			loss_tr[i].append(np.average(losses[:,i]))
	
	for epoch in range(min_epoch, max_epoch):
		filename = os.path.join(LOG_DIR, experiment, 'validation_epoch%d.dat'%epoch)
		losses_rank = np.array(read_epoch_output(filename, compute_loss))
		loss_val.append(np.average(losses_rank[:, 0]))

	return loss_tr, loss_val

def get_scores_gap(experiment, max_epoch=50):
	output_gap_av = []
	output_gap_std = []
	for epoch in range(max_epoch):
		filename = os.path.join(LOG_DIR, experiment, 'training_epoch%d.dat'%epoch)
		proteins, data, losses = read_epoch_data(filename)
		gap = []
		for protein in proteins:
			pos_score = []
			neg_score = []
			for model_out, gdt in data[protein]:
				if gdt>0:
					pos_score.append(model_out)
				else:
					neg_score.append(model_out)
			gap.append(np.mean(neg_score) - np.mean(pos_score))
		
		output_gap_av.append(np.mean(gap))
		output_gap_std.append(np.std(gap))
	
	return output_gap_av, output_gap_std


def plot_local(loss_tr, loss_val, output_gap_av, output_gap_std, experiment):
	f = plt.figure(figsize=(6,4))
	plt.plot(loss_tr[1], label='training rank')
	plt.plot(loss_val, label='validation rank')
	
	plt.yscale('log')
	plt.xlabel('Epoch number',fontsize=16)
	plt.ylabel('Ranking loss',fontsize=16)
	plt.legend(prop={'size':16})
	plt.tick_params(axis='x', which='major', labelsize=10)
	plt.tick_params(axis='y', which='major', labelsize=10)
	plt.tight_layout()
	plt.legend()
	plt.savefig('Fig/%s_training.png'%(experiment), format='png', dpi=600)

	f = plt.figure(figsize=(6,4))
	plt.plot(loss_tr[2], label='training clash')
	plt.plot(loss_tr[3], label='training neg')
	plt.plot(loss_tr[4], label='training zero')
	plt.plot(output_gap_av, label='Output gap average')
	plt.plot(output_gap_std, label='Output gap std')
	plt.yscale('log')
	plt.xlabel('Epoch number',fontsize=16)
	plt.ylabel('Ranking loss',fontsize=16)
	plt.tight_layout()
	plt.legend()
	plt.savefig('Fig/%s_addLosses.png'%(experiment), format='png', dpi=600)


def custom_plot_local():
	se3_loss_tr, se3_loss_val = get_training_losses(experiment = 'LocalSE3MultiResReprScalar', max_epoch=150)
	e3_loss_tr, e3_loss_val = get_training_losses(experiment = 'LocalE3MultiResRepr4x4', max_epoch=150)

	f = plt.figure(figsize=(6,4))
	plt.plot(se3_loss_tr[1], '.-', label='SE(3) training')
	plt.plot(se3_loss_val, '.-', label='SE(3) validation')
	plt.plot(e3_loss_tr[1], '--', label='ConvNet training')
	plt.plot(e3_loss_val, '--', label='ConvNet validation')
	ax = plt.gca()
	
	plt.yscale('log')
	ax.set_yticks([1.0, 0.8, 0.6, 0.4, 0.2, 0.1])
	ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
	plt.xlabel('Epoch number',fontsize=16)
	plt.ylabel('Ranking loss',fontsize=16)
	plt.legend(prop={'size':16})
	plt.tick_params(axis='x', which='major', labelsize=10)
	plt.tick_params(axis='y', which='major', labelsize=10)
	plt.ylim([0.0, 1.0])
	plt.tight_layout()
	plt.legend()
	plt.savefig('Fig/Total_training.png', format='png', dpi=600)
	sys.exit()

if __name__=='__main__':
	custom_plot_local()
	
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-experiment', default='LocalDebug', help='Experiment name')
	parser.add_argument('-type', default='local', help='Experiment name')
	parser.add_argument('-max_epoch', default=100, help='Experiment name', type=int)
	args = parser.parse_args()

	if not os.path.exists('Fig'):
		os.mkdir('Fig')
	
	loss_tr, loss_val = get_training_losses(experiment = args.experiment, max_epoch=args.max_epoch)
	print("Minimum loss validation: %f, epoch = %d"%(np.min(loss_val), np.argmin(loss_val)))
	output_gap_av, output_gap_std = get_scores_gap(experiment = args.experiment, max_epoch=args.max_epoch)
	plot_local(loss_tr, loss_val, output_gap_av, output_gap_std, args.experiment)



	
	
