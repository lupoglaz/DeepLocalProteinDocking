import torch
from torch.autograd import Variable, Function
import torch.nn as nn
import numpy as np


class BatchRankingLossFunction(Function):
	def __init__(self, gap=1.0, threshold=0.1):
		super(BatchRankingLossFunction, self).__init__()
		self.gap = gap
		self.threshold = threshold
	
	
	def forward(self, net_output, labels):
		net_output = torch.squeeze(net_output)
		batch_size = net_output.size(0)
		
		loss = torch.zeros(1, dtype=torch.float, device='cuda')
		self.dfdo = torch.zeros(batch_size, dtype=torch.float, device='cuda')
		N = 0
		for i in range(batch_size):
			for j in range(batch_size):
				if i==j: continue
				N += 1
				tm_i = labels[i]
				tm_j = labels[j]
				
				if tm_i<tm_j:
					y_ij = -1
				else:
					y_ij = 1
				
				if torch.abs(tm_i-tm_j) > self.threshold:
					example_weight = 1.0
				else:
					example_weight = 0.0
				
				dL = example_weight*max(0, self.gap + y_ij*(net_output[i] - net_output[j]))
				if dL>0:
					self.dfdo[i] += example_weight*y_ij
					self.dfdo[j] -= example_weight*y_ij

				loss[0] += dL

		loss /= float(N)
		self.dfdo /= float(N)

		return loss
	
	
	def backward(self, input):
		return self.dfdo, None
		# return torch.unsqueeze(self.dfdo, dim=1), None


class BatchRankingLoss(nn.Module):
	def __init__(self, gap=1.0, threshold=0.1):
		super(BatchRankingLoss, self).__init__()
		self.gap = gap
		self.threshold = threshold

	def forward(self, input, gdt_ts):
		return BatchRankingLossFunction(self.gap, self.threshold)(input, gdt_ts)




if __name__=='__main__':
	outputs = Variable(torch.randn(10).cuda(), requires_grad=True)
	gdts = Variable(torch.randn(10).cuda())

	loss = BatchRankingLoss()
	y = loss(outputs, gdts)
	y.backward()
	print(y, outputs.grad)