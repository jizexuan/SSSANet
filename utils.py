import os
import sys 
import torch
import shutil
import numpy as np 
from config import config
from torch import nn
import torch.nn.functional as F 
from sklearn.metrics import f1_score
import random
from scipy.io import loadmat

"https://github.com/Bjarten/early-stopping-pytorch"
class EarlyStopping:
	"""Early stops the training if validation loss doesn't improve after a given patience."""
	def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
		"""
		Args:
			patience (int): How long to wait after last time validation loss improved.
							Default: 7
			verbose (bool): If True, prints a message for each validation loss improvement. 
							Default: False
			delta (float): Minimum change in the monitored quantity to qualify as an improvement.
							Default: 0
			path (str): Path for the checkpoint to be saved to.
							Default: 'checkpoint.pt'
		"""
		self.patience = patience
		self.verbose = verbose
		self.counter = 0
		self.best_score = None
		self.early_stop = False
		self.val_loss_min = np.Inf
		self.delta = delta
		self.path = path

	def __call__(self, val_loss, model):

		score = -val_loss

		if self.best_score is None:
			self.best_score = score
			self.save_checkpoint(val_loss, model)
		elif score < self.best_score + self.delta:
			self.counter += 1
			print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
			if self.counter >= self.patience:
				self.early_stop = True
		else:
			self.best_score = score
			self.save_checkpoint(val_loss, model)
			self.counter = 0

	def save_checkpoint(self, val_loss, model):
		'''Saves model when validation loss decrease.'''
		if self.verbose:
			print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
		torch.save(model.state_dict(), self.path)
		self.val_loss_min = val_loss
# save best model
def save_checkpoint(state, is_best_acc,is_best_loss,is_best_f1,fold):
	filename = config.weights + config.model_name + os.sep +str(fold) + os.sep + "checkpoint.pth.tar"
	torch.save(state, filename)
	if is_best_acc:
		shutil.copyfile(filename,"%s/%s_fold_%s_model_best_acc.pth.tar"%(config.best_models,config.model_name,str(fold)))
	if is_best_loss:
		shutil.copyfile(filename,"%s/%s_fold_%s_model_best_loss.pth.tar"%(config.best_models,config.model_name,str(fold)))
	if is_best_f1:
		shutil.copyfile(filename,"%s/%s_fold_%s_model_best_f1.pth.tar"%(config.best_models,config.model_name,str(fold)))

# evaluate meters
class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

# print logger
class Logger(object):
	def __init__(self):
		self.terminal = sys.stdout  #stdout
		self.file = None

	def open(self, file, mode=None):
		if mode is None: mode ='w'
		self.file = open(file, mode)

	def write(self, message, is_terminal=1, is_file=1 ):
		if '\r' in message: is_file=0

		if is_terminal == 1:
			self.terminal.write(message)
			self.terminal.flush()
			#time.sleep(1)

		if is_file == 1:
			self.file.write(message)
			self.file.flush()

	def flush(self):
		# this flush method is needed for python 3 compatibility.
		# this handles the flush command by doing nothing.
		# you might want to specify some extra behavior here.
		pass
		

def get_learning_rate(optimizer):
	lr=[]
	for param_group in optimizer.param_groups:
	   lr +=[ param_group['lr'] ]

	#assert(len(lr)==1) #we support only one param_group
	lr = lr[0]

	return lr

def time_to_str(t, mode='min'):
	if mode=='min':
		t  = int(t)/60
		hr = t//60
		min = t%60
		return '%2d hr %02d min'%(hr,min)

	elif mode=='sec':
		t   = int(t)
		min = t//60
		sec = t%60
		return '%2d min %02d sec'%(min,sec)
	else:
		raise NotImplementedError
	
def readMFile(datapath):
	m = loadmat(datapath)
	T11 = m['T11']
	T12 = m['T12']
	T13 = m['T13']
	T22 = m['T22']
	T23 = m['T23']
	T33 = m['T33']
	return T11,T12,T13,T22,T23,T33