import os
import torch
import random
from nets import SSSNet
import pandas as pd
import numpy as np
import utils
from tqdm import tqdm
from torch import nn,optim
from config import config
from torchsummary import summary
from torch.optim import SGD,Adam
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from timeit import default_timer as timer
from sklearn.metrics import f1_score,accuracy_score
from thop import profile
from pthflops import count_ops

class SARDataset(Dataset):
	def __init__(self,images_df,datapath,labelpath,winsize=36,mode="2D",norm=False):
		self.images_df = images_df.copy() #csv
		data = np.load(datapath)
		if norm:
			data = utils.Norm_channel(data)
		self.data = data.transpose([2,0,1]).astype('float32')
		self.label = np.load(labelpath)
		self.mode = mode
		self.winsize = winsize
	def __len__(self):
		return len(self.images_df)
		
	def __getitem__(self,index):
		r = self.images_df.iloc[index].i
		c = self.images_df.iloc[index].j
		X = self.data[:,r:r+self.winsize,c:c+self.winsize]
		y = self.label[r,c]
		return X.reshape(1,9,self.winsize,self.winsize),y-1

def train(train_loader,model,criterion,optimizer,epoch,valid_metrics,best_results,start):
	losses = utils.AverageMeter()
	f1 = utils.AverageMeter()
	acc = utils.AverageMeter()
	model.train()
	for i,(images,target) in enumerate(train_loader):
		images = images.to(device)
		indx_target=target.clone()
		target = torch.from_numpy(np.array(target)).long().to(device)
		# compute output
		output = model(images)
		loss = criterion(output,target)
		losses.update(loss.item(),images.size(0))
		f1_batch = f1_score(target.cpu().data.numpy(),np.argmax(F.softmax(output).cpu().data.numpy(),axis=1),average='macro')
		acc_score=accuracy_score(target.cpu().data.numpy(),np.argmax(F.softmax(output).cpu().data.numpy(),axis=1))
		f1.update(f1_batch,images.size(0))
		acc.update(acc_score,images.size(0))
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		print('\r',end='',flush=True)
		message = '%s %5.1f %6.1f	  |   %0.3f  %0.3f  %0.3f  | %0.3f  %0.3f  %0.4f   | %s  %s  %s |   %s' % (\
				"train", i/len(train_loader) + epoch, epoch,
				acc.avg, losses.avg, f1.avg,
				valid_metrics[0], valid_metrics[1],valid_metrics[2],
				str(best_results[0])[:8],str(best_results[1])[:8],str(best_results[2])[:8],
				utils.time_to_str((timer() - start),'min'))
		print(message , end='',flush=True)
	return [acc.avg,losses.avg,f1.avg]

# 2. evaluate function
def evaluate(val_loader,model,criterion,epoch,train_metrics,best_results,start):
	# only meter loss and f1 score
	losses = utils.AverageMeter()
	f1 = utils.AverageMeter()
	acc= utils.AverageMeter()
	# switch mode for evaluation
	model.to(device)
	model.eval()
	with torch.no_grad():
		for i, (images,target) in enumerate(val_loader):
			images_var = images.to(device)
			indx_target=target.clone()
			target = torch.from_numpy(np.array(target)).long().to(device)
			output = model(images_var)
			loss = criterion(output,target)
			losses.update(loss.item(),images_var.size(0))
			f1_batch = f1_score(target.cpu().data.numpy(),np.argmax(F.softmax(output).cpu().data.numpy(),axis=1),average='macro')
			acc_score=accuracy_score(target.cpu().data.numpy(),np.argmax(F.softmax(output).cpu().data.numpy(),axis=1))		
			f1.update(f1_batch,images.size(0))
			acc.update(acc_score,images.size(0))
			print('\r',end='',flush=True)
			message = '%s   %5.1f %6.1f	 |	 %0.3f  %0.3f   %0.3f	| %0.3f  %0.3f  %0.4f  | %s  %s  %s  |  %s' % (\
					"val", i/len(val_loader) + epoch, epoch,					
					acc.avg,losses.avg,f1.avg,
					train_metrics[0], train_metrics[1],train_metrics[2],
					str(best_results[0])[:8],str(best_results[1])[:8],str(best_results[2])[:8],
					utils.time_to_str((timer() - start),'min'))
			print(message, end='',flush=True)
	return [acc.avg,losses.avg,f1.avg]

# 3. test model on public dataset and save the probability matrix
def test(test_loader,model):
	labels= []
	model.to(device)
	model.eval()
	for i,(input,filepath) in tqdm(enumerate(test_loader)):
		#3.2 change everything to cuda and get only basename
		with torch.no_grad():
			image_var = input.to(device)
			y_pred = model(image_var)
			label = np.argmax(F.softmax(y_pred).cpu().data.numpy(),axis=1).tolist()
			labels.extend(label)
	plabel = np.array(labels).flatten().reshape([-1,1])
	print(plabel.shape)
	return plabel

def getDataset(dataset,winsize):#san
	datapath = 'DATA/San_padding'
	labelpath = 'DATA/Label_SanFrancisco.npy'
	csvpath = './data_inx/San'
	config.num_classes = 5
	config.data_name = "san"
	config.lr = 0.01
	return datapath+str(winsize)+'.npy',labelpath,csvpath
	
# 4. main function
def run(model,net,datapath,labelpath,csvpath,winsize=48):
	fold = 0
	# 4.1 mkdirs
	if not os.path.exists(config.weights + config.model_name + os.sep +str(fold)):
		os.makedirs(config.weights + config.model_name + os.sep +str(fold))
	if not os.path.exists(config.best_models):
		os.mkdir(config.best_models)
	if not os.path.exists(config.results):
		os.mkdir(config.results)
	#4.3 optim & criterion
	optimizer = optim.SGD(model.parameters(),lr = config.lr,momentum=0.9,weight_decay=1e-4)

	criterion=nn.CrossEntropyLoss().to(device)
	start_epoch = 0
	best_acc=0
	best_loss = np.inf
	best_f1 = 0
	best_results = [0,np.inf,0]
	val_metrics = [0,np.inf,0]
	
	model.to(device)
	train_lst = pd.read_csv(csvpath+"train.csv")
	train_gen = SARDataset(train_lst,datapath,labelpath,winsize,net)
	train_loader = DataLoader(train_gen,batch_size=config.batch_size,shuffle=True,pin_memory=True,num_workers=0) #num_worker is limited by shared memory in Docker!
	
	val_lst = pd.read_csv(csvpath+"val.csv")
	val_gen = SARDataset(val_lst,datapath,labelpath,winsize,net)
	val_loader = DataLoader(val_gen,batch_size=config.batch_size,shuffle=False,pin_memory=True,num_workers=0)

	start = timer()
	#train
	for epoch in range(0,config.epochs):#config.epochs
		# train
		train_metrics = train(train_loader,model,criterion,optimizer,epoch,val_metrics,best_results,start)
		# val
		val_metrics = evaluate(val_loader,model,criterion,epoch,train_metrics,best_results,start)
		# check results
		is_best_acc = val_metrics[0] > best_results[0] 
		best_results[0] = max(val_metrics[0],best_results[0])
		is_best_loss = val_metrics[1] < best_results[1]
		best_results[1] = min(val_metrics[1],best_results[1])
		is_best_f1 = val_metrics[2] > best_results[2]
		best_results[2] = max(val_metrics[2],best_results[2]) 
		# save model
		utils.save_checkpoint({
					"epoch":epoch + 1,
					"model_name":config.model_name,
					"state_dict":model.state_dict(),
					"best_acc":best_results[0],
					"best_loss":best_results[1],
					"optimizer":optimizer.state_dict(),
					"fold":fold,
					"best_f1":best_results[2],
		},is_best_acc,is_best_loss,is_best_f1,fold)
		print('\r',end='',flush=True)
		print('%s  %5.1f %6.1f	  |   %0.3f   %0.3f   %0.3f	 |  %0.3f   %0.3f	%0.3f	|   %s  %s  %s | %s' % (\
				"best", epoch, epoch,					
				train_metrics[0], train_metrics[1],train_metrics[2],
				val_metrics[0],val_metrics[1],val_metrics[2],
				str(best_results[0])[:8],str(best_results[1])[:8],str(best_results[2])[:8],
				utils.time_to_str((timer() - start),'min'))
			)
	test_lst = pd.read_csv(csvpath+"test.csv")
	test_gen = SARDataset(test_lst,datapath,labelpath,winsize,net)
	test_loader = DataLoader(test_gen,500,shuffle=False,pin_memory=True,num_workers=0)
	for point in ['loss']:#'loss','acc','f1'
		best_model = torch.load("%s/%s_fold_%s_model_best_%s.pth.tar"%(config.best_models,config.model_name,str(fold),point))
		model.load_state_dict(best_model["state_dict"])
		plabel = test(test_loader,model)
		np.save(config.results+config.model_name+ str(point) + '.npy',plabel)

def main(net,dataset,inx,winsize=48):
	datapath,labelpath,csvpath = getDataset(dataset,winsize)
	input_channel = 1
	model = SSSNet.SSSNet(num_classes=config.num_classes,input_channel=input_channel) #,nclip=9
	config.model_name = 'SSSNet'+ config.data_name
	config.model_name=config.model_name + '_' + str(winsize)
	run(model,net,datapath,labelpath,csvpath,winsize)
	#summary(model.to(device), (1,9, 36, 36)) #total params
	#input = torch.randn(1,1,9,36, 36).to(device)
	#flops, params = profile(model.to(device), inputs=(input, ))
	#count_ops(model, input)

if __name__ == "__main__":
	#4.2 get model
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	net = 'SSSNet'
	dataset = 'san'
	SEED = 1
	np.random.seed(SEED)
	torch.manual_seed(SEED)
	torch.cuda.manual_seed_all(SEED)
	torch.backends.cudnn.deterministic=True
	main(net,dataset,SEED,winsize=36)