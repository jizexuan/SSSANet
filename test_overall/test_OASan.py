import numpy as np
from scipy.io import loadmat,savemat
from PIL import Image
from sklearn.metrics import f1_score,accuracy_score
from sklearn import metrics
def kappa(confusion_matrix, k):
	dataMat = np.mat(confusion_matrix)
	P0 = 0.0
	for i in range(k):
		P0 += dataMat[i, i]*1.0
	xsum = np.sum(dataMat, axis=1)
	ysum = np.sum(dataMat, axis=0)
	Pe= float(ysum*xsum)/np.sum(dataMat)**2
	OA = float(P0/np.sum(dataMat)*1.0)
	cohens_coefficient = float((OA-Pe)/(1-Pe))
	return cohens_coefficient
	
class EvaluationSan():
	def __init__(self,groundTruth,plabel):
		self.groundTruth = groundTruth
		self.plabel = plabel
		self.mpGroundTruth,self.mlabel,self.pl_matrix,self.mpl_matrix,self.mpGroundTruth_matrix = self.getLabeled()
		self.num_classes = 5 
	def showGroundTruth(self,groundTruth,Black=True):
		m,n = groundTruth.shape
		if Black:
			img = np.zeros([m,n,3])
		else:
			img = np.ones([m,n,3])*255
		for i in range(m):
			for j in range(n):
				pos = groundTruth[i,j]
				if pos == 1:
					img[i,j,0] = 0
					img[i,j,1] = 255
					img[i,j,2] = 255
				elif  pos == 2:
					img[i,j,0] = 255
					img[i,j,1] = 255
					img[i,j,2] = 0
				elif pos == 3 :
					img[i,j,0] = 0
					img[i,j,1] = 0
					img[i,j,2] = 255
				elif pos == 4 :
					img[i,j,0] = 255
					img[i,j,1] = 0
					img[i,j,2] = 0
				elif pos == 5 :
					img[i,j,0] = 0
					img[i,j,1] = 255
					img[i,j,2] = 0
		im = Image.fromarray(np.uint8(img))
		#im.show()
		return im
	
	def getLabeled(self):
		plabel = self.plabel.reshape([300,342])
		plabel = plabel.repeat(3,axis=0).repeat(3,axis=1)
		plabel = plabel[1:,1:-1] #test data is  obtained by a step 3
		label = self.groundTruth[:-1,:]
		mpGroundTruth_matrix=label
		mplabel = plabel*label.astype('bool')
		label = label.flatten()
		inx = (label.astype('bool'))
		mpGroundTruth = plabel.flatten()[inx]
		mlabel = label[inx]
		return mpGroundTruth,mlabel,plabel,mplabel,mpGroundTruth_matrix
		
	def test(self,detailed=False):
		y_true, y_pred = self.mpGroundTruth,self.mlabel
		overall_accuracy = metrics.accuracy_score(y_true, y_pred)
		report={'overall_accuracy':overall_accuracy}
		
		if detailed:
			classify_report = metrics.classification_report(y_true, y_pred)
			confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
			acc_for_each_class = metrics.precision_score(y_true, y_pred, average=None)
			average_accuracy = np.mean(acc_for_each_class)
			kappa_coefficient = kappa(confusion_matrix, self.num_classes)
			
			report['classify_report']=classify_report
			report['confusion_matrix']=confusion_matrix
			report['acc_for_each_class']=acc_for_each_class
			report['average_accuracy']=average_accuracy
			report['kappa_coefficient']=kappa_coefficient
		return report