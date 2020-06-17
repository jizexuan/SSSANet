import numpy as np
import test_overall.test_OAGer as tger
import fnmatch
import os

def test(path,groundTruthger):
	files = os.listdir(path)
	for file in files:
		fullpath = os.path.join(path,file)
		if fnmatch.fnmatch(fullpath,'*.npy'):#是指定文件
			plabel = np.load(fullpath)+1
			if fnmatch.fnmatch(file,'*ger*'):
				eva = tger.EvaluationGer(groundTruthger,plabel)
			report = eva.test(True)
			im = eva.showGroundTruth(eva.mpl_matrix,Black=True) #预测图 ，有标签部分
			im.save(path+file[:-4]+'overlaid.BMP')
			im = eva.showGroundTruth(eva.pl_matrix,Black=True)	#预测图，全部
			im.save(path+file[:-4]+'all.BMP')
			b = (eva.mpGroundTruth_matrix!=eva.mpl_matrix).astype('int')
			im = eva.showGroundTruth(eva.mpGroundTruth_matrix*b,Black=False)#错误部分
			im.save(path+file[:-4]+'bwrong.BMP')
			print(file,report['overall_accuracy'],report['average_accuracy'],report['kappa_coefficient'])
			print(file,report['acc_for_each_class'])
			#matrix=report['confusion_matrix'].T
			#print(matrix)
			
if __name__ == '__main__':
	labelPathger = './DATA/Label_Germany.npy'
	groundTruthger = np.load(labelPathger)
	path = './results/'
	test(path,groundTruthger)