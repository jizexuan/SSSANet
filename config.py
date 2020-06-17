#coding=utf-8
import warnings

class DefaultConfigs(object):
	env='default'
	model_name = "cnn"
	data_name='ger'
	weights = "./checkpoints/"
	results = "./results/"
	best_models = "./checkpoints/best_models/"
	debug_file='./tmp/debug'
	num_classes = 3
	img_weight = 36
	img_height = 36
	lr = 0.01
	lr_decay = 0.5
	weight_decay =0e-5
	batch_size = 64
	epochs = 100
	
def parse(self, kwargs):
	"""
	update config by kwargs
	"""
	for k, v in kwargs.items():
		if not hasattr(self, k):
			warnings.warn("Warning: opt has not attribut %s" % k)
		setattr(self, k, v)

	print('user config:')
	for k, v in self.__class__.__dict__.items():
		if not k.startswith('__'):
			print(k, getattr(self, k))


DefaultConfigs.parse = parse
config = DefaultConfigs()
