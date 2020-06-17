import torch.nn as nn
import math
import torch

def conv_3x3(inp, oup, stride):
	return nn.Sequential(
		nn.Conv3d(inp, oup, 3, stride, 1, bias=False),
		nn.ReLU(inplace=True)
	)

def conv_1x1(inp, oup):
	return nn.Sequential(
		nn.Conv3d(inp, oup, 1, 1, 0, bias=False),
		nn.ReLU(inplace=True)
	)
		
class SSSA(nn.Module):
	def __init__(self,features,oup,r,stride=1,L=32):
		super(SSSA, self).__init__()
		d = max(int(features / r), L)
		self.SpatialConv = nn.Sequential(
			nn.Conv3d(features,oup, (1,3,3),stride,(0,1,1),bias=False),
			nn.ReLU(inplace=True),
		)
		self.TemporalConv = nn.Sequential(
			nn.Conv3d(oup,oup,(3,1,1),1,(1,0,0),bias=False),
			nn.ReLU(inplace=True),
		)
		self.gap = nn.AdaptiveAvgPool3d(1)
		self.fc = nn.Linear(oup, d)
		self.fcs1 = nn.Linear(d, oup)
		self.fcs2 = nn.Linear(d, oup)
		self.softmax = nn.Softmax(dim=1)
		self.nonlinear=	nn.Sequential(
				nn.BatchNorm3d(oup),
				nn.ReLU(inplace=True)
				)
	def forward(self, x):
		x = self.SpatialConv(x)
		fea1 = x
		fea2 = self.TemporalConv(x)
		fea_s = self.gap(fea2).squeeze_()
		fea_z = self.fc(fea_s)
		vector1 = self.fcs1(fea_z)
		vector2 = self.fcs2(fea_z)
		vector1 = self.softmax(vector1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
		vector2 = self.softmax(vector2).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
		fea_v = fea1*vector1+fea2*vector2
		fea_v = self.nonlinear(fea_v)
		return fea_v

class InvertedResidual(nn.Module):
	def __init__(self, inp, stride, expand_ratio):
		super(InvertedResidual, self).__init__()
		assert stride in [1, 2]

		hidden_dim = round(inp * expand_ratio)
		self.identity = stride == 1
		self.conv = nn.Sequential(
				nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False),
				nn.ReLU(inplace=True),
				SSSA(hidden_dim,hidden_dim,r=4,stride=stride, L=6),
				nn.Conv3d(hidden_dim, inp, 1, 1, 0, bias=False),
				#linear activation after point-wise convolution
		)
	def forward(self, x):
		if self.identity:
			return x + self.conv(x)
		else:
			return self.conv(x)

class SSSNet(nn.Module):
	def __init__(self, num_classes=1000, input_channel=1):
		super(SSSNet, self).__init__()
		# setting of inverted residual blocks
		conv_3x3_outChannel = 8
		layers = [conv_3x3(input_channel, conv_3x3_outChannel, 2)]
		# building inverted residual blocks
		block = InvertedResidual
		input_channel = conv_3x3_outChannel
		
		layers.append(block(input_channel, 1, 3))
		layers.append(block(input_channel, 1, 3))
		layers.append(block(input_channel, 1, 3))
		
		self.features = nn.Sequential(*layers)
		# building last several layers
		output_channel = 128
		self.conv = conv_1x1(input_channel, output_channel)
		self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
		self.classifier = nn.Linear(output_channel, num_classes)
		self._initialize_weights()

	def forward(self, x):
		x = self.features(x)
		x = self.conv(x)
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x
		
	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv3d):
				nn.init.xavier_normal_(m.weight.data)
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm3d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				nn.init.xavier_normal_(m.weight.data)
				m.bias.data.zero_()