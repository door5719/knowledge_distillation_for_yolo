import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, gradcheck
from torch.autograd.gradcheck import gradgradcheck
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import torchvision.utils as vutils
import time
import pdb

"""
context attentnion module
"""
class _ContextAttentionModule(nn.Module):
	def __init__(self, in_channels, inter_channels=None, dimension=2,mode='dot_product', sub_sample=True, bn_layer=True):
		super(_ContextAttentionModule, self).__init__()

		assert dimension in [1, 2, 3]

		self.dimension = dimension
		self.sub_sample = sub_sample

		self.in_channels = in_channels
		self.inter_channels = inter_channels

		if self.inter_channels is None:
			self.inter_channels = in_channels // 2
			if self.inter_channels == 0:
				self.inter_channels = 1

		if dimension == 3:
			conv_nd = nn.Conv3d
			max_pool = nn.MaxPool3d
			bn = nn.BatchNorm3d
		elif dimension == 2:
			conv_nd = nn.Conv2d
			max_pool = nn.MaxPool2d
			bn = nn.BatchNorm2d
		else:
			conv_nd = nn.Conv1d
			max_pool = nn.MaxPool1d
			bn = nn.BatchNorm1d

		self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
						 kernel_size=1, stride=1, padding=0)

		if bn_layer:
			self.W = nn.Sequential(
				conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
						kernel_size=1, stride=1, padding=0),
				bn(num_features=self.in_channels)
			)
			nn.init.constant(self.W[1].weight, 0)
			nn.init.constant(self.W[1].bias, 0)
		else:
			self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
							 kernel_size=1, stride=1, padding=0)
			nn.init.constant(self.W.weight, 0)
			nn.init.constant(self.W.bias, 0)

		self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
							 kernel_size=1, stride=1, padding=0)
		self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
						   kernel_size=1, stride=1, padding=0)

		if sub_sample:
			self.g = nn.Sequential(self.g, max_pool(kernel_size=2))
			self.phi = nn.Sequential(self.phi, max_pool(kernel_size=2))
		if mode=='dot_product':
			self.operation_function = self._dot_product
		elif mode=='embedded_gaussian':
			self.operation_function = self._embedded_gaussian

	def forward(self, x1,x2):
		'''
		:param x: (b, c, t, h, w)
		:return:
		'''
		output = self.operation_function(x1,x2)
		return output

	def _embedded_gaussian(self, xi,xj):  #xi is the original feature map , xj is the context one 
		'''
		:param x: (b, c, t, h, w)
		:return:
		'''

		batch_size = xi.size(0)

		g_x = self.g(xj).view(batch_size, self.inter_channels, -1)
		g_x = g_x.permute(0, 2, 1)

		theta_x = self.theta(xi).view(batch_size, self.inter_channels, -1)
		theta_x = theta_x.permute(0, 2, 1)
		phi_x = self.phi(xj).view(batch_size, self.inter_channels, -1)
		f = torch.matmul(theta_x, phi_x)

		f_div_C = F.softmax(f,dim=-1)

		y = torch.matmul(f_div_C, g_x)
		y = y.permute(0, 2, 1).contiguous()
		y = y.view(batch_size, self.inter_channels, *xi.size()[2:])
		W_y = self.W(y)
		z = W_y + xi

		return z

	def _dot_product(self, xi,xj):
		batch_size = xi.size(0)
		# g=>(b, c, t, h, w)->(b, 0.5c, t, h, w)->(b, thw, 0.5c)
		g_x = self.g(xj).view(batch_size, self.inter_channels, -1)
		g_x = g_x.permute(0, 2, 1)

		theta_x = self.theta(xi).view(batch_size, self.inter_channels, -1)
		theta_x = theta_x.permute(0, 2, 1)
		phi_x = self.phi(xj).view(batch_size, self.inter_channels, -1)

		f = torch.matmul(theta_x, phi_x)
		N = f.size(-1)
		f_div_C = f / N

		y = torch.matmul(f_div_C, g_x)
		y = y.permute(0, 2, 1).contiguous()
		y = y.view(batch_size, self.inter_channels, *xi.size()[2:])
		W_y = self.W(y)  #conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0)
		z = W_y + xi

		return z

"""DCAS"""
class DilatedContextAttentionModule(nn.Module):
	def __init__(self,mode='dot_product',channels=256):
		super(DilatedContextAttentionModule, self).__init__()
		self.in_channels=channels
		self.inter_channels=channels
		self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
						 kernel_size=1, stride=1, padding=0)

		self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
							 kernel_size=1, stride=1, padding=0)

		self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
							 kernel_size=1, stride=1, padding=0)
		self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
						   kernel_size=1, stride=1, padding=0)
		if mode=='dot_product':
			self.operation_function = self._dot_product
		elif mode=='embedded_gaussian':
			self.operation_function = self._embedded_gaussian

	def forward(self, x1,x2):
		'''
		:param x: (b, c, t, h, w)
		:return:
		'''
		output = self.operation_function(x1,x2)
		return output

	def _embedded_gaussian(self, xi,xj):  #x is the original feature map , xj is the context one 
		'''
		:param x: (b, c, t, h, w)
		:return:
		'''
		batch_size = xi.size(0)
		g_x = self.g(xj).view(batch_size, self.inter_channels, -1)
		g_x = g_x.permute(0, 2, 1)

		theta_x = self.theta(xi).view(batch_size, self.inter_channels, -1)
		theta_x = theta_x.permute(0, 2, 1)
		phi_x = self.phi(xj).view(batch_size, self.inter_channels, -1)
		f = torch.matmul(theta_x, phi_x)
		f_div_C = F.softmax(f,dim=-1)
		y = torch.matmul(f_div_C, g_x)
		y = y.permute(0, 2, 1).contiguous()
		y = y.view(batch_size, self.inter_channels, *xi.size()[2:])
		W_y = self.W(y)
		z = W_y + xi
		return z

	def _dot_product(self, xi,xj):
		batch_size = xi.size(0)
		# g=>(b, c, t, h, w)->(b, 0.5c, t, h, w)->(b, thw, 0.5c)
		g_x = self.g(xj).view(batch_size, self.inter_channels, -1)
		g_x = g_x.permute(0, 2, 1)

		theta_x = self.theta(xi).view(batch_size, self.inter_channels, -1)
		theta_x = theta_x.permute(0, 2, 1)
		phi_x = self.phi(xj).view(batch_size, self.inter_channels, -1)

		f = torch.matmul(theta_x, phi_x)
		N = f.size(-1)
		f_div_C = f / N

		y = torch.matmul(f_div_C, g_x)
		y = y.permute(0, 2, 1).contiguous()
		y = y.view(batch_size, self.inter_channels, *xi.size()[2:])
		W_y = self.W(y)  #conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0)
		z = W_y + xi

		return z

class Dilated_Context_Attention_Sum(nn.Module):

	def __init__(self,padding_dilation,input_channels):
		super(Dilated_Context_Attention_Sum, self).__init__()
		self.branch0 = nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=padding_dilation,dilation=padding_dilation,bias=True)
		self.relu_b=nn.ReLU(inplace=True)
		self.cam=DilatedContextAttentionModule(mode='dot_product',channels=input_channels)
	
	def forward(self, x):
		x0 = self.branch0(x)
		x0 = self.relu_b(x0)
		out=self.cam(x0,x)
		return out

