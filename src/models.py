import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)

class Generator(nn.Module):

	def __init__(self, args):
		super(Generator, self).__init__()
		self.image_size = 64
		self.num_channels = 3
		self.noise_dim = args.nz
		self.embed_dim = args.nembedding
		self.projected_embed_dim = 64
		self.latent_dim = self.noise_dim + self.projected_embed_dim
		self.ngf = 64

		self.projection = nn.Sequential(
			nn.Linear(in_features=self.embed_dim, out_features=self.projected_embed_dim),
			nn.BatchNorm1d(num_features=self.projected_embed_dim),
			nn.LeakyReLU(negative_slope=0.2, inplace=True)
			)

		# based on: https://github.com/pytorch/examples/blob/master/dcgan/main.py
		self.netG = nn.Sequential(
			nn.ConvTranspose2d(self.latent_dim, self.ngf * 8, 4, 1, 0, bias=False),
			nn.BatchNorm2d(self.ngf * 8),
			nn.ReLU(True),
			# state size. (ngf*8) x 4 x 4
			nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ngf * 4),
			nn.ReLU(True),
			# state size. (ngf*4) x 8 x 8
			nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ngf * 2),
			nn.ReLU(True),
			# state size. (ngf*2) x 16 x 16
			nn.ConvTranspose2d(self.ngf * 2,self.ngf, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ngf),
			nn.ReLU(True),
			# state size. (ngf) x 32 x 32
			nn.ConvTranspose2d(self.ngf, self.num_channels, 4, 2, 1, bias=False),
			nn.Tanh()
			 # state size. (num_channels) x 64 x 64
			)

		self.apply(weights_init)

	def forward(self, input):

		# dim of z : 1*128
		projected_embed = self.projection(embed_vector).unsqueeze(2).unsqueeze(3)
		latent_vector = torch.cat([projected_embed, z], 1)
		output = self.netG(latent_vector)

		return output


class Discriminator(nn.Module):
	def __init__(self, ngpu):
	super(Discriminator, self).__init__()
	self.ngpu = ngpu
	self.main = nn.Sequential(
		# input is (nc) x 64 x 64
		nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
		nn.LeakyReLU(0.2, inplace=True),
		# state size. (ndf) x 32 x 32
		nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
		nn.BatchNorm2d(ndf * 2),
		nn.LeakyReLU(0.2, inplace=True),
		# state size. (ndf*2) x 16 x 16
		nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
		nn.BatchNorm2d(ndf * 4),
		nn.LeakyReLU(0.2, inplace=True),
		# state size. (ndf*4) x 8 x 8
		nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
		nn.BatchNorm2d(ndf * 8),
		nn.LeakyReLU(0.2, inplace=True),
		# state size. (ndf*8) x 4 x 4
		nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
		nn.Sigmoid()
	)

	def forward(self, input):
	if input.is_cuda and self.ngpu > 1:
		output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
	else:
		output = self.main(input)

	return output.view(-1, 1).squeeze(1)