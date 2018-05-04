from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutil
from dataset import Sound2ImageDataset
from models import Generator,Discriminator
from torch.utils.data import DataLoader




opt = parser.parse_args()

if torch.cuda.is_available() and not opt.cuda:
	print("WARNING: You have a CUDA device, so you should probably run with --cuda")

class Trainer(object):

	def __init__(self,args):
		self.dataset = dataset
		self.device = args.device

		self.generator = torch.nn.DataParallel(Generator(args.ngpu).to(args.device))
		self.discriminator = torch.nn.DataParallel(Discriminator(args.ngpu).to(args.device))

		self.dataloader = DataLoader()

		self.optimizerD = optim.Adam(self.discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
		self.optimizerG = optim.Adam(self.generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

		self.criterion = nn.BCELoss()

	def train(self,args):

		self.generator.train()
		self.discriminator.train()

		for epoch in range(args.nepochs):
			for i, sample in enumerate(dataloader):

				batch_size = right_images.size(0)
				right_images = sample['right_images']
				right_embed = sample['right_embed']
				wrong_images = sample['wrong_images']

				right_images = Variable(right_images.float()).to(self.device)
				right_embed = Variable(right_embed.float()).to(self.device)
				wrong_images = Variable(wrong_images.float()).to(self.device)

				real_labels = torch.ones(batch_size).to(device)
				fake_labels = torch.zeros(batch_size).to(device)

				##############################################

				self.optimizerD.zero_grad()
				real_score = self.discriminator(right_images, right_embed)
				real_loss = self.criterion(real_score,real_labels)		#CHECK : smoothed real labels
				#real_loss.backward()					#neeeded or not

				wrong_score = self.discriminator(wrong_images, right_embed)
				wrong_loss = self.criterion(wrong_score,fake_labels)*0.5
				#wrong_loss.backward() 

				noise = Variable(torch.randn(batch_size, args.nz)).to(self.device) 					#CHECK : normal distr
				noise = noise.view(noise.size(0),noise.size(1), 1, 1) #TODO: dimensions
				fake_images = self.generator(right_embed, noise)
				fake_score = self.discriminator(fake_images.detach(), right_embed)
				fake_loss = criterion(outputs, fake_labels) * 0.5
				#fake_loss.backward()

				d_loss = real_loss + fake_loss + wrong_loss
				d_loss.backward()
				self.optimizerD.step()

				########################################################

				self.optimizerG.zero_grad()
				g_real_labels = torch.ones(batch_size).to(device)
				noise = Variable(torch.randn(batch_size, args.nz)).to(self.device) 					#CHECK : normal distr
				noise = noise.view(noise.size(0),noise.size(1), 1, 1) #TODO: dimensions
				
				generated_images = self.generator(right_embed, noise)
				fake_score = self.discriminator(generated_images,right_embed)
				g_loss = self.criterion(fake_score,g_real_labels)
				g_loss.backward()
				self.optimizerG.step()

			torch.save(self.generator.state_dict(), '%s/netG_epoch_%d.pth' % (args.outf, epoch))
			torch.save(self.discriminator.state_dict(), '%s/netD_epoch_%d.pth' % (args.outf, epoch))
			#if (epoch) % 10 == 0:
				# save model
				#Utils.save_checkpoint(self.discriminator, self.generator, self.checkpoints_path, self.save_path, epoch)










