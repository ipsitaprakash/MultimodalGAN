from __future__ import print_function
import pdb
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
import torchvision.utils as vutils
from models import Generator,Discriminator
from torch.utils.data import DataLoader
from torch.autograd import Variable
import h5py
import numpy as np
class Trainer(object):

	def __init__(self,dataset,args):

		self.dataset = dataset
		self.device = args.device

		self.generator = torch.nn.DataParallel(Generator(args).to(args.device))
		self.discriminator = torch.nn.DataParallel(Discriminator(args).to(args.device))				
		if args.netG!='':
                        self.generator.load_state_dict(torch.load(args.netG))
		if args.netD!='':
                        self.discriminator.load_state_dict(torch.load(args.netD))
		self.dataloader = DataLoader(dataset, batch_size=args.batchSize,shuffle=True, num_workers=int(args.workers))
		self.optimizerD = optim.Adam(self.discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
		self.optimizerG = optim.Adam(self.generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

		self.ngpu = int(args.ngpu)

		self.criterion = nn.BCELoss()

	def train(self,args):

		#self.generator.train()
		#self.discriminator.train()
		self.generator.eval()
		self.discriminator.eval()

		fixed_noise = torch.randn(args.batchSize, args.nz, 1, 1, device=self.device)
		gen_imgs = []
		class_imgs = []
		for epoch in range(args.nepochs):

			for i, sample in enumerate(self.dataloader):

				right_images = sample['right_images']
				right_embed = sample['right_embed']
				wrong_images = sample['wrong_images']
				batch_size = right_images.size(0)
				print('Batch size {}'.format(batch_size))
				right_images = Variable(right_images.float()).to(self.device)
				right_embed = Variable(right_embed.float()).to(self.device)
				wrong_images = Variable(wrong_images.float()).to(self.device)

				real_labels = torch.ones(batch_size).to(self.device)
				fake_labels = torch.zeros(batch_size).to(self.device)

				print("".join(['#']*50))
				#print("Right img :{}".format(right_images.size()))
				#print("Right embed : {}".format(right_embed.size()))
				#print("Wrong image: {}".format(wrong_images.size()))

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
				fake_loss = self.criterion(fake_score, fake_labels) * 0.5
				#fake_loss.backward()

				d_loss = real_loss + fake_loss + wrong_loss
				d_loss.backward()
				self.optimizerD.step()

				########################################################

				self.optimizerG.zero_grad()
				g_real_labels = torch.ones(batch_size).to(self.device)
				noise = Variable(torch.randn(batch_size, args.nz)).to(self.device) 					#CHECK : normal distr
				noise = noise.view(noise.size(0),noise.size(1), 1, 1) #TODO: dimensions
				
				generated_images = self.generator(right_embed, noise)
				fake_score = self.discriminator(generated_images,right_embed)
				g_loss = self.criterion(fake_score,g_real_labels)
				g_loss.backward()
				self.optimizerG.step()

				print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'% (epoch, args.nepochs, i, len(self.dataloader),d_loss,g_loss))
				
				if i > 380:
					continue
				else:
					fake = self.generator(right_embed, fixed_noise)
					gen_imgs.append(fake.cpu().data.numpy())
					class_imgs.append(sample['class'].cpu().data.numpy())
				"""	
				if i % 100 == 0:
					vutils.save_image(right_images, '%s/real_samples_epoch_%03d.png' % (args.outf,epoch), normalize=True)
					fake = self.generator(right_embed,fixed_noise)
					vutils.save_image(fake.detach(), '%s/fake_samples_epoch_%03d.png' % (args.outf, epoch), normalize=True)
				"""
			#torch.save(self.generator.state_dict(), '%s/netG_epoch_%d.pth' % (args.outf, epoch))
			#torch.save(self.discriminator.state_dict(), '%s/netD_epoch_%d.pth' % (args.outf, epoch))
				if i == 380:
					pdb.set_trace()
					#gen_imgs = torch.stack(gen_imgs).data.numpy()
					#class_imgs = torch.stack(class_imgs).data.numpy()
					gen_imgs = np.array(gen_imgs)
					class_imgs = np.array(class_imgs)
					f = h5py.File('../saved_frames/generator_samples.hdf5', mode='w')
					f.create_dataset('img', data=gen_imgs.reshape(gen_imgs.shape[0]*gen_imgs.shape[1], gen_imgs.shape[2], gen_imgs.shape[3],  gen_imgs.shape[4]), dtype="float64")
					f.create_dataset('class', data=class_imgs.reshape(class_imgs.shape[0]*class_imgs.shape[1], 1))	
					f.close()
					exit()
