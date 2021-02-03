"""
Super-resolution of CelebA using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
(if not available there see if options are listed at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the sript using command 'python3 srgan.py'
"""

import argparse
import os
import numpy as np
import math
import itertools
import sys
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import pprint
from torchvision.utils import save_image, make_grid
import cv2
from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

  
  
def Unnormalize(img):
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  unorm0 = img[0,...] * std[0] + mean[0]
  unorm1 = img[1,...] * std[1] + mean[1]
  unorm2 = img[2,...] * std[2] + mean[2]
  unorm = np.dstack((unorm0,unorm1,unorm2))
  return unorm
  
  
def srgan(opt, dataloader):
  #print(opt)
  cuda = torch.cuda.is_available()
  
  hr_shape = (opt.hr_height, opt.hr_width)
  print(hr_shape)
  # Initialize generator and discriminator
  generator = GeneratorResNet()
  discriminator = Discriminator(input_shape=(opt.channels, *hr_shape))
  feature_extractor = FeatureExtractor()
  
  # Set feature extractor to inference mode
  feature_extractor.eval()
  
  # Losses
  criterion_GAN = torch.nn.MSELoss()
  criterion_content = torch.nn.L1Loss()
  
  if cuda:
      generator = generator.cuda()
      discriminator = discriminator.cuda()
      feature_extractor = feature_extractor.cuda()
      criterion_GAN = criterion_GAN.cuda()
      criterion_content = criterion_content.cuda()
  
  if opt.epoch != 0:
      # Load pretrained models
      #generator.load_state_dict(torch.load("saved_model/generator_%d.pth"%opt.epoch))
      #discriminator.load_state_dict(torch.load("saved_model/discriminator_%d.pth"%opt.epoch))
      ###### for test_net_srgan.py
      generator.load_state_dict(torch.load("srgan/saved_models/generator_%d.pth"%opt.epoch))
      discriminator.load_state_dict(torch.load("srgan/saved_models/discriminator_%d.pth"%opt.epoch))
  
  # Optimizers
  optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
  optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
  
  Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor  
  
  # ----------
  #  Training
  # ----------
  img_count = 0
  #j=1
  img_loader = []
  for epoch in range(opt.epoch, opt.n_epochs):
      for i, imgs in enumerate(dataloader):
          #print(imgs)
          
          # Configure model input
          imgs_lr = Variable(imgs["lr"].type(Tensor))
          imgs_hr = Variable(imgs["hr"].type(Tensor))
          imgs_names = imgs["name"]
          #plt.imshow(imgs["lr"].eval(session=sess))
          #plt.imshow(sess.run(imgs["lr"]))
          #exit()
          # Adversarial ground truths
          valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
          fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
          #print(*discriminator.output_shape)
          # ------------------
          #  Train Generators
          # ------------------
  
          optimizer_G.zero_grad()
  
          # Generate a high resolution image from low resolution input
          #print(imgs_lr.size())
          gen_hr = generator(imgs_lr)
  
          # Adversarial loss
          #print(valid.size())
          #print(discriminator(gen_hr).size())
          loss_GAN = criterion_GAN(discriminator(gen_hr), valid)
  
          # Content loss
          gen_features = feature_extractor(gen_hr)
          real_features = feature_extractor(imgs_hr)
          loss_content = criterion_content(gen_features, real_features.detach())
  
          # Total loss
          loss_G = loss_content + 1e-3 * loss_GAN
  
          loss_G.backward()
          optimizer_G.step()
  
          # ---------------------
          #  Train Discriminator
          # ---------------------
  
          optimizer_D.zero_grad()
  
          # Loss of real and fake images
          loss_real = criterion_GAN(discriminator(imgs_hr), valid)
          loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)
  
          # Total loss
          loss_D = (loss_real + loss_fake) / 2
  
          loss_D.backward()
          optimizer_D.step()
  
          # --------------
          #  Log Progress
          # --------------
  
          sys.stdout.write(
              "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] \n"
              % (epoch, opt.n_epochs, i, len(dataloader), loss_D.item(), loss_G.item())
          )
  
          batches_done = epoch * len(dataloader) + i
          
          if epoch == opt.n_epochs-1:
              for k in range(gen_hr.shape[0]):
  
                  arr = np.append(imgs["origin_size"][0][k].numpy(), imgs["origin_size"][1][k].numpy())
                  gen_hr_os = F.interpolate(gen_hr, size=(int(arr[0]*1.2),int(arr[1]*1.2)), mode='bilinear') 
                  gen_hr_os_numpy = Unnormalize(gen_hr_os[k].detach().cpu().numpy())
                  gen_hr_os_numpy = gen_hr_os_numpy[:,:,::-1]
                  #print(gen_hr_os_numpy.shape)
                  cv2.imwrite("srgan/srgan_output/%s"% imgs_names[k], gen_hr_os_numpy*255)
                  #save_image(gen_hr_os,"srgan/srgan_output/%s" % imgs_names[k],normalize=True)
                  print("saved srgan output: %s"% imgs_names[k])
                  #img_loader.append(gen_hr_os)
  
  #return img_loader
                  
          
          
          #if batches_done % opt.sample_interval == 0:
  '''
          if epoch % 10 == 0:
              # Save image grid with upsampled inputs and SRGAN outputs
              imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
              gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
              imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
              img_grid = torch.cat((imgs_lr, gen_hr), -1)
              save_image(img_grid, "images_0612/%d.JPG" % (epoch*100+j), normalize=False)
              j+=1
             
      j = 1
            
  
      if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
          # Save model checkpoints
          torch.save(generator.state_dict(), "saved_models/generator_%d.pth" % epoch)
          torch.save(discriminator.state_dict(), "saved_models/discriminator_%d.pth" % epoch)
  '''
if __name__ == "__main__":
  dataloader = DataLoader(
      ImageDataset("srgan_input", hr_shape=(512,512)),
      batch_size=3,
      shuffle=True,
      num_workers=0,
  )
  args_sr = get_parse()
  gan_result = srgan(args_sr, dataloader)
  print(gan_result.shape)

