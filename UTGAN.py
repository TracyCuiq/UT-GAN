import torch.nn as nn
import torch
import numpy as np

import torch.nn.functional as F
import torchvision
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torch.autograd import Variable
import datetime
from gen_models import Generator_Unet
from dis_models import Discriminator_steg
from utils import imsave_singel, show_result


models_path = './models/'
img_prob_train_path = './train_result/'
img_prob_eval_path = './eval_result/'

# custom weights initialization called on netG and netD

def weights_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            #m.weight.data.normal_(0, 0.02)
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.ConvTranspose2d):
            #nn.init.normal_(m.weight.data, 0.0, 0.02)
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
            m.bias.requires_grad = False
    
class UTGAN:
    def __init__(self, device, img_nc=1, lr=0.0001, payld=0.4, bilinear=False):

        self.device = device
        self.img_nc = img_nc
        self.lr = lr
        self.payld = payld
        self.bilinear = bilinear
        self.netG = Generator_Unet(self.img_nc, bilinear=self.bilinear).to(self.device)
        self.netDisc = Discriminator_steg(self.img_nc).to(self.device)

        # initialize all weights
        self.netG.apply(weights_init)
        self.netDisc.apply(weights_init)
        
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), self.lr)
        self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(), self.lr)

        if not os.path.exists(models_path):
            os.makedirs(models_path)
        if not os.path.exists(img_prob_train_path):
            os.makedirs(img_prob_train_path)
        if not os.path.exists(img_prob_eval_path):
            os.makedirs(img_prob_eval_path)

    def train_batch(self, cover, TANH_LAMBDA=60, D_LAMBDA=1, Payld_LAMBDA=1e-7):
        # optimize D
        for i in range(1):
            self.optimizer_D.zero_grad()
            with torch.no_grad():
                prob_pred = self.netG(Variable(cover))
            data_noise = np.random.rand(prob_pred.shape[0], prob_pred.shape[1], prob_pred.shape[2], prob_pred.shape[3])
            tensor_noise = torch.from_numpy(data_noise).float().to(self.device)
            modi_map = 0.5*(torch.tanh((prob_pred+2.*tensor_noise-2)*TANH_LAMBDA) - torch.tanh((prob_pred-2.*tensor_noise)*TANH_LAMBDA))
            stego = (cover*255 + modi_map)/255.
            data = torch.stack((cover, stego))

            data_shape = list(data.size())
            data = data.reshape(data_shape[0] * data_shape[1], *data_shape[2:])
            data_group = data.to(self.device)

            label_zeros = np.zeros(cover.shape[0])
            label_ones = np.ones(cover.shape[0])
            label = np.stack((label_zeros, label_ones))
            label = torch.from_numpy(label).long()
            label = Variable(label).to(self.device)
            label_group = label.view(-1)

            pred_D = self.netDisc(data_group.detach())
            loss_D = self.criterion(pred_D, label_group)
            loss_D.backward()
            self.optimizer_D.step()

        # optimize G
        for i in range(1):
            img_size = cover.shape[2]
            batch_size = cover.shape[0]
            self.optimizer_G.zero_grad()

            prob_pred = self.netG(Variable(cover))
            modi_map = 0.5*(torch.tanh((prob_pred+2.*tensor_noise-2)*TANH_LAMBDA) - torch.tanh((prob_pred-2.*tensor_noise)*TANH_LAMBDA))
            stego = (cover*255 + modi_map)/255.
            data = torch.stack((cover, stego))
            data_shape = list(data.size())
            data = data.reshape(data_shape[0] * data_shape[1], *data_shape[2:])
            data_group = data.to(self.device)
            pred_D = self.netDisc(data_group)
            loss_D = self.criterion(pred_D, label_group)

            # cal G's loss in GAN
            prob_chanP = prob_pred / 2.0 + 1e-5
            prob_chanM = prob_pred / 2.0 + 1e-5
            prob_unchan = 1 - prob_pred + 1e-5

            cap_entropy = torch.sum( (-prob_chanP * torch.log2(prob_chanP)
                -prob_chanM * torch.log2(prob_chanM)
                -prob_unchan * torch.log2(prob_unchan) ),
                dim=(1,2,3)
                )

            payld_gen = torch.sum((cap_entropy), dim=0) / (img_size * img_size * batch_size)
            cap = img_size * img_size * self.payld
            loss_entropy = torch.mean(torch.pow(cap_entropy - cap, 2), dim=0)

            loss_G = D_LAMBDA * (-loss_D) + Payld_LAMBDA * loss_entropy
            loss_G.backward()
            self.optimizer_G.step()

        return loss_D.data[0], loss_G.data[0]

    def train(self, train_dataloader, epochs):
    	data_iter = iter(train_dataloader)
        sample_batch = data_iter.next()
        data_fixed = sample_batch['img'][0:]
        data_fixed = Variable(data_fixed.cuda())
        noise_fixed = np.random.rand(data_fixed.shape[0], data_fixed.shape[1], data_fixed.shape[2], data_fixed.shape[3])
        noise_fixed = torch.from_numpy(noise_fixed).float().to(self.device)
        noise_fixed = Variable(noise_fixed.cuda())

        for epoch in range(1, epochs+1):
            loss_D_sum = 0
            loss_G_sum = 0
            
            for i, data in enumerate(train_dataloader, start=0):
                images = data['img']
                images = images.to(self.device)
                
                loss_D_batch, loss_G_batch = self.train_batch(images)
                loss_D_sum += loss_D_batch
                loss_G_sum += loss_G_batch
            # print statistics
            num_batch = len(train_dataloader)
            print("epoch %d:\nloss_D: %.6f, loss_G: %.6f" %
                  (epoch, loss_D_sum/num_batch, loss_G_sum/num_batch))
            # save generator
            if epoch%1==0:
                with torch.no_grad():
                    modi_map_fixed = 0.5*(torch.tanh((self.netG(data_fixed)+2.*noise_fixed-2)*60) - torch.tanh((self.netG(data_fixed)-2.*noise_fixed)*60))
                    stego_fixed = (data_fixed*255 + modi_map_fixed)/255.
                show_result(epoch, self.netG(data_fixed), save=True, path=img_prob_train_path+str(epoch)+'prob.png')
                show_result(epoch, stego_fixed, save=True, path=img_prob_train_path+str(epoch)+'steg.png')
                show_result(epoch, modi_map_fixed, save=True, path=img_prob_train_path+str(epoch)+'modi.png')
                show_result(epoch, data_fixed, save=True, path=img_prob_train_path+str(epoch)+'cover.png')
            if epoch%100==0:
                netG_file_name = models_path + 'netG_epoch_' + str(epoch) + '.pth'
                torch.save(self.netG.state_dict(), netG_file_name)
