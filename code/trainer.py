from __future__ import print_function
from six.moves import range
import sys
import numpy as np
import os
import random
import time
from PIL import Image
from copy import deepcopy

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision.utils as vutils
from torch.nn.functional import softmax, log_softmax
from torch.nn.functional import cosine_similarity
from tensorboardX import summary
from tensorboardX import FileWriter

from miscc.config import cfg
from miscc.utils import mkdir_p

from model import G_NET, D_NET


# ################## Shared functions ###################

def child_to_parent(child_c_code, classes_child, classes_parent):

    ratio = classes_child / classes_parent
    arg_parent = torch.argmax(child_c_code,  dim = 1) / ratio
    parent_c_code = torch.zeros([child_c_code.size(0), classes_parent]).cuda()
    for i in range(child_c_code.size(0)):
        parent_c_code[i][arg_parent[i]] = 1
    return parent_c_code


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten

def load_network(gpus):
    netG = G_NET()
    netG.apply(weights_init)
    netG = torch.nn.DataParallel(netG, device_ids=gpus)
    print(netG)

    netsD = []
    for i in range(3): # 3 discriminators for background, parent and child stage
        netsD.append(D_NET(i))

    for i in range(len(netsD)):
        netsD[i].apply(weights_init)
        netsD[i] = torch.nn.DataParallel(netsD[i], device_ids=gpus)

    count = 0

    if cfg.TRAIN.NET_G != '':
        state_dict = torch.load(cfg.TRAIN.NET_G)
        netG.load_state_dict(state_dict)
        print('Load ', cfg.TRAIN.NET_G)

        istart = cfg.TRAIN.NET_G.rfind('_') + 1
        iend = cfg.TRAIN.NET_G.rfind('.')
        count = cfg.TRAIN.NET_G[istart:iend]
        count = int(count) + 1

    if cfg.TRAIN.NET_D != '':
        for i in range(len(netsD)):
            print('Load %s_%d.pth' % (cfg.TRAIN.NET_D, i))
            state_dict = torch.load('%s_%d.pth' % (cfg.TRAIN.NET_D, i))
            netsD[i].load_state_dict(state_dict)

    if cfg.CUDA:
        netG.cuda()
        for i in range(len(netsD)):
            netsD[i].cuda()

    return netG, netsD, len(netsD), count


def define_optimizers(netG, netsD):
    optimizersD = []
    num_Ds = len(netsD)
    for i in range(num_Ds):      
        opt = optim.Adam(netsD[i].parameters(),
                         lr=cfg.TRAIN.DISCRIMINATOR_LR,
                         betas=(0.5, 0.999))
        optimizersD.append(opt)

    optimizerG = []
    optimizerG.append(optim.Adam(netG.parameters(),
                            lr=cfg.TRAIN.GENERATOR_LR,
                            betas=(0.5, 0.999)))

    for i in range(num_Ds):
        if i==1:   
                opt = optim.Adam(netsD[i].parameters(),
                lr=cfg.TRAIN.GENERATOR_LR,
                betas=(0.5, 0.999))
                optimizerG.append(opt)
        elif i==2:
                opt = optim.Adam([{'params':netsD[i].module.jointConv.parameters()},{'params':netsD[i].module.logits.parameters()}],
                lr=cfg.TRAIN.GENERATOR_LR,
                betas=(0.5, 0.999))
                optimizerG.append(opt)

    return optimizerG, optimizersD


def save_model(netG, avg_param_G, netsD, epoch, model_dir):
    load_params(netG, avg_param_G)
    torch.save(
        netG.state_dict(),
        '%s/netG_%d.pth' % (model_dir, epoch))
    for i in range(len(netsD)):
        netD = netsD[i]
        torch.save(
            netD.state_dict(),
            '%s/netD%d.pth' % (model_dir, i))
    print('Save G/Ds models.')


def save_img_results(imgs_tcpu, fake_imgs, num_imgs,
                     count, image_dir, summary_writer):
    num = cfg.TRAIN.VIS_COUNT

    real_img = imgs_tcpu[-1][0:num]
    vutils.save_image(
        real_img, '%s/real_samples%09d.png' % (image_dir,count),
        normalize=True)
    real_img_set = vutils.make_grid(real_img).numpy()
    real_img_set = np.transpose(real_img_set, (1, 2, 0))
    real_img_set = real_img_set * 255
    real_img_set = real_img_set.astype(np.uint8)

    for i in range(len(fake_imgs)):
        fake_img = fake_imgs[i][0:num]

        vutils.save_image(
            fake_img.data, '%s/count_%09d_fake_samples%d.png' %
            (image_dir, count, i), normalize=True)

        fake_img_set = vutils.make_grid(fake_img.data).cpu().numpy()

        fake_img_set = np.transpose(fake_img_set, (1, 2, 0))
        fake_img_set = (fake_img_set + 1) * 255 / 2
        fake_img_set = fake_img_set.astype(np.uint8)
        summary_writer.flush()



class FineGAN_trainer(object):
    def __init__(self, output_dir, data_loader, imsize):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            self.log_dir = os.path.join(output_dir, 'Log')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
            mkdir_p(self.log_dir)
            self.summary_writer = FileWriter(self.log_dir)

        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True

        self.batch_size = cfg.TRAIN.BATCH_SIZE * self.num_gpus
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)



    def prepare_data(self, data):
        fimgs, cimgs, c_code, _, warped_bbox = data

        real_vfimgs, real_vcimgs = [], []
        if cfg.CUDA:
            vc_code = Variable(c_code).cuda()
	    for i in range(len(warped_bbox)):
		warped_bbox[i] = Variable(warped_bbox[i]).float().cuda()

        else:
            vc_code = Variable(c_code)
	    for i in range(len(warped_bbox)):
		warped_bbox[i] = Variable(warped_bbox[i])

        if cfg.CUDA:
            real_vfimgs.append(Variable(fimgs[0]).cuda())
            real_vcimgs.append(Variable(cimgs[0]).cuda())
        else:
            real_vfimgs.append(Variable(fimgs[0]))
            real_vcimgs.append(Variable(cimgs[0]))

        return fimgs, real_vfimgs, real_vcimgs, vc_code, warped_bbox

    def train_Dnet(self, idx, count):
      if idx == 0 or idx == 2: # Discriminator is only trained in background and child stage. (NOT in parent stage)
        flag = count % 100
        batch_size = self.real_fimgs[0].size(0)
        criterion, criterion_one = self.criterion, self.criterion_one

        netD, optD = self.netsD[idx], self.optimizersD[idx] 
	if idx == 0:
        	real_imgs = self.real_fimgs[0]
	
	elif idx == 2:
		real_imgs = self.real_cimgs[0]
       
        fake_imgs = self.fake_imgs[idx]
        netD.zero_grad()        
        real_logits = netD(real_imgs)

	if idx == 2:
		fake_labels = torch.zeros_like(real_logits[1])
		real_labels = torch.ones_like(real_logits[1])
	elif idx == 0:

		fake_labels = torch.zeros_like(real_logits[1])
		ext, output = real_logits
		weights_real = torch.ones_like(output)
		real_labels = torch.ones_like(output)
            
                for i in range(batch_size):
                        x1 =  self.warped_bbox[0][i]
                        x2 =  self.warped_bbox[2][i]
                        y1 =  self.warped_bbox[1][i]
                        y2 =  self.warped_bbox[3][i]

                        a1 = max(torch.tensor(0).float().cuda(), torch.ceil((x1 - self.recp_field)/self.patch_stride))
                        a2 = min(torch.tensor(self.n_out - 1).float().cuda(), torch.floor((self.n_out - 1) - ((126 - self.recp_field) - x2)/self.patch_stride)) + 1
                        b1 = max(torch.tensor(0).float().cuda(), torch.ceil((y1 - self.recp_field)/self.patch_stride))
                        b2 = min(torch.tensor(self.n_out - 1).float().cuda(), torch.floor((self.n_out - 1) - ((126 - self.recp_field) - y2)/self.patch_stride)) + 1

			if (x1 != x2 and y1 != y2):
                        	weights_real[i, :, a1.type(torch.int) : a2.type(torch.int) , b1.type(torch.int) : b2.type(torch.int)] = 0.0

                norm_fact_real = weights_real.sum()
                norm_fact_fake = weights_real.shape[0]*weights_real.shape[1]*weights_real.shape[2]*weights_real.shape[3]
		real_logits = ext, output

        fake_logits = netD(fake_imgs.detach())


	  
	if idx == 0: # Background stage

            errD_real_uncond = criterion(real_logits[1], real_labels)  # Real/Fake loss for 'real background' (on patch level)
	    errD_real_uncond = torch.mul(errD_real_uncond, weights_real)  # Masking output units which correspond to receptive fields which lie within the boundin box
	    errD_real_uncond = errD_real_uncond.mean()

            errD_real_uncond_classi = criterion(real_logits[0], weights_real)  # Background/foreground classification loss
	    errD_real_uncond_classi = errD_real_uncond_classi.mean()
	   
            errD_fake_uncond = criterion(fake_logits[1], fake_labels)  # Real/Fake loss for 'fake background' (on patch level)
	    errD_fake_uncond = errD_fake_uncond.mean()

            if (norm_fact_real > 0):    # Normalizing the real/fake loss for background after accounting the number of masked members in the output.
            	errD_real = errD_real_uncond * ((norm_fact_fake * 1.0) /(norm_fact_real * 1.0))
	    else:
		errD_real = errD_real_uncond

            errD_fake = errD_fake_uncond
	    errD = ((errD_real + errD_fake) * cfg.TRAIN.BG_LOSS_WT) + errD_real_uncond_classi

        if idx == 2:
	
            errD_real = criterion_one(real_logits[1], real_labels) # Real/Fake loss for the real image
            errD_fake = criterion_one(fake_logits[1], fake_labels) # Real/Fake loss for the fake image   
            errD = errD_real + errD_fake

        if (idx == 0 or idx == 2):
              errD.backward()
              optD.step()

        if (flag == 0):
            summary_D = summary.scalar('D_loss%d' % idx, errD.data[0])
            self.summary_writer.add_summary(summary_D, count)
            summary_D_real = summary.scalar('D_loss_real_%d' % idx, errD_real.data[0])
            self.summary_writer.add_summary(summary_D_real, count)
            summary_D_fake = summary.scalar('D_loss_fake_%d' % idx, errD_fake.data[0])
            self.summary_writer.add_summary(summary_D_fake, count)

        return errD

    def train_Gnet(self, count):
        self.netG.zero_grad()
        for myit in range(len(self.netsD)): 
             self.netsD[myit].zero_grad()

        errG_total = 0
        flag = count % 100
        batch_size = self.real_fimgs[0].size(0)
        criterion_one, criterion_class, c_code, p_code = self.criterion_one, self.criterion_class, self.c_code, self.p_code

        for i in range(self.num_Ds):
	  
	    outputs = self.netsD[i](self.fake_imgs[i]) 	

            if i == 0 or i == 2:  # real/fake loss for background (0) and child (2) stage
		real_labels = torch.ones_like(outputs[1])
	    	errG = criterion_one(outputs[1], real_labels) 
		if i==0:
			errG = errG * cfg.TRAIN.BG_LOSS_WT
			errG_classi = criterion_one(outputs[0], real_labels) # Background/Foreground classification loss for the fake background image (on patch level)
			errG = errG + errG_classi
	    	errG_total = errG_total + errG	

            if i == 1: # Mutual information loss for the parent stage (1)
                    pred_p = self.netsD[i](self.fg_mk[i-1])
                    errG_info = criterion_class(pred_p[0], torch.nonzero(p_code.long())[:,1])
            elif i == 2: # Mutual information loss for the child stage (2)
                    pred_c = self.netsD[i](self.fg_mk[i-1])
                    errG_info = criterion_class(pred_c[0], torch.nonzero(c_code.long())[:,1])

            if(i>0):
                errG_total = errG_total + errG_info

            if flag == 0:
		if i>0:
                  summary_D_class = summary.scalar('Information_loss_%d' % i, errG_info.data[0])
                  self.summary_writer.add_summary(summary_D_class, count)

		if i == 0 or i == 2:
                  summary_D = summary.scalar('G_loss%d' % i, errG.data[0])
                  self.summary_writer.add_summary(summary_D, count) 

        errG_total.backward()
        for myit in range(len(self.netsD)): 
        	self.optimizerG[myit].step()
        return errG_total

    def train(self):
        self.netG, self.netsD, self.num_Ds, start_count = load_network(self.gpus)
        avg_param_G = copy_G_params(self.netG)

        self.optimizerG, self.optimizersD = \
            define_optimizers(self.netG, self.netsD)

        self.criterion = nn.BCELoss(reduce=False)
	self.criterion_one = nn.BCELoss()
        self.criterion_class = nn.CrossEntropyLoss()

        self.real_labels = \
            Variable(torch.FloatTensor(self.batch_size).fill_(1))
        self.fake_labels = \
            Variable(torch.FloatTensor(self.batch_size).fill_(0))
	
        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(self.batch_size, nz))
        fixed_noise = \
            Variable(torch.FloatTensor(self.batch_size, nz).normal_(0, 1))
        hard_noise = \
            Variable(torch.FloatTensor(self.batch_size, nz).normal_(0, 1)).cuda()
	
	self.patch_stride = float(4)    # Receptive field stride given the current discriminator architecture for background stage 
	self.n_out = 24                 # Output size of the discriminator at the background stage; N X N where N = 24
	self.recp_field = 34            # Receptive field of each of the member of N X N


        if cfg.CUDA:
            self.criterion.cuda()
            self.criterion_one.cuda()
            self.criterion_class.cuda()
            self.real_labels = self.real_labels.cuda()
            self.fake_labels = self.fake_labels.cuda()
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
        
        print ("Starting normal FineGAN training..") 
        count = start_count
        start_epoch = start_count // (self.num_batches)

        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()
            
            for step, data in enumerate(self.data_loader, 0):
               
                self.imgs_tcpu, self.real_fimgs, self.real_cimgs, \
                    self.c_code, self.warped_bbox = self.prepare_data(data)

                # Feedforward through Generator. Obtain stagewise fake images
                noise.data.normal_(0, 1)
                self.fake_imgs, self.fg_imgs, self.mk_imgs, self.fg_mk = \
                    self.netG(noise, self.c_code)

                # Obtain the parent code given the child code
		self.p_code = child_to_parent(self.c_code, cfg.FINE_GRAINED_CATEGORIES, cfg.SUPER_CATEGORIES) 

                # Update Discriminator networks 
                errD_total = 0
                for i in range(self.num_Ds):
		  if i == 0 or i == 2: # only at parent and child stage
                    errD = self.train_Dnet(i, count)
                    errD_total += errD

                # Update the Generator networks
                errG_total = self.train_Gnet(count)
                for p, avg_p in zip(self.netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                count = count + 1

                if count % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:
                    backup_para = copy_G_params(self.netG)
                    save_model(self.netG, avg_param_G, self.netsD, count, self.model_dir)
                    # Save images
                    load_params(self.netG, avg_param_G)
                    
                    self.fake_imgs, self.fg_imgs, self.mk_imgs, self.fg_mk = \
                        self.netG(fixed_noise, self.c_code)
                    save_img_results(self.imgs_tcpu, (self.fake_imgs + self.fg_imgs + self.mk_imgs + self.fg_mk), self.num_Ds,
                                     count, self.image_dir, self.summary_writer)
                    #
                    load_params(self.netG, backup_para)

            end_t = time.time()
            print('''[%d/%d][%d]
                         Loss_D: %.2f Loss_G: %.2f Time: %.2fs
                      '''  
                  % (epoch, self.max_epoch, self.num_batches,
                     errD_total.data[0], errG_total.data[0],
                     end_t - start_t))

        save_model(self.netG, avg_param_G, self.netsD, count, self.model_dir)

	print ("Done with the normal training. Now performing hard negative training..")
        count = 0
        start_t = time.time()
        for step, data in enumerate(self.data_loader, 0):

            self.imgs_tcpu, self.real_fimgs, self.real_cimgs, \
                self.c_code, self.warped_bbox = self.prepare_data(data)

            if (count % 2) == 0: # Train on normal batch of images

                    # Feedforward through Generator. Obtain stagewise fake images
                    noise.data.normal_(0, 1)
                    self.fake_imgs, self.fg_imgs, self.mk_imgs, self.fg_mk = \
                        self.netG(noise, self.c_code)

                    self.p_code = child_to_parent(self.c_code, cfg.FINE_GRAINED_CATEGORIES, cfg.SUPER_CATEGORIES)

                    # Update discriminator networks
                    errD_total = 0
                    for i in range(self.num_Ds):
                            if i == 0 or i == 2:
                                    errD = self.train_Dnet(i, count)
                                    errD_total += errD


                    # Update the generator network
                    errG_total = self.train_Gnet(count)

            else: # Train on degenerate images
                    repeat_times=10
                    all_hard_z = Variable(torch.zeros(self.batch_size * repeat_times, nz)).cuda()
                    all_hard_class = Variable(torch.zeros(self.batch_size * repeat_times, cfg.FINE_GRAINED_CATEGORIES)).cuda()
                    all_logits = Variable(torch.zeros(self.batch_size * repeat_times,)).cuda()
                    
                    for hard_it in range(repeat_times):
                            hard_noise = hard_noise.data.normal_(0,1)
                            hard_class = Variable(torch.zeros([self.batch_size, cfg.FINE_GRAINED_CATEGORIES])).cuda()
                            my_rand_id=[]

                            for c_it in range(self.batch_size):
                                    rand_class = random.sample(range(cfg.FINE_GRAINED_CATEGORIES),1);
                                    hard_class[c_it][rand_class] = 1
                                    my_rand_id.append(rand_class)

                            all_hard_z[self.batch_size * hard_it : self.batch_size * (hard_it + 1)] = hard_noise.data
                            all_hard_class[self.batch_size * hard_it : self.batch_size * (hard_it + 1)] = hard_class.data
                            self.fake_imgs, self.fg_imgs, self.mk_imgs, self.fg_mk = self.netG(hard_noise.detach(), hard_class.detach())

                            fake_logits = self.netsD[2](self.fg_mk[1].detach())
                            smax_class = softmax(fake_logits[0], dim = 1)

                            for b_it in range(self.batch_size):	
                                    all_logits[(self.batch_size * hard_it) + b_it] = smax_class[b_it][my_rand_id[b_it]]
                            
                    sorted_val, indices_hard = torch.sort(all_logits)
                    noise = all_hard_z[indices_hard[0 : self.batch_size]]
                    self.c_code = all_hard_class[indices_hard[0 : self.batch_size]]

                    self.fake_imgs, self.fg_imgs, self.mk_imgs, self.fg_mk = \
                        self.netG(noise, self.c_code)

                    self.p_code = child_to_parent(self.c_code, cfg.FINE_GRAINED_CATEGORIES, cfg.SUPER_CATEGORIES)

                    # Update Discriminator networks
                    errD_total = 0
                    for i in range(self.num_Ds):
                            if i == 0 or i == 2:
                                    errD = self.train_Dnet(i, count)
                                    errD_total += errD

                    # Update generator network
                    errG_total = self.train_Gnet(count)

            for p, avg_p in zip(self.netG.parameters(), avg_param_G):
                        avg_p.mul_(0.999).add_(0.001, p.data)
            count = count + 1

            if count % cfg.TRAIN.SNAPSHOT_INTERVAL_HARDNEG == 0:
		backup_para = copy_G_params(self.netG)
                save_model(self.netG, avg_param_G, self.netsD, count+500000, self.model_dir)
                load_params(self.netG, avg_param_G)

                self.fake_imgs, self.fg_imgs, self.mk_imgs, self.fg_mk = \
                    self.netG(fixed_noise, self.c_code)
                save_img_results(self.imgs_tcpu, (self.fake_imgs + self.fg_imgs + self.mk_imgs + self.fg_mk), self.num_Ds,
                                 count, self.image_dir, self.summary_writer)
                #
                load_params(self.netG, backup_para)

            end_t = time.time()

            if (count % 100) == 0:
                print('''[%d/%d][%d]
                             Loss_D: %.2f Loss_G: %.2f Time: %.2fs
                          '''  
                      % (count, cfg.TRAIN.HARDNEG_MAX_ITER, self.num_batches,
                         errD_total.data[0], errG_total.data[0],
                         end_t - start_t))

            if (count == cfg.TRAIN.HARDNEG_MAX_ITER): # Hard negative training complete
                    break

        save_model(self.netG, avg_param_G, self.netsD, count, self.model_dir)
        self.summary_writer.close()



class FineGAN_evaluator(object):

    def __init__(self):

        self.save_dir = os.path.join(cfg.SAVE_DIR, 'images')
        mkdir_p(self.save_dir)
        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True
        self.batch_size = cfg.TRAIN.BATCH_SIZE * self.num_gpus


    def evaluate_finegan(self):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for model not found!')
        else:
            # Build and load the generator
            netG = G_NET()
            netG.apply(weights_init)
            netG = torch.nn.DataParallel(netG, device_ids=self.gpus)
            model_dict = netG.state_dict()

            state_dict = \
                torch.load(cfg.TRAIN.NET_G,
                           map_location=lambda storage, loc: storage)

            state_dict = {k: v for k, v in state_dict.items() if k in model_dict}

            model_dict.update(state_dict)
            netG.load_state_dict(model_dict)
            print('Load ', cfg.TRAIN.NET_G)

            # Uncomment this to print Generator layers
            # print(netG)
            
            nz = cfg.GAN.Z_DIM
            noise = torch.FloatTensor(self.batch_size, nz)
            noise.data.normal_(0, 1)

            if cfg.CUDA:
                netG.cuda()
                noise = noise.cuda()

            netG.eval()

            background_class = cfg.TEST_BACKGROUND_CLASS 
            parent_class = cfg.TEST_PARENT_CLASS 
            child_class = cfg.TEST_CHILD_CLASS
            bg_code = torch.zeros([self.batch_size, cfg.FINE_GRAINED_CATEGORIES])
            p_code = torch.zeros([self.batch_size, cfg.SUPER_CATEGORIES])
            c_code = torch.zeros([self.batch_size, cfg.FINE_GRAINED_CATEGORIES])

            for j in range(self.batch_size):
                bg_code[j][background_class] = 1
                p_code[j][parent_class] = 1
                c_code[j][child_class] = 1

            fake_imgs, fg_imgs, mk_imgs, fgmk_imgs = netG(noise, c_code, p_code, bg_code) # Forward pass through the generator

            self.save_image(fake_imgs[0][0], self.save_dir, 'background')
            self.save_image(fake_imgs[1][0], self.save_dir, 'parent_final')
            self.save_image(fake_imgs[2][0], self.save_dir, 'child_final')
            self.save_image(fg_imgs[0][0], self.save_dir, 'parent_foreground')
            self.save_image(fg_imgs[1][0], self.save_dir, 'child_foreground')
            self.save_image(mk_imgs[0][0], self.save_dir, 'parent_mask')
            self.save_image(mk_imgs[1][0], self.save_dir, 'child_mask')
            self.save_image(fgmk_imgs[0][0], self.save_dir, 'parent_foreground_masked')
            self.save_image(fgmk_imgs[1][0], self.save_dir, 'child_foreground_masked')


    def save_image(self, images, save_dir, iname):
        
        img_name = '%s.png' % (iname)
        full_path = os.path.join(save_dir, img_name)
        
        if (iname.find('mask') == -1) or (iname.find('foreground') != -1):
            img = images.add(1).div(2).mul(255).clamp(0, 255).byte()
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(full_path)

        else:
            img = images.mul(255).clamp(0, 255).byte()
            ndarr = img.data.cpu().numpy()
            ndarr = np.reshape(ndarr, (ndarr.shape[-1], ndarr.shape[-1], 1))
            ndarr = np.repeat(ndarr, 3, axis=2)
            im = Image.fromarray(ndarr)
            im.save(full_path)


