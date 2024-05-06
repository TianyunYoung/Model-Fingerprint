import os
import time
import random
import numpy as np
from itertools import cycle

import torch

from utils.loss import TripletLoss
from utils.logger import Progbar, AverageMeter
from models.denoiser import get_denoiser
from models.classifier import UnetResNet50


class Trainer(): 
    def __init__(self, train_loaders, syn_models, device, config, writer, logger, model_dir):
        
        # image dataset, models
        self.train_loaders = train_loaders
        self.syn_models = syn_models
        self.model = UnetResNet50(class_num=len(self.syn_models))
        self.model = self.model.to(device)
        self.denoise_func = get_denoiser(1, device)

        # optimization
        self.optimizer = torch.optim.AdamW(self.model.parameters(), \
                                           lr=config.init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, \
                                                                              T_0=config.T_0, T_mult=config.T_mult, eta_min=config.eta_min)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.criterionMetric = TripletLoss(margin=config.margin)

        # others
        self.config = config
        self.device = device
        self.logger = logger
        self.writer = writer
        self.model_dir = model_dir
        self.board_num = 0


    def train_epoch(self, epoch):
        progbar = Progbar(len(self.train_loaders[0]), stateful_metrics=['epoch'])
        batch_time = AverageMeter()
        end = time.time()

        if len(self.train_loaders) == 1:
            self.data_zip = zip(self.train_loaders[0])
        elif len(self.train_loaders) == 2:
            self.data_zip = zip(self.train_loaders[0], cycle(self.train_loaders[1]))

        self.model.train()
        for _, batches in enumerate(self.data_zip):
            # sample real images
            real_img_batch = random.choice(batches)[0]
            real_imgs = real_img_batch.reshape((-1, 3, real_img_batch.size(-2), real_img_batch.size(-1))).to(self.device)

            # sample synthetic models 
            selection_num = range(len(real_imgs))
            sample_list = [i for i in range(len(self.syn_models))]
            selected_indexs = [random.choice(sample_list) for _ in selection_num] 
            selected_syn_models = [self.syn_models[idx] for idx in selected_indexs]

            # generate fingerprinted images
            fingerprinted_imgs, labels = [], []
            for enu_idx, (model_idx, syn_model) in enumerate(zip(selected_indexs, selected_syn_models)):
                real_img = real_imgs[enu_idx].unsqueeze(0)
                fingerprinted_img = syn_model(real_img)
                label = torch.ones(real_img.size(0), dtype=torch.long) * model_idx
                
                fingerprinted_imgs.append(fingerprinted_img)
                labels.append(label)

            fingerprinted_imgs, labels = torch.cat(fingerprinted_imgs), torch.cat(labels).to(self.device)

            # train fingerprint extractor  
            probs, features = self.model(fingerprinted_imgs, denoise_func=self.denoise_func)

            loss_cls = self.criterion(probs, labels)
            loss_metric_cls = self.criterionMetric(features, labels)                      
            loss = self.config.w_cls * loss_cls + self.config.w_metric * loss_metric_cls
            losses = {'loss_cls':loss_cls.item(), 'loss_metric_cls':loss_metric_cls.item()}
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            if torch.isinf(loss) or torch.isnan(loss):
                print("Skipping batch due to infinite loss")
                continue  # Skip the rest of this iteration and move to the next batch

            loss.backward()
            self.optimizer.step()
            
            # logs
            progbar.add(1, values=[('epoch', epoch)]+[(loss_key,losses[loss_key]) for loss_key in losses.keys()]+[('lr', self.scheduler.get_lr()[0])])
            for loss_key in losses.keys():
                self.writer.add_scalars(loss_key, {'loss_key': losses[loss_key]}, self.board_num)
            
            batch_time.update(time.time() - end)
            end = time.time()
    
        self.scheduler.step()        


    def predict_on_real(self, dataloader, run_type='test', vis=False): 
        self.model.eval()
        progbar = Progbar(len(dataloader), stateful_metrics=['run-type'])
        with torch.no_grad():
            features = None
            for batch_idx, batch in enumerate(dataloader):
                input_img_batch, label_batch, _ = batch 
                input_imgs = input_img_batch.reshape((-1, 3, input_img_batch.size(-2), input_img_batch.size(-1))).to(self.device)
                labels = label_batch.reshape((-1)).to(self.device)

                _, features, _ = self.model(input_imgs, denoise_func = self.denoise_func)
                progbar.add(1, values=[('run-type', run_type)])

                if batch_idx == 0:
                    all_gt_labels = labels
                    all_features = features.cpu().numpy()
                else:
                    all_gt_labels = torch.cat([all_gt_labels, labels])
                    all_features=np.vstack((all_features, features.cpu().numpy()))

        all_gt_labels = all_gt_labels.cpu().numpy()

        return all_features, all_gt_labels


    def save_model(self, epoch, save_suffix='model.pth'):
            torch.save({
            'epoch': epoch,
            'model': self.model.__class__.__name__,
            'state_dict': self.model.state_dict()
        }, os.path.join(self.model_dir, save_suffix))
    
