import os
import shutil
import sys
import json
import argparse
import time
import copy
from tensorboardX import SummaryWriter

import torch 
from torch.utils.data import DataLoader

from utils.common import load_config
from utils.logger import create_logger

from data.dataset_real import load_folder_dataset
from models.syn_model import SynModel
from train.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--config_name', type=str, help='model configuration file')
    parser.add_argument('--device', default='cuda:0', type=str, help='cuda:n or cpu')
    args = parser.parse_args()
    return args


def get_syn_models(syn_model_configs, save_dir, device, model_num=20):

    arch_seed_names = []
    syn_models = []
    for syn_model_config in syn_model_configs:
        syn_model_dir = os.path.join(save_dir, syn_model_config.name)
        os.makedirs(syn_model_dir, exist_ok=True)
        print(syn_model_dir)              
        syn_network = SynModel(syn_model_config)
        for s in range(model_num):
            single_syn_model = copy.deepcopy(syn_network)
            model_path = os.path.join(syn_model_dir, 'seed{}'.format(s), 'model.pth')
            syn_weight = torch.load(model_path, map_location='cpu')['state_dict']
            single_syn_model.load_state_dict(syn_weight)
            single_syn_model.eval()
            single_syn_model = single_syn_model.to(device)
            syn_models.append(single_syn_model)
            arch_seed_names.append(f'{syn_model_config.name}-{s}')


    return syn_models, arch_seed_names


class SynModelConfig:
    def __init__(self, arch_id='000000', kernel_size=3, in_channels=32):

        LAYER_NUM = {0: 1, 1: 2}
        CONV_TYPE = {0: "post", 1: "pre"}
        UP_TYPE = {0: "bilinear", 1: "nearest", 2: "deconv"}
        ACT_TYPE = {0: None, 1: "relu", 2: "sig", 3: "tanh"}
        NORM_TYPE = {0: None, 1: "bn", 2: "in"}
        BLOCK_NUM = {0: 1, 1: 2}
        
        block_id,layer_id,conv_type_id, up_id, act_id, norm_id = [int(i) for i in arch_id]
        self.layer_num=LAYER_NUM[layer_id]
        self.conv_type=CONV_TYPE[conv_type_id]
        self.up_type=UP_TYPE[up_id]
        self.act_type=ACT_TYPE[act_id]
        self.norm_type=NORM_TYPE[norm_id]
        self.block_num=BLOCK_NUM[block_id]

        self.out_channels=3
        self.in_channels=in_channels
        self.kernel_size=kernel_size

        self.name = 'SynModel_{}_k{}_c{}'.format(arch_id, kernel_size, in_channels)


if __name__ == '__main__':
    # load configs
    opt = parse_args()
    config = load_config('configs.{}'.format(opt.config_name))
    config_attr = dir(config)
    config_params = {config_attr[i]: getattr(config, config_attr[i]) for i in range(len(config_attr)) if config_attr[i][:2] != '__'}

    # setup gpu device
    torch.cuda.set_device(int(opt.device.split(':')[1]))
    device = torch.device(opt.device)

    # load synthetic models
    syn_model_configs = [SynModelConfig(arch_id, kernel_size=3) for arch_id in config.arch_id_list]
    syn_models, arch_seed_names = get_syn_models(syn_model_configs, config.syn_model_dir, device, model_num=config.model_num)

    # setup real datasets
    train_sets = []
    for source in config.img_sources:
        if source == 'lsun_objects':
            train_sets.append(load_folder_dataset('./dataset/real_images/lsun'))
        elif source == 'celeba':
            train_sets.append(load_folder_dataset('./dataset/real_images/img_align_celeba'))

    sources = '_'.join(config.img_sources)
    train_loaders = []
    for train_set in train_sets:
        print(len(train_set))
        train_loader = DataLoader(
            dataset=train_set,
            num_workers=config.num_workers,
            batch_size=config.batch_size,
            pin_memory=True,
            shuffle=True,
            drop_last=False,
        )
        train_loaders.append(train_loader)

    # setup model save dir
    time_str = time.strftime('%Y-%m-%d-%H-%M')    
    month_day = time_str.split('-')[1]+time_str.split('-')[2]
    run_id = f'{month_day}/{sources}_b{config.batch_size}'
    model_dir = os.path.join('results', run_id)
    os.makedirs(os.path.join(model_dir,'logs'),exist_ok=True)
    
    writer = SummaryWriter(logdir=os.path.join(model_dir,'logs'))
    logger = create_logger(os.path.join(model_dir,'logs'), log_name='train')    

    logger.info('model dir: %s' % model_dir)
    logger.info('syn model num: %d' % len(syn_model_configs))
    for syn_model_config in syn_model_configs:
        logger.info(syn_model_config.name)

    # save configs
    options_file = os.path.join(model_dir, 'options.json')
    with open(options_file, 'w') as fp:
        json.dump(vars(opt), fp, indent=4)
    shutil.copy('configs/{}.py'.format(opt.config_name),os.path.join(model_dir, 'configs.py'))
    logger.info('options: %s',opt)
    logger.info('config_params: %s',config_params)

    # setup trainer
    trainer = Trainer(train_loaders, syn_models, device, config, writer, logger, model_dir)   

    # begin to train
    logger.info("begin to train!")
    for epoch in range(config.max_epochs):
        trainer.train_epoch(epoch)
    
        if (epoch+1) % config.save_interval == 0: 
            trainer.save_model(epoch, 'model_epoch{}.pth'.format(epoch))
