import os
import argparse
import numpy as np

import torch 
from torch.utils.data import DataLoader

from utils.logger import Progbar
from utils.common import load_config, make_verify_file, read_annotations, plot_hist, tsne_analyze
from utils.evaluation import get_feature_dict, test_performance
from models.denoiser import get_denoiser
from models.classifier import UnetResNet50
from data.dataset import ImageDataset


class Tester(): 
    def __init__(self, model, denoise_func, save_dir, config, args):
        self.model = model
        self.denoise_func = denoise_func
        
        self.device = args.device
        self.window_slide = args.window_slide
        self.config = config
        self.save_dir = save_dir


    def get_feature(self, dataloader, run_type='test'): 
        progbar = Progbar(len(dataloader), stateful_metrics=['run-type'])
        self.model.eval()
        with torch.no_grad():
            features = None
            for batch_idx, batch in enumerate(dataloader):
                input_img_batch, label_batch, _ = batch 
                label = label_batch.reshape((-1)).to(self.device)

                if len(input_img_batch.shape) == 4:
                    input_img = input_img_batch.to(self.device)
                    _, feature = self.model(input_img, denoise_func = self.denoise_func)
                else:
                    input_img = input_img_batch.reshape(-1, 3, input_img_batch.shape[-2], input_img_batch.shape[-1]).to(self.device)
                    _, feature_crops = self.model(input_img, denoise_func = self.denoise_func)
                    feature_crops = feature_crops.reshape(input_img_batch.shape[0], input_img_batch.shape[1], -1)
                    feature = torch.mean(feature_crops, 1)

                if batch_idx == 0:
                    gt_labels = label
                    features = feature
                else:
                    gt_labels = torch.cat([gt_labels, label])
                    features = torch.cat(([features, feature]))
                progbar.add(1, values=[('run-type', run_type)])

        features = features.cpu().numpy()
        labels = gt_labels.cpu().numpy()

        return features, labels

    def tsne_analysis(self, test_data_path, run_type):

        test_set = ImageDataset(read_annotations(test_data_path), self.config, window_slide=self.window_slide)
        test_loader = DataLoader(
            dataset=test_set,
            num_workers=self.config.num_workers,
            batch_size=self.config.batch_size,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )
        features, labels = self.get_feature(test_loader, run_type='tsne')

        os.makedirs(os.path.join(self.save_dir,'tsne'), exist_ok=True)
        tsne_analyze(features, labels, classes=[str(i) for i in range(len(set(list(labels))))], feature_num=1000, \
            save_path = os.path.join(self.save_dir,'tsne','{}.png'.format(run_type)), do_fit = True)


    def test_verification(self, test_data_path, run_type):

        verify_path = test_data_path.replace('.txt','_verify.txt')
        if not os.path.isfile(verify_path):
            print('make_verify_file', verify_path)
            make_verify_file(test_data_path)

        with open(verify_path, 'r') as fd:
            pairs = fd.readlines()
        data_list = []
        for pair in pairs:
            splits = pair.split('\t')
            if splits[0] not in data_list:
                data_list.append(splits[0])
            if splits[1] not in data_list:
                data_list.append(splits[1])

        verify_set = ImageDataset([[d,0] for d in data_list], self.config, window_slide=self.window_slide)
        verify_loader = DataLoader(
            dataset=verify_set,
            num_workers=self.config.num_workers,
            batch_size=self.config.batch_size,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )

        verify_features, _ = self.get_feature(verify_loader, run_type=run_type)
        print('verify_features', verify_features.shape)

        fe_dict = get_feature_dict(data_list, verify_features)
        auc, acc, th, sims, labels = test_performance(fe_dict, verify_path, return_detail=True)
        
        os.makedirs(os.path.join(self.save_dir,'hist'),exist_ok=True)
        plot_hist(sims, labels, os.path.join(self.save_dir,'hist','{}.png'.format(run_type)))

        auc = round(auc*100,2)
        acc = round(acc*100,2)

        return auc, acc, th


def parse_args():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--pretrain_model_path', type=str, default='')
    parser.add_argument('--device', default='cuda:0', type=str, help='cuda:n or cpu')
    parser.add_argument('--test_data_paths', nargs='+', type=str)
    parser.add_argument('--window_slide', action='store_true', default=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # load configs
    args = parse_args()
    config = load_config('configs.{}'.format('test'))

    pre_model = torch.load(args.pretrain_model_path, map_location='cpu')
    config.class_num = int(pre_model['state_dict']['model.classification_head.3.weight'].shape[0])
    model = UnetResNet50(class_num=config.class_num)
    model.load_state_dict(pre_model['state_dict'])
    model = model.to(args.device)
    denoise_func = get_denoiser(1, args.device)

    save_dir = os.path.join('results', 'pred')
    tester = Tester(model, denoise_func, save_dir, config, args)
    test_names = [os.path.basename(test_data_path)[:-4] for test_data_path in args.test_data_paths]

    for test_idx, test_data_path in enumerate(args.test_data_paths):
        tester.tsne_analysis(test_data_path, run_type=test_names[test_idx])
        test_auc, test_acc, best_th = tester.test_verification(test_data_path, run_type=test_names[test_idx])
        print('%s \t \t test auc: %.2f, test acc: %.2f, best th: %.2f' %
                (test_names[test_idx], test_auc, test_acc, best_th))
