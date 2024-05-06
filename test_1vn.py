import os
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader

from utils.logger import Progbar
from utils.common import read_annotations, load_config, make_identification_set
from utils.evaluation import cosin_metric, evaluate_multiclass
from models.denoiser import get_denoiser
from models.classifier import UnetResNet50
from data.dataset import ImageDataset
from sklearn.metrics import confusion_matrix

class Tester(): 
    def __init__(self, model, denoise_func, config, args):
        self.model = model
        self.denoise_func = denoise_func
        
        self.device = args.device
        self.config = config
        self.window_slide = args.window_slide

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


    def calculate_result(self, test_probs, test_labels):
        pred_labels = np.argmax(test_probs, axis=1)
        results = evaluate_multiclass(test_labels, pred_labels)
        print(confusion_matrix(test_labels, pred_labels))

        return round(results['acc'], 4)*100, round(results['f1'], 4)*100

def parse_args():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--pretrain_model_path', type=str, default='', required=True)
    parser.add_argument('--test_data_paths', nargs='+', type=str)
    parser.add_argument('--window_slide', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda:0', help='device', required=True)
    parser.add_argument('--nshot', default=10, help='use how many images to construct the fingerprint in the gallery', type=int)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    config = load_config('configs.{}'.format('test'))
    print(args.pretrain_model_path)
    pre_model = torch.load(args.pretrain_model_path, map_location='cpu')
    config.class_num = int(pre_model['state_dict']['model.classification_head.3.weight'].shape[0])
    model = UnetResNet50(class_num=config.class_num)
    model.load_state_dict(pre_model['state_dict'])
    model = model.to(args.device)
    denoise_func = get_denoiser(1, args.device)

    tester = Tester(model, denoise_func, config, args)
    test_names = [os.path.basename(test_data_path)[:-4] for test_data_path in args.test_data_paths]

    for test_idx, test_data_path in enumerate(args.test_data_paths):
        support_data_path = test_data_path.replace('.txt', '_support.txt')
        test_sample_data_path = test_data_path.replace('.txt', '_test.txt')
        if not os.path.exists(support_data_path) or not os.path.exists(test_sample_data_path):
            print(support_data_path)
            print(test_sample_data_path)
            make_identification_set(test_data_path, support_data_path, test_sample_data_path, args.nshot)

        support_set = ImageDataset(read_annotations(support_data_path), config, window_slide=args.window_slide)
        support_loader = DataLoader(
            dataset=support_set,
            num_workers=config.num_workers,
            batch_size=config.batch_size,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )
    
        test_set = ImageDataset(read_annotations(test_sample_data_path), config, window_slide=args.window_slide)
        test_loader = DataLoader(
            dataset=test_set,
            num_workers=config.num_workers,
            batch_size=config.batch_size,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )
        
        support_features, support_labels  = tester.get_feature(support_loader, run_type='support')
        galley_features = [np.mean(support_features[support_labels==i],0) for i in range(len(set(list(support_labels))))]
        
        test_features, test_labels = tester.get_feature(test_loader, run_type='test')
         
        test_probs = []
        for fe_1 in test_features:
            sims = [cosin_metric(fe_1, fe_2) for fe_2 in galley_features]
            prob = np.array(sims)
            test_probs.append(prob)
        test_probs = np.array(test_probs)

        acc, f1 = tester.calculate_result(test_probs, test_labels)
        print('%s \t \t test acc: %.2f, test f1: %.2f' %
                (test_names[test_idx], acc, f1))

if __name__=='__main__':
    main()

