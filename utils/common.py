import os
import random
import importlib
import numpy as np

import torch
import matplotlib.pyplot as plt
from MulticoreTSNE import MulticoreTSNE as TSNE

import torch.nn.functional as F

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def load_config(config_path):
    module = importlib.import_module(config_path)
    return module.Config()

def read_annotations(data_path, debug=False, shuffle=True):
    lines = map(str.strip, open(data_path).readlines())
    data = []
    for line in lines:
        if len(line.split('\t'))==1:
            sample_path = line.split('\t')[0]
            label = 0
        else:
            sample_path, label = line.split('\t')
        label = int(label)
        data.append((sample_path, label))
    if shuffle:
        random.shuffle(data)
    if debug:
        data=data[:1000]

    return data

def plot_confusion_matrix(confusion, labels_name, save_path):
    plt.figure(figsize=(12, 12))
    plt.imshow(confusion, cmap=plt.cm.Blues)  # 在特定的窗口上显示图像
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)  # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)  # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    for first_index in range(len(confusion)):
        for second_index in range(len(confusion[first_index])):
            plt.text(first_index - 0.3, second_index, confusion[first_index][second_index])
    plt.tight_layout()
    plt.savefig(save_path, format='png')
    plt.close()

def plot_ROC_curve(results, save_path):
    plt.figure(figsize=(8,8))
    lw = 2
    plt.plot(results['fpr'], results['tpr'], color='darkorange',
            lw=lw, label='ROC curve (area = %0.3f)' % results['AUROC']) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(save_path)

def tsne_analyze(features, labels, classes, save_path, feature_num=None ,do_fit = True):
    print(save_path)
    save_dir = os.path.split(save_path)[0]
    save_name = os.path.basename(save_path)[:-4]
    if do_fit:
        print(f">>> t-SNE fitting")
        embeddings = TSNE(n_jobs=4).fit_transform(features)
        print(f"<<< fitting over")
        np.save(os.path.join(save_dir,f'embeddings_{save_name}.npy'), embeddings)
        np.save(os.path.join(save_dir,f'labels_{save_name}.npy'), labels)
    else:
        embeddings=np.load(os.path.join(save_dir,f'embeddings_{save_name}.npy'))
        labels=np.load(os.path.join(save_dir,f'labels_{save_name}.npy'))
    index = [i for i in range(len(embeddings))]
    random.shuffle(index)
    embeddings = np.array([embeddings[index[i]] for i in range(len(index))])
    labels = [labels[index[i]] for i in range(len(index))]
    if feature_num is not None:
        embeddings = embeddings[:feature_num]
        labels = labels[:feature_num]

    vis_x = embeddings[:, 0]
    vis_y = embeddings[:, 1]
    plt.figure(figsize=(5,5))
    plt.gca().xaxis.set_major_locator(plt.NullLocator()) 
    plt.gca().yaxis.set_major_locator(plt.NullLocator()) 
     
    num_classes =  len(set(labels))
    print('num_classes', num_classes)
    for i, lab in enumerate(list(range(num_classes))):
        if i < 20:
            color = plt.cm.tab20(i)
        elif i<40:
            color = plt.cm.tab20b(i-20)
        else:
            color = plt.cm.tab20c(i-40)
        class_index = [j for j,v in enumerate(labels) if v == lab]
        plt.scatter(vis_x[class_index], vis_y[class_index], color = color, alpha=1, marker='*')

    plt.xticks([])
    plt.yticks([])
    plt.legend(classes, loc='upper right')
    plt.savefig(save_path, bbox_inches='tight')
    plt.savefig(save_path.replace('pdf','png'), bbox_inches='tight')
    plt.close()


def plot_hist(sims, labels, save_path):

    plt.figure(figsize=(8,8))
    plt.hist([sims[labels==0], sims[labels==1]], rwidth=0.8, bins=50, alpha=0.7, label=['same','diff'])
    plt.legend()
    plt.savefig(save_path)


def get_input_data(input_img, denoise_func):
    
    input_img = (input_img+1)/2
    with torch.no_grad():
        noise_img = denoise_func.network(input_img)
    
    dft_img = torch.abs(torch.fft.fft2(noise_img.to(torch.float32)))
    dft_img = F.interpolate(dft_img, size=[input_img.shape[2], input_img.shape[3]], mode='nearest')

    return dft_img


def make_identification_set(test_data_path, support_data_path, test_sample_data_path, nshot):

    annotations, labels = [], []
    with open(test_data_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            img_path, label = line.split('\t')[0], int(line.split('\t')[1])
            annotations.append([img_path, label])
            labels.append(label)
    
    label_num = len(set(labels))
    support_data = [[element for element in annotations if element[1]==i][:nshot] for i in range(label_num)]
    test_sample_data = [[element for element in annotations if element[1]==i][nshot:] for i in range(label_num)]
    
    with open(support_data_path, 'w') as f:
        for single_class_data in support_data:
            for element in single_class_data:
                f.write(element[0]+'\t'+str(element[1])+'\n')
    
    with open(test_sample_data_path, 'w') as f:
        for single_class_data in test_sample_data:
            for element in single_class_data:
                f.write(element[0]+'\t'+str(element[1])+'\n')
    

def make_verify_file(test_data_path):
    lines = map(str.strip, open(test_data_path).readlines())
    test_X, test_y = [], []
    for line in lines:
        sample_path, label = line.split('\t')
        label = int(label)
        test_X.append(sample_path)
        test_y.append(label)

    pair_num = 10000
    total_img_num = len(test_X)
    indexs = [i for i in range(total_img_num)]
    positive_pairs,negative_pairs =[],[]
    while len(positive_pairs)<pair_num//2 and len(positive_pairs)<pair_num//2:
        sampled_pair_index = random.sample(indexs, 2)
        pair=[test_X[sampled_pair_index[0]], test_X[sampled_pair_index[1]]]
        if test_y[sampled_pair_index[0]] == test_y[sampled_pair_index[1]]:
            if pair not in positive_pairs and len(positive_pairs)<pair_num//2:
                positive_pairs.append(pair+[1])
        else:
            if pair not in negative_pairs and len(negative_pairs)<pair_num//2:
                negative_pairs.append(pair+[0])
    
    pairs = positive_pairs + negative_pairs
    random.shuffle(pairs)
    verify_save_path = test_data_path.replace('.txt','_verify.txt')
    with open(verify_save_path, "w") as f:
        for i in range(len(pairs)):
            f.write(str(pairs[i][0]) + "\t" + str(pairs[i][1]) + "\t"+ str(pairs[i][2])+"\n")
        