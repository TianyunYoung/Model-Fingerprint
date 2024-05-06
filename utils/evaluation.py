import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def cal_accuracy(y_score, y_true):

    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return (best_acc, best_th)

def get_feature_dict(test_list, features):
    fe_dict = {}
    for i, each in enumerate(test_list):
        fe_dict[each] = features[i]
    return fe_dict


def test_performance(fe_dict, pair_list, return_detail=False):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()

    sims = []
    labels = []
    for pair in pairs:
        splits = pair.split()
        fe_1 = fe_dict[splits[0]]
        fe_2 = fe_dict[splits[1]]
        label = int(splits[2])
        sim = cosin_metric(fe_1, fe_2)

        sims.append(sim)
        labels.append(label)

    sims = np.asarray(sims)
    labels = np.asarray(labels)
    
    acc, th = cal_accuracy(sims, labels)
    auc = roc_auc_score(labels, sims)

    if return_detail:
        return auc, acc, th, sims, labels
    else:
        return auc, acc, th



def evaluate_multiclass(gt_labels, pred_labels):
    acc = accuracy_score(gt_labels, pred_labels)
    f1 = f1_score(gt_labels, pred_labels, average='macro')
    recall = recall_score(gt_labels, pred_labels, average='macro')
    recalls = recall_score(gt_labels, pred_labels, average = None)  # 每一类recall返回
    return {'recalls':recalls,'recall':recall,'f1':f1,'acc':acc}
