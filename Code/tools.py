import os
import yaml
import torch
import random
import numpy as np

from sklearn import metrics
from torch.utils.data import Dataset

def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

class Train_Dataset(Dataset):
    def __init__(self, path: str, num_bags: int, bag_idx: int):       
        super(Train_Dataset, self).__init__()
        data_list = os.path.join(path, 'trains.txt')
        self.num_bags = num_bags
        self.bag_idx = bag_idx
        self.an_videos = []
        self.no_videos = []
        with open(data_list, 'r') as f:
            for line in f:
                video_name = line.split( )[0]
                save_path = os.path.join(path, video_name + '.pt')
                if video_name[0] == 'N':
                    self.no_videos.append(save_path)
                else:
                    self.an_videos.append(save_path)
    
    def __len__(self):
        return min(len(self.an_videos), len(self.no_videos))

    def __getitem__(self, idx):  
        an_tokens = torch.load(self.an_videos[idx])
        no_tokens = torch.load(self.no_videos[idx])
        label = torch.tensor([an_tokens.shape[0], no_tokens.shape[0]])
        pair = torch.cat([an_tokens, no_tokens],dim=0)
        return pair, label

    def __call__(self):
        random.shuffle(self.no_videos)
        random.shuffle(self.an_videos)

class Test_Dataset(Dataset):
    def __init__(self, path: str, num_bags: int, bag_idx: int):       
        super(Test_Dataset, self).__init__()
        data_list = os.path.join(path, 'vals.txt')
        self.num_bags = num_bags
        self.bag_idx = bag_idx
        self.videos = []
        self.labels = []
        with open(data_list, 'r') as f:
            for line in f:
                line_items = line.split( )
                video_name = line_items[0]
                save_path = os.path.join(path, video_name + '.pt')
                gt = np.zeros(int(line_items[1]))
                an_num = int((len(line_items)-2)/2)
                for i in range(an_num):
                    gt[int(line_items[2 + 2*i]):int(line_items[3+ 2*i])] = 1
                self.videos.append(save_path)
                self.labels.append(gt)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):  
        tokens = torch.load(self.videos[idx])
        label = self.labels[idx]
        return tokens, label

def get_false_alarm_rate(ground_true, pred):
    pred = ((pred > 0.5) + 0.)#.tolist()
    fa_rate = np.count_nonzero(pred > ground_true)/len(pred)
    return fa_rate

def get_report(ground_true, pred, target_names=['0', '1']):
    pred = ((pred > 0.5) + 0.).tolist()
    return metrics.classification_report(ground_true, pred, target_names=target_names)

def min_max_norm(input):
    '''input: torch.tensor'''
    norm = input - input.min()
    norm = norm / norm.max()
    return norm.cpu().detach().numpy()

def MIL(y_anomaly, y_normal, normal_loss = False):
    loss = torch.tensor(0.).cuda()
    normal = torch.tensor(0.).cuda()

    y_anomaly_max = torch.max(y_anomaly) # anomaly
    y_normal_max = torch.max(y_normal) # normal

    loss = 1-y_anomaly_max+y_normal_max

    anormaly = torch.sum(y_anomaly)
    normal = torch.sum(y_normal)
    if normal_loss:
        final_loss = loss + 1*normal + 1*(anormaly/y_anomaly.shape[0])
    else: 
        final_loss = loss
    return final_loss

def test(net, test_loader, anomaly_idx, num_frames):
    i = 0
    net.eval()
    gt_all = np.array([])
    score_all = np.array([])

    for X, y in test_loader:
        i += 1
        frames = len(y[0])
        label = y[0].numpy()
        X = X.squeeze(0).cuda()
        score = net(X)
        score = score.detach().cpu()
        score_list = np.zeros(frames)
       
        for j in range(score.shape[0]):
            if j == (score.shape[0] - 1):
                score_list[num_frames*j:] = score[j]
            else:
                score_list[num_frames*j:num_frames*(j+1)] = score[j]

        gt_all = np.concatenate((gt_all, label), axis=0)
        score_all = np.concatenate((score_all, score_list), axis=0)
        if i == anomaly_idx:
            gt_an = gt_all
            score_an = score_all
    
    return gt_all, score_all, gt_an, score_an