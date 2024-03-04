import time
import torch
import random
import numpy as np

from sklearn import metrics
from torch.utils.data import DataLoader
from TA_NET import TA_NET
from tools import Train_Dataset, Test_Dataset, test, read_yaml_file, MIL


def train(cfg):
    # dataset choose, TAD or UCF-crime-traffic
    if cfg['TRAIN']['dataset'] == 'TAD':
        dataset_cfg = cfg['TAD']
    else:
        dataset_cfg = cfg['UCF']

    train_cfg = cfg['TRAIN']
    net_cfg = cfg['TA_NET']
    num_bags = net_cfg['num_bags']
    bag_idx = net_cfg['bag_idx']
    dataset_path = train_cfg['dataset_path']

    SEED = train_cfg['SEED']
    random.seed(train_cfg['SEED_dataset'])
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

    # load TA-NET
    net = TA_NET(num_bags=net_cfg['num_bags'], 
                    bag_idx=net_cfg['bag_idx'], 
                    width=net_cfg['width'], 
                    layers=net_cfg['layers'], 
                    heads=net_cfg['heads'], 
                    layers_1=net_cfg['layers_1'], 
                    heads_1=net_cfg['heads_1'])
    net = net.cuda()

    reg_optimizer = torch.optim.Adamax(net.parameters(), lr=train_cfg['learning_rate'], weight_decay=train_cfg['reg_weight'])

    # load datasets
    train_dataset = Train_Dataset(path=dataset_path, num_bags=num_bags, bag_idx=bag_idx)

    test_dataset = Test_Dataset(path=dataset_path, num_bags=num_bags, bag_idx=bag_idx)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # training
    start = time.time()
    for epoch in range(train_cfg['epoch_num']):
        net.train()
        epoch_loss = 0

        reg_optimizer.zero_grad()
        batch_count = 0

        train_dataset()
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, drop_last=True, num_workers=0)
        for X, y in train_loader:
            batch_count += 1

            X = X.squeeze(0).cuda()
            y = y.squeeze()  
            
            score = net(X)

            score_an = score[0:y[0]]
            score_no = score[y[0]:]
            
            if epoch+1 < 30:
                loss = MIL(score_an, score_no, False)
            else:
                loss = MIL(score_an, score_no, True)
            epoch_loss += loss.item()

            loss.backward()
            
            if batch_count == train_cfg['batch_size']:
                batch_count = 0
                reg_optimizer.step()
                reg_optimizer.zero_grad()

        #.............test............#
        gt_all, score_all, gt_an, score_an = test(net, test_loader, dataset_cfg['anomaly'], dataset_cfg['num_frames'])

        fpr_an, tpr_an, thresholds_an = metrics.roc_curve(gt_an, score_an, pos_label=1)
        auc_an = metrics.auc(fpr_an, tpr_an)
        fpr, tpr, thresholds = metrics.roc_curve(gt_all, score_all, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        
        #print results
        if (epoch%1) == 0:
            print('SEED%d Epoch:%d/%d Loss: %.4f anomaly subset auc: %.2f overall set auc: %.2f time: %.1f sec'
                %(SEED, epoch + 1, train_cfg['epoch_num'], epoch_loss, auc_an*100, auc*100, time.time() - start))
            start = time.time()

if __name__ == "__main__":
    # load config
    file_path = 'train_config.yaml'  
    cfg = read_yaml_file(file_path)

    train(cfg)