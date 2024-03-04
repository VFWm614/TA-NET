import torch

from sklearn import metrics
from torch.utils.data import DataLoader
from TA_NET import TA_NET
from tools import Test_Dataset, test, get_report, read_yaml_file

def test_net(cfg):
    # dataset choose, TAD or UCF-crime-traffic
    if cfg['TEST']['dataset'] == 'TAD':
        dataset_cfg = cfg['TAD']
    else:
        dataset_cfg = cfg['UCF']

    dataset_path = cfg['TEST']['dataset_path']
    model_path = cfg['TEST']['model_path']
    net_cfg = cfg['TA_NET']

    # load pretrained model
    net = TA_NET(num_bags=net_cfg['num_bags'], 
                    bag_idx=net_cfg['bag_idx'], 
                    width=net_cfg['width'], 
                    layers=net_cfg['layers'], 
                    heads=net_cfg['heads'], 
                    layers_1=net_cfg['layers_1'], 
                    heads_1=net_cfg['heads_1'])
    net.load_state_dict(torch.load(model_path))
    net = net.cuda()

    # load test dataset
    test_dataset = Test_Dataset(path=dataset_path, num_bags=net_cfg['num_bags'], bag_idx=net_cfg['bag_idx'])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # test
    gt_all, score_all_best, gt_an, score_an = test(net, test_loader, dataset_cfg['anomaly'], dataset_cfg['num_frames'])

    # print result
    print('--Test on Anomaly Subset--')
    fpr_an, tpr_an, thresholds_an = metrics.roc_curve(gt_an, score_an, pos_label=1)
    auc_an = metrics.auc(fpr_an, tpr_an)
    print(get_report(gt_an, score_an))
    print('AUC of the Anomaly Subset is {: .2f}%'.format(auc_an*100))

    print('--Test on Overall Set--')
    fpr, tpr, thresholds = metrics.roc_curve(gt_all, score_all_best, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print(get_report(gt_all, score_all_best))
    print('AUC of the Overall Set is {: .2f}%'.format(auc*100))

if __name__ == "__main__":
    # load config
    file_path = 'test_config.yaml'  
    cfg = read_yaml_file(file_path)

    test_net(cfg)