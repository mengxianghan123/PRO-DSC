import torch
from torch.utils.data import DataLoader
import numpy as np
from data.data_utils import FeatureDataset
import argparse
from model.model import PRO_DSC
import os
import torch.nn.functional as F
from metrics.clustering import spectral_clustering_metrics



@torch.no_grad()
def evaluate(args, model):
    feature_dict = torch.load(args.data_dir)

    clip_features = feature_dict['features']
    clip_labels = feature_dict['ys']
    if args.data == 'cifar20':
        def sparse2coarse(targets):
            """CIFAR100 Coarse Labels. """
            coarse_targets = [ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  3, 14,  9, 18,  7, 11,  3,
                            9,  7, 11,  6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  0, 11,  1, 10,
                            12, 14, 16,  9, 11,  5,  5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 16,
                            4, 17,  4,  2,  0, 17,  4, 18, 17, 10,  3,  2, 12, 12, 16, 12,  1,
                            9, 19,  2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 16, 19,  2,  4,  6,
                            19,  5,  5,  8, 19, 18,  1,  2, 15,  6,  0, 17,  8, 14, 13]
            return np.array(coarse_targets)[targets.numpy()]
        clip_labels = torch.from_numpy(sparse2coarse(clip_labels))
        
    clip_feature_set = FeatureDataset(clip_features, clip_labels)
    test_loader = DataLoader(clip_feature_set, batch_size=args.bs, shuffle=False, drop_last=False, num_workers=8)

    label_list = list()
    c_list = list()
    model.eval()
    for data, labels in test_loader:
        label_list.append(labels)
        data = data.float().to(args.gpu)
        _, c_norm = model(data)
        c_list.append(c_norm)
    c = torch.cat(c_list, dim=0)
    labels = torch.cat(label_list, dim=0).cpu().numpy()

    s_matrix = torch.abs(c.matmul(c.T))

    acc_lst, nmi_lst, pred_lst = spectral_clustering_metrics(s_matrix.cpu().numpy(), args.n_clusters, labels, verbose=False)
    print('mean acc: {} max acc: {}, mean nmi: {} max nmi: {}'.format(np.mean(acc_lst), np.max(acc_lst),np.mean(nmi_lst), np.max(nmi_lst)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unsupervised Learning')
    parser.add_argument('--data', type=str, default="cifar10",
                        help='data (default: cifar10)')
    parser.add_argument('--pth', type=str)
    parser.add_argument('--data_dir', type=str, default="./data/datasets/cifar10_clip_60000.pt",
                        help='data dir')    
    parser.add_argument('--hidden_dim', type=int, default=4096,
                        help='dimension of hidden state')
    parser.add_argument('--z_dim', type=int, default=128,
                        help='dimension of subspace feature dimension')
    parser.add_argument('--bs', type=int, default=1024,
                        help='batch size (default: 1024)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu for training')
    args = parser.parse_args()
    if args.data == 'cifar10':
        args.n_clusters = 10
    elif args.data == 'cifar20':
        args.n_clusters = 20
    elif args.data == 'cifar100':
        args.n_clusters = 100
    elif args.data == 'imagenet':
        args.n_clusters = 1000
    model = PRO_DSC(input_dim=768, hidden_dim = args.hidden_dim, z_dim = args.z_dim).to(args.gpu)
    model.load_state_dict(torch.load(args.pth))
    evaluate(args=args, model=model)
