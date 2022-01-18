# -*- coding: utf-8 -*-
import argparse
import datetime
import os.path


def opts():
    now = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='SRDC', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    # datasets

    parser.add_argument('--dataset_name', type=str, default='OfficeHome',
                        help='')
    parser.add_argument('--data_path_source', type=str, default='./data/datasets/OfficeHome/',
                        help='root of source training set')
    parser.add_argument('--data_path_target', type=str, default='./data/datasets/OfficeHome/',
                        help='root of target training set')
    parser.add_argument('--data_path_target_t', type=str, default='./data/datasets/OfficeHome/',
                        help='root of target test set')
    parser.add_argument('--src', type=str, default='Art', help='source training set')
    parser.add_argument('--tar', type=str, default='Product', help='target training set')
    parser.add_argument('--tar_t', type=str, default='Product', help='target test set')
    parser.add_argument('--num_classes', type=int, default=65, help='class number')
    # general optimization options
    parser.add_argument('--optim_name', type=str, default='SGD', help='name of optim')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--workers', type=int, default=8, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--warmup', type=int, default=0, help='learning rate warmup epoch')
    parser.add_argument('--lr_plan', type=str, default='dao', help='learning rate decay plan of step or dao or cos')
    parser.add_argument('--schedule', type=int, nargs='+', default=[30, 60, 90],
                        help='decrease learning rate at these epochs for step decay')
    parser.add_argument('--lr_alpha', type=float, default=0.01, help='lr decay level for cos decay')
    parser.add_argument('--lr_decay_epoch', type=int, default=20, help='lr decay epoch for cos decay')
    parser.add_argument('--dao_lr_alpha', type=float, default=10, help='lr decay level for dao decay')
    parser.add_argument('--dao_lr_beta', type=float, default=0.75, help='lr decay epoch for dao decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay (L2 penalty)')
    parser.add_argument('--nesterov', action='store_true', help='whether to use nesterov SGD')
    parser.add_argument('--eps', type=float, default=1e-6, help='a small value to prevent underflow')
    # specific optimization options

    parser.add_argument('--mirror_kl_loss', type=bool, default=True, help='mirror_loss_weight')
    parser.add_argument('--mirror_on_src', type=bool, default=False, help='use mirror loss on src data')
    parser.add_argument('--label_softmax', type=bool, default=True,
                        help='whether the label has been softmax in mirror loss')
    parser.add_argument('--mirror_loss_weight', type=float, default=1, help='mirror_loss_weight')
    parser.add_argument('--mirror_loss_weight_first_feature', type=float, default=1.0,
                        help='weight for first feature mirror loss')
    parser.add_argument('--mirror_loss_weight_second_feature', type=float, default=1.0,
                        help='weight for second feature mirror loss')
    parser.add_argument('--mirror_top_n', type=int, default=3, help='mirror sample number in source domain')
    parser.add_argument('--mirror_sample_weight', action='store_true',
                        help='whether give a weight for every mirror sample')
    parser.add_argument('--mirror_sample_weight_version', type=int, default=1, help='version of sample weight ')
    parser.add_argument('--mirror_kl_loss_start_epoch', type=int, default=0, help='when to use mirror loss ')
    parser.add_argument('--mirror_distence_method', type=str, default='Euclidean', help='distence method')
    parser.add_argument('--mirror_kernel', type=str, default='Gaussian_Kernel',
                        help='kernel type, Gaussian_Kernel or Exponential_Kernel or Cauchy_Kernel or Polynomial_Kernel or Sigmoid_Kernel or Multiquadric_Kernel')
    parser.add_argument('--mirror_gamma', type=float, default=0.7, help='')
    parser.add_argument('--mirror_c', type=float, default=5.0, help='')
    parser.add_argument('--mirror_d', type=float, default=2.0, help='')

    parser.add_argument('--use_SRDC_loss', type=bool, default=True, help='whether to use SRDC loss')

    parser.add_argument('--ao', action='store_true', help='whether to use alternative optimization')
    parser.add_argument('--cluster_method', type=str, default='kmeans',
                        help='clustering method of kmeans or spherical_kmeans or kernel_kmeans to choose')
    parser.add_argument('--cluster_iter', type=int, default=5, help='number of iterations of K-means')
    parser.add_argument('--cluster_kernel', type=str, default='rbf', help='kernel to choose when using kernel K-means')
    parser.add_argument('--gamma', type=float, default=None,
                        help='bandwidth for rbf or polynomial kernel when using kernel K-means')
    parser.add_argument('--sample_weight', action='store_true',
                        help='whether to adapt sample weight when using kernel K-means')
    parser.add_argument('--initial_cluster', type=int, default=1,
                        help='target or source class centroids for initialization of K-means')
    parser.add_argument('--init_cen_on_st', action='store_true',
                        help='whether to initialize learnable cluster centers on both source and target instances')
    parser.add_argument('--src_cen_first', action='store_true',
                        help='whether to use source class centroids as initial target cluster centers at the first epoch')
    parser.add_argument('--src_cls', action='store_true',
                        help='whether to classify source instances when clustering target instances')
    parser.add_argument('--src_fit', action='store_true',
                        help='whether to use convex combination of true label vector and predicted label vector as training guide')
    parser.add_argument('--src_pretr_first', action='store_true',
                        help='whether to perform clustering over features extracted by source pre-trained model at the first epoch')
    parser.add_argument('--alpha', type=float, default=1.0, help='degrees of freedom of Student\'s t-distribution')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='weight of auxiliary target distribution or assigned cluster labels')
    parser.add_argument('--embed_softmax', action='store_true',
                        help='whether to use softmax to normalize soft cluster assignments for embedding clustering')
    parser.add_argument('--div', type=str, default='kl',
                        help='measure of prediction divergence between one target instance and its perturbed counterpart')
    parser.add_argument('--sigma', type=float, default=0.1,
                        help='standard deviation of Gaussian for data augmentation operation of blurring')

    # checkpoints+
    parser.add_argument('--log', type=str, default='log',
                        help='log folder')
    parser.add_argument('--stop_epoch', type=int, default=200, metavar='N',
                        help='stop epoch for early stop (default: 200)')
    # architecture
    parser.add_argument('--arch', type=str, default='resnet50', help='model name')
    parser.add_argument('--num_neurons', type=int, default=128, help='number of neurons of fc1')
    parser.add_argument('--conv_feature_len', type=int, default=2048, help='number of neurons of conv feature')
    parser.add_argument('--fc_feature_len', type=int, default=512, help='number of neurons of conv feature')
    parser.add_argument('--pretrained', action='store_true', help='whether to use pretrained model')
    # i/o
    parser.add_argument('--print_freq', type=int, default=10, metavar='N', help='print frequency (default: 10)')

    args = parser.parse_args()
    args.pretrained = True
    args.init_cen_on_st = True
    if args.src.find('webcam') != -1:
        args.beta = 0.5
    if "VisDA" == args.data_path_source.split('/')[-2]:
        args.initial_cluster = 2
    args.src_cls = True
    args.src_cen_first = True
    args.embed_softmax = True
    return args


if __name__ == "__main__":
    args = opts()
    print args

