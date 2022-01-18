# -*- coding: utf-8 -*-
"""
# @Time    : 2022/1/6 11:21 上午
# @Author  : wmq
# @File    : train.py
# @Software: PyCharm
"""
import os
import gc
import time
import json
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn
from opts import opts
from torch.autograd import Variable
from Models.Mirror_Model import Get_Model
from Data.prepare_data import generate_dataloader
from trainer import validate_compute_cen, validate, k_means, kernel_k_means, spherical_k_means, train_epoch

args = opts()
best_prec1 = 0
best_test_prec1 = 0
cond_best_test_prec1 = 0
best_cluster_acc = 0
best_cluster_acc_2 = 0
local_code_root = os.path.dirname(os.path.abspath(__file__))


def set_seed(seed=666):
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def count_epoch_on_large_dataset(train_loader_target, train_loader_source, args):
    batch_number_t = len(train_loader_target)
    batch_number = batch_number_t
    if args.src_cls:
        batch_number_s = len(train_loader_source)
        if batch_number_s > batch_number_t:
            batch_number = batch_number_s

    return batch_number


def main():
    global args, best_prec1, best_test_prec1, cond_best_test_prec1, best_cluster_acc, best_cluster_acc_2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # define model
    model = Get_Model(args).to(device)

    # define learnable cluster centers
    learn_cen = Variable(torch.FloatTensor(args.num_classes, args.conv_feature_len).fill_(0)).to(device)
    learn_cen.requires_grad_(True)
    learn_cen_2 = Variable(torch.FloatTensor(args.num_classes, args.fc_feature_len).fill_(0)).to(device)
    learn_cen_2.requires_grad_(True)

    # define class centers
    src_centers = np.zeros((args.num_classes, args.fc_feature_len), dtype=np.float)
    tgt_centers = np.zeros((args.num_classes, args.fc_feature_len), dtype=np.float)

    # define loss function/criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    # apply different learning rates to different layer only SGD optimizer
    optimizer = torch.optim.SGD([
        {'params': model.conv1.parameters(), 'name': 'conv'},
        {'params': model.bn1.parameters(), 'name': 'conv'},
        {'params': model.layer1.parameters(), 'name': 'conv'},
        {'params': model.layer2.parameters(), 'name': 'conv'},
        {'params': model.layer3.parameters(), 'name': 'conv'},
        {'params': model.layer4.parameters(), 'name': 'conv'},
        {'params': model.fc1.parameters(), 'name': 'ca_cl'},
        {'params': model.fc2.parameters(), 'name': 'ca_cl'},
        {'params': learn_cen, 'name': 'conv'},
        {'params': learn_cen_2, 'name': 'conv'}
    ],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov)

    epoch = 0

    # make log directory
    if not os.path.exists(args.log):
        os.makedirs(args.log)

    log = open(os.path.join(args.log, 'log.txt'), 'a')
    state = {k: v for k, v in args._get_kwargs()}
    log.write(json.dumps(state) + '\n')
    log.close()

    # start time
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    log.write('\n-------------------------------------------\n')
    log.write(time.asctime(time.localtime(time.time())))
    log.write('\n-------------------------------------------')
    log.close()

    cudnn.benchmark = True

    # process data and prepare dataloaders
    train_loader_source, train_loader_target, val_loader_source, val_loader_target = generate_dataloader(
        args, local_code_root)

    train_loader_target.dataset.tgts = list(np.array(
        torch.LongTensor(train_loader_target.dataset.tgts).fill_(-1)))  # avoid using ground truth labels of target

    print('begin training')
    batch_number = count_epoch_on_large_dataset(train_loader_target, train_loader_source, args)
    num_itern_total = args.epochs * batch_number

    new_epoch_flag = False  # if new epoch, new_epoch_flag=True
    test_flag = False  # if test, test_flag=True
    src_cs = torch.FloatTensor(len(train_loader_source.dataset.tgts)).fill_(1).to(device)  # initialize source weights
    count_itern_each_epoch = 0
    for itern in range(epoch * batch_number, num_itern_total):
        # evaluate on the target training and test data
        if (itern == 0) or (count_itern_each_epoch == batch_number):
            prec1, c_s, c_s_2, c_t, c_t_2, c_srctar, c_srctar_2, source_features, source_features_2, source_targets, target_features, target_features_2, target_targets, pseudo_labels = validate_compute_cen(
                train_loader_target, train_loader_source, model, criterion, epoch, args, device=device)
            test_acc = validate(device, val_loader_target, model, criterion, args)
            test_flag = True

            g_source_features = source_features
            g_source_features_2 = source_features_2

            g_target_features = target_features
            g_target_features_2 = target_features_2

            # K-means clustering or its variants
            if ((itern == 0) and args.src_cen_first) or (args.initial_cluster == 2):
                cen = c_s
                cen_2 = c_s_2
            else:
                cen = c_t
                cen_2 = c_t_2
            if (itern != 0) and (args.initial_cluster != 0) and (args.cluster_method == 'kernel_kmeans'):
                cluster_acc, c_t = kernel_k_means(target_features, target_targets, pseudo_labels, train_loader_target,
                                                  epoch, model, args, best_cluster_acc)
                cluster_acc_2, c_t_2 = kernel_k_means(target_features_2, target_targets, pseudo_labels,
                                                      train_loader_target, epoch, model, args, best_cluster_acc_2,
                                                      change_target=False)
            elif args.cluster_method != 'spherical_kmeans':
                cluster_acc, c_t = k_means(target_features, target_targets, train_loader_target, epoch, model, cen,
                                           args, best_cluster_acc)
                cluster_acc_2, c_t_2 = k_means(target_features_2, target_targets, train_loader_target, epoch, model,
                                               cen_2, args, best_cluster_acc_2, change_target=False)
            else:
                cluster_acc, c_t = spherical_k_means(target_features, target_targets, train_loader_target, epoch, model,
                                                     cen, args, best_cluster_acc)
                cluster_acc_2, c_t_2 = spherical_k_means(target_features_2, target_targets, train_loader_target, epoch,
                                                         model, cen_2, args, best_cluster_acc_2, change_target=False)

            # record the best accuracy of K-means clustering
            log = open(os.path.join(args.log, 'log.txt'), 'a')
            if cluster_acc != best_cluster_acc:
                best_cluster_acc = cluster_acc
                log.write('\n best_cluster acc: %3f' % best_cluster_acc)
            if cluster_acc_2 != best_cluster_acc_2:
                best_cluster_acc_2 = cluster_acc_2
                log.write('\n best_cluster_2 acc: %3f' % best_cluster_acc_2)
            log.close()

            # re-initialize learnable cluster centers
            if args.init_cen_on_st:
                cen = (c_t + c_s) / 2  # or c_srctar
                cen_2 = (c_t_2 + c_s_2) / 2  # or c_srctar_2
            else:
                cen = c_t
                cen_2 = c_t_2
            # if itern == 0:
            learn_cen.data = cen.data.clone()
            learn_cen_2.data = cen_2.data.clone()

            if itern != 0:
                count_itern_each_epoch = 0
                epoch += 1
            batch_number = count_epoch_on_large_dataset(train_loader_target, train_loader_source, args)
            train_loader_target_batch = enumerate(train_loader_target)
            train_loader_source_batch = enumerate(train_loader_source)

            new_epoch_flag = True

            del source_features
            del source_features_2
            del source_targets
            del target_features
            del target_features_2
            del target_targets
            del pseudo_labels
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
        elif (args.data_path_source.find('VisDA') != -1) and (itern % int(num_itern_total / 5) == 0):
            prec1, _, _, _, _, _, _, _, _, _, _, _, _, _ = validate_compute_cen(val_loader_target, val_loader_source,
                                                                                model, criterion, epoch, args,
                                                                                compute_cen=False, device=device)
            test_acc = validate(device, val_loader_target, model, criterion, args)
            test_flag = True
        if test_flag:
            # record the best prec1 and save checkpoint
            log = open(os.path.join(args.log, 'log.txt'), 'a')
            if prec1 > best_prec1:
                best_prec1 = prec1
                cond_best_test_prec1 = 0
                log.write('\n best val acc till now: %3f' % best_prec1)
            if test_acc > best_test_prec1:
                best_test_prec1 = test_acc
                log.write('\n best test acc till now: %3f' % best_test_prec1)
            # ipdb.set_trace()
            is_cond_best = ((prec1 == best_prec1) and (test_acc > cond_best_test_prec1))
            if is_cond_best:
                cond_best_test_prec1 = test_acc
                log.write('\n cond best test acc till now: %3f' % cond_best_test_prec1)
            log.close()
            save_checkpoint({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'learn_cen': learn_cen,
                'learn_cen_2': learn_cen_2,
                'best_prec1': best_prec1,
                'best_test_prec1': best_test_prec1,
                'cond_best_test_prec1': cond_best_test_prec1,
                'src_centers': src_centers,
                'tgt_centers': tgt_centers
            }, test_acc, epoch)

            test_flag = False

        # early stop
        if epoch > args.stop_epoch:
            break
        # train for one iteration
        train_loader_source_batch, train_loader_target_batch, src_centers, tgt_centers, optimizer, loss = train_epoch(
            train_loader_source, train_loader_source_batch, train_loader_target, train_loader_target_batch,
            model, learn_cen, learn_cen_2, g_source_features, g_source_features_2, g_target_features, g_target_features_2,
            optimizer, itern, epoch, new_epoch_flag, src_cs, src_centers, tgt_centers, args)

        model = model.to(device)
        new_epoch_flag = False
        count_itern_each_epoch += 1
        # loss backward and network update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = open(os.path.join(args.log, 'log.txt'), 'a')
    log.write('\n***   best val acc: %3f   ***' % best_prec1)
    log.write('\n***   best test acc: %3f   ***' % best_test_prec1)
    log.write('\n***   cond best test acc: %3f   ***' % cond_best_test_prec1)
    # end time
    log.write('\n-------------------------------------------\n')
    log.write(time.asctime(time.localtime(time.time())))
    log.write('\n-------------------------------------------\n')
    log.close()


def save_checkpoint(state, test_acc, epoch):
    filename = os.path.join(local_code_root, 'checkpoint', 'checkpoint_%03d_%.4f.pth.tar' % (epoch, test_acc))
    torch.save(state, filename)


if __name__ == '__main__':
    if args.seed is not None:
        set_seed(args.seed)
    main()
