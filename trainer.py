# -*- coding: utf-8 -*-

from __future__ import division
import gc
import os
import time
import math
import torch
import numpy as np
import torch.nn.functional as F
from Utils import kernel_function
from torch.autograd import Variable
from Utils.kernel_kmeans import KernelKMeans


def validate(device, val_loader, model, criterion, args):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    total_vector = torch.FloatTensor(args.num_classes).fill_(0).to(device)
    correct_vector = torch.FloatTensor(args.num_classes).fill_(0).to(device)

    for i, (input, target, _) in enumerate(val_loader):
        target = target.to(device)
        input = input.to(device)
        input_var = Variable(input)
        target_var = Variable(target)

        # forward
        with torch.no_grad():
            _, _, output = model(input_var)
            loss = criterion(output, target_var)

        # compute and record loss and accuracy
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

    print(' * Test on T test set - Prec@1 {top1.avg:.3f}, Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    if args.dataset_name.find('VisDA') != -1:
        total_vector, correct_vector = accuracy_for_each_class(output.data, target, total_vector,
                                                               correct_vector)  # compute class-wise accuracy
        acc_for_each_class = 100.0 * correct_vector / total_vector
        for i in range(args.num_classes):
            if i == 0:
                print ("%dst: %3f" % (i + 1, acc_for_each_class[i]))
            elif i == 1:
                print (",  %dnd: %3f" % (i + 1, acc_for_each_class[i]))
            elif i == 2:
                print (", %drd: %3f" % (i + 1, acc_for_each_class[i]))
            else:
                print (", %dth: %3f" % (i + 1, acc_for_each_class[i]))
        print ("\n                          Avg. over all classes: %3f" % acc_for_each_class.mean())
        return acc_for_each_class.mean()
    else:
        return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    if maxk > output.size(1):
        maxk = output.size(1)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        if k > output.size(1):
            k = output.size(1) - 1
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


def accuracy_for_each_class(output, target, total_vector, correct_vector):
    """Computes the precision for each class"""
    batch_size = target.size(0)

    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1)).float().cpu().squeeze()
    for i in range(batch_size):
        total_vector[target[i]] += 1

        correct_vector[torch.LongTensor([target[i]])] += correct[i]

    return total_vector, correct_vector


def validate_compute_cen(val_loader_target, val_loader_source, model, criterion, epoch, args, compute_cen=True, device='cpu'):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    # compute source class centroids
    source_features = torch.FloatTensor(len(val_loader_source.dataset.imgs), args.conv_feature_len).fill_(0).to(device)
    source_features_2 = torch.FloatTensor(len(val_loader_source.dataset.imgs), args.fc_feature_len).fill_(0).to(device)
    source_targets = torch.LongTensor(len(val_loader_source.dataset.imgs)).fill_(0).to(device)
    c_src = torch.FloatTensor(args.num_classes, args.conv_feature_len).fill_(0).to(device)
    c_src_2 = torch.FloatTensor(args.num_classes, args.fc_feature_len).fill_(0).to(device)
    count_s = torch.FloatTensor(args.num_classes, 1).fill_(0).to(device)
    if compute_cen:
        for i, (input, target, index) in enumerate(val_loader_source):  # the iterarion in the source dataset
            input_var = Variable(input)
            if torch.cuda.is_available():
                input_var = input_var.cuda(device='cuda:0')
                target = target.cuda(device='cuda:0')
            with torch.no_grad():
                feature, feature_2, output = model(input_var)
            source_features[index.to(device), :] = feature.data.clone()
            source_features_2[index.to(device), :] = feature_2.data.clone()
            source_targets[index.to(device)] = target.clone()
            target_ = torch.FloatTensor(output.size()).fill_(0).to(device)
            target_.scatter_(1, target.unsqueeze(1), torch.ones(output.size(0), 1).to(device))
            if args.cluster_method == 'spherical_kmeans':
                c_src += ((feature / feature.norm(p=2, dim=1, keepdim=True)).unsqueeze(1) * target_.unsqueeze(2)).sum(0)
                c_src_2 += ((feature_2 / feature_2.norm(p=2, dim=1, keepdim=True)).unsqueeze(1) * target_.unsqueeze(
                    2)).sum(0)
            else:
                c_src += (feature.unsqueeze(1) * target_.unsqueeze(2)).sum(0)
                c_src_2 += (feature_2.unsqueeze(1) * target_.unsqueeze(2)).sum(0)
                count_s += target_.sum(0).unsqueeze(1)

    target_features = torch.FloatTensor(len(val_loader_target.dataset.imgs), args.conv_feature_len).fill_(0).to(device)
    target_features_2 = torch.FloatTensor(len(val_loader_target.dataset.imgs), args.fc_feature_len).fill_(0).to(device)
    target_targets = torch.LongTensor(len(val_loader_target.dataset.imgs)).fill_(0).to(device)
    pseudo_labels = torch.LongTensor(len(val_loader_target.dataset.imgs)).fill_(0).to(device)
    c_tar = torch.FloatTensor(args.num_classes, args.conv_feature_len).fill_(0).to(device)
    c_tar_2 = torch.FloatTensor(args.num_classes, args.fc_feature_len).fill_(0).to(device)
    count_t = torch.FloatTensor(args.num_classes, 1).fill_(0).to(device)

    total_vector = torch.FloatTensor(args.num_classes).fill_(0).to(device)
    correct_vector = torch.FloatTensor(args.num_classes).fill_(0).to(device)

    end = time.time()
    for i, (input, target, index) in enumerate(val_loader_target):  # the iterarion in the target dataset
        data_time.update(time.time() - end)

        input_var = Variable(input).to(device)
        target_var = Variable(target).to(device)
        target = target.to(device)

        with torch.no_grad():
            feature, feature_2, output = model(input_var)
        target_features[index.to(device), :] = feature.data.clone()  # index:a tensor
        target_features_2[index.to(device), :] = feature_2.data.clone()
        target_targets[index.to(device)] = target.clone()
        pseudo_labels[index.to(device)] = output.softmax(1).argmax(1).data.clone()

        if compute_cen:  # compute target class centroids
            pred = output.data.max(1)[1]

            pred_ = torch.FloatTensor(output.size()).fill_(0).to(device)
            pred_.scatter_(1, pred.unsqueeze(1), torch.ones(output.size(0), 1).to(device))

            if args.cluster_method == 'spherical_kmeans':
                c_tar += ((feature / feature.norm(p=2, dim=1, keepdim=True)).unsqueeze(1) * pred_.unsqueeze(2)).sum(0)
                c_tar_2 += ((feature_2 / feature_2.norm(p=2, dim=1, keepdim=True)).unsqueeze(1) * pred_.unsqueeze(
                    2)).sum(0)
            else:
                c_tar += (feature.unsqueeze(1) * pred_.unsqueeze(2)).sum(0)
                c_tar_2 += (feature_2.unsqueeze(1) * pred_.unsqueeze(2)).sum(0)
                count_t += pred_.sum(0).unsqueeze(1)

    # compute global class centroids
    c_srctar = (c_src + c_tar) / (count_s + count_t)
    c_srctar_2 = (c_src_2 + c_tar_2) / (count_s + count_t)
    c_src /= count_s
    c_src_2 /= count_s
    c_tar /= (count_t + args.eps)
    c_tar_2 /= (count_t + args.eps)

    if args.data_path_source.find('VisDA') != -1:
        total_vector, correct_vector = accuracy_for_each_class(output.data, target, total_vector,
                                                               correct_vector)  # compute class-wise accuracy
        acc_for_each_class = 100.0 * correct_vector / total_vector
        return acc_for_each_class.mean(), c_src, c_src_2, c_tar, c_tar_2, c_srctar, c_srctar_2, source_features, source_features_2, source_targets, target_features, target_features_2, target_targets, pseudo_labels
    else:

        return top1.avg, c_src, c_src_2, c_tar, c_tar_2, c_srctar, c_srctar_2, source_features, source_features_2, source_targets, target_features, target_features_2, target_targets, pseudo_labels


def kernel_k_means(target_features, target_targets, pseudo_labels, train_loader_target, epoch, model, args, best_prec,
                   change_target=True):
    # define kernel k-means clustering
    kkm = KernelKMeans(n_clusters=args.num_classes, max_iter=args.cluster_iter, random_state=args.seed,
                       kernel=args.cluster_kernel, gamma=args.gamma, verbose=1)

    kkm.fit(np.array(target_features.cpu()), initial_label=np.array(pseudo_labels.long().cpu()),
            true_label=np.array(target_targets.cpu()), args=args, epoch=epoch)

    idx_sim = torch.from_numpy(kkm.labels_)
    if torch.cuda.is_available():
        c_tar = torch.cuda.FloatTensor(args.num_classes, target_features.size(1)).fill_(0)
        count = torch.cuda.FloatTensor(args.num_classes, 1).fill_(0)
    else:
        c_tar = torch.FloatTensor(args.num_classes, target_features.size(1)).fill_(0)
        count = torch.FloatTensor(args.num_classes, 1).fill_(0)
    for i in range(target_targets.size(0)):
        c_tar[idx_sim[i]] += target_features[i]
        count[idx_sim[i]] += 1
        if change_target:
            train_loader_target.dataset.tgts[i] = idx_sim[i].item()
    c_tar /= (count + args.eps)

    prec1 = kkm.prec1_
    is_best = prec1 > best_prec
    if is_best:
        best_prec = prec1
        # torch.save(c_tar, os.path.join(args.log, 'c_t_kernel_kmeans_cluster_best.pth.tar'))
        # torch.save(model.state_dict(), os.path.join(args.log, 'checkpoint_kernel_kmeans_cluster_best.pth.tar'))

    del target_features
    del target_targets
    del pseudo_labels
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()

    return best_prec, c_tar


def k_means(target_features, target_targets, train_loader_target, epoch, model, c, args, best_prec, change_target=True):
    batch_time = AverageMeter()

    c_tar = c.data.clone()
    end = time.time()
    for itr in range(args.cluster_iter):
        torch.cuda.empty_cache()
        if args.data_path_source.find('VisDA') != -1 or args.data_path_source.find(
                'Gender') != -1 or args.data_path_source.find('Emotion') != -1 \
                or args.data_path_source.find('Digits') != -1:
            dist_xt_ct = []
            for b_idx in range(target_features.size(0) // 1000):
                b_idx = b_idx * 1000
                if b_idx + 1000 > target_features.size(0):

                    e_idx = target_features.size(0)
                else:
                    e_idx = b_idx + 1000

                dist_xt_ct_temp = (
                    (target_features[b_idx:e_idx].unsqueeze(1) - c_tar.unsqueeze(0)).pow(2).sum(
                        dim=2))
                dist_xt_ct.append(dist_xt_ct_temp)
            if e_idx < target_features.size(0) - 1:
                dist_xt_ct_temp = (
                    (target_features[e_idx:].unsqueeze(1) - c_tar.unsqueeze(0)).pow(2).sum(
                        dim=2))
                dist_xt_ct.append(dist_xt_ct_temp)
            dist_xt_ct = torch.cat(dist_xt_ct, dim=0)
        else:
            dist_xt_ct_temp = target_features.unsqueeze(1) - c_tar.unsqueeze(0)
            dist_xt_ct = dist_xt_ct_temp.pow(2).sum(2)
        _, idx_sim = (-1 * dist_xt_ct).data.topk(1, 1, True, True)
        prec1 = accuracy(-1 * dist_xt_ct.data, target_targets, topk=(1,))[0].item()
        is_best = prec1 > best_prec
        if is_best:
            best_prec = prec1

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        print('Epoch %d, K-means clustering %d, Average clustering time %.3f, Prec@1 %.3f' % (
            epoch, itr, batch_time.avg, prec1))
        log = open(os.path.join(args.log, 'log.txt'), 'a')
        log.write('\nEpoch %d, K-means clustering %d, Average clustering time %.3f, Prec@1 %.3f' % (
            epoch, itr, batch_time.avg, prec1))
        if args.data_path_source.find('VisDA') != -1:
            total_vector_dist = torch.FloatTensor(args.num_classes).fill_(0)
            correct_vector_dist = torch.FloatTensor(args.num_classes).fill_(0)
            total_vector_dist, correct_vector_dist = accuracy_for_each_class(-1 * dist_xt_ct.data, target_targets,
                                                                             total_vector_dist, correct_vector_dist)
            acc_for_each_class_dist = 100.0 * correct_vector_dist / (total_vector_dist + args.eps)
            log.write("\nAcc_dist for each class: ")
            for i in range(args.num_classes):
                if i == 0:
                    log.write("%dst: %3f" % (i + 1, acc_for_each_class_dist[i]))
                elif i == 1:
                    log.write(",  %dnd: %3f" % (i + 1, acc_for_each_class_dist[i]))
                elif i == 2:
                    log.write(", %drd: %3f" % (i + 1, acc_for_each_class_dist[i]))
                else:
                    log.write(", %dth: %3f" % (i + 1, acc_for_each_class_dist[i]))
            log.write("\n                          Avg_dist. over all classes: %3f" % acc_for_each_class_dist.mean())
        log.close()
        if torch.cuda.is_available():
            c_tar_temp = torch.cuda.FloatTensor(args.num_classes, c_tar.size(1)).fill_(0)
            count = torch.cuda.FloatTensor(args.num_classes, 1).fill_(0)
        else:
            c_tar_temp = torch.FloatTensor(args.num_classes, c_tar.size(1)).fill_(0)
            count = torch.FloatTensor(args.num_classes, 1).fill_(0)
        for k in range(args.num_classes):
            c_tar_temp[k] += target_features[idx_sim.squeeze(1) == k].sum(0)
            count[k] += (idx_sim.squeeze(1) == k).float().sum()
        c_tar_temp /= (count + args.eps)

        if (itr == (args.cluster_iter - 1)) and change_target:
            for i in range(idx_sim.size(0)):
                train_loader_target.dataset.tgts[i] = int(idx_sim[i])

        c_tar = c_tar_temp.clone()

        del dist_xt_ct_temp
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

    del target_features
    del target_targets
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()

    return best_prec, c_tar


def spherical_k_means(target_features, target_targets, train_loader_target, epoch, model, c, args, best_prec,
                      change_target=True):
    batch_time = AverageMeter()

    c_tar = c.data.clone()
    end = time.time()
    for itr in range(args.cluster_iter):
        torch.cuda.empty_cache()
        dist_xt_ct_temp = target_features.unsqueeze(1) * c_tar.unsqueeze(0)
        dist_xt_ct = 0.5 * (1 - dist_xt_ct_temp.sum(2) / (
                target_features.norm(2, dim=1, keepdim=True) * c_tar.norm(2, dim=1, keepdim=True).t() + args.eps))
        _, idx_sim = (-1 * dist_xt_ct).data.topk(1, 1, True, True)
        prec1 = accuracy(-1 * dist_xt_ct.data, target_targets, topk=(1,))[0].item()
        is_best = prec1 > best_prec
        if is_best:
            best_prec = prec1
            # torch.save(c_tar, os.path.join(args.log, 'c_t_spherical_kmeans_cluster_best.pth.tar'))
            # torch.save(model.state_dict(), os.path.join(args.log, 'checkpoint_spherical_kmeans_cluster_best.pth.tar'))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        print('Epoch %d, Spherical K-means clustering %d, Average clustering time %.3f, Prec@1 %.3f' % (
            epoch, itr, batch_time.avg, prec1))
        log = open(os.path.join(args.log, 'log.txt'), 'a')
        log.write('\nEpoch %d, Spherical K-means clustering %d, Average clustering time %.3f, Prec@1 %.3f' % (
            epoch, itr, batch_time.avg, prec1))
        if args.data_path_source.find('VisDA') != -1:
            total_vector_dist = torch.FloatTensor(args.num_classes).fill_(0)
            correct_vector_dist = torch.FloatTensor(args.num_classes).fill_(0)
            total_vector_dist, correct_vector_dist = accuracy_for_each_class(-1 * dist_xt_ct.data, target_targets,
                                                                             total_vector_dist, correct_vector_dist)
            acc_for_each_class_dist = 100.0 * correct_vector_dist / (total_vector_dist + args.eps)
            log.write("\nAcc_dist for each class: ")
            for i in range(args.num_classes):
                if i == 0:
                    log.write("%dst: %3f" % (i + 1, acc_for_each_class_dist[i]))
                elif i == 1:
                    log.write(",  %dnd: %3f" % (i + 1, acc_for_each_class_dist[i]))
                elif i == 2:
                    log.write(", %drd: %3f" % (i + 1, acc_for_each_class_dist[i]))
                else:
                    log.write(", %dth: %3f" % (i + 1, acc_for_each_class_dist[i]))
            log.write("\n                          Avg_dist. over all classes: %3f" % acc_for_each_class_dist.mean())
        log.close()
        c_tar_temp = torch.cuda.FloatTensor(args.num_classes, c_tar.size(1)).fill_(0)
        for k in range(args.num_classes):
            c_tar_temp[k] += (target_features[idx_sim.squeeze(1) == k] / (
                    target_features[idx_sim.squeeze(1) == k].norm(2, dim=1, keepdim=True) + args.eps)).sum(0)

        if (itr == (args.cluster_iter - 1)) and change_target:
            for i in range(target_targets.size(0)):
                train_loader_target.dataset.tgts[i] = int(idx_sim[i])

        c_tar = c_tar_temp.clone()

        del dist_xt_ct_temp
        gc.collect()
        torch.cuda.empty_cache()

    del target_features
    del target_targets
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()

    return best_prec, c_tar


def adjust_learning_rate(optimizer, epoch, args):
    """Adjust the learning rate according the epoch"""
    if epoch < args.warmup:
        lr = args.lr * epoch / args.warmup

    else:

        if args.lr_plan == 'step':
            exp = epoch > args.schedule[1] and 2 or epoch > args.schedule[0] and 1 or 0
            lr = args.lr * (0.1 ** exp)
        elif args.lr_plan == 'dao':
            lr = args.lr / math.pow((1 + args.dao_lr_alpha * epoch / args.epochs), args.dao_lr_beta)
        elif args.lr_plan == 'cos':
            lr_alpha = args.lr_alpha
            global_epoch = min(epoch, args.lr_decay_epoch)
            weight = 0.5 * (1 + math.cos(math.pi * (global_epoch / args.lr_decay_epoch)))
            lr_weight = (1 - lr_alpha) * weight + lr_alpha
            lr = args.lr * lr_weight
        elif args.lr_plan == 'no':
            lr = args.lr
    for param_group in optimizer.param_groups:
        if param_group['name'] == 'conv':
            param_group['lr'] = lr * 0.1
        elif param_group['name'] == 'ca_cl':
            param_group['lr'] = lr
        else:
            raise ValueError('The required parameter group does not exist.')


def mirror_kl_loss_v2(args, features, mirror_features, last_centers):

    if args.mirror_distence_method == 'Kernel':
        kf = getattr(kernel_function, args.mirror_kernel)
        dist_matrix = kf(features, mirror_features, args.mirror_gamma, args.mirror_c, args.mirror_d)
        if args.mirror_kernel in ["Gaussian_Kernel", "Exponential_Kernel", "Cauchy_Kernel"]:
            select_dist, mirror_idx = dist_matrix.topk(k=args.mirror_top_n, dim=1, largest=True)
        else:
            select_dist, mirror_idx = dist_matrix.topk(k=args.mirror_top_n, dim=1, largest=False)
    elif args.mirror_distence_method == 'Euclidean':
        dist_matrix = ((features.unsqueeze(1) - mirror_features.unsqueeze(0)).pow(2).sum(dim=2))
        select_dist, mirror_idx = dist_matrix.topk(k=args.mirror_top_n, dim=1, largest=False)
    elif args.mirror_distence_method == 'Normal_Euclidean':
        features_norm = torch.norm(features, dim=1, keepdim=True)
        mirror_features_norm = torch.norm(mirror_features, dim=1, keepdim=True)
        dist_matrix = ((features_norm.unsqueeze(1) - mirror_features_norm.unsqueeze(0)).pow(2).sum(dim=2))
        select_dist, mirror_idx = dist_matrix.topk(k=args.mirror_top_n, dim=1, largest=False)
    elif args.mirror_distence_method == 'Cos':
        features_norm = torch.norm(features, dim=1, keepdim=True)
        mirror_features_norm = torch.norm(mirror_features, dim=1, keepdim=True)
        dist_matrix = (1 + features.mm(mirror_features.t()) / (1e-6 + features_norm.mm(mirror_features_norm.t()))) / 2.0
        select_dist, mirror_idx = dist_matrix.topk(k=args.mirror_top_n, dim=1, largest=True)

    if args.mirror_sample_weight:
        if args.mirror_sample_weight_version == 1:
            if args.mirror_distence_method != 'Cos':
                sample_weight = (1 / (1 + select_dist)) / (1 / (1 + select_dist)).sum(dim=1).unsqueeze(1)
            else:
                sample_weight = select_dist / select_dist.sum(dim=1).unsqueeze(1)
        else:
            if args.mirror_distence_method != 'Cos':
                sample_weight = (1 / (1 + select_dist)).softmax(dim=1)
            else:
                sample_weight = select_dist.softmax(dim=1)
    else:
        if torch.cuda.is_available():
            sample_weight = torch.cuda.FloatTensor(features.size(0), args.mirror_top_n).fill_(
                1 / args.mirror_top_n)
        else:
            sample_weight = torch.FloatTensor(features.size(0), args.mirror_top_n).fill_(
                1 / args.mirror_top_n)
    mirror_select_features = (mirror_features[mirror_idx] * sample_weight.unsqueeze(2)).sum(dim=1)

    # prob_source:
    mirror_pred = (1 + (mirror_select_features.unsqueeze(1) - last_centers.unsqueeze(0)).pow(2).sum(
        2) / args.alpha).pow(
        - (args.alpha + 1) / 2)

    if args.label_softmax:
        mirror_pred = F.softmax(mirror_pred, dim=1)
    else:
        mirror_pred = mirror_pred / mirror_pred.sum(1, keepdim=True)

    # tgt_pred
    cur_pred = (1 + (features.unsqueeze(1) - last_centers.unsqueeze(0)).pow(2).sum(
        2) / args.alpha).pow(
        - (args.alpha + 1) / 2)
    if args.label_softmax:
        cur_pred = F.softmax(cur_pred, dim=1)
    else:
        cur_pred = cur_pred / cur_pred.sum(1, keepdim=True)

    loss = mirror_pred * torch.log(
        torch.clamp(mirror_pred / torch.clamp(cur_pred, min=1e-7), min=1e-7))

    return args.mirror_loss_weight * loss.sum(1).mean()


def TarDisClusterLoss(args, epoch, output, target, softmax=True, em=False):
    if softmax:
        prob_p = F.softmax(output, dim=1)
    else:
        prob_p = output / output.sum(1, keepdim=True)
    if em:
        prob_q = prob_p
    else:

        if torch.cuda.is_available():
            prob_q1 = Variable(torch.cuda.FloatTensor(prob_p.size()).fill_(0))
            prob_q1.scatter_(1, target.unsqueeze(1), torch.ones(prob_p.size(0), 1).cuda())  # assigned pseudo labels
        else:
            prob_q1 = Variable(torch.FloatTensor(prob_p.size()).fill_(0))
            prob_q1.scatter_(1, target.unsqueeze(1), torch.ones(prob_p.size(0), 1))  # assigned pseudo labels
        if (epoch == 0) or args.ao:
            prob_q = prob_q1
        else:
            prob_q2 = prob_p / prob_p.sum(0, keepdim=True).pow(0.5)
            prob_q2 /= prob_q2.sum(1, keepdim=True)
            prob_q = (1 - args.beta) * prob_q1 + args.beta * prob_q2
    if softmax:
        loss = - (prob_q * F.log_softmax(output, dim=1)).sum(1).mean()
    else:
        loss = - (prob_q * prob_p.log()).sum(1).mean()

    return loss


def SrcClassifyLoss(args, output, target, index, src_cs, lam, softmax=True, fit=False):
    if softmax:
        prob_p = F.softmax(output, dim=1)
    else:
        prob_p = output / output.sum(1, keepdim=True)
    if torch.cuda.is_available():
        prob_q = Variable(torch.cuda.FloatTensor(prob_p.size()).fill_(0))
        prob_q.scatter_(1, target.unsqueeze(1), torch.ones(prob_p.size(0), 1).cuda())
    else:
        prob_q = Variable(torch.FloatTensor(prob_p.size()).fill_(0))
        prob_q.scatter_(1, target.unsqueeze(1), torch.ones(prob_p.size(0), 1))
    if fit:
        prob_q = (1 - prob_p) * prob_q + prob_p * prob_p

    src_weights = src_cs[index]

    if softmax:
        loss = - (src_weights * (prob_q * F.log_softmax(output, dim=1)).sum(1)).mean()
    else:
        loss = - (src_weights * (prob_q * prob_p.log()).sum(1)).mean()

    return loss


def train_epoch(train_loader_source, train_loader_source_batch, train_loader_target, train_loader_target_batch, model,
                learn_cen, learn_cen_2, last_source_features, last_source_features_2, last_target_features,
                last_target_features_2, optimizer, itern, epoch, new_epoch_flag, src_cs, src_centers, tgt_centers, args):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1_source = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    lam = 2 / (1 + math.exp(-1 * 10 * epoch / args.epochs)) - 1  # penalty parameter
    # lam = 1.0
    if args.src_cls:
        weight = lam
    else:
        weight = 1.0
    adjust_learning_rate(optimizer, epoch, args)  # adjust learning rate

    end = time.time()
    # prepare target data
    try:
        (input_target, target_target, _) = train_loader_target_batch.next()[1]
    except StopIteration:
        train_loader_target_batch = enumerate(train_loader_target)
        (input_target, target_target, _) = train_loader_target_batch.next()[1]

    input_target_var = Variable(input_target)
    target_target_var = Variable(target_target)
    if torch.cuda.is_available():
        target_target = target_target.cuda(device='cuda:0')
        input_target_var = input_target_var.cuda(device='cuda:0')
        target_target_var = target_target_var.cuda(device='cuda:0')

    # model forward on target
    f_t, f_t_2, ca_t = model(input_target_var)

    loss = 0

    if args.use_SRDC_loss:
        SRDC_loss = weight * TarDisClusterLoss(args, epoch, ca_t, target_target, em=(args.cluster_method == 'em'))
        loss += SRDC_loss
    else:
        SRDC_loss = 0

    if args.mirror_kl_loss and epoch >= args.mirror_kl_loss_start_epoch:

        mirror_kl_loss_ = 0

        tgt_mirror_kl_loss_ = mirror_kl_loss_v2(args, f_t_2, last_source_features_2, learn_cen_2) * args.mirror_loss_weight_second_feature + mirror_kl_loss_v2(
            args, f_t, last_source_features, learn_cen) * args.mirror_loss_weight_first_feature

        mirror_kl_loss_ += tgt_mirror_kl_loss_
        loss += tgt_mirror_kl_loss_
    else:
        mirror_kl_loss_ = 0

    if args.src_cls:
        # prepare source data
        try:
            (input_source, target_source, index) = train_loader_source_batch.next()[1]
        except StopIteration:
            train_loader_source_batch = enumerate(train_loader_source)
            (input_source, target_source, index) = train_loader_source_batch.next()[1]
        input_source_var = Variable(input_source)
        target_source_var = Variable(target_source)
        if torch.cuda.is_available():
            target_source = target_source.cuda()
            input_source_var = input_source_var.cuda()
            target_source_var = target_source_var.cuda()

        # model forward on source
        f_s, f_s_2, ca_s = model(input_source_var)
        prec1_s = accuracy(ca_s.data, target_source, topk=(1,))[0]
        top1_source.update(prec1_s.item(), input_source.size(0))

        src_task_loss = SrcClassifyLoss(args, ca_s, target_source, index, src_cs, lam, fit=args.src_fit)

        loss += src_task_loss

        if args.mirror_on_src:
            src_mirror_kl_loss_ = mirror_kl_loss_v2(args, f_s_2, last_target_features_2, learn_cen_2) * args.mirror_loss_weight_second_feature + mirror_kl_loss_v2(
                args, f_s, last_target_features, learn_cen) * args.mirror_loss_weight_first_feature
            mirror_kl_loss_ += src_mirror_kl_loss_
            loss += src_mirror_kl_loss_

    losses.update(loss.data.item(), input_target.size(0))

    batch_time.update(time.time() - end)
    if itern % args.print_freq == 0:
        print('Train - epoch [{0}/{1}]\t'
              'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'S@1 {s_top1.val:.3f} ({s_top1.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            .format(
            epoch, args.epochs, batch_time=batch_time,
            data_time=data_time, s_top1=top1_source, loss=losses))
        log = open(os.path.join(args.log, 'log.txt'), 'a')
        log.write(
            "\nTrain - epoch: %d, top1_s acc: %3f, loss: %4f task_loss:%.4f, mirror_loss: %.4f, SRDC_loss:%.4f" % (
            epoch, top1_source.avg, losses.avg, src_task_loss, mirror_kl_loss_, SRDC_loss
            ))
        log.close()
    if new_epoch_flag:
        print('The penalty weight is %3f' % weight)

    return train_loader_source_batch, train_loader_target_batch, src_centers, tgt_centers, optimizer, loss
