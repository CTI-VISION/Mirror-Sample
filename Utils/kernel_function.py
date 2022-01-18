# -*- coding: utf-8 -*-

"""
# @Author  : wmq
# @Software: PyCharm
# @File    : kernel_function.py
# @Time    : 2020-09-14 15:35
"""
from __future__ import division
import torch
import numpy as np


def Gaussian_Kernel(x, y, gamma=1.0, c=1.0, d=3):
    dist_matrix = ((x.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(dim=2))

    dist_matrix = torch.exp(-1 * gamma * dist_matrix)

    return dist_matrix


def Exponential_Kernel(x, y, gamma=1.0, c=1.0, d=3):
    dist_matrix = ((x.unsqueeze(1) - y.unsqueeze(0)).sum(dim=2))

    dist_matrix = torch.exp(-1 * gamma * dist_matrix)

    return dist_matrix


def Cauchy_Kernel(x, y, gamma=1.0, c=1.0, d=3):
    dist_matrix = ((x.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(dim=2))

    dist_matrix = 1 / (1 + dist_matrix * gamma)

    return dist_matrix


def Polynomial_Kernel(x, y, gamma=1.0, c=1.0, d=3):
    dist_matrix = x.mm(y.t())

    dist_matrix = torch.pow((gamma * dist_matrix + c), d)

    return dist_matrix


def Sigmoid_Kernel(x, y, gamma=1.0, c=1.0, d=3):
    dist_matrix = x.mm(y.t())

    dist_matrix = torch.tanh(gamma * dist_matrix + c)

    return dist_matrix


def Multiquadric_Kernel(x, y, gamma=1.0, c=1.0, d=3):
    dist_matrix = ((x.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(dim=2))

    dist_matrix = torch.pow((dist_matrix + c * c), 0.5)

    return dist_matrix


def cal_pca(x):
    dim = x.shape[0]
    N = x.shape[1]
    mean = x.mean(dim=1)
    x = x - mean
    x = x.detach().cpu().numpy()

    if dim <= N:
        cov_mat = np.cov(x, rowvar=True)
    else:
        cov_mat = np.cov(x, rowvar=False)

    eignvalue, featurevector = np.linalg.eig(cov_mat)

    idx = np.argsort(featurevector)
    idx = idx[::-1]
    EigenValues = featurevector[idx]
    ProjectMatrix = eignvalue[idx]

    return ProjectMatrix, EigenValues


def GFK(x, y, dim):
    Ps, pre_XA = cal_pca(x.t())
    Pt, pre_XB = cal_pca(y.t())
    Pst, _ = cal_pca(torch.cat((x, y), 0).t())

    optimal_dim = dim
