# -*- coding: utf-8 -*-

"""
four graphs used in this experiment are provided in the following paper:
    [1] Arias-Castro, Ery, Emmanuel J. Candes, and Arnaud Durand. Detection of
        an anomalous cluster in a network. The Annals of Statistics (2011).
"""

import os
import time
import pickle
import random
import numpy as np
import multiprocessing
from itertools import product
from numpy.random import normal

from os import sys, path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from algo_wrapper.base import node_pre_rec_fm
from algo_wrapper.base import logistic_predict
from algo_wrapper.base import simu_graph_rectangle
from algo_wrapper.algo_wrapper import algo_online_rda_l1
from algo_wrapper.algo_wrapper import algo_online_da_gl
from algo_wrapper.algo_wrapper import algo_online_adam
from algo_wrapper.algo_wrapper import algo_online_da_sgl
from algo_wrapper.algo_wrapper import algo_online_ada_grad
from algo_wrapper.algo_wrapper import algo_online_da_iht
from algo_wrapper.algo_wrapper import algo_online_sto_iht
from algo_wrapper.algo_wrapper import algo_online_best_subset

# see the reference in [1]
bench_data = {
    # figure 1 in [1], it has 26 nodes.
    'fig_1': [475, 505, 506, 507, 508, 509, 510, 511, 512, 539, 540, 541, 542,
              543, 544, 545, 576, 609, 642, 643, 644, 645, 646, 647, 679, 712],
    # figure 2 in [1], it has 46 nodes.
    'fig_2': [439, 440, 471, 472, 473, 474, 504, 505, 506, 537, 538, 539, 568,
              569, 570, 571, 572, 600, 601, 602, 603, 604, 605, 633, 634, 635,
              636, 637, 666, 667, 668, 698, 699, 700, 701, 730, 731, 732, 733,
              763, 764, 765, 796, 797, 798, 830],
    # figure 3 in [1], it has 92 nodes.
    'fig_3': [151, 183, 184, 185, 217, 218, 219, 251, 252, 285, 286, 319, 320,
              352, 353, 385, 386, 405, 406, 407, 408, 409, 419, 420, 437, 438,
              439, 440, 441, 442, 443, 452, 453, 470, 471, 475, 476, 485, 486,
              502, 503, 504, 507, 508, 509, 518, 519, 535, 536, 541, 550, 551,
              568, 569, 583, 584, 601, 602, 615, 616, 617, 635, 636, 648, 649,
              668, 669, 670, 680, 681, 702, 703, 704, 711, 712, 713, 736, 737,
              738, 739, 740, 741, 742, 743, 744, 745, 771, 772, 773, 774, 775,
              776],
    # figure 4 in [1], it has 132 nodes.
    'fig_4': [244, 245, 246, 247, 248, 249, 254, 255, 256, 277, 278, 279, 280,
              281, 282, 283, 286, 287, 288, 289, 290, 310, 311, 312, 313, 314,
              315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 343, 344, 345,
              346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 377,
              378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390,
              411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423,
              448, 449, 450, 451, 452, 453, 454, 455, 456, 481, 482, 483, 484,
              485, 486, 487, 488, 489, 514, 515, 516, 517, 518, 519, 520, 521,
              547, 548, 549, 550, 551, 552, 553, 579, 580, 581, 582, 583, 584,
              585, 586, 613, 614, 615, 616, 617, 618, 646, 647, 648, 649, 650,
              680, 681],
    'mu_list': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    # 4 figures.
    'fig_list': [('fig_1', 1), ('fig_2', 2), ('fig_3', 3), ('fig_4', 4)],
    # grid size (length).
    'length': 33,
    # grid width (width).
    'width': 33,
    # the dimension of grid graph is 33 x 33.
    'p': 33 * 33,
    # sparsity list of these 4 figures.
    's': {'fig_1': 26, 'fig_2': 46, 'fig_3': 92, 'fig_4': 132},
    # positive label.
    'posi_label': 1,
    # negative label.
    'nega_label': -1,
    'noise_mu': 0.0,
    'noise_std': 1.0,
    'num_tr': 1000,
    'num_va': 400,
    'num_te': 400,
    'num_trials': 1  # TODO here, we only try one trial
}


def get_para_space(fig_i):
    model_paras = {
        # baseline-01
        'rda-l1': {
            # lambda: to control the sparsity, original paper: [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.]
            'lambda_list': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 3e-2, 5e-2, 1e-1,
                            3e-1, 5e-1, 1e0, 3e0, 5e0, 1e1],
            # gamma: to control the learning rate. original paper [5000.]
            'gamma_list': [1e0, 5e0, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4],
            # rho: to control the sparsity-enhancing parameter. original paper is: [0.0, 0.005]
            'rho_list': [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2,
                         1e-1, 5e-1, 1e0]},
        # baseline-02
        'da-gl': {
            # lambda: to control the sparsity
            'lambda_list': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 3e-2, 5e-2, 1e-1,
                            3e-1, 5e-1, 1e0, 3e0, 5e0, 1e1],
            # gamma: to control the learning rate.
            'gamma_list': [1e0, 5e0, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4]},
        # baseline-03
        'da-sgl': {
            # lambda: to control the sparsity
            'lambda_list': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 3e-2, 5e-2, 1e-1,
                            3e-1, 5e-1, 1e0, 3e0, 5e0, 1e1],
            # gamma: to control the learning rate.
            'gamma_list': [1e0, 5e0, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4]},
        # baseline-04
        'adagrad': {
            # lambda: to control the sparsity
            'lambda_list': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 3e-2, 5e-2, 1e-1,
                            3e-1, 5e-1, 1e0, 3e0, 5e0, 1e1],
            # eta: to control the learning rate.
            'eta_list': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0,
                         5e0, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3],
            # to avoid divided by zero.
            'epsilon_list': [1e-8]},
        # baseline-05
        'sto-iht': {
            # s: to control the sparsity
            's_list': [5, 10, 15, 20, 25, 26, 30, 35, 40, 45, 46, 50, 55, 60,
                       65, 70, 75, 80, 85,
                       90, 92, 95, 100, 105, 110, 115, 120, 125, 130, 132, 135,
                       140, 145, 150],
            # lr: to control the learning rate of sto-iht
            'lr_list': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1,
                        5e-1],
            # l2-regularization parameter.
            'l2_lambda_list': [0e0]},
        # baseline-06
        'da-iht': {
            's_list': [5, 10, 15, 20, 25, 26, 30, 35, 40, 45, 46, 50, 55, 60,
                       65, 70, 75, 80, 85,
                       90, 92, 95, 100, 105, 110, 115, 120, 125, 130, 132, 135,
                       140, 145, 150],
            # gamma: to control the learning rate.
            'gamma_list': [1e0, 5e0, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4],
            # l2-regularization parameter.
            'l2_lambda_list': [0.0]},
        # baseline-07
        'adam': {
            # alpha: to control the learning rate
            'alpha_list': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1],
            # the parameter for the first moment.
            'beta1_list': [0.9],
            # the parameter for the second moment.
            'beta2_list': [0.999],
            # the parameter to avoid divided by zero.
            'epsilon_list': [1e-8]},
        'best-subset': {
            # gamma: to control the learning rate.
            'gamma_list': [1e0, 5e0, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4],
            # the parameter of l2 regularization parameter.
            'l2_lambda_list': [0.0],
            # the best subset of features
            'best_subset': np.asarray(bench_data[fig_i], dtype=np.int32),
            # sparsity list
            's_list': [len(bench_data[fig_i])]},
        # baseline-08
        'graph-sto-iht': {
            's_list': [5, 10, 15, 20, 25, 26, 30, 35, 40, 45, 46, 50, 55, 60,
                       65, 70, 75, 80, 85,
                       90, 92, 95, 100, 105, 110, 115, 120, 125, 130, 132, 135,
                       140, 145, 150],
            'l2_lambda_list': [0.0],
            # to control the learning rate
            'lr_list': [1e-3, 1e-2, 1e-1, 1e0],
            'ratio_list': [0.1],
            'max_num_iter_list': [20],
            'num_clusters_list': [1]},
        'graph-da-iht': {
            's_list': [5, 10, 15, 20, 25, 26, 30, 35, 40, 45, 46, 50, 55, 60,
                       65, 70, 75, 80, 85,
                       90, 92, 95, 100, 105, 110, 115, 120, 125, 130, 132, 135,
                       140, 145, 150],
            'l2_lambda_list': [0.0],
            # to control the learning rate.
            'gamma_list': [1e0, 5e0, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4],
            'ratio_list': [0.1],
            'max_num_iter_list': [20],
            'num_clusters_list': [1]}
    }
    return model_paras


def show_figures_data(root, data, mu, fig_id):
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from pylab import rcParams
    font = {'size': 14}
    plt.rc('font', **font)
    plt.rc('font', **font)
    rc('text', usetex=True)
    rcParams['figure.figsize'] = 12, 4
    fig, ax = plt.subplots(1, 3)
    w = np.zeros(data['p'])
    w[data['subgraph']] = mu
    length, width = bench_data['length'], bench_data['width']
    for i in range(len(data['x_tr'])):
        if data['y_tr'][i] == 1:
            ax[0].imshow(w.reshape(length, width))
            ax[0].set_title('Graph G(V,E)')
            break
    for i in range(len(data['x_tr'])):
        if data['y_tr'][i] == 1:
            ax[1].imshow(data['x_tr'][i].reshape(length, width))
            ax[1].set_title('Positive label(mu=%.1f)' % mu)
            break
    for i in range(len(data['x_tr'])):
        if data['y_tr'][i] == -1:
            ax[2].imshow(data['x_tr'][i].reshape(length, width))
            ax[2].set_title('Negative label(mu=%.1f)' % mu)
            break
    fig = plt.gcf()
    fig.set_figheight(4.0)
    fig.set_figwidth(12.0)
    if not os.path.exists(root + 'figs/data_%s_%s.pdf' % (fig_id, str(mu))):
        fig.savefig(root + 'figs/data_%s_%s.pdf' % (fig_id, str(mu)),
                    dpi=600, bbox_inches='tight', pad_inches=0, format='pdf')
    plt.close()


def generate_grid_data(width, height, num_tr, num_va, num_te, noise_mu,
                       noise_std, mu, sub_graph, fig_id, trial_i, num_trials,
                       mu_strategy='constant', show_fig=False):
    p = int(width * height)
    posi_label = bench_data['posi_label']
    nega_label = bench_data['nega_label']
    edges, weis = simu_graph_rectangle(width, height)  # get grid graph
    s, n = len(sub_graph), (num_tr + num_va + num_te)
    num_posi, num_nega = n / 2, n / 2
    # generate training samples and labels
    labels = [posi_label] * num_posi + [nega_label] * num_nega
    y_labels = np.asarray(labels, dtype=np.float64)
    x_data = normal(noise_mu, noise_std, n * p).reshape(n, p)
    if mu_strategy == 'constant':
        _ = s * num_posi
        anomalous_data = normal(mu, noise_std, _).reshape(num_posi, s)
    else:
        anomalous_data = np.zeros(shape=(num_posi, s))
        for ind, rand_ in enumerate(normal(0.0, 1.0, s)):
            if rand_ < 0.0:
                anomalous_data[:, ind] = normal(-mu, 1.0, num_posi)
            else:
                anomalous_data[:, ind] = normal(mu, 1.0, num_posi)
    x_data[:num_posi, sub_graph] = anomalous_data
    rand_indices = np.random.permutation(len(y_labels))
    x_tr, y_tr = x_data[rand_indices], y_labels[rand_indices]
    # normalize data by z-score
    x_mean = np.tile(np.mean(x_tr, axis=0), (len(x_tr), 1))
    x_std = np.tile(np.std(x_tr, axis=0), (len(x_tr), 1))
    x_tr = np.nan_to_num(np.divide(x_tr - x_mean, x_std))
    data = {'x_tr': x_tr[:num_tr],
            'y_tr': y_tr[:num_tr],
            'x_va': x_tr[num_tr:num_tr + num_va],
            'y_va': y_tr[num_tr:num_tr + num_va],
            'x_te': x_tr[num_tr + num_va:],
            'y_te': y_tr[num_tr + num_va:],
            'subgraph': sub_graph,
            'edges': edges,
            'weights': weis,
            'mu': mu,
            'noise_mu': noise_mu,
            'noise_std': noise_std,
            'mu_strategy': mu_strategy,
            'fig_id': fig_id,
            'trial_i': trial_i,
            'num_trials': num_trials,
            'num_tr': num_tr,
            'num_va': num_va,
            'num_te': num_te,
            'p': p}
    if show_fig:
        show_figures_data('', data, mu, fig_id)
    return data


def get_best_re(current, best_acc, tag):
    # first time to have the result.
    if best_acc is None:
        best_acc = current
    # select by acc, if it is equal, # just choose the one which has less missed samples.
    cond_1 = best_acc['va_acc_%s' % tag] < current['va_acc_%s' % tag]
    cond_2 = best_acc['va_acc_%s' % tag] == current['va_acc_%s' % tag]
    cond_3 = best_acc['missed_%s' % tag][-1] > current['missed_%s' % tag][-1]
    if cond_1 or (cond_2 and cond_3):
        best_acc = current
    return best_acc


def generate_re(para, result, algo_para):
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import roc_curve
    wt, wt_bar, subgraph = result['wt'], result['wt_bar'], para['subgraph']
    x_te, x_va = para['x_te'], para['x_va']
    y_te, y_va = para['y_te'], para['y_va']

    # to use model wt
    te_pred_prob_wt, te_pred_y_wt = logistic_predict(x_te, wt)
    va_pred_prob_wt, va_pred_y_wt = logistic_predict(x_va, wt)
    va_acc_wt = accuracy_score(y_true=y_va, y_pred=va_pred_y_wt)
    te_acc_wt = accuracy_score(y_true=y_te, y_pred=te_pred_y_wt)
    va_auc_wt = roc_auc_score(y_true=y_va, y_score=va_pred_y_wt)
    te_auc_wt = roc_auc_score(y_true=y_te, y_score=te_pred_y_wt)
    va_roc_wt = roc_curve(y_true=y_va, y_score=va_pred_prob_wt)
    te_roc_wt = roc_curve(y_true=y_te, y_score=te_pred_prob_wt)
    pred_nodes_wt = np.nonzero(wt[:para['p']])[0]
    n_pre_rec_fm_wt = node_pre_rec_fm(subgraph, pred_nodes_wt)
    node_fm_wt = n_pre_rec_fm_wt[2]
    te_acc_node_fm_wt = 2. * (te_acc_wt * node_fm_wt) / (
            te_acc_wt + node_fm_wt)
    va_acc_node_fm_wt = 2. * (va_acc_wt * node_fm_wt) / (
            va_acc_wt + node_fm_wt)

    # to use model wt_bar
    te_pred_prob_wt_bar, te_pred_y_wt_bar = logistic_predict(x_te, wt_bar)
    va_pred_prob_wt_bar, va_pred_y_wt_bar = logistic_predict(x_va, wt_bar)
    va_acc_wt_bar = accuracy_score(y_true=y_va, y_pred=va_pred_y_wt_bar)
    te_acc_wt_bar = accuracy_score(y_true=y_te, y_pred=te_pred_y_wt_bar)
    va_auc_wt_bar = roc_auc_score(y_true=y_va, y_score=va_pred_y_wt_bar)
    te_auc_wt_bar = roc_auc_score(y_true=y_te, y_score=te_pred_y_wt_bar)
    va_roc_wt_bar = roc_curve(y_true=y_va, y_score=va_pred_prob_wt_bar)
    te_roc_wt_bar = roc_curve(y_true=y_te, y_score=te_pred_prob_wt_bar)
    pred_nodes_wt_bar = np.nonzero(wt_bar[:para['p']])[0]
    n_pre_rec_fm_wt_bar = node_pre_rec_fm(subgraph, pred_nodes_wt_bar)
    node_fm_wt_bar = n_pre_rec_fm_wt_bar[2]
    va_acc_node_fm_wt_bar = 2. * (va_acc_wt_bar * node_fm_wt_bar) / (
            va_acc_wt_bar + node_fm_wt_bar)
    te_acc_node_fm_wt_bar = 2. * (te_acc_wt_bar * node_fm_wt_bar) / (
            te_acc_wt_bar + node_fm_wt_bar)
    re = {'mu': para['mu'],
          'fig_id': para['fig_id'],
          'subgraph': para['subgraph'],
          'trial_i': para['trial_i'],
          'n_tr': para['num_tr'],
          'p': para['p'],
          'method_name': '',
          'para': algo_para,

          # validation metric results for model wt
          'va_auc_wt': va_auc_wt,
          'va_acc_wt': va_acc_wt,
          'va_roc_wt': va_roc_wt,
          'va_node_fm_wt': node_fm_wt,
          'va_acc_node_fm_wt': va_acc_node_fm_wt,

          # test metric results for model wt
          'te_auc_wt': te_auc_wt,
          'te_acc_wt': te_acc_wt,
          'te_roc_wt': te_roc_wt,
          'te_node_fm_wt': node_fm_wt,
          'te_acc_node_fm_wt': te_acc_node_fm_wt,
          'te_n_pre_rec_fm_wt': n_pre_rec_fm_wt,

          # validation metric results for model wt_bar
          'va_auc_wt_bar': va_auc_wt_bar,
          'va_acc_wt_bar': va_acc_wt_bar,
          'va_roc_wt_bar': va_roc_wt_bar,
          'va_node_fm_wt_bar': node_fm_wt_bar,
          'va_acc_node_fm_wt_bar': va_acc_node_fm_wt_bar,

          # test metric results for model wt_bar
          'te_auc_wt_bar': te_auc_wt_bar,
          'te_acc_wt_bar': te_acc_wt_bar,
          'te_roc_wt_bar': te_roc_wt_bar,
          'te_node_fm_wt_bar': node_fm_wt_bar,
          'te_acc_node_fm_wt_bar': te_acc_node_fm_wt_bar,
          'te_n_pre_rec_fm_wt_bar': n_pre_rec_fm_wt_bar,

          # some result parameters
          'wt': result['wt'],  # including intercept
          'wt_bar': result['wt_bar'],  # including intercept
          'losses': result['losses'],
          'run_time': result['total_time'],
          'missed_wt': result['missed_wt'],
          'missed_wt_bar': result['missed_wt_bar']}
    return re


# regularized dual averaging.
def algo_rda_l1(data):
    results = []
    para = data['para_space']['rda-l1']
    for (lambda_, gamma, rho) in product(para['lambda_list'],
                                         para['gamma_list'], para['rho_list']):
        result = algo_online_rda_l1(
            data['x_tr'], data['y_tr'], np.zeros(data['p'] + 1), lambda_,
            gamma, rho, 'logistic', 0)
        results.append(generate_re(data, result, (lambda_, gamma, rho)))
    return results


# regularized dual averaging.
def algo_da_gl(data):
    def group_select():
        group_list_ = []
        cube = [0, 1, 2, 33, 34, 35, 66, 67, 68]
        width, height = 33, int(data['p'] / 33)
        sub_group = np.asarray(cube, dtype=np.int32)
        num_group_ = int(width / 3) * int(height / 3)
        for i in range(int(height / 3)):
            for j in range(int(width / 3)):
                group_list_.extend(sub_group + j * 3 + i * 3 * 33)
        group_list_ = np.asarray(group_list_, dtype=np.int32)
        group_size_list_ = np.asarray([9] * 11 * 11, dtype=np.int32)
        return group_list_, group_size_list_, num_group_

    results = []
    para = data['para_space']['da-gl']
    group_list, group_size_list, num_group = group_select()
    for (lambda_, gamma) in product(para['lambda_list'], para['gamma_list']):
        result = algo_online_da_gl(
            data['x_tr'], data['y_tr'], np.zeros(data['p'] + 1), lambda_,
            gamma, group_list, group_size_list, num_group, 'logistic', 0)
        results.append(generate_re(data, result, (lambda_, gamma)))
    return results


# regularized dual averaging.
def algo_da_sgl(data):
    def group_select():
        group_list_ = []
        cube = [0, 1, 2, 33, 34, 35, 66, 67, 68]
        width, height = 33, int(data['p'] / 33)
        sub_group = np.asarray(cube, dtype=np.int32)
        num_group_ = int(width / 3) * int(height / 3)
        for i in range(int(height / 3)):
            for j in range(int(width / 3)):
                group_list_.extend(sub_group + j * 3 + i * 3 * 33)
        group_list_ = np.asarray(group_list_, dtype=np.int32)
        group_size_list_ = np.asarray([9] * 11 * 11, dtype=np.int32)
        return group_list_, group_size_list_, num_group_

    results = []
    para = data['para_space']['da-sgl']
    group_list, group_size_list, num_group = group_select()
    r = np.asarray([1.] * num_group, dtype=np.int32)
    for (lambda_, gamma) in product(para['lambda_list'], para['gamma_list']):
        result = algo_online_da_sgl(
            data['x_tr'], data['y_tr'], np.zeros(data['p'] + 1), lambda_,
            gamma, group_list, group_size_list, r, num_group, 'logistic', 0)
        results.append(generate_re(data, result, (lambda_, gamma)))
    return results


# regularized dual averaging.
def algo_adam(data):
    results = []
    para = data['para_space']['adam']
    for (alpha, beta1, beta2, epsilon) in product(
            para['alpha_list'], para['beta1_list'], para['beta2_list'],
            para['epsilon_list']):
        result = algo_online_adam(
            data['x_tr'], data['y_tr'], np.zeros(data['p'] + 1), alpha, beta1,
            beta2, epsilon, 'logistic', 0)
        results.append(generate_re(data, result, alpha))
    return results


def algo_adagrad(data):
    results = []
    para = data['para_space']['adagrad']
    for (lambda_, eta, epsilon) in product(para['lambda_list'],
                                           para['eta_list'],
                                           para['epsilon_list']):
        result = algo_online_ada_grad(
            data['x_tr'], data['y_tr'], np.zeros(data['p'] + 1), lambda_, eta,
            epsilon, 'logistic', 0)
        results.append(generate_re(data, result, (lambda_, eta)))
    return results


def algo_best_subset(data):
    results = []
    para = data['para_space']['best-subset']
    best_subset = para['best_subset']
    for (s, gamma, l2_lambda) in product(para['s_list'], para['gamma_list'],
                                         para['l2_lambda_list']):
        result = algo_online_best_subset(
            data['x_tr'], data['y_tr'], np.zeros(data['p'] + 1), best_subset,
            gamma, l2_lambda, s, 'logistic', 0)
        results.append(generate_re(data, result, (s, gamma, l2_lambda)))
    return results


def algo_sto_iht(data):
    results = []
    para = data['para_space']['sto-iht']
    for (lr, l2_lambda, s) in product(para['lr_list'], para['l2_lambda_list'],
                                      para['s_list']):
        result = algo_online_sto_iht(
            data['x_tr'], data['y_tr'], np.zeros(data['p'] + 1), lr, l2_lambda,
            s, 'logistic', 0)
        results.append(generate_re(data, result, (lr, l2_lambda)))
    return results


def algo_da_iht(data):
    results = []
    para = data['para_space']['da-iht']
    for (gamma, l2_lambda, s) in product(para['gamma_list'],
                                         para['l2_lambda_list'],
                                         para['s_list']):
        result = algo_online_da_iht(
            data['x_tr'], data['y_tr'], np.zeros(data['p'] + 1), gamma,
            l2_lambda, s, 'logistic', 0)
        results.append(generate_re(data, result, (gamma, l2_lambda, s)))
    return results


def algo_graph_sto_iht(data, para_pair):
    s, l2_lambda, lr, ratio, max_num_iter, num_clusters = para_pair
    result = algo_online_graph_sto_iht(
        data['x_tr'], data['y_tr'], np.zeros(data['p'] + 1),
        lr, l2_lambda, data['edges'], data['weights'], num_clusters, s, ratio,
        max_num_iter, 'logistic', 0)
    para = (s, l2_lambda, lr, ratio, max_num_iter, num_clusters)
    re = generate_re(data, result, para)
    return [re]


def algo_graph_da_iht(data, para_pair):
    s, l2_lambda, gamma, ratio, max_num_iter, num_clusters = para_pair
    result = algo_online_graph_da(
        data['x_tr'], data['y_tr'], np.zeros(data['p'] + 1),
        gamma, l2_lambda, data['edges'], data['weights'], num_clusters, s,
        ratio, max_num_iter, 'logistic', 0)
    para = (gamma, l2_lambda, s, ratio, max_num_iter, num_clusters)
    re = generate_re(data, result, para)
    return [re]


def run_single_method(para):
    data, method, para_pair = para
    print('testing: %s' % method)
    if method == 'rda-l1':
        results = algo_rda_l1(data)
    elif method == 'sto-iht':
        results = algo_sto_iht(data)
    elif method == 'da-iht':
        results = algo_da_iht(data)
    elif method == 'da-gl':
        results = algo_da_gl(data)
    elif method == 'adam':
        results = algo_adam(data)
    elif method == 'da-sgl':
        results = algo_da_sgl(data)
    elif method == 'adagrad':
        results = algo_adagrad(data)
    elif method == 'graph-sto-iht':
        results = algo_graph_sto_iht(data, para_pair)
    elif method == 'graph-da-iht':
        results = algo_graph_da_iht(data, para_pair)
    else:
        results = None
    return [(re, method) for re in results]


def get_input_paras(data, method_list):
    input_paras = []
    for method in method_list:
        if method != 'graph-da-iht' and method != 'graph-sto-iht':
            input_paras.append((data, method, 'none'))
    para = data['para_space']['graph-da-iht']
    for para_pair in product(
            para['s_list'],
            para['l2_lambda_list'],
            para['gamma_list'],
            para['ratio_list'],
            para['max_num_iter_list'],
            para['num_clusters_list']):
        input_paras.append((data, 'graph-da-iht', para_pair))
    para = data['para_space']['graph-sto-iht']
    for para_pair in product(
            para['s_list'],
            para['l2_lambda_list'],
            para['lr_list'],
            para['ratio_list'],
            para['max_num_iter_list'],
            para['num_clusters_list']):
        input_paras.append((data, 'graph-sto-iht', para_pair))
    return input_paras


def model_selection(candidate_results):
    acc_wt = None
    acc_wt_bar = None
    for re in candidate_results:
        acc_wt = get_best_re(re, acc_wt, 'wt')
        acc_wt_bar = get_best_re(re, acc_wt_bar, 'wt_bar')
    return {'best-acc-wt': acc_wt,
            'best-acc-wt-bar': acc_wt_bar}


def get_results_pool(results_pool, data, method_list):
    summary_results = []
    for method in method_list:
        results = [_[0] for _ in results_pool if _[1] == method]
        summary_results.append([model_selection(results), method])
    method_labels = {'rda-l1': 'RDA-L1',
                     'da-iht': 'DA-IHT',
                     'adagrad': 'AdaGrad',
                     'sto-iht': 'StoIHT',
                     'graph-sto-iht': 'GraphStoIHT',
                     'da-gl': 'DA-GL',
                     'da-sgl': 'DA-SGL',
                     'adam': 'ADAM',
                     'graph-da-iht': 'GraphDA-IHT',
                     'best-subset': 'BestSubset'}
    out = {'summary_results': summary_results,
           'trial_i': data['trial_i'],
           'method_list': method_list,
           'method_labels': method_labels,
           'noise_mu': data['noise_mu'],
           'noise_std': data['noise_std'],
           'mu': data['mu'],
           'mu_strategy': data['mu_strategy']}
    return out


def run_exp_fix_tr_mu(root, trial_i, num_cpus):
    """ experiment on a fixed dataset.
    :param root:
    :param trial_i:
    :param num_cpus:
    :return:
    """
    method_list = ['rda-l1', 'da-iht', 'adagrad', 'sto-iht',
                   'graph-sto-iht', 'da-gl', 'da-sgl', 'adam', 'graph-da-iht']
    root_in = root + 'input/'
    mu, num_tr = 0.3, 400
    for (fig_i, fig_ind) in bench_data['fig_list']:
        start_time = time.time()
        f_name = root_in + 'fig%d_mu_%.1f_trial_%02d.pkl' % (
            fig_ind, mu, trial_i)
        if not os.path.exists(f_name):
            print('cannot find file: %s' % f_name)
            exit(0)
        data = pickle.load(open(f_name))
        data['para_space'] = get_para_space(fig_i)
        data['x_tr'] = data['x_tr'][:num_tr]
        data['y_tr'] = data['y_tr'][:num_tr]
        pool = multiprocessing.Pool(processes=num_cpus)
        input_paras = get_input_paras(data, method_list)
        results_pool = pool.map(run_single_method, input_paras)
        results_pool = [_ for results in results_pool for _ in results]
        pool.close()
        pool.join()
        print('run time: %.4f seconds of trail: %d' % (
            time.time() - start_time, trial_i))
        out = get_results_pool(results_pool, data, method_list)
        f_name = '../results/benchmark/fix_tr_mu/fig%d_mu_%.1f_trial_%02d_fix_tr_mu.pkl' % (
            fig_ind, mu, trial_i)
        pickle.dump(out, open(f_name, 'wb'))


def run_exp_diff_tr(root, trial_i, num_cpus):
    """
    experiment on performance w.r.t different sample size.
    :param root: root path
    :param trial_i: i-th trial
    :param num_cpus: number of cpus.
    :return:
    """
    method_list = ['rda-l1', 'da-iht', 'adagrad', 'sto-iht', 'graph-sto-iht',
                   'da-gl', 'da-sgl', 'adam', 'graph-da-iht']
    root_in = root + 'input/'
    mu, num_trials = 0.3, 20
    num_tr_list = [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100]
    for (fig_i, fig_ind) in bench_data['fig_list']:
        f_name = root_in + 'fig%d_mu_%.1f_trial_%02d.pkl' % (
            fig_ind, mu, trial_i)
        if not os.path.exists(f_name):
            print('cannot find file: %s' % f_name)
            exit(0)
        data = pickle.load(open(f_name))
        data['para_space'] = get_para_space(fig_i)
        all_results = dict()
        for num_tr in num_tr_list:  # use subset of training samples.
            data['num_tr'] = num_tr
            data['x_tr'] = data['x_tr'][:num_tr]
            data['y_tr'] = data['y_tr'][:num_tr]
            start_time = time.time()
            pool = multiprocessing.Pool(processes=num_cpus)
            input_paras = get_input_paras(data, method_list)
            results_pool = pool.map(run_single_method, input_paras)
            results_pool = [_ for results in results_pool for _ in results]
            pool.close()
            pool.join()
            print('run time: %.4f seconds of trail: %d with n: %d' % (
                time.time() - start_time, trial_i, num_tr))
            out = get_results_pool(results_pool, data, method_list)
            all_results[num_tr] = out
        f_name = '../results/benchmark/diff_tr/fig%d_mu_%.1f_trial_%02d_diff_tr.pkl' % (
            fig_ind, mu, trial_i)
        pickle.dump(all_results, open(f_name, 'wb'))


def run_exp_diff_mu(root, trial_i, num_cpus):
    """
    # experiment on performance w.r.t different sample size.
    :param root:
    :param trial_i:
    :param num_cpus:
    :return:
    """
    method_list = ['rda-l1', 'da-iht', 'adagrad', 'sto-iht', 'graph-sto-iht',
                   'da-gl', 'da-sgl', 'adam',
                   'graph-da-iht']
    root_in = root + 'input/'
    num_tr, num_va, num_te = 400, 400, 400
    mu_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    for (fig_i, fig_ind) in bench_data['fig_list']:
        for mu in mu_list:
            f_name = root_in + 'fig%d_mu_%.1f_trial_%02d.pkl' % (
                fig_ind, mu, trial_i)
            if not os.path.exists(f_name):
                print('cannot find file: %s, please generate dataset' % f_name)
                exit(0)
            data = pickle.load(open(f_name))
            data['para_space'] = get_para_space(fig_i)
            data['x_tr'] = data['x_tr'][:num_tr]
            data['y_tr'] = data['y_tr'][:num_tr]
            pool = multiprocessing.Pool(processes=num_cpus)
            input_paras = get_input_paras(data, method_list)
            results_pool = pool.map(run_single_method, input_paras)
            results_pool = [_ for results in results_pool for _ in results]
            pool.close()
            pool.join()
            out = get_results_pool(results_pool, data, method_list)
            f = '../results/benchmark/diff_mu/fig%d_mu_%.1f_trial_%02d_diff_mu.pkl' % (
                fig_ind, mu, trial_i)
            print('save results to: %s' % f)
            pickle.dump(out, open(f, 'wb'))


def run_exp_diff_s(root, trial_i, num_cpus):
    method_list, mu = ['adam', 'graph-da-iht'], 0.3
    num_tr, num_va, num_te = 400, 400, 400
    figure_sparsity_list = {'fig_1': range(2, 60, 3),
                            'fig_2': range(5, 101, 5),
                            'fig_3': range(10, 201, 10),
                            'fig_4': range(15, 263, 13)}
    root_in = root + 'input/'
    for (fig_i, i) in bench_data['fig_list']:
        f_name = root_in + 'fig%d_mu_%.1f_trial_%02d.pkl' % (i, mu, trial_i)
        data = pickle.load(open(f_name))
        data['num_tr'] = num_tr
        data['x_tr'] = data['x_tr'][:num_tr]
        data['y_tr'] = data['y_tr'][:num_tr]
        print('load file: %s ' % f_name)
        if not os.path.exists(f_name):
            print('cannot find file: %s' % f_name)
            exit(0)
        all_results = dict()
        for s in figure_sparsity_list[fig_i]:
            para = {
                'adam': {
                    # lambda: to control the sparsity
                    'alpha_list': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1,
                                   5e-1],
                    # the parameter for the first moment.
                    'beta1_list': [0.9],
                    # the parameter for the second moment.
                    'beta2_list': [0.999],
                    # the parameter to avoid divided by zero.
                    'epsilon_list': [1e-8]},
                'graph-da-iht': {
                    's_list': [s],
                    # to control l2-regularization parameter
                    'l2_lambda_list': [0.0],
                    # to control the learning rate.
                    'gamma_list': [1e0, 5e0, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3,
                                   1e4],
                    # head-tail budget para
                    'ratio_list': [0.1],
                    'max_num_iter_list': [20],
                    'num_clusters_list': [1]}}
            data['para_space'] = para
            start_time = time.time()
            pool = multiprocessing.Pool(processes=num_cpus)
            input_paras = [(data, 'adam', 'none')]
            para = data['para_space']['graph-da-iht']
            for para_pair in product(
                    para['s_list'],
                    para['l2_lambda_list'],
                    para['gamma_list'],
                    para['ratio_list'],
                    para['max_num_iter_list'],
                    para['num_clusters_list']):
                input_paras.append((data, 'graph-da-iht', para_pair))
            results_pool = pool.map(run_single_method, input_paras)
            results_pool = [_ for results in results_pool for _ in results]
            pool.close()
            pool.join()
            print('run time: %.4f seconds of trail: %d with s: %d' % (
                time.time() - start_time, trial_i, s))
            out = get_results_pool(results_pool, data, method_list)
            all_results[s] = out
        f_name = '../results/benchmark/diff_s/fig%d_mu_%.1f_trial_%02d_diff_s.pkl' % (
            i, mu, trial_i)
        print('save file: %s' % f_name)
        pickle.dump(all_results, open(f_name, 'wb'))


def print_result(method_labels, all_methods, all_results):
    # model selected by best accuracy.
    for metric in ['best-acc']:
        print('-' * 100)
        print('-' * 36 + ' selected by %s ' % metric + '-' * 36)
        print(
            'Method & \\textit{Pre}$\pm$std & \\textit{Rec}$\pm$std & \\textit{F1}$\pm$std & '
            '${AUC}_{{\\bf w}_t,{\\bar{\\bf w}}_t}$ & '
            '${Acc}_{{\\bf w}_t,{\\bar{\\bf w}}_t}$ & '
            '${Miss}_{{\\bf w}_t,{\\bar{\\bf w}}_t}$ & '
            '${NN}_{{\\bf w}_t,{\\bar{\\bf w}}_t}$\\\\')
        for method in all_methods:
            best_n_fm_wt = [re[0]['%s-wt' % metric] for case in all_results for
                            re in case if re[1] == method]
            p = best_n_fm_wt[0]['p']
            best_n_fm_wt_bar = [re[0]['%s-wt-bar' % metric] for case in
                                all_results for re in case if re[1] == method]
            node_pre = [_['te_n_pre_rec_fm_wt'][0] for _ in best_n_fm_wt]
            node_rec = [_['te_n_pre_rec_fm_wt'][1] for _ in best_n_fm_wt]
            node_fm = [_['te_n_pre_rec_fm_wt'][2] for _ in best_n_fm_wt]
            auc_wt = [_['te_auc_wt'] for _ in best_n_fm_wt]
            auc_wt_bar = [_['te_auc_wt_bar'] for _ in best_n_fm_wt_bar]
            acc_wt = [_['te_acc_wt'] for _ in best_n_fm_wt]
            acc_wt_bar = [_['te_acc_wt_bar'] for _ in best_n_fm_wt_bar]
            missed_wt = [_['missed_wt'][-1] for _ in best_n_fm_wt]
            missed_wt_bar = [_['missed_wt_bar'][-1] for _ in best_n_fm_wt]
            nonzero_ratio1 = [
                float(len(np.nonzero(_['wt'][:p])[0])) * 100. / float(p) for _
                in best_n_fm_wt]
            nonzero_ratio2 = [
                float(len(np.nonzero(_['wt_bar'][:p])[0])) * 100. / float(p)
                for _ in best_n_fm_wt]
            if nonzero_ratio1 == 1.0 and nonzero_ratio2 == 1.0:
                p_str = '%12s & %.3f$\pm$%.2f & %.3f$\pm$%.2f & %.3f$\pm$%.2f & (%.3f, %.3f) & (%.3f, ' \
                        '%.3f) & (%3.2f, %3.2f) & (%2.2f\\%%, %2.2f\\%%) \\\\'
            elif nonzero_ratio1 == 1.0 and nonzero_ratio2 != 1.0:
                p_str = '%12s & %.3f$\pm$%.2f & %.3f$\pm$%.2f & %.3f$\pm$%.2f & (%.3f, %.3f) & (%.3f, ' \
                        '%.3f) & (%3.2f, %3.2f) & (%2.2f\\%%, %2.2f\\%%) \\\\'
            elif nonzero_ratio1 != 1.0 and nonzero_ratio2 == 1.0:
                p_str = '%12s & %.3f$\pm$%.2f & %.3f$\pm$%.2f & %.3f$\pm$%.2f & (%.3f, %.3f) & (%.3f, ' \
                        '%.3f) & (%3.2f, %3.2f) & (%2.2f\\%%, %2.2f\\%%) \\\\'
            else:
                p_str = '%12s & %.3f$\pm$%.2f & %.3f$\pm$%.2f & %.3f$\pm$%.2f & (%.3f, %.3f) & (%.3f, ' \
                        '%.3f) & (%3.2f, %3.2f) & (%2.2f\\%%, %2.2f\\%%) \\\\'
            print(p_str %
                  (method_labels[method],
                   float(np.mean(node_pre)), float(np.std(node_pre)),
                   float(np.mean(node_rec)), float(np.std(node_rec)),
                   float(np.mean(node_fm)), float(np.std(node_fm)),
                   float(np.mean(auc_wt)), float(np.mean(auc_wt_bar)),
                   float(np.mean(acc_wt)), float(np.mean(acc_wt_bar)),
                   float(np.mean(missed_wt)),
                   float(np.mean(missed_wt_bar)),
                   float(np.mean(nonzero_ratio1)),
                   float(np.mean(nonzero_ratio2))))


def print_result_2(method_labels, all_methods, all_results):
    # model selected by best accuracy.
    for metric in ['best-acc']:
        print('-' * 100)
        print('-' * 36 + ' selected by %s ' % metric + '-' * 36)
        print('Method & \\textit{Pre} & \\textit{Rec} & \\textit{F1} & '
              '${AUC}_{{\\bf w}_t,{\\bar{\\bf w}}_t}$ & '
              '${Acc}_{{\\bf w}_t,{\\bar{\\bf w}}_t}$ & '
              '${Miss}_{{\\bf w}_t,{\\bar{\\bf w}}_t}$ & '
              '${NN}_{{\\bf w}_t,{\\bar{\\bf w}}_t}$\\\\')
        for method in all_methods:
            best_n_fm_wt = [re[0]['%s-wt' % metric] for case in all_results for
                            re in case if re[1] == method]
            p = best_n_fm_wt[0]['p']
            best_n_fm_wt_bar = [re[0]['%s-wt-bar' % metric] for case in
                                all_results for re in case if re[1] == method]
            node_pre = [_['te_n_pre_rec_fm_wt'][0] for _ in best_n_fm_wt]
            node_rec = [_['te_n_pre_rec_fm_wt'][1] for _ in best_n_fm_wt]
            node_fm = [_['te_n_pre_rec_fm_wt'][2] for _ in best_n_fm_wt]
            auc_wt = [_['te_auc_wt'] for _ in best_n_fm_wt]
            auc_wt_bar = [_['te_auc_wt_bar'] for _ in best_n_fm_wt_bar]
            acc_wt = [_['te_acc_wt'] for _ in best_n_fm_wt]
            acc_wt_bar = [_['te_acc_wt_bar'] for _ in best_n_fm_wt_bar]
            missed_wt = [_['missed_wt'][-1] for _ in best_n_fm_wt]
            missed_wt_bar = [_['missed_wt_bar'][-1] for _ in best_n_fm_wt]
            nonzero_ratio1 = [
                float(len(np.nonzero(_['wt'][:p])[0])) * 100. / float(p) for _
                in best_n_fm_wt]
            nonzero_ratio2 = [
                float(len(np.nonzero(_['wt_bar'][:p])[0])) * 100. / float(p)
                for _ in best_n_fm_wt]
            if nonzero_ratio1 == 1.0 and nonzero_ratio2 == 1.0:
                p_str = '%12s & %.3f & %.3f & %.3f & (%.3f, %.3f) & (%.3f, ' \
                        '%.3f) & (%3.f, %3.f) & (%2.2f\\%%, %2.2f\\%%) \\\\'
            elif nonzero_ratio1 == 1.0 and nonzero_ratio2 != 1.0:
                p_str = '%12s & %.3f & %.3f & %.3f & (%.3f, %.3f) & (%.3f, ' \
                        '%.3f) & (%3.f, %3.f) & (%2.2f\\%%, %2.2f\\%%) \\\\'
            elif nonzero_ratio1 != 1.0 and nonzero_ratio2 == 1.0:
                p_str = '%12s & %.3f & %.3f & %.3f & (%.3f, %.3f) & (%.3f, ' \
                        '%.3f) & (%3.f, %3.f) & (%2.2f\\%%, %2.2f\\%%) \\\\'
            else:
                p_str = '%12s & %.3f & %.3f & %.3f & (%.3f, %.3f) & (%.3f, ' \
                        '%.3f) & (%3.f, %3.f) & (%2.2f\\%%, %2.2f\\%%) \\\\'
            print(p_str %
                  (method_labels[method],
                   float(np.mean(node_pre)),
                   float(np.mean(node_rec)),
                   float(np.mean(node_fm)),
                   float(np.mean(auc_wt)), float(np.mean(auc_wt_bar)),
                   float(np.mean(acc_wt)), float(np.mean(acc_wt_bar)),
                   float(np.mean(missed_wt)),
                   float(np.mean(missed_wt_bar)),
                   float(np.mean(nonzero_ratio1)),
                   float(np.mean(nonzero_ratio2))))


def show_wt(fig_id, trial_i, model):
    import matplotlib.pyplot as plt
    length, width = 33, 33
    # we use the first figure to do the experiments.
    f_name = 'results/fix_tr_mu/fig%d_mu_0.3_trial_%02d_fix_tr_mu.pkl' % (
        fig_id, trial_i)
    results = pickle.load(open(f_name))['summary_results']
    results = {_[1]: _[0] for _ in results}
    method_list = ['adam', 'rda-l1', 'adagrad', 'da-gl', 'da-sgl', 'sto-iht',
                   'graph-sto-iht', 'da-iht',
                   'graph-da-iht']
    label_list = ['ADAM', 'RDA-L1', 'AdaGrad', 'DA-GL', 'DA-SGL', 'StoIHT',
                  'GraphIHT', 'DA-IHT', 'GraphDA']
    from matplotlib import rc
    from pylab import rcParams
    font = {'size': 12}
    plt.rc('font', **font)
    plt.rc('font', **font)
    rc('text', usetex=True)
    rcParams['figure.figsize'] = 12, 4
    fig, ax = plt.subplots(2, 5)
    for i, method in enumerate(method_list):
        re = results[method]['best-node-fm-wt']
        w = re[model][:length * width]
        if model == 'wt':
            w[w > 0] = 1.
            w[w < 0] = -1.
            ax[i / 5, i % 5].imshow(w.reshape(length, width), cmap='gray',
                                    vmin=-1, vmax=1)
        else:
            ax[i / 5, i % 5].imshow(w.reshape(length, width))
        ax[i / 5, i % 5].set_title('%s' % label_list[i])
        ax[i / 5, i % 5].set_yticklabels('')
        ax[i / 5, i % 5].set_xticklabels('')
    w = np.zeros(33 * 33)
    w[bench_data['fig_%d' % fig_id]] = 1.0
    if model == 'wt':
        ax[1, 4].imshow(w.reshape(length, width), cmap='gray', vmin=-1, vmax=1)
    else:
        ax[1, 4].imshow(w.reshape(length, width))
    ax[1, 4].set_title('True Model')
    ax[1, 4].set_yticklabels('')
    ax[1, 4].set_xticklabels('')
    fig = plt.gcf()
    f_name = 'results/figs/fix_tr_mu_fig%d_mu_0.3_trial_%02d_%s.pdf' % (
        fig_id, trial_i, model)
    print('save fig to: %s' % f_name)
    fig.savefig(f_name, dpi=600, bbox_inches='tight', pad_inches=0,
                format='pdf')
    plt.close()


def show_wt_2(fig_id, trial_i):
    import matplotlib.pyplot as plt
    length, width = 33, 33
    # we use the first figure to do the experiments.
    f_name = 'results/fix_tr_mu/fig%d_mu_0.3_trial_%02d_fix_tr_mu.pkl' % (
        fig_id, trial_i)
    results = pickle.load(open(f_name))['summary_results']
    results = {_[1]: _[0] for _ in results}
    method_list = ['adam', 'rda-l1', 'adagrad', 'da-gl', 'da-sgl', 'sto-iht',
                   'graph-sto-iht', 'da-iht',
                   'graph-da-iht']
    label_list = ['\\textbf{\\textsc{ADAM}}', '\\textbf{\\textsc{RDA-L1}}',
                  '\\textbf{\\textsc{AdaGrad}}', '\\textbf{\\textsc{DA-GL}}',
                  '\\textbf{\\textsc{DA-SGL}}', '\\textbf{\\textsc{StoIHT}}',
                  '\\textbf{\\textsc{GraphStoIHT}}',
                  '\\textbf{\\textsc{DA-IHT}}', '\\textbf{\\textsc{GraphDA}}']
    from matplotlib import rc
    from pylab import rcParams
    font = {'size': 12}
    plt.rc('font', **font)
    plt.rc('font', **font)
    rc('text', usetex=True)
    rcParams['figure.figsize'] = 17, 3.4
    fig, ax = plt.subplots(2, 10)
    for i, j in product(range(2), range(10)):
        ax[i, j].get_xaxis().set_visible(False)
        if j != 0:
            ax[i, j].get_yaxis().set_visible(False)
        for _ in ['top', 'bottom', 'right', 'left']:
            ax[i, j].spines[_].set_visible(False)
            ax[i, j].spines[_].set_visible(False)
    for i, method in enumerate(method_list):
        re = results[method]['best-node-fm-wt']
        w = re['wt'][:length * width]
        w[w > 0.0] = 1.0
        w[w < 0.0] = -1.0
        ax[0, i].imshow(w.reshape(length, width), cmap='gray', vmin=-1, vmax=1)
        w = re['wt_bar'][:length * width]
        w[w > 0.0] = 1.0
        w[w < 0.0] = -1.0
        ax[1, i].imshow(w.reshape(length, width), cmap='gray', vmin=-1, vmax=1)
        ax[0, i].set_title('%s' % label_list[i])
    ax[0, 0].set_ylabel(r"$\displaystyle {\bf w}_t$")
    ax[1, 0].set_ylabel(r"$\displaystyle \overline{{\bf w}}_t$", labelpad=2)
    plt.setp(ax[0, 0].get_xticklabels(), visible=False)
    plt.setp(ax[0, 0].get_yticklabels(), visible=False)
    plt.setp(ax[1, 0].get_xticklabels(), visible=False)
    plt.setp(ax[1, 0].get_yticklabels(), visible=False)
    ax[0, 0].tick_params(axis='both', which='both', length=0)
    ax[1, 0].tick_params(axis='both', which='both', length=0)

    w = np.zeros(33 * 33)
    w[bench_data['fig_%d' % fig_id]] = 1.0
    w[w > 0.0] = 1.0
    w[w < 0.0] = -1.0
    ax[0, 9].imshow(w.reshape(length, width), cmap='gray', vmin=-1, vmax=1)
    w[bench_data['fig_%d' % fig_id]] = 1.0
    w[w > 0.0] = 1.0
    w[w < 0.0] = -1.0
    ax[1, 9].imshow(w.reshape(length, width), cmap='gray', vmin=-1, vmax=1)
    ax[0, 9].set_title('\\textbf{\\textsc{True Model}}')
    fig = plt.gcf()
    f_name = 'results/figs/fix_tr_mu_fig%d_mu_03_trial_%02d.pdf' % (
        fig_id, trial_i)
    print('save fig to: %s' % f_name)
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    fig.savefig(f_name, dpi=600, bbox_inches='tight', pad_inches=0,
                format='pdf')
    plt.close()


def show_mistakes(root, fig_id, trial_i):
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from pylab import rcParams
    plt.rc('font', **{'size': 16})
    rc('text', usetex=True)
    rcParams['figure.figsize'] = 16, 8
    f_name = root + 'output/simu_fig%d_mu_0.3_trial_%02d_fix_tr_mu.pkl' % (
        fig_id, trial_i)
    results = pickle.load(open(f_name))['summary_results']
    results = {_[1]: _[0] for _ in results}
    method_list = ['adam', 'rda-l1', 'adagrad', 'da-gl', 'da-sgl',
                   'sto-iht', 'graph-sto-iht', 'da-iht', 'graph-da-iht']
    label_list = ['ADAM', 'RDA-L1', 'AdaGrad', 'DA-GL', 'DA-SGL',
                  'StoIHT', 'GraphIHT', 'DA-IHT', 'Graph-DA']
    marker_list = ['p', '>', '<', 'X', 'o', '*', 'v', '^', 's']
    color_list = ['b', 'g', 'y', 'c', 'gray', 'k', 'indianred', 'm', 'r']
    # generate first figure
    metric_list = ['best-node-fm-wt', 'best-acc-wt', 'best-acc-node-fm-wt']
    metric_title = ['Node-F1-Score', 'Accuracy', 'Accuracy-F1-Score']
    fig, ax = plt.subplots(nrows=2, ncols=3)
    for ind_method, method in enumerate(method_list):
        for ind_metric, metric in enumerate(metric_list):
            re = results[method][metric]
            x = np.arange(0, 401, 30)
            ax[0, ind_metric].plot(
                x, re['missed_wt'][x],
                marker=marker_list[ind_method], markersize=8.,
                color=color_list[ind_method],
                label=label_list[ind_method], linewidth=1.)
            ax[1, 1].legend(loc='upper center',
                            bbox_to_anchor=(0.5, -0.18), ncol=5)
            if ind_metric == 0:
                ax[0, ind_metric].set_ylabel(
                    r"Mistakes($\displaystyle {\bf w}_t $)")
                ax[1, ind_metric].set_ylabel(
                    r"Mistakes($\displaystyle \bar{{\bf w}_t} $)")
            else:
                ax[0, ind_metric].set_yticklabels('')
                ax[1, ind_metric].set_yticklabels('')
            ax[0, ind_metric].set_title(metric_title[ind_metric])
            ax[1, ind_metric].errorbar(
                x, re['missed_wt_bar'][x],
                marker=marker_list[ind_method], markersize=8.,
                color=color_list[ind_method], label=label_list[ind_method],
                linewidth=1.)
            ax[1, ind_metric].set_xlabel("Samples Seen")

    plt.subplots_adjust(wspace=0, hspace=0)
    if not os.path.exists(root + 'figs'):
        os.mkdir(root + 'figs')
    f_name = root + 'figs/simu_fig%d_mu_0.3_trial_%02d_mistakes.pdf' % (
        fig_id, trial_i)
    print('save fig to: %s' % f_name)
    fig.savefig(f_name, dpi=600, bbox_inches='tight', pad_inches=0,
                format='pdf')
    plt.close()


def show_figure_1():
    """
    Figure 1 is generated by the latex code at: figs/figure_01.tex
    """
    print('please check the latex code at: figs/figure_01.tex')


def show_figure_2():
    """
    To generate Figure 2, you may need tex: https://www.tug.org/texlive/
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    from matplotlib import rc
    #
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 18
    rc('text', usetex=True)
    edges, weights = gen_grid_graph(width=33, height=33)
    height, width = 33, 33
    sub_graph_edges = []
    p = 33 * 33
    plt.figure(figsize=(2, 2))
    for iii, fig_i in enumerate(['fig_1', 'fig_2', 'fig_3', 'fig_4']):
        pos, graph = dict(), nx.Graph()
        black_edges = []
        red_edges = []
        red_edge_list = []
        for edge in edges:
            graph.add_edge(edge[0], edge[1])
            if (edge[0], edge[1]) in sub_graph_edges:
                red_edges.append('r')
                red_edge_list.append((edge[0], edge[1]))
            else:
                black_edges.append('k')
        for index, (i, j) in enumerate(product(range(height), range(width))):
            graph.add_node(index)
            pos[index] = (j, height - i)
        print('subgraph, which has %s nodes.' % bench_data['s'][fig_i])
        nx.draw_networkx_nodes(graph, pos, node_size=5, nodelist=range(p),
                               linewidths=.2, node_color='w',
                               edgecolors='gray', font_size=6)
        nx.draw_networkx_nodes(graph, pos, node_size=5,
                               nodelist=bench_data[fig_i], linewidths=.2,
                               node_color=[1.] * len(bench_data[fig_i]),
                               cmap='gray', edgecolors='gray', font_size=6)
        nx.draw_networkx_edges(graph, pos, alpha=0.4, width=0.5,
                               edge_color='gray', font_size=6)
        nx.draw_networkx_edges(graph, pos, alpha=0.8, width=0.5,
                               edgelist=red_edge_list, edge_color='r',
                               font_size=6)
        plt.axis('off')
        fig = plt.gcf()
        fig.set_figheight(2)
        fig.set_figwidth(2)
        plt.subplots_adjust(0, 0, 1, 1, 0, 0)
        for ax in fig.axes:
            ax.axis('off')
            ax.margins(0.02, 0.02)
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
        f_name = 'figs/figure_02_%d.pdf' % (iii + 1)
        fig.savefig(f_name, dpi=1200, pad_inches=0.0, format='pdf')
        plt.close()


def show_4_tables():
    all_methods = ['adam', 'rda-l1', 'adagrad', 'da-gl', 'da-sgl', 'sto-iht',
                   'graph-sto-iht', 'da-iht',
                   'graph-da-iht']
    method_labels = {
        'rda-l1': '\\textbf{$\displaystyle \ell_1$-\\textsc{RDA}}',
        'da-iht': '\\textbf{\\textsc{DA-IHT}}',
        'adagrad': '\\textbf{\\textsc{AdaGrad}}',
        'sto-iht': '\\textbf{\\textsc{StoIHT}}',
        'graph-sto-iht': '\\textbf{\\textsc{GraphStoIHT}}',
        'da-gl': '\\textbf{\\textsc{DA-GL}}',
        'da-sgl': '\\textbf{\\textsc{DA-SGL}}',
        'adam': '\\textbf{\\textsc{ADAM}}',
        'graph-da-iht': '\\textbf{\\textsc{GraphDA}}',
        'best-subset': '\\textbf{\\textsc{BestSubset}}'}
    num_trials, root = 20, '../results/benchmark/fix_tr_mu/'
    for (fig_i, i) in bench_data['fig_list']:
        print('\n\n--- %s ---' % fig_i)
        all_results = []
        for trial_i in range(num_trials):
            f_name = root + 'fig%d_mu_0.3_trial_%02d_fix_tr_mu.pkl' % (
                i, trial_i)
            out = pickle.load(open(f_name))
            all_results.append(out['summary_results'])
        if fig_i == 'fig_1':
            print_result(method_labels, all_methods, all_results)
        else:
            print_result_2(method_labels, all_methods, all_results)


def show_figure_3():
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from pylab import rcParams
    plt.rc('font', **{'size': 14})
    rc('text', usetex=True)
    rcParams['figure.figsize'] = 8.3, 4
    method_list = ['adam', 'rda-l1', 'adagrad', 'da-gl', 'graph-sto-iht',
                   'sto-iht', 'da-sgl', 'da-iht', 'graph-da-iht']
    label_list = ['ADAM', r'$\displaystyle \ell_1$-RDA', 'AdaGrad', 'DA-GL',
                  'GraphStoIHT',
                  'StoIHT', 'DA-SGL', 'DA-IHT', 'GraphDA']
    marker_list = ['p', '>', '<', 'X', 'o', '*', 'v', '^', 's']
    color_list = ['b', 'g', 'y', 'c', 'gray', 'k', 'indianred', 'm', 'r']
    # generate figure 3 shown in the paper.
    metric_list = ['best-acc-wt']
    fig_title = [r'(a) \textit{Graph01}', r'(b) \textit{Graph02}',
                 r'(c) \textit{Graph03}', r'(d) \textit{Graph04}']
    fig, ax = plt.subplots(nrows=2, ncols=4, sharex='all', sharey='all')
    for i, j in product(range(2), range(4)):
        ax[i, j].grid(b=True, which='both', color='gray', linestyle='--',
                      axis='both')
    for fig_id in [1, 2, 3, 4]:
        average_results = dict()
        for ind_method, method in enumerate(method_list):
            average_results[method] = dict()
            for ind_metric, metric in enumerate(metric_list):
                average_results[method][metric] = dict()
                average_results[method][metric]['wt'] = np.zeros(
                    shape=(20, 400))
                average_results[method][metric]['wt_bar'] = np.zeros(
                    shape=(20, 400))
        for trial_i in range(20):
            f_name = '../results/benchmark/fix_tr_mu/fig%d_mu_0.3_trial_%02d_fix_tr_mu.pkl' % (
                fig_id, trial_i)
            results = pickle.load(open(f_name))['summary_results']
            results = {_[1]: _[0] for _ in results}
            for ind_method, method in enumerate(method_list):
                for ind_metric, metric in enumerate(metric_list):
                    average_results[method][metric]['wt'][trial_i] = \
                        results[method][metric]['missed_wt']
                    average_results[method][metric]['wt_bar'][trial_i] = \
                        results[method][metric]['missed_wt_bar']
        for ind_method, method in enumerate(method_list):
            for ind_metric, metric in enumerate(metric_list):
                re = np.mean(average_results[method][metric]['wt'], axis=0)
                x = np.arange(100, 400, 20)
                ax[0, 0].set_xticks([150, 250, 350])
                ax[0, 1].set_yticks([1.7, 1.9, 2.1])
                ax[0, 0].set_yticks([1.7, 1.9, 2.1])
                ax[0, 0].set_ylim([1.6, 2.2])
                ax[0, 1].set_ylim([1.6, 2.2])
                ax[0, fig_id - 1].plot(
                    x, np.log10(re[x]), marker=marker_list[ind_method],
                    markersize=4.,
                    markerfacecolor='white', markeredgewidth=1.,
                    color=color_list[ind_method], label=label_list[ind_method],
                    linewidth=1.)
                ax[1, 1].legend(loc='lower center',
                                bbox_to_anchor=(0.9, -0.65), ncol=5,
                                fontsize=14,
                                borderpad=0.01, columnspacing=0.1,
                                labelspacing=0.02,
                                handletextpad=0.01, framealpha=1.0)
                if ind_metric == 0:
                    ax[0, ind_metric].set_ylabel(
                        r"$\displaystyle \log (\textit{Miss}_{{\bf w}_t}$)",
                        fontsize=16, labelpad=0.01)
                    ax[1, ind_metric].set_ylabel(
                        r"$\displaystyle \log (\textit{Miss}_{\bar{{\bf w}_t}} $)",
                        fontsize=16, labelpad=0.01)
                else:
                    ax[0, fig_id - 1].set_yticklabels('')
                    ax[1, fig_id - 1].set_yticklabels('')
                re = np.mean(average_results[method][metric]['wt_bar'], axis=0)
                ax[0, fig_id - 1].set_title(fig_title[fig_id - 1])
                ax[1, fig_id - 1].plot(
                    x, np.log10(re[x]), marker=marker_list[ind_method],
                    markersize=4.,
                    markerfacecolor='white', markeredgewidth=1.,
                    color=color_list[ind_method], label=label_list[ind_method],
                    linewidth=1.)
                ax[1, fig_id - 1].set_xlabel("Samples Seen", labelpad=0.03)

    plt.subplots_adjust(wspace=0, hspace=0)
    f_name = '../results/benchmark/figs/figure_3.pdf'
    print('save fig to: %s' % f_name)
    fig.savefig(f_name, dpi=1200, bbox_inches='tight', pad_inches=0.03,
                format='pdf')
    plt.close()


def show_figure_4(trial_i=0):
    import matplotlib.pyplot as plt
    length, width = 33, 33
    from matplotlib import rc
    from pylab import rcParams
    font = {'size': 12}
    plt.rc('font', **font)
    plt.rc('font', **font)
    rc('text', usetex=True)
    rcParams['figure.figsize'] = 16.9, 6.8
    fig, ax = plt.subplots(4, 10)
    # we use the first figure to do the experiments.
    method_list = ['adam', 'rda-l1', 'adagrad', 'da-gl', 'da-sgl', 'sto-iht',
                   'graph-sto-iht', 'da-iht', 'graph-da-iht']
    label_list = ['\\textsc{Adam}',
                  r'$\displaystyle \ell_1$' + '\\textsc{-RDA}',
                  '\\textsc{AdaGrad}', '\\textsc{DA-GL}',
                  '\\textsc{DA-SGL}', '\\textsc{StoIHT}',
                  '\\textsc{GraphStoIHT}', '\\textsc{DA-IHT}',
                  '\\textsc{GraphDA}']
    for fig_id in range(1, 5):
        f_name = '../results/benchmark/fix_tr_mu/fig%d_mu_0.3_trial_%02d_fix_tr_mu.pkl' % (
            fig_id, trial_i)
        results = pickle.load(open(f_name))['summary_results']
        results = {_[1]: _[0] for _ in results}
        for i, j in product(range(4), range(10)):
            ax[i, j].get_xaxis().set_visible(False)
            if j != 0:
                ax[i, j].get_yaxis().set_visible(False)
            for _ in ['top', 'bottom', 'right', 'left']:
                ax[i, j].spines[_].set_visible(False)
                ax[i, j].spines[_].set_visible(False)
        for i, method in enumerate(method_list):
            re = results[method]['best-node-fm-wt']
            w = re['wt'][:length * width]
            w[w > 0.0] = 1.0
            w[w < 0.0] = -1.0
            ax[fig_id - 1, i].imshow(w.reshape(length, width), cmap='gray',
                                     vmin=-1, vmax=1)
            ax[0, i].set_title('%s' % label_list[i])
        ax[fig_id - 1, 0].margins(y=20)
        ax[fig_id - 1, 0].set_ylabel(
            r"\textit{Graph%02d}-$\displaystyle {\bf w}_t$" % fig_id,
            fontsize=18,
            labelpad=0)

        plt.setp(ax[fig_id - 1, 0].get_xticklabels(), visible=False)
        plt.setp(ax[fig_id - 1, 0].get_yticklabels(), visible=False)
        ax[fig_id - 1, 0].tick_params(axis='both', which='both', length=0)
        ax[fig_id - 1, 0].tick_params(axis='both', which='both', length=0)

        w = np.zeros(33 * 33)
        w[bench_data['fig_%d' % fig_id]] = 1.0
        w[w > 0.0] = 1.0
        w[w < 0.0] = -1.0
        ax[fig_id - 1, 9].imshow(w.reshape(length, width), cmap='gray',
                                 vmin=-1, vmax=1)
        w[bench_data['fig_%d' % fig_id]] = 1.0
        ax[0, 9].set_title('\\textsc{True Model}')
        fig = plt.gcf()
    f_name = '../results/benchmark/figs/figure_4.pdf'
    print('save fig to: %s' % f_name)
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    fig.savefig(f_name, dpi=600, bbox_inches='tight', pad_inches=0.03,
                format='pdf')
    plt.close()


def show_figure_5():
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from pylab import rcParams
    plt.rc('font', **{'size': 10})
    rc('text', usetex=True)
    rcParams['figure.figsize'] = 8.3, 2.0
    # we use the first figure to do the experiments.
    method_list = ['adam', 'graph-da-iht']
    label_list = [r'\textsc{Adam}', r'\textsc{GraphDA}']
    color_list = ['b', 'r']
    marker_list = ['o', 's']
    sparsity_list = {
        'fig_1': range(2, 60, 3),
        'fig_2': range(5, 101, 5),
        'fig_3': range(10, 201, 10),
        'fig_4': range(15, 263, 13)}
    mu, num_trials = 0.3, 20
    fig, ax = plt.subplots(1, 4)
    for i in range(4):
        ax[i].grid(color='gray', linestyle='--')
    title_list = [r'(a) \textit{Graph01}', r'(b) \textit{Graph02}',
                  r'(c) \textit{Graph03}', r'(d) \textit{Graph04}']
    ax[0].set_xticks(range(12, 60, 12))
    ax[1].set_xticks(range(20, 90, 20))
    ax[2].set_xticks(range(40, 180, 40))
    ax[3].set_xticks(range(50, 260, 60))
    ax[0].set_ylim([0.2, 0.45])
    ax[0].set_yticks([0.25, 0.30, 0.35, 0.40])
    ax[1].set_ylim([0.12, 0.42])
    ax[1].set_yticks([0.18, 0.24, 0.30, 0.36])
    ax[2].set_ylim([0.05, 0.30])
    ax[2].set_yticks([0.10, 0.15, 0.20, 0.25])
    ax[3].set_ylim([0.02, 0.25])
    ax[3].set_yticks([0.06, 0.11, 0.16, 0.21])
    for (fig_i, fig_ind) in bench_data['fig_list']:
        all_results = {_: {method: [] for method in method_list} for _ in
                       sparsity_list[fig_i]}
        for trial_i in range(num_trials):
            f_name = '../results/benchmark/diff_s/fig%d_mu_%.1f_trial_%02d_diff_s.pkl' % (
                fig_ind, mu, trial_i)
            results = pickle.load(open(f_name))
            for method in method_list:
                str_ = []
                for s in sparsity_list[fig_i]:
                    result = results[s]['summary_results']
                    result = {_[1]: _[0] for _ in result}
                    error_rate = 1. - result[method]['best-acc-wt'][
                        'te_acc_wt']
                    all_results[s][method].append(error_rate)
                    str_.append('%.3f' % error_rate)
        for ind_method, method in enumerate(method_list):
            str_ = []
            mean_re = []
            std_re = []
            for s in sparsity_list[fig_i]:
                mean_re.append(np.mean(all_results[s][method]))
                std_re.append(np.std(all_results[s][method]))
                str_.append('%.3f' % np.mean(all_results[s][method]))
            ax[fig_ind - 1].errorbar(x=sparsity_list[fig_i],
                                     y=np.asarray(mean_re),
                                     yerr=np.asarray(std_re),
                                     color=color_list[ind_method],
                                     marker=marker_list[ind_method],
                                     markersize=3.,
                                     markerfacecolor='white',
                                     label=label_list[ind_method],
                                     linewidth=1.0)
            ax[3].legend(loc='upper right', fontsize=10,
                         borderpad=0.2, columnspacing=0.2, labelspacing=0.2,
                         handletextpad=0.2, framealpha=1.0)
            ax[0].set_ylabel("Test-set Error rate", fontsize=14, labelpad=0.02)
            ax[fig_ind - 1].set_title(title_list[fig_ind - 1])
            ax[fig_ind - 1].set_xlabel("Sparsity", fontsize=14, labelpad=0.02)
    plt.subplots_adjust(wspace=0.3, hspace=0)
    f_name = '../results/benchmark/figs/figure_5.pdf'
    print('save file to: %s' % f_name)
    plt.savefig(f_name, dpi=1200, bbox_inches='tight', pad_inches=0,
                format='pdf')
    plt.close()


def show_figure_6_all():
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from pylab import rcParams
    plt.rc('font', **{'size': 15})
    rc('text', usetex=True)
    rcParams['figure.figsize'] = 16, 6
    method_list = ['adam', 'rda-l1', 'adagrad', 'da-gl', 'graph-sto-iht',
                   'sto-iht', 'da-sgl', 'da-iht',
                   'graph-da-iht']
    label_list = [r'\textsc{Adam}', 'RDA-L1', 'AdaGrad', 'DA-GL', 'GraphIHT',
                  'StoIHT', 'DA-SGL', 'DA-IHT', 'GraphDA']
    marker_list = ['p', '>', '<', 'X', 'o', '*', 'v', '^', 's']
    color_list = ['b', 'g', 'y', 'c', 'gray', 'k', 'indianred', 'm', 'r']
    fig_title = [r'(a) \textit{Graph01}', r'(b) \textit{Graph02}',
                 r'(c) \textit{Graph03}', r'(d) \textit{Graph04}']
    num_tr_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    num_trials = 20
    fig, ax = plt.subplots(nrows=2, ncols=4)
    for i in range(2):
        for j in range(4):
            ax[i, j].grid(b=True, which='both', color='gray', linestyle='--',
                          axis='both')
            ax[i, j].set_xticks(np.arange(100, 1001, 100.))
    for i in range(2):
        for j in range(4):
            ax[i, j].set_yticks([0.6, 0.7, 0.8, 0.9])
            ax[i, j].set_xticks([200, 400, 600, 800])
            ax[i, j].set_xlim([0, 1000])
            ax[i, j].set_ylim([0.5, 1.])
    for fig_ind, fig_id in enumerate([1, 2, 3, 4]):
        raw_data = dict()
        for trial_i in range(num_trials):
            f_name = '../results/benchmark/diff_tr/fig%d_mu_0.3_trial_%02d_diff_tr.pkl' % (
                fig_id, trial_i)
            results = pickle.load(open(f_name))
            raw_data[trial_i] = results
        for tag_ind, (tag1, tag2) in enumerate(
                zip(['wt', 'wt-bar'], ['wt', 'wt_bar'])):
            all_results = {_: {method: [] for method in method_list} for _ in
                           num_tr_list}
            for trial_i in range(num_trials):
                results = raw_data[trial_i]
                for n in num_tr_list:
                    result = results[n]['summary_results']
                    result = {_[1]: _[0] for _ in result}
                    for method in method_list:
                        all_results[n][method].append(
                            result[method]['best-acc-%s' % tag1][
                                'te_acc_%s' % tag2])
            for n in num_tr_list:
                for method in method_list:
                    all_results[n][method] = np.mean(all_results[n][method])
            for ind_method, method in enumerate(method_list):
                ax[tag_ind, fig_ind].plot(num_tr_list,
                                          [all_results[_][method] for _ in
                                           num_tr_list],
                                          color=color_list[ind_method],
                                          marker=marker_list[ind_method],
                                          label=label_list[ind_method],
                                          linewidth=1.5)
                if tag_ind == 0:
                    ax[tag_ind, fig_ind].set_title(fig_title[fig_ind])
                if fig_ind != 0:
                    ax[tag_ind, fig_ind].set_yticklabels('')
                    ax[tag_ind, fig_ind].set_yticklabels('')
            ax[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
                            ncol=5)
            for j in range(4):
                ax[1, j].set_xlabel('Samples Seen')
            ax[0, 0].set_ylabel("Accuracy")
            ax[1, 0].set_ylabel("F1-Score")
    plt.subplots_adjust(wspace=0, hspace=0)
    f_name = '../results/benchmark/figs/fix_tr_mu_03_diff_tr.pdf'
    print('save fig to: %s' % f_name)
    plt.savefig(f_name, dpi=600, bbox_inches='tight', pad_inches=0,
                format='pdf')
    plt.close()


def show_figure_6(fig_id=1):
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from pylab import rcParams
    plt.rc('font', **{'size': 15})
    rc('text', usetex=True)
    rcParams['figure.figsize'] = 8, 3
    method_list = ['adam', 'rda-l1', 'adagrad', 'da-gl', 'graph-sto-iht',
                   'sto-iht', 'da-sgl', 'da-iht',
                   'graph-da-iht']
    label_list = [r'\textsc{Adam}', r'$\displaystyle \ell_1$-\textsc{RDA}',
                  r'\textsc{AdaGrad}',
                  r'\textsc{DA-GL}', r'\textsc{GraphStoIHT}',
                  r'\textsc{StoIHT}', r'\textsc{DA-SGL}', r'\textsc{DA-IHT}',
                  r'\textsc{GraphDA}']
    marker_list = ['p', '>', '<', 'X', 'o', '*', 'v', '^', 's']
    color_list = ['b', 'g', 'y', 'c', 'gray', 'k', 'indianred', 'm', 'r']
    num_tr_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    num_trials = 20
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey='all')
    for i in range(2):
        ax[i].grid(b=True, which='both', color='gray', linestyle='--',
                   axis='both')
        ax[i].set_xticks(np.arange(100, 1001, 100.))
    if fig_id == 1:
        ax[1].set_yticks([0.55, 0.6, 0.65, 0.7, 0.75])
        for i in product(range(2)):
            ax[i].set_ylim([0.5, 0.8])
    if fig_id == 2:
        ax[1].set_yticks([0.55, 0.65, 0.75, 0.85])
        for i in product(range(2)):
            ax[i].set_ylim([0.5, 0.9])
    if fig_id == 3:
        ax[1].set_yticks([0.6, 0.7, 0.8, 0.9])
        for i in product(range(2)):
            ax[i].set_ylim([0.55, 0.95])
    if fig_id == 4:
        ax[1].set_yticks([0.65, 0.75, 0.85, 0.95])
        for i in product(range(2)):
            ax[i].set_ylim([0.6, 0.99])

    for i in product(range(2)):
        ax[i].set_xticks([100, 300, 500, 700, 900])
        ax[i].set_xlim([50, 1050])

    raw_data = dict()
    for trial_i in range(num_trials):
        f_name = '../results/benchmark/diff_tr/fig%d_mu_0.3_trial_%02d_diff_tr.pkl' % (
            fig_id, trial_i)
        results = pickle.load(open(f_name))
        raw_data[trial_i] = results
    for tag_ind, (tag1, tag2) in enumerate(
            zip(['wt', 'wt-bar'], ['wt', 'wt_bar'])):
        for metric_ind, metric in enumerate(['te_acc_%s' % tag2]):
            all_results = {_: {method: [] for method in method_list} for _ in
                           num_tr_list}
            for trial_i in range(num_trials):
                results = raw_data[trial_i]
                for n in num_tr_list:
                    result = results[n]['summary_results']
                    result = {_[1]: _[0] for _ in result}
                    for method in method_list:
                        all_results[n][method].append(
                            result[method]['best-acc-%s' % tag1][metric])
            for n in num_tr_list:
                for method in method_list:
                    all_results[n][method] = np.mean(all_results[n][method])
            for ind_method, method in enumerate(method_list):
                ax[tag_ind].plot(
                    num_tr_list, [all_results[_][method] for _ in num_tr_list],
                    color=color_list[ind_method],
                    marker=marker_list[ind_method], markerfacecolor='white',
                    markeredgewidth=1.5,
                    label=label_list[ind_method], linewidth=1.5)
        ax[1].legend(loc='center right', bbox_to_anchor=(1.73, 0.5), ncol=1,
                     fontsize=16,
                     borderpad=0.2, columnspacing=0.5, labelspacing=0.2,
                     handletextpad=0.2, framealpha=1.0)
        for j in range(2):
            ax[j].set_xlabel('Samples Seen')
        ax[0].set_title(r"(a) $\displaystyle Acc_{{\bf w}_t}$", fontsize=15)
        ax[1].set_title(r"(b) $\displaystyle Acc_{{\bar{\bf w}}_t}$",
                        fontsize=15)
    plt.subplots_adjust(wspace=0, hspace=0.1)
    f_name = '../results/benchmark/figs/figure_6.pdf'
    print('save fig to: %s' % f_name)
    plt.subplots_adjust(wspace=0.0, hspace=0)
    plt.savefig(f_name, dpi=1200, bbox_inches='tight', pad_inches=0.05,
                format='pdf')
    plt.close()


def show_figure_7(fig_id=1):
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from pylab import rcParams
    plt.rc('font', **{'size': 12})
    rc('text', usetex=True)
    rcParams['figure.figsize'] = 8, 3
    method_list = ['adam', 'rda-l1', 'adagrad', 'da-gl', 'graph-sto-iht',
                   'sto-iht', 'da-sgl', 'da-iht',
                   'graph-da-iht']
    label_list = [r'\textsc{Adam}', r'$\displaystyle \ell_1$-\textsc{RDA}',
                  r'\textsc{AdaGrad}',
                  r'\textsc{DA-GL}', r'\textsc{GraphStoIHT}',
                  r'\textsc{StoIHT}', r'\textsc{DA-SGL}', r'\textsc{DA-IHT}',
                  r'\textsc{GraphDA}']
    marker_list = ['p', '>', '<', 'X', 'o', '*', 'v', '^', 's']
    color_list = ['b', 'g', 'y', 'c', 'gray', 'k', 'indianred', 'm', 'r']
    num_trials = 20
    mu_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey='all')
    for i in range(2):
        ax[i].grid(b=True, which='both', color='gray', linestyle='--',
                   axis='both')
        ax[i].set_xticks([0.2, 0.4, 0.6, 0.8])
    ax[0].set_yticks([0.6, 0.7, 0.8, 0.9])
    for i in range(2):
        ax[i].set_ylim([0.5, 1.0])
        ax[i].set_xlim([0.0, 1.0])
    raw_data = dict()
    for trial_i in range(num_trials):
        results = dict()
        for mu in mu_list:
            f_name = '../results/benchmark/diff_mu/fig%d_mu_%.1f_trial_%02d_diff_mu.pkl' % (
                fig_id, mu, trial_i)
            results[mu] = pickle.load(open(f_name))
        raw_data[trial_i] = results
    for tag_ind, (tag1, tag2) in enumerate(
            zip(['wt', 'wt-bar'], ['wt', 'wt_bar'])):
        for metric_ind, metric in enumerate(['te_acc_%s' % tag2]):
            all_results = {_: {__: [] for __ in method_list} for _ in mu_list}
            for trial_i in range(num_trials):
                results = raw_data[trial_i]
                for mu in mu_list:
                    result = results[mu]['summary_results']
                    result = {_[1]: _[0] for _ in result}
                    for method in method_list:
                        all_results[mu][method].append(
                            result[method]['best-acc-%s' % tag1][metric])
            for mu, method in product(mu_list, method_list):
                all_results[mu][method] = np.mean(all_results[mu][method])
            for ind_method, method in enumerate(method_list):
                ax[tag_ind].plot(mu_list,
                                 [all_results[_][method] for _ in mu_list],
                                 color=color_list[ind_method],
                                 marker=marker_list[ind_method],
                                 markerfacecolor='white', markeredgewidth=1.5,
                                 label=label_list[ind_method], linewidth=1.5)
    ax[1].legend(loc='center right', bbox_to_anchor=(1.73, 0.5), ncol=1,
                 fontsize=16,
                 borderpad=0.2, columnspacing=0.5, labelspacing=0.2,
                 handletextpad=0.2, framealpha=1.0)
    for j in range(2):
        ax[j].set_xlabel(r"$\displaystyle \mu $", labelpad=-1, fontsize=14)
    ax[0].set_title(r"(a) $\displaystyle Acc_{{\bf w}_t}$", fontsize=15)
    ax[1].set_title(r"(b) $\displaystyle Acc_{{\bar{\bf w}}_t}$", fontsize=15)
    f_name = '../results/benchmark/figs/figure_7.pdf'
    print('save file to %s' % f_name)
    plt.subplots_adjust(wspace=0, hspace=0.05)
    plt.savefig(f_name, dpi=1200, bbox_inches='tight', pad_inches=0.05,
                format='pdf')
    plt.close()


def print_help():
    print(
        '%s is a wrong option, you have the following options:' % sys.argv[1])
    print(
        '\n'.join(['python exp_logit_benchmark.py run_fix_n trial_i num_cpus',
                   'python exp_logit_benchmark.py rep trial_i num_cpus',
                   'python exp_logit_benchmark.py show_run',
                   'python exp_logit_benchmark.py show_rep']))


def generate_dataset(root):
    num_trials = bench_data['num_trials']
    num_tr = bench_data['num_tr']
    num_va = bench_data['num_va']
    num_te = bench_data['num_te']
    noise_mu = bench_data['noise_mu']
    noise_std = bench_data['noise_std']
    # consider 20 different mu values.
    mu_list = bench_data['mu_list']
    # num_trials: we repeat the experiments 20 times.
    # the final results will averaged on these 20 experiments.
    for ((fig_i, i), mu) in product(bench_data['fig_list'], mu_list):
        for trial_i in range(num_trials):
            data = generate_grid_data(width=33, height=33, num_tr=num_tr,
                                      num_va=num_va, num_te=num_te,
                                      noise_mu=noise_mu, noise_std=noise_std,
                                      mu=mu, sub_graph=bench_data[fig_i],
                                      fig_id=i, trial_i=trial_i,
                                      num_trials=num_trials,
                                      mu_strategy='constant')
            f_name = root + 'input/fig%d_mu_%.1f_trial_%02d.pkl' % (
                i, mu, trial_i)
            print('save file to: %s' % f_name)
            if not os.path.exists(f_name):
                pickle.dump(data, open(f_name, 'wb'))


def gen_grid_graph(width, height, rand_weight=False):
    """ Generate a grid graph with size, width x height. Totally there will be
        width x height number of nodes in this generated graph.
    :param width:       the width of the grid graph.
    :param height:      the height of the grid graph.
    :param rand_weight: the edge costs in this generated grid graph.
    :return:            1.  list of edges
                        2.  list of edge costs
    """
    np.random.seed()
    if width < 0 and height < 0:
        print('Error: width and height should be positive.')
        return [], []
    width, height = int(width), int(height)
    edges, weights = [], []
    index = 0
    for i in range(height):
        for j in range(width):
            if (index % width) != (width - 1):
                edges.append((index, index + 1))
                if index + width < int(width * height):
                    edges.append((index, index + width))
            else:
                if index + width < int(width * height):
                    edges.append((index, index + width))
            index += 1
    edges = np.asarray(edges, dtype=int)
    # random generate costs of the graph
    if rand_weight:
        weights = []
        while len(weights) < len(edges):
            weights.append(random.uniform(1., 2.0))
        weights = np.asarray(weights, dtype=np.float64)
    else:  # set unit weights for edge costs.
        weights = np.ones(len(edges), dtype=np.float64)
    return edges, weights


def main():
    root = '../dataset/benchmark/'
    command = sys.argv[1]
    if command == 'gen_dataset':
        generate_dataset(root=root)
    elif command == 'run_fix_tr_mu':
        trial_i, num_cpus = int(sys.argv[2]), int(sys.argv[3])
        run_exp_fix_tr_mu(root=root, trial_i=trial_i, num_cpus=num_cpus)
    elif command == 'run_diff_tr':
        trial_i, num_cpus = int(sys.argv[2]), int(sys.argv[3])
        run_exp_diff_tr(root=root, trial_i=trial_i, num_cpus=num_cpus)
    elif command == 'run_diff_mu':
        trial_i, num_cpus = int(sys.argv[2]), int(sys.argv[3])
        run_exp_diff_mu(root=root, trial_i=trial_i, num_cpus=num_cpus)
    elif command == 'run_diff_s':
        trial_i, num_cpus = int(sys.argv[2]), int(sys.argv[3])
        run_exp_diff_s(root=root, trial_i=trial_i, num_cpus=num_cpus)
    elif command == 'show_figure_1':
        show_figure_1()
    elif command == 'show_figure_2':
        show_figure_2()
    elif command == 'show_4_tables':
        show_4_tables()
    elif command == 'show_figure_3':
        show_figure_3()
    elif command == 'show_figure_4':
        for trial_i in range(1):
            show_figure_4(trial_i=trial_i)
    elif command == 'show_figure_5':
        show_figure_5()
    elif command == 'show_figure_6':
        show_figure_6()
    elif command == 'show_figure_7':
        show_figure_7()
    else:
        print_help()


if __name__ == '__main__':
    main()
