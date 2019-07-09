# -*- coding: utf-8 -*-

"""
References:
    [1] Xiao, Lin. "Dual averaging methods for regularized stochastic
        learning and online optimization." Journal of Machine Learning
        Research 11.Oct (2010): 2543-2596.
    [2] Yang, Haiqin, Zenglin Xu, Irwin King, and Michael R. Lyu.
        "Online learning for group lasso." In Proceedings of the 27th
        International Conference on Machine Learning (ICML-10),
        pp. 1191-1198. 2010.
    [3] Duchi, John, Elad Hazan, and Yoram Singer. "Adaptive subgradient
        methods for online learning and stochastic optimization."
        Journal of Machine Learning Research 12.Jul (2011): 2121-2159.
"""

import os
import pickle
import numpy as np
import multiprocessing
from itertools import product
from os import sys, path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from algo_wrapper.base import node_pre_rec_fm
from algo_wrapper.base import least_square_predict
from algo_wrapper.algo_wrapper import algo_online_rda_l1
from algo_wrapper.algo_wrapper import algo_online_da_gl
from algo_wrapper.algo_wrapper import algo_online_da_sgl
from algo_wrapper.algo_wrapper import algo_online_ada_grad
from algo_wrapper.algo_wrapper import algo_online_da_iht
from algo_wrapper.algo_wrapper import algo_online_graph_da


def group_select(style):
    if style == 'line':
        group_list_ = np.asarray(range(784), dtype=np.int32)
        group_size_list_ = np.asarray([28] * 28, dtype=np.int32)
        num_group_ = 28
    elif style == 'grid':
        group_list_ = []
        cube = [0, 1, 28, 29]
        sub_group = np.asarray(cube, dtype=np.int32)
        for i in range(14):
            for j in range(14):
                group_list_.extend(sub_group + j * 2 + i * 2 * 28)
        group_list_ = np.asarray(group_list_, dtype=np.int32)
        group_size_list_ = np.asarray([4] * 14 * 14, dtype=np.int32)
        num_group_ = 14 * 14
    elif style == 'gird-2':
        group_list_ = []
        cube = [0, 1, 3, 4,
                28, 29, 30, 31,
                56, 57, 58, 59,
                84, 85, 86, 87]
        sub_group = np.asarray(cube, dtype=np.int32)
        for i in range(7):
            for j in range(7):
                group_list_.extend(sub_group + j * 4 + i * 4 * 28)
        group_list_ = np.asarray(group_list_, dtype=np.int32)
        group_size_list_ = np.asarray([16] * 7 * 7, dtype=np.int32)
        num_group_ = 7 * 7
    else:
        group_list_ = np.asarray(range(784), dtype=np.int32)
        group_size_list_ = np.asarray([28] * 28, dtype=np.int32)
        num_group_ = 28
    return group_list_, group_size_list_, num_group_


def generate_re(para, result, algo_para, tag):
    pred_nodes = np.nonzero(result['wt'])[0]
    pred_y = least_square_predict(para['x_te'], result['wt'])
    re = {'subgraph': para['subgraph'],
          'trial_i': para['trial_i'],
          'method_name': 'rda-l1',
          'para': algo_para,
          'mse_va': np.mean((pred_y - para[tag[0] + '_va']) ** 2.),
          'mse_te': np.mean((pred_y - para[tag[0] + '_te']) ** 2.),
          'n_pre_rec_fm': node_pre_rec_fm(para['subgraph'], pred_nodes),
          'wt': result['wt'],  # including intercept
          'losses': result['losses'],
          'wt_true': para[tag[1]],
          'run_time': result['total_time'],
          'missed_wt': result['missed_wt'],
          'missed_wt_bar': result['missed_wt_bar']}
    return re


def get_best_re(current, best_node_fm, best_mse):
    # the first result of node-fm
    if best_node_fm is None:
        best_node_fm = current
    # the first result of mse
    if best_mse is None:
        best_mse = current
    # for f1-score of node performance, the higher, the better.
    if best_node_fm['n_pre_rec_fm'][2] < current['n_pre_rec_fm'][2]:
        best_node_fm = current
    # for mse(mean squared error), the lower, the better.
    if best_mse['mse_va'] > current['mse_va']:
        best_mse = current
    return best_node_fm, best_mse


def append_true_model(data, best_node_fm, best_mse):
    best_node_fm['w1'] = data['w1']
    best_node_fm['w2'] = data['w2']
    best_node_fm['w3'] = data['w3']
    best_mse['w1'] = data['w1']
    best_mse['w2'] = data['w2']
    best_mse['w3'] = data['w3']
    return best_node_fm, best_mse


def algo_rda_l1(para):
    data, tag, img_id, n_tr = para
    best_node_fm, best_mse = None, None
    # lambda: to control the sparsity
    lambda_list = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 3e-2, 5e-2, 1e-1, 3e-1, 5e-1, 1e0, 3e0, 5e0, 1e1]
    # gamma: to control the learning rate. (it cannot be too small)
    gamma_list = [1e0, 5e0, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4]
    # rho: to control the sparsity-enhancing parameter.
    rho_list = [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0]
    for (lambda_, gamma, rho) in product(lambda_list, gamma_list, rho_list):
        print(lambda_, gamma, rho)
        x_tr = data['x_tr'][:n_tr]
        y_tr = data[tag[0] + '_tr'][:n_tr]
        w0 = np.zeros(data['p'] + 1)
        result = algo_online_rda_l1(x_tr, y_tr, w0, lambda_, gamma, rho, 'least_square', 0)
        re = generate_re(data, result, (lambda_, gamma, rho), tag)
        best_node_fm, best_mse = get_best_re(re, best_node_fm, best_mse)
    print(n_tr, best_node_fm['n_pre_rec_fm'], best_mse['mse_te'])
    best_node_fm, best_mse = append_true_model(data, best_node_fm, best_mse)
    return best_node_fm, best_mse, tag, n_tr, data['trial_i']


def algo_da_iht(para):
    data, tag, img_id, n_tr = para
    best_node_fm, best_mse = None, None
    s_list = range(30, 100, 2)
    gamma_list = [1e0, 5e0, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4]
    l2_lambda = [0.0]
    for (gamma, lambda_, s) in product(gamma_list, l2_lambda, s_list):
        x_tr = data['x_tr'][:n_tr]
        y_tr = data[tag[0] + '_tr'][:n_tr]
        w0 = np.zeros(data['p'] + 1)
        result = algo_online_da_iht(x_tr, y_tr, w0, gamma, lambda_, s, 'least_square', 0)
        re = generate_re(data, result, (gamma, s), tag)
        best_node_fm, best_mse = get_best_re(re, best_node_fm, best_mse)
    print(n_tr, best_node_fm['n_pre_rec_fm'], best_mse['mse_te'])
    best_node_fm, best_mse = append_true_model(data, best_node_fm, best_mse)
    return best_node_fm, best_mse, tag, n_tr, data['trial_i']


def algo_graph_da_iht(para):
    data, tag, img_id, n_tr = para
    best_node_fm, best_mse = None, None
    s_list = range(30, 100, 2)
    gamma_list = [1e0, 5e0, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4]
    l2_lambda = [0.0]
    ratio_list = [0.1]
    max_num_iter_list = [20]
    for (gamma, lambda_, s, ratio, max_iter) in product(gamma_list, l2_lambda, s_list, ratio_list, max_num_iter_list):
        result = algo_online_graph_da(
            data['x_tr'][:n_tr], data[tag[0] + '_tr'][:n_tr], np.zeros(data['p'] + 1), gamma, lambda_, data['edges'],
            data['weights'], 1, s, ratio, max_iter, 'least_square', 0)
        re = generate_re(data, result, (gamma, lambda_, s), tag)
        best_node_fm, best_mse = get_best_re(re, best_node_fm, best_mse)
    print(n_tr, best_node_fm['n_pre_rec_fm'], best_mse['mse_te'])
    best_node_fm, best_mse = append_true_model(data, best_node_fm, best_mse)
    return best_node_fm, best_mse, tag, n_tr, data['trial_i']


def algo_ada_grad(para):
    data, tag, img_id, n_tr = para
    best_node_fm, best_mse = None, None
    # lambda: to control the sparsity
    lambda_list = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 3e-2, 5e-2, 1e-1, 3e-1, 5e-1, 1e0, 3e0, 5e0, 1e1]
    # eta: to control the learning rate. (it cannot be too small)
    eta_list = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3]
    epsilon_list = [1e-8]
    for (lambda_, eta, epsilon) in product(lambda_list, eta_list, epsilon_list):
        x_tr = data['x_tr'][:n_tr]
        y_tr = data[tag[0] + '_tr'][:n_tr]
        w0 = np.zeros(data['p'] + 1)
        result = algo_online_ada_grad(x_tr, y_tr, w0, lambda_, eta, epsilon, 'least_square', 0)
        re = generate_re(data, result, (lambda_, eta), tag)
        best_node_fm, best_mse = get_best_re(re, best_node_fm, best_mse)
    print(n_tr, best_node_fm['n_pre_rec_fm'], best_mse['mse_te'])
    best_node_fm, best_mse = append_true_model(data, best_node_fm, best_mse)
    return best_node_fm, best_mse, tag, n_tr, data['trial_i']


def algo_da_gl(para):
    data, tag, img_id, n_tr = para
    best_node_fm, best_mse = None, None
    # lambda: to control the sparsity
    lambda_list = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 3e-2, 5e-2, 1e-1, 3e-1, 5e-1, 1e0, 3e0, 5e0, 1e1]
    # gamma: to control the learning rate. (it cannot be too small)
    gamma_list = [1e0, 5e0, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4]
    group_list, group_size_list, num_group = group_select(style='grid')
    for (lambda_, gamma) in product(lambda_list, gamma_list):
        x_tr = data['x_tr'][:n_tr]
        y_tr = data[tag[0] + '_tr'][:n_tr]
        w0 = np.zeros(data['p'] + 1)
        result = algo_online_da_gl(
            x_tr, y_tr, w0, lambda_, gamma, group_list,
            group_size_list, num_group, 'least_square', 0)
        re = generate_re(data, result, (lambda_, gamma), tag)
        best_node_fm, best_mse = get_best_re(re, best_node_fm, best_mse)
    print(n_tr, best_node_fm['n_pre_rec_fm'], best_mse['mse_te'])
    best_node_fm, best_mse = append_true_model(data, best_node_fm, best_mse)
    return best_node_fm, best_mse, tag, n_tr, data['trial_i']


def algo_da_sgl(para):
    data, tag, img_id, n_tr = para
    best_node_fm, best_mse = None, None
    # lambda: to control the sparsity
    lambda_list = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 3e-2, 5e-2, 1e-1, 3e-1, 5e-1, 1e0, 3e0, 5e0, 1e1]
    # gamma: to control the learning rate. (it cannot be too small)
    gamma_list = [1e0, 5e0, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4]
    group_list, group_size_list, num_group = group_select(style='grid')
    r = np.asarray([1.] * num_group, dtype=np.int32)
    for (lambda_, gamma) in product(lambda_list, gamma_list):
        x_tr = data['x_tr'][:n_tr]
        y_tr = data[tag[0] + '_tr'][:n_tr]
        w0 = np.zeros(data['p'] + 1)
        result = algo_online_da_sgl(
            x_tr, y_tr, w0, lambda_, gamma, group_list, group_size_list, r, num_group, 'least_square', 0)
        re = generate_re(data, result, (lambda_, gamma), tag)
        best_node_fm, best_mse = get_best_re(re, best_node_fm, best_mse)
    print(n_tr, best_node_fm['n_pre_rec_fm'], best_mse['mse_te'])
    best_node_fm, best_mse = append_true_model(data, best_node_fm, best_mse)
    return best_node_fm, best_mse, tag, n_tr, data['trial_i']


def generate_single_data(n, image_id):
    def simu_graph(num_nodes):
        edges_, weights_ = [], []
        length = int(np.sqrt(num_nodes))
        width_, index_ = length, 0
        for i in range(length):
            for j in range(width_):
                if (index_ % length) != (length - 1):
                    edges_.append((index_, index_ + 1))
                    weights_.append(1.0)
                    if index_ + length < int(width_ * length):
                        edges_.append((index_, index_ + length))
                        weights_.append(1.0)
                else:
                    if index_ + length < int(width_ * length):
                        edges_.append((index_, index_ + length))
                        weights_.append(1.0)
                index_ += 1
        return np.asarray(edges_, dtype=int), np.asarray(weights_)

    p = 28 * 28
    edges, weights = simu_graph(p)  # get grid graph
    import networkx as nx
    g = nx.Graph()
    for edge in edges:
        g.add_edge(edge[0], edge[1])
    all_data = []
    sparse_images = dict()
    for i in range(10):
        sparse_images[i] = dict()
        online_data = pickle.load(open('../dataset/mnist/mnist_img_%d.pkl' % i))
        for item in online_data:
            img, img_id = item[0], item[1]
            if len(img.nonzero()[0]) <= 100:
                h = g.subgraph(img.nonzero()[0])
                if nx.number_connected_components(h) == 1:
                    sparse_images[i][len(img.nonzero()[0])] = img
    for i in range(10):
        min_id = min(sparse_images[i].keys())
        sparse_images[i] = sparse_images[i][min_id]
        print(min_id)
    for trial_i in range(20):
        p = 28 * 28
        edges, weights = simu_graph(p)  # get grid graph
        img = sparse_images[image_id]
        w1, w2, w3 = np.zeros(p), np.zeros(p), np.zeros(p)
        w1 = img / 255.  # to normalized to [0,1]
        w2[img.nonzero()[0]] = np.ones(len(img.nonzero()[0]))
        w3[img.nonzero()[0]] = np.random.normal(0., 1., len(img.nonzero()[0]))
        x = (np.random.normal(0., 1., (n * p)) / np.sqrt(n)).reshape((n, p))
        y1, y2, y3 = np.dot(x, w1), np.dot(x, w2), np.dot(x, w3)
        data = {'x_tr': x[:n - 400],
                'x_va': x[n - 400:n - 200],
                'x_te': x[n - 200:n],
                'y1_tr': y1[:n - 400],
                'y2_tr': y2[:n - 400],
                'y3_tr': y3[:n - 400],
                'y1_va': y1[n - 400:n - 200],
                'y2_va': y2[n - 400:n - 200],
                'y3_va': y3[n - 400:n - 200],
                'y1_te': y1[n - 200:n],
                'y2_te': y2[n - 200:n],
                'y3_te': y3[n - 200:n],
                'w1': w1,
                'w2': w2,
                'w3': w3,
                'n': n,
                'p': p,
                's': len(img.nonzero()[0]),
                'subgraph': img.nonzero()[0],
                'edges': edges,
                'weights': weights,
                'trial_i': trial_i,
                'img_id': image_id}
        all_data.append(data)
    return all_data


def show_figure_8():
    from matplotlib import rc
    import matplotlib.pyplot as plt
    from pylab import rcParams
    import matplotlib.gridspec as gridspec
    plt.rc('font', **{'size': 14})
    rc('text', usetex=True)
    rcParams['figure.figsize'] = 7.8, 8

    sparse_mnist = pickle.load(open('../dataset/mnist/sparse_mnist_images.pkl'))
    for item in sparse_mnist:
        img = sparse_mnist[item]
        sparse_mnist[item] = dict()
        for tag in ['w1', 'w2', 'w3']:
            if tag == 'w1':
                sparse_mnist[item][tag] = img / 255.
            if tag == 'w2':
                nonzeros = np.nonzero(img)
                sparse_mnist[item][tag] = np.zeros(784)
                sparse_mnist[item][tag][nonzeros] = 1.0
            if tag == 'w3':
                nonzeros = np.nonzero(img)
                sparse_mnist[item][tag] = np.zeros(784)
                sparse_mnist[item][tag][nonzeros] = np.random.normal(loc=0.0, scale=1.0, size=len(nonzeros[0]))
    num_trials, p = 20, len(range(50, 1001, 50))
    n_tr_indices = {key: _ for _, key in enumerate(range(50, 1001, 50))}

    grid = gridspec.GridSpec(4, 15)
    ax00 = plt.subplot(grid[0, 0:4])
    plt.xticks(())
    plt.yticks(())
    ax01 = plt.subplot(grid[0, 4:8])
    plt.xticks(())
    plt.yticks(())
    ax02 = plt.subplot(grid[0, 8:12])
    plt.xticks(())
    plt.yticks(())
    ax03 = plt.subplot(grid[0, 12:15])
    ax03.plot()
    ax03.set_xticks([])
    ax03.set_yticks([])
    for str_ in ['right', 'top', 'left']:
        ax03.spines[str_].set_visible(False)
    plt.xticks(())
    plt.yticks(())
    ax10 = plt.subplot(grid[1, 0:5])
    ax11 = plt.subplot(grid[1, 5:10])
    ax12 = plt.subplot(grid[1, 10:15])
    ax20 = plt.subplot(grid[2, 0:5])
    ax21 = plt.subplot(grid[2, 5:10])
    ax22 = plt.subplot(grid[2, 10:15])
    ax30 = plt.subplot(grid[3, 0:5])
    ax31 = plt.subplot(grid[3, 5:10])
    ax32 = plt.subplot(grid[3, 10:15])
    ax = np.asarray([[ax00, ax01, ax02], [ax10, ax11, ax12], [ax20, ax21, ax22], [ax30, ax31, ax32]])

    for i in range(1, 4):
        for j in range(3):
            ax[i, j].grid(b=True, which='both', color='lightgray', linestyle='--', axis='both')
            ax[i, j].set_xlim([0, 1050])
            ax[i, j].set_xticks([200, 400, 600, 800])
            ax[i, j].set_ylim([0., 1.])
            ax[i, j].set_yticks([0.2, 0.4, 0.6, 0.8])

    ax[1, 0].set_ylabel(r"\textit{Normalized Model}", fontsize=14, labelpad=0.02)
    ax[2, 0].set_ylabel(r"\textit{Constant Model}", fontsize=14, labelpad=0.02)
    ax[3, 0].set_ylabel(r"\textit{Gaussian Model}", fontsize=14, labelpad=0.02)
    ax[3, 0].set_xlabel(r"Samples Seen", fontsize=14)
    ax[3, 1].set_xlabel(r"Samples Seen", fontsize=14)
    ax[3, 2].set_xlabel(r"Samples Seen", fontsize=14)
    ax[1, 0].set_xticklabels('')
    ax[2, 0].set_xticklabels('')
    for j in range(3):
        ax[0, j].set_xticklabels('')
        ax[0, j].set_yticklabels('')
        ax[0, j].tick_params(axis='both', which='both', length=0)
    for j in range(1, 3):
        for i in range(1, 4):
            ax[i, j].tick_params(axis='both', which='both', length=0)
        ax[1, j].set_xticklabels('')
        ax[2, j].set_xticklabels('')
        ax[1, j].set_yticklabels('')
        ax[2, j].set_yticklabels('')
        ax[3, j].set_yticklabels('')
    color_list = ['g', 'm', 'y', 'brown', 'b', 'r']
    marker_list = ['v', '^', '>', '<', 'D', 's']
    label_list = [r'$\displaystyle \ell_1$-\textsc{RDA}', r'\textsc{DA-IHT}', r'\textsc{AdaGrad}',
                  r'\textsc{DA-GL}', r'\textsc{DA-SGL}', r'\textsc{GraphDA}']
    method_list = ['rda-l1', 'da-iht', 'ada-grad', 'da-gl', 'da-sgl', 'graph-da-iht']
    for img_id_ind, img_id in enumerate([0, 4, 5]):
        for method_ind, method in enumerate(method_list):
            results = {'mse_te': {'w1': np.zeros(shape=(num_trials, p)),
                                  'w2': np.zeros(shape=(num_trials, p)),
                                  'w3': np.zeros(shape=(num_trials, p))},
                       'node-fm': {'w1': np.zeros(shape=(num_trials, p)),
                                   'w2': np.zeros(shape=(num_trials, p)),
                                   'w3': np.zeros(shape=(num_trials, p))}}
            results_pool = []
            for trial_i in range(num_trials):
                _ = pickle.load(open('../results/mnist/fig%d_trial_%02d_%s.pkl' % (img_id, trial_i, method)))
                results_pool.extend(_)
            print(method)
            for result in results_pool:
                re_node_fm, re_mse, tag, n_tr, trial_i = result
                results['mse_te'][tag[1]][trial_i][n_tr_indices[n_tr]] = re_mse['mse_te']
                results['node-fm'][tag[1]][trial_i][n_tr_indices[n_tr]] = re_node_fm['n_pre_rec_fm'][2]
                if (method == 'da-gl' or method == 'da-sgl') and (tag[1] == 'w1'):
                    print(method, n_tr, tag, re_node_fm['para'], re_node_fm['n_pre_rec_fm'])
            ax[0, img_id_ind].imshow(sparse_mnist[img_id]['w1'].reshape(28, 28), cmap='gray')
            for ind, tag in enumerate(['w1', 'w2', 'w3']):
                print(method, np.mean(results['mse_te'][tag], axis=0))
                ax[ind + 1, img_id_ind].errorbar(range(50, 1001, 50),
                                                 y=np.mean(results['node-fm'][tag], axis=0),
                                                 yerr=np.std(results['node-fm'][tag], axis=0),
                                                 marker=marker_list[method_ind], markersize=4.,
                                                 markerfacecolor='white',
                                                 color=color_list[method_ind], linewidth=1.,
                                                 label=label_list[method_ind])
    ax[1, 2].legend(loc='center right', fontsize=12,
                    bbox_to_anchor=(1.05, 1.45),
                    frameon=True, borderpad=0.1, labelspacing=0.2,
                    handletextpad=0.1, markerfirst=True)
    ax[1, 0].set_xlabel('Training Samples')
    ax[1, 1].set_xlabel('Training Samples')
    ax[1, 2].set_xlabel('Training Samples')
    plt.subplots_adjust(wspace=.0, hspace=0)
    plt.savefig('../results/mnist/figs/figure_8.pdf',
                dpi=1200, bbox_inches='tight', pad_inches=0, format='pdf')
    plt.close()


def show_figure_11():
    sparse_mnist = pickle.load(open('../dataset/mnist/sparse_mnist_images.pkl'))
    for item in sparse_mnist:
        img = sparse_mnist[item]
        sparse_mnist[item] = dict()
        for tag in ['w1', 'w2', 'w3']:
            if tag == 'w1':
                sparse_mnist[item][tag] = img / 255.
            if tag == 'w2':
                nonzeros = np.nonzero(img)
                sparse_mnist[item][tag] = np.zeros(784)
                sparse_mnist[item][tag][nonzeros] = 1.0
            if tag == 'w3':
                nonzeros = np.nonzero(img)
                sparse_mnist[item][tag] = np.zeros(784)
                sparse_mnist[item][tag][nonzeros] = np.random.normal(loc=0.0, scale=1.0, size=len(nonzeros[0]))
    num_trials, p = 20, len(range(50, 1001, 50))
    n_tr_indices = {key: _ for _, key in enumerate(range(50, 1001, 50))}
    from matplotlib import rc
    import matplotlib.pyplot as plt
    from pylab import rcParams

    plt.rc('font', **{'size': 12})
    rc('text', usetex=True)
    rcParams['figure.figsize'] = 16, 8
    fig, ax = plt.subplots(4, 7)
    for i in range(1, 4):
        for j in range(7):
            ax[i, j].grid(b=True, which='both', color='lightgray', linestyle='--', axis='both')
            ax[i, j].set_xlim([0, 1050])
            ax[i, j].set_xticks([200, 400, 600, 800])
            ax[i, j].set_ylim([0., 1.])
            ax[i, j].set_yticks([0.2, 0.4, 0.6, 0.8])

    ax[1, 0].set_ylabel(r"\textit{Normalized Model}", fontsize=14)
    ax[2, 0].set_ylabel(r"\textit{Constant Model}", fontsize=14)
    ax[3, 0].set_ylabel(r"\textit{Gaussian Model}", fontsize=14)
    for i in range(7):
        ax[3, i].set_xlabel(r"Samples Seen", fontsize=14)
    for i in range(1, 3):
        ax[i, 0].set_xticklabels('')
    for j in range(7):
        ax[0, j].set_xticklabels('')
        ax[0, j].set_yticklabels('')
        ax[0, j].tick_params(axis='both', which='both', length=0)
    for i in range(1, 3):
        for j in range(1, 7):
            ax[i, j].tick_params(axis='both', which='both', length=0)
            ax[i, j].set_xticklabels('')
            ax[i, j].set_yticklabels('')
    for j in range(1, 7):
        ax[3, j].set_yticklabels('')

    color_list = ['g', 'm', 'y', 'brown', 'b', 'r']
    marker_list = ['v', '^', '>', '<', 'D', 's']
    label_list = [r'$\displaystyle \ell_1$-\textsc{RDA}', r'\textsc{DA-IHT}', r'\textsc{AdaGrad}',
                  r'\textsc{DA-GL}', r'\textsc{DA-SGL}', r'\textsc{GraphDA}']
    method_list = ['rda-l1', 'da-iht', 'ada-grad', 'da-gl', 'da-sgl', 'graph-da-iht']
    for img_id_ind, img_id in enumerate([1, 2, 3, 6, 7, 8, 9]):
        for method_ind, method in enumerate(method_list):
            results = {'mse_te': {'w1': np.zeros(shape=(num_trials, p)),
                                  'w2': np.zeros(shape=(num_trials, p)),
                                  'w3': np.zeros(shape=(num_trials, p))},
                       'node-fm': {'w1': np.zeros(shape=(num_trials, p)),
                                   'w2': np.zeros(shape=(num_trials, p)),
                                   'w3': np.zeros(shape=(num_trials, p))}}
            results_pool = []
            for trial_i in range(num_trials):
                _ = pickle.load(open('../results/mnist/fig%d_trial_%02d_%s.pkl' % (img_id, trial_i, method)))
                results_pool.extend(_)
            print(method)
            for result in results_pool:
                re_node_fm, re_mse, tag, n_tr, trial_i = result
                results['mse_te'][tag[1]][trial_i][n_tr_indices[n_tr]] = re_mse['mse_te']
                results['node-fm'][tag[1]][trial_i][n_tr_indices[n_tr]] = re_node_fm['n_pre_rec_fm'][2]
                if (method == 'da-gl' or method == 'da-sgl') and (tag[1] == 'w1'):
                    print(method, n_tr, tag, re_node_fm['para'], re_node_fm['n_pre_rec_fm'])
            ax[0, img_id_ind].imshow(sparse_mnist[img_id]['w1'].reshape(28, 28), cmap='gray')
            for ind, tag in enumerate(['w1', 'w2', 'w3']):
                print(method, np.mean(results['mse_te'][tag], axis=0))
                ax[ind + 1, img_id_ind].errorbar(
                    range(50, 1001, 50), y=np.mean(results['node-fm'][tag], axis=0),
                    yerr=np.std(results['node-fm'][tag], axis=0), marker=marker_list[method_ind], markersize=4.,
                    markerfacecolor='white',
                    color=color_list[method_ind], linewidth=1., label=label_list[method_ind])
    ax[3, 3].legend(loc='lower center', fontsize=14, bbox_to_anchor=(0.5, -0.6), ncol=6,
                    frameon=True, borderpad=0.1, labelspacing=0.2, handletextpad=0.1, markerfirst=True)
    plt.subplots_adjust(wspace=.0, hspace=0)
    plt.savefig('../results/mnist/figs/figure_11.pdf', dpi=600, bbox_inches='tight', pad_inches=0, format='pdf')
    plt.close()


def run_test():
    method = sys.argv[2]
    num_cpus = int(sys.argv[3])
    start_img_id = int(sys.argv[4])
    end_img_id = int(sys.argv[5])
    for img_id in range(start_img_id, end_img_id):
        all_data = generate_single_data(n=1400, image_id=img_id)
        all_tags = [('y1', 'w1', 0), ('y2', 'w2', 1), ('y3', 'w3', 2)]
        input_paras = [(data, tag, img_id, n_tr) for data, tag, n_tr in
                       product(all_data, all_tags, range(50, 1001, 50))]
        pool = multiprocessing.Pool(processes=num_cpus)
        if method == 'rda-l1':
            results_pool = pool.map(algo_rda_l1, input_paras)
        elif method == 'da-iht':
            results_pool = pool.map(algo_da_iht, input_paras)
        elif method == 'ada-grad':
            results_pool = pool.map(algo_ada_grad, input_paras)
        elif method == 'da-gl':
            results_pool = pool.map(algo_da_gl, input_paras)
        elif method == 'da-sgl':
            results_pool = pool.map(algo_da_sgl, input_paras)
        elif method == 'graph-da-iht':
            results_pool = pool.map(algo_graph_da_iht, input_paras)
        else:
            results_pool = None
        pool.close()
        pool.join()
        for trial_i in range(20):
            results = []
            for result in results_pool:
                if result[-1] == trial_i:
                    results.append(result)
            f_name = '../results/mnist/fig%d_trial_%02d_%s.pkl' % (img_id, trial_i, method)
            pickle.dump(results, open(f_name, 'wb'))


def main():
    command = sys.argv[1]
    if command == 'show_figure_8':
        show_figure_8()
    elif command == 'show_figure_11':
        show_figure_11()
    elif command == 'run_test':
        run_test()


if __name__ == '__main__':
    main()
