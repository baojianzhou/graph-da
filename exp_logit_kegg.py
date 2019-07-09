# -*- coding: utf-8 -*-
import sys
import time
import pickle
import numpy as np
from os import path
import multiprocessing
from itertools import product

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from algo_wrapper.base import node_pre_rec_fm
from algo_wrapper.base import logistic_predict

from algo_wrapper.algo_wrapper import algo_online_rda_l1
from algo_wrapper.algo_wrapper import algo_online_da_gl
from algo_wrapper.algo_wrapper import algo_online_da_sgl
from algo_wrapper.algo_wrapper import algo_online_ada_grad
from algo_wrapper.algo_wrapper import algo_online_da_iht
from algo_wrapper.algo_wrapper import algo_online_best_subset
from algo_wrapper.algo_wrapper import algo_online_graph_da

num_tr, num_va, num_te = 200, 200, 200
strategies = ['s1', 's2']


def generate_re(para, result, algo_para):
    x_va, y_va = para['x_va'], para['y_va']
    x_te, y_te = para['x_te'], para['y_te']
    te_pre_prob, te_y_pred = logistic_predict(x_te, result['wt'])
    va_pre_prob, va_y_pred = logistic_predict(x_va, result['wt'])
    pred_nodes = np.nonzero(result['wt'][:para['p']])[0]
    pred_nodes_dict = {para['nodes'][_]: '' for _ in range(len(para['nodes'])) if _ in pred_nodes}
    pred_subgraph = {'nodes': set(pred_nodes_dict.keys()), 'edges': [], 'weights': []}
    for edge_ind, edge in enumerate(para['edges']):
        if edge[0] in pred_nodes_dict and edge[1] in pred_nodes_dict:
            pred_subgraph['edges'].append(edge)
            pred_subgraph['weights'].append(para['weights'][edge_ind])
    re = {'strategy': para['strategy'],
          'pathway_id': para['pathway_id'],
          'subgraph': para['subgraph'],
          'n_tr': algo_para[-2],
          'trial_i': algo_para[-1],
          'method_name': '--',
          'para': algo_para,
          'pred_subgraph': pred_subgraph,
          'auc': roc_auc_score(y_true=y_te, y_score=te_y_pred),
          'acc': accuracy_score(y_true=y_te, y_pred=te_y_pred),
          'roc': roc_curve(y_true=y_te, y_score=te_pre_prob),
          'va_acc': accuracy_score(y_true=y_va, y_pred=va_y_pred),
          'va_fm': node_pre_rec_fm(para['subgraph'], pred_nodes)[2],
          'n_pre_rec_fm': node_pre_rec_fm(para['subgraph'], pred_nodes),
          'wt': result['wt'],  # including intercept
          'losses': result['losses'],
          'run_time': result['total_time'],
          'missed_wt': result['missed_wt'],
          'missed_wt_bar': result['missed_wt_bar'],
          'nonzeros_wt': result['nonzeros_wt'],
          'nonzeros_wt_bar': result['nonzeros_wt_bar']}
    return re


def generate_re_overlap_group_lasso(para, result, algo_para, extended_features):
    x_va, y_va = para['x_va'], para['y_va']
    x_te, y_te = para['x_te'], para['y_te']
    x_va_extend = np.zeros(shape=(len(x_va), para['p']))
    x_te_extend = np.zeros(shape=(len(x_te), para['p']))
    for ind, gene in enumerate(extended_features):
        x_va_extend[:, ind] = x_va[:, para['node_indices'][gene]]
        x_te_extend[:, ind] = x_te[:, para['node_indices'][gene]]
    te_pre_prob, te_y_pred = logistic_predict(x_te_extend, result['wt'])
    va_pre_prob, va_y_pred = logistic_predict(x_va_extend, result['wt'])
    pred_nodes = [para['node_indices'][extended_features[_]] for _ in np.nonzero(result['wt'][:para['p']])[0]]
    pred_nodes_dict = {para['nodes'][_]: '' for _ in range(len(para['nodes'])) if _ in pred_nodes}
    pred_subgraph = {'nodes': set(pred_nodes_dict.keys()), 'edges': [], 'weights': []}
    for edge_ind, edge in enumerate(para['edges']):
        if edge[0] in pred_nodes_dict and edge[1] in pred_nodes_dict:
            pred_subgraph['edges'].append(edge)
            pred_subgraph['weights'].append(para['weights'][edge_ind])
    re = {'strategy': para['strategy'],
          'pathway_id': para['pathway_id'],
          'subgraph': para['subgraph'],
          'n_tr': algo_para[-2],
          'trial_i': algo_para[-1],
          'method_name': '--',
          'para': algo_para,
          'pred_subgraph': pred_subgraph,
          'auc': roc_auc_score(y_true=y_te, y_score=te_y_pred),
          'acc': accuracy_score(y_true=y_te, y_pred=te_y_pred),
          'roc': roc_curve(y_true=y_te, y_score=te_pre_prob),
          'va_acc': accuracy_score(y_true=y_va, y_pred=va_y_pred),
          'va_fm': node_pre_rec_fm(para['subgraph'], pred_nodes)[2],
          'n_pre_rec_fm': node_pre_rec_fm(para['subgraph'], pred_nodes),
          'wt': result['wt'],  # including intercept
          'losses': result['losses'],
          'run_time': result['total_time'],
          'missed_wt': result['missed_wt'],
          'missed_wt_bar': result['missed_wt_bar'],
          'nonzeros_wt': result['nonzeros_wt'],
          'nonzeros_wt_bar': result['nonzeros_wt_bar']}
    return re


def get_best_re(current, best_acc):
    # the first result of mse
    if best_acc is None:
        best_acc = current
    # for auc, the higher, the better.
    if best_acc['va_acc'] <= current['va_acc']:
        best_acc = current
    return best_acc


def print_re(best_auc_re, best_node_fm, pathway_id, sub_graph):
    print('-' * 80)
    print('pathway_id: %s, number of genes: %d' % (pathway_id, len(sub_graph)))
    print('auc: %.4f -- %.4f' % (best_auc_re['auc'], best_node_fm['auc']))
    print('n_pre_rec_fm: (%.4f,%.4f,%.4f) -- (%.4f,%.4f,%.4f) ' %
          (best_auc_re['n_pre_rec_fm'][0], best_auc_re['n_pre_rec_fm'][1],
           best_auc_re['n_pre_rec_fm'][2], best_node_fm['n_pre_rec_fm'][0],
           best_node_fm['n_pre_rec_fm'][1], best_node_fm['n_pre_rec_fm'][2]))
    print('para: ', best_auc_re['para'], best_node_fm['para'])
    print('-' * 80)


def algo_rda_l1(para):
    input_p, strategy, pathway_id, trial_i, n_tr = para
    file_name = 'input/data_%s_%s_%d.pkl' % (strategy, pathway_id, trial_i)
    para = pickle.load(open(input_p + file_name))
    best_acc_re = None
    # lambda: to control the sparsity
    lambda_list = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 3e-2, 5e-2, 1e-1, 3e-1, 5e-1, 1e0, 3e0, 5e0, 1e1]
    # gamma: to control the learning rate. (it cannot be too small)
    gamma_list = [1e0, 5e0, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4]
    # rho: to control the sparsity-enhancing parameter.
    rho_list = [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0]
    for (lambda_, gamma, rho) in product(lambda_list, gamma_list, rho_list):
        x_tr, y_tr = para['x_tr'][:n_tr], para['y_tr'][:n_tr]
        w0 = np.zeros(para['p'] + 1)
        result = algo_online_rda_l1(x_tr, y_tr, w0, lambda_, gamma, rho, 'logistic', 0)
        # para, result, algo_para, tag
        re = generate_re(para, result, (lambda_, gamma, rho, n_tr, trial_i))
        # tuning parameters by auc and node f-measure
        best_acc_re = get_best_re(re, best_acc_re)
    print('-' * 100)
    print('---best-node-fm---')
    lambda_, gamma, rho = best_acc_re['para'][:3]
    print('best parameter pair (lambda_, gamma, rho, n_tr):'
          '(%.2e,%.2e,%.2e,%3d)' % (lambda_, gamma, rho, n_tr))
    print('node-(pre,rec,fm,non-zero):( %.4f %.4f %.4f %4d )' %
          (best_acc_re['n_pre_rec_fm'][0], best_acc_re['n_pre_rec_fm'][1],
           best_acc_re['n_pre_rec_fm'][2], best_acc_re['nonzeros_wt'][-1]))
    print('auc: %.4f acc: %.4f' % (best_acc_re['auc'], best_acc_re['acc']))
    return best_acc_re


def algo_da_iht(para):
    input_p, strategy, pathway_id, trial_i, n_tr = para
    file_name = 'input/data_%s_%s_%d.pkl' % (strategy, pathway_id, trial_i)
    para = pickle.load(open(input_p + file_name))
    best_acc_re = None
    s_list = [40, 45, 50, 55, 60]
    gamma_list = [1e0, 5e0, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4]
    l2_lambda_list = [0.0]
    for (s, gamma, l2_lambda) in product(s_list, gamma_list, l2_lambda_list):
        x_tr, y_tr = para['x_tr'][:n_tr], para['y_tr'][:n_tr]
        w0 = np.zeros(para['p'] + 1)
        result = algo_online_da_iht(x_tr, y_tr, w0, gamma, l2_lambda, s, 'logistic', 0)
        # para, result, algo_para, tag
        re = generate_re(para, result, (s, gamma, l2_lambda, n_tr, trial_i))
        # tuning parameters by acc
        best_acc_re = get_best_re(re, best_acc_re)
    print('-' * 100)
    s, gamma, l2_lambda = best_acc_re['para'][:3]
    print('best parameter pair (s, gamma, l2_lambda, n_tr):'
          '(%d,%.2e,%.2e,%3d)' % (s, gamma, l2_lambda, n_tr))
    print('node-(pre,rec,fm,non-zero):( %.4f %.4f %.4f %4d )' %
          (best_acc_re['n_pre_rec_fm'][0], best_acc_re['n_pre_rec_fm'][1],
           best_acc_re['n_pre_rec_fm'][2], best_acc_re['nonzeros_wt'][-1]))
    print('auc: %.4f acc: %.4f' % (best_acc_re['auc'], best_acc_re['acc']))
    return best_acc_re


def algo_ada_grad(para):
    input_p, strategy, pathway_id, trial_i, n_tr = para
    file_name = 'input/data_%s_%s_%d.pkl' % (strategy, pathway_id, trial_i)
    para = pickle.load(open(input_p + file_name))
    best_acc_re = None
    # lambda: to control the sparsity
    lambda_list = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 3e-2, 5e-2, 1e-1, 3e-1, 5e-1, 1e0, 3e0, 5e0, 1e1]
    # eta: to control the learning rate. (it cannot be too small)
    eta_list = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3]
    epsilon_list = [1e-8]
    for (lambda_, eta, delta) in product(lambda_list, eta_list, epsilon_list):
        x_tr, y_tr = para['x_tr'][:n_tr], para['y_tr'][:n_tr]
        w0 = np.zeros(para['p'] + 1)
        result = algo_online_ada_grad(x_tr, y_tr, w0, lambda_, eta, delta, 'logistic', 0)
        re = generate_re(para, result, (lambda_, eta, n_tr, trial_i))
        best_acc_re = get_best_re(re, best_acc_re)
    print('-' * 100)
    print('---best-node-fm---')
    lambda_, eta = best_acc_re['para'][:2]
    print('best parameter pair (lambda_, eta, n_tr):'
          '(%.2e,%.2e,%3d)' % (lambda_, eta, n_tr))
    print('node-(pre,rec,fm,non-zero):( %.4f %.4f %.4f %4d )' %
          (best_acc_re['n_pre_rec_fm'][0], best_acc_re['n_pre_rec_fm'][1],
           best_acc_re['n_pre_rec_fm'][2], best_acc_re['nonzeros_wt'][-1]))
    print('auc: %.4f acc: %.4f' % (best_acc_re['auc'], best_acc_re['acc']))
    return best_acc_re


def algo_da_gl(para):
    def group_select():
        extended_features_ = []
        pathway_ids = list(para['pathways'].keys())
        group_size_list_ = []
        for id_ in pathway_ids:
            extended_features_.extend(para['pathways'][id_])
            group_size_list_.append(len(para['pathways'][id_]))
        group_size_list_ = np.asarray(group_size_list_)
        group_list_ = range(len(extended_features_))
        group_list_ = np.asarray(group_list_, dtype=np.int32)
        num_group_ = len(para['pathways'])
        return group_list_, group_size_list_, num_group_, extended_features_

    input_p, strategy, pathway_id, trial_i, n_tr = para
    file_name = 'input/data_%s_%s_%d.pkl' % (strategy, pathway_id, trial_i)
    para = pickle.load(open(input_p + file_name))
    best_acc_re = None
    # lambda: to control the sparsity
    lambda_list = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 3e-2, 5e-2, 1e-1, 3e-1, 5e-1, 1e0, 3e0, 5e0, 1e1]
    # gamma: to control the learning rate. (it cannot be too small)
    gamma_list = [1e0, 5e0, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4]
    g_list, g_size_list, num_group, extended_features = group_select()
    para['p'] = len(g_list)
    for (lambda_, gamma) in product(lambda_list, gamma_list):
        x_tr = para['x_tr'][:n_tr]
        y_tr = para['y_tr'][:n_tr]
        # this will be extended by groups
        x_tr_ext = np.zeros(shape=(n_tr, para['p']))
        for ind, gene in enumerate(extended_features):
            x_tr_ext[:, ind] = x_tr[:, para['node_indices'][gene]]
        w0 = np.zeros(para['p'] + 1)
        result = algo_online_da_gl(x_tr_ext, y_tr, w0, lambda_, gamma, g_list, g_size_list, num_group, 'logistic', 0)
        re = generate_re_overlap_group_lasso(para, result, (lambda_, gamma, n_tr, trial_i), extended_features)
        best_acc_re = get_best_re(re, best_acc_re)
    print('-' * 100)
    print('---best-node-fm---')
    lambda_, gamma = best_acc_re['para'][:2]
    print('best parameter pair (lambda_, gamma, n_tr):'
          '(%.2e,%.2e,%3d)' % (lambda_, gamma, n_tr))
    print('node-(pre,rec,fm,non-zero):( %.4f %.4f %.4f %4d )' %
          (best_acc_re['n_pre_rec_fm'][0], best_acc_re['n_pre_rec_fm'][1],
           best_acc_re['n_pre_rec_fm'][2], best_acc_re['nonzeros_wt'][-1]))
    print('auc: %.4f acc: %.4f' % (best_acc_re['auc'], best_acc_re['acc']))
    return best_acc_re


def algo_da_sgl(para):
    def group_select():
        extended_features_ = []
        pathway_ids = list(para['pathways'].keys())
        group_size_list_ = []
        for id_ in pathway_ids:
            extended_features_.extend(para['pathways'][id_])
            group_size_list_.append(len(para['pathways'][id_]))
        group_size_list_ = np.asarray(group_size_list_)
        group_list_ = range(len(extended_features_))
        group_list_ = np.asarray(group_list_, dtype=np.int32)
        num_group_ = len(para['pathways'])
        return group_list_, group_size_list_, num_group_, extended_features_

    input_p, strategy, pathway_id, trial_i, n_tr = para
    file_name = 'input/data_%s_%s_%d.pkl' % (strategy, pathway_id, trial_i)
    para = pickle.load(open(input_p + file_name))
    best_acc_re = None
    # lambda: to control the sparsity
    lambda_list = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 3e-2, 5e-2, 1e-1, 3e-1, 5e-1, 1e0, 3e0, 5e0, 1e1]
    # gamma: to control the learning rate. (it cannot be too small)
    gamma_list = [1e0, 5e0, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4]
    g_list, g_s_list, num_group, extended_features = group_select()
    para['p'] = len(g_list)
    r = np.asarray([1.] * num_group, dtype=np.int32)
    for (lambda_, gamma) in product(lambda_list, gamma_list):
        x_tr = para['x_tr'][:n_tr]
        y_tr = para['y_tr'][:n_tr]
        # this will be extended by groups
        x_tr_ext = np.zeros(shape=(n_tr, para['p']))
        for ind, gene in enumerate(extended_features):
            x_tr_ext[:, ind] = x_tr[:, para['node_indices'][gene]]
        w0 = np.zeros(para['p'] + 1)
        result = algo_online_da_sgl(x_tr_ext, y_tr, w0, lambda_, gamma, g_list, g_s_list, r, num_group, 'logistic', 0)
        re = generate_re_overlap_group_lasso(para, result, (lambda_, gamma, n_tr, trial_i), extended_features)
        best_acc_re = get_best_re(re, best_acc_re)
    print('-' * 100)
    print('---best-node-fm---')
    lambda_, gamma = best_acc_re['para'][:2]
    print('best parameter pair (lambda_, gamma, n_tr):'
          '(%.2e,%.2e,%3d)' % (lambda_, gamma, n_tr))
    print('node-(pre,rec,fm,non-zero):( %.4f %.4f %.4f %4d )' %
          (best_acc_re['n_pre_rec_fm'][0], best_acc_re['n_pre_rec_fm'][1],
           best_acc_re['n_pre_rec_fm'][2], best_acc_re['nonzeros_wt'][-1]))
    print('auc: %.4f acc: %.4f' % (best_acc_re['auc'], best_acc_re['acc']))
    return best_acc_re


def algo_best_subset(para):
    input_p, strategy, pathway_id, trial_i, n_tr = para
    file_name = 'input/data_%s_%s_%d.pkl' % (strategy, pathway_id, trial_i)
    para = pickle.load(open(input_p + file_name))
    x_tr, y_tr = para['x_tr'][:n_tr], para['y_tr'][:n_tr]
    w0 = np.zeros(para['p'] + 1)
    gamma_list = [1e0, 5e0, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4]
    l2_lambda_list = [0.0]
    best_acc_re = None
    # 886, 894, 955, 1033, 1034, 1035, 1041, 1084, 1085, 1086, 1122,
    # 1123, 1133, 1135, 1136, 1159, 1165, 1183, 1192, 1228, 1250, 1251,
    # 1268, 1273, 1275, 1286, 1291, 1303, 1304, 1305, 1306, 1307, 1308,
    # 1443, 1445, 1957, 2062, 2096, 2444, 3546, 3547, 3548, 3549, 3568,
    # 3586, 3587, 3588, 3790, 3791, 3792]
    best_subset = np.asarray(
        [1033, 1034, 1035, 3546, 3547, 1273, 3548, 3549, 1275, 1041, 2444,
         2062, 1443, 3790, 3791, 3792, 1445, 1084, 1085, 1086, 1286, 1957,
         1122, 1291, 1123, 886, 1133, 3568, 1135, 1136, 1159, 1165, 955, 1183,
         1192, 894, 1303, 1304, 1305, 1306, 1307, 1308, 2096, 1228, 1250, 1251,
         3586, 3587, 3588, 1268], dtype=np.int32)
    s_list = [len(best_subset)]
    for (s, gamma, l2_lambda) in product(s_list, gamma_list, l2_lambda_list):
        result = algo_online_best_subset(x_tr, y_tr, w0, best_subset, gamma, l2_lambda, s, 'logistic', 0)
        re = generate_re(para, result, (s, gamma, l2_lambda, n_tr, trial_i))
        best_acc_re = get_best_re(re, best_acc_re)
    print('-' * 100)
    print('---best-node-fm---')
    s, gamma, l2_lambda = best_acc_re['para'][:3]
    print('best parameter pair (s, gamma, l2_lambda, n_tr):'
          '(%d,%.2e,%.2e,%3d)' % (s, gamma, l2_lambda, n_tr))
    print('node-(pre,rec,fm,non-zero):( %.4f %.4f %.4f %4d )' %
          (best_acc_re['n_pre_rec_fm'][0],
           best_acc_re['n_pre_rec_fm'][1],
           best_acc_re['n_pre_rec_fm'][2],
           best_acc_re['nonzeros_wt'][-1]))
    print('auc: %.4f acc: %.4f' % (best_acc_re['auc'], best_acc_re['acc']))
    return best_acc_re


def algo_graph_da_single(para):
    start_time = time.time()
    input_p, strategy, pathway_id, s, n_tr, trial_i, gamma, l2_lambda = para
    file_name = 'input/data_%s_%s_%d.pkl' % (strategy, pathway_id, trial_i)
    para = pickle.load(open(input_p + file_name))
    nodes = {key: ind for ind, key in enumerate(para['nodes'])}
    edges = [[nodes[_[0]], nodes[_[1]]] for _ in para['edges']]
    edges = np.asarray(edges, dtype=int)
    weights = np.asarray(para['weights'], dtype=np.float64)
    x_tr, y_tr = para['x_tr'][:n_tr], para['y_tr'][:n_tr]
    w0 = np.zeros(para['p'] + 1)
    result = algo_online_graph_da(
        x_tr, y_tr, w0, gamma, l2_lambda, edges, weights, 1, s, 0.1, 30, 'logistic', 0)
    re = generate_re(para, result, (s, gamma, l2_lambda, n_tr, trial_i))
    run_time = time.time() - start_time
    print('n_tr: %3d s: %2d gamma: %.4f l2_lambda: %.4f run_time: %.4f node-f1-score: %.4f auc: %.4f'
          % (n_tr, s, gamma, l2_lambda, run_time, re['n_pre_rec_fm'][2], re['auc']))
    return re, n_tr, trial_i


def test_baselines():
    method_name = sys.argv[1]
    num_cpus = int(sys.argv[2])
    input_p = '../dataset/kegg/'
    pathway_id, num_trials = 'hsa05213', 20
    n_tr_list = range(20, 201, 10)
    for strategy in ['s1', 's2']:
        for trial_id in range(num_trials):
            input_paras = [(input_p, strategy, pathway_id, trial_id, n_tr) for n_tr in n_tr_list]
            pool = multiprocessing.Pool(processes=num_cpus)
            if method_name == 'rda-l1':
                results_pool = pool.map(algo_rda_l1, input_paras)
            elif method_name == 'da-iht':
                results_pool = pool.map(algo_da_iht, input_paras)
            elif method_name == 'da-gl':
                results_pool = pool.map(algo_da_gl, input_paras)
            elif method_name == 'da-sgl':
                results_pool = pool.map(algo_da_sgl, input_paras)
            elif method_name == 'adagrad':
                results_pool = pool.map(algo_ada_grad, input_paras)
            elif method_name == 'best-subset':
                results_pool = pool.map(algo_best_subset, input_paras)
            else:
                results_pool = None
                print('cannot find %s method' % method_name)
            pool.close()
            pool.join()
            f_name = '../results/kegg/results_%s_trial_%02d_%s.pkl' % (strategy, trial_id, method_name)
            if results_pool is not None:
                pickle.dump(results_pool, open(f_name, 'wb'))


def test_graph_da():
    method_name = sys.argv[2]
    num_cpus = int(sys.argv[3])
    strategy = sys.argv[4]
    trial_id = int(sys.argv[5])
    input_p = '../dataset/kegg/'
    pathway_id, num_trials = 'hsa05213', 20
    n_tr_list = range(20, 201, 10)
    pool = multiprocessing.Pool(processes=num_cpus)
    s_list = [40, 45, 50, 55, 60]
    lambda_list = [0.0]
    gamma_list = [1e0, 5e0, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4, 5e4]
    input_paras = [(input_p, strategy, pathway_id, s, n_tr, trial_id, gamma, l2_lambda)
                   for s, gamma, l2_lambda, n_tr in product(s_list, gamma_list, lambda_list, n_tr_list)]
    results_pool = pool.map(algo_graph_da_single, input_paras)
    results = {key: None for key in n_tr_list}
    for key in results:
        best_acc_re = None
        for re, n_tr, trial_i in results_pool:
            if n_tr == key:
                best_acc_re = get_best_re(re, best_acc_re)
        results[key] = best_acc_re
    pool.close()
    pool.join()
    f_name = 'results/kegg/results_%s_trial_%02d_%s.pkl' % (strategy, trial_id, method_name)
    pickle.dump(results.values(), open(f_name, 'wb'))


def show_figure_9():
    from matplotlib import rc
    import matplotlib.pyplot as plt
    from pylab import rcParams
    plt.rc('font', **{'size': 16})
    rc('text', usetex=True)
    rcParams['figure.figsize'] = 8.2, 3.2

    num_trials = 20
    n_tr_list = range(20, 201, 10)
    fig, ax = plt.subplots(1, 2)
    for i in range(2):
        ax[i].grid(b=True, which='both', color='gray', linestyle='dotted', axis='both')
        ax[i].set_xlim([10, 210])
        ax[i].set_xticks([50, 90, 130, 170])
    # -----------
    ax[0].set_ylim([0.0, 0.5])
    ax[0].set_yticks([0.1, 0.2, 0.3, 0.4])
    ax[1].set_ylim([0.02, 1.02])
    ax[1].set_yticks([0.2, 0.4, 0.6, 0.8])
    label_list = [r'$\displaystyle \ell_1$-\textsc{RDA}', r'\textsc{DA-IHT}', r'\textsc{AdaGrad}',
                  r'\textsc{DA-GL}', r'\textsc{DA-SGL}', r'\textsc{GraphDA}']
    method_list = ['rda-l1', 'da-iht', 'adagrad', 'da-gl', 'da-sgl', 'graph-da-iht']
    color_list = ['g', 'm', 'y', 'brown', 'b', 'r']
    marker_list = ['v', '^', '>', '<', 'D', 's']
    n_tr_indices = {key: _ for _, key in enumerate(n_tr_list)}
    for strategy_ind, strategy in enumerate(['s1', 's2']):
        for method_ind, method in enumerate(method_list):
            print('processing %s %s' % (strategy, method))
            results = {'acc': {'node-f1': np.zeros(shape=(num_trials, len(n_tr_list))),
                               'auc': np.zeros(shape=(num_trials, len(n_tr_list))),
                               'missed': np.zeros(shape=(num_trials, len(n_tr_list)))}}
            results_pool = []
            for trial_i in range(num_trials):
                _ = pickle.load(open('../results/kegg/results_%s_trial_%02d_%s.pkl' % (strategy, trial_i, method)))
                results_pool.extend(_)
            for result in results_pool:
                re_acc = result
                n_tr, trial_i = n_tr_indices[re_acc['n_tr']], re_acc['trial_i']
                results['acc']['node-f1'][trial_i][n_tr] = re_acc['n_pre_rec_fm'][2]
                results['acc']['auc'][trial_i][n_tr] = re_acc['auc']
                results['acc']['missed'][trial_i][n_tr] = re_acc['missed_wt'][-1]
            ax[strategy_ind].errorbar(n_tr_list, y=np.mean(results['acc']['node-f1'], axis=0),
                                      yerr=np.std(results['acc']['node-f1'], axis=0),
                                      marker=marker_list[method_ind], markersize=6., markerfacecolor='white',
                                      color=color_list[method_ind], linewidth=1.0, label=label_list[method_ind])

        ax[1].legend(fontsize=16, bbox_to_anchor=(0.95, 0.9), ncol=1,
                     borderpad=0.2, columnspacing=1.5, labelspacing=0.4,
                     handletextpad=0.2, framealpha=1.0)
        ax[0].set_title(r'(a) \textit{Strategy 1}', fontsize=16)
        ax[1].set_title(r'(b) \textit{Strategy 2}', fontsize=16)
        ax[0].set_ylabel(r'\textit{F1} Score', fontsize=16)
        ax[1].set_ylabel(r'\textit{F1} Score', fontsize=16)
        ax[0].set_xlabel('Samples Seen', fontsize=16)
        ax[1].set_xlabel('Samples Seen', fontsize=16)
    f_name = '../results/kegg/figs/figure_9.pdf'
    plt.subplots_adjust(wspace=0.25, hspace=0.04)
    fig.savefig(f_name, dpi=600, bbox_inches='tight', pad_inches=0, format='pdf')


def generate_detected_genes():
    num_trials = 10  # we just show 10 trials. The other 10 trials have the similar pattern.
    strategy, pathway_id, trial_i = 's1', 'hsa05213', 0
    para = pickle.load(open('../dataset/kegg/' + 'input/data_%s_%s_%d.pkl' % (strategy, pathway_id, trial_i)))
    all_subgraphs = {'nodes': para['nodes'],
                     'edges': para['edges'],
                     'weights': para['weights'],
                     'hsa05213': {'nodes': para['pathways_graph']['hsa05213']['graph']['nodes'],
                                  'edges': [],
                                  'weights': []}}
    for edge_ind, edge in enumerate(para['edges']):
        if edge[0] in all_subgraphs['hsa05213']['nodes'] and edge[1] in all_subgraphs['hsa05213']['nodes']:
            all_subgraphs['hsa05213']['edges'].append(edge)
            all_subgraphs['hsa05213']['weights'].append(para['weights'][edge_ind])
    method_list = ['rda-l1', 'da-iht', 'adagrad', 'da-gl', 'da-sgl', 'best-subset', 'graph-da-iht']
    for strategy_ind, strategy in enumerate(['s1', 's2']):
        all_subgraphs[strategy] = dict()
        for method_ind, method in enumerate(method_list):
            all_subgraphs[strategy][method] = dict()
            for trial_i in range(num_trials):
                print(strategy, method, trial_i)
                f_name = '../results/kegg/results_%s_trial_%02d_%s.pkl'
                for __ in pickle.load(open(f_name % (strategy, trial_i, method))):
                    if __['n_tr'] == 200:
                        all_subgraphs[strategy][method][trial_i] = __['pred_subgraph']
                        break
    pickle.dump(all_subgraphs, open('../results/kegg/generated_subgraphs.pkl', 'wb'))


def show_figure_10():
    import networkx as nx
    import matplotlib.pyplot as plt
    from pylab import rcParams
    rcParams['figure.figsize'] = 8, 5.
    from matplotlib import rc
    from matplotlib.font_manager import FontProperties
    plt.rc('font', **{'size': 12, 'weight': 'bold'})
    rc('text', usetex=True)
    font0 = FontProperties()
    font0.set_weight('bold')
    all_subgraphs = pickle.load(open('../results/kegg/generated_subgraphs.pkl'))
    method_list = ['rda-l1', 'da-iht', 'adagrad', 'da-gl', 'da-sgl', 'graph-da-iht']
    label_list = [r'$\displaystyle \ell_1$-\textsc{RDA}', r'\textsc{DA-IHT}', r'\textsc{AdaGrad}',
                  r'\textsc{DA-GL}', r'\textsc{DA-SGL}', r'\textsc{GraphDA}']
    fig, ax = plt.subplots(5, 6)
    for trial_i_ind, trial_i in enumerate(range(0, 5)):
        for method_ind, method in enumerate(method_list):
            print(trial_i, method)
            g = nx.Graph()
            detected_graph = all_subgraphs['s1'][method][trial_i]
            detected_nodes = list(detected_graph['nodes'])
            true_nodes = all_subgraphs['hsa05213']['nodes']
            intersect = list(set(true_nodes).intersection(detected_graph['nodes']))
            for edge_ind, edge in enumerate(detected_graph['edges']):
                g.add_edge(edge[0], edge[1], weight=detected_graph['weights'][edge_ind])
            for node in list(detected_graph['nodes']):
                g.add_node(node)
            for node in list(true_nodes):
                g.add_node(node)
            g = nx.minimum_spanning_tree(g)
            print('method: %s pre: %.3f rec: %.3f fm: %.3f' %
                  (method, float(len(intersect)) / float(len(detected_nodes)),
                   float(len(intersect)) / float(len(true_nodes)),
                   float(2.0 * len(intersect)) / float(len(detected_nodes) + len(true_nodes))))
            color_list = []
            for node in detected_nodes:
                if node in intersect:
                    color_list.append('r')
                else:
                    color_list.append('b')
            nx.draw_spring(g, ax=ax[trial_i_ind, method_ind], node_size=10, edge_color='black', edge_width=2.,
                           font_size=4, node_edgecolor='black', node_facecolor='white',
                           node_edgewidth=1., k=10.0, nodelist=detected_nodes, node_color=color_list)
            ax[trial_i_ind, method_ind].axis('on')
            ax[0, method_ind].set(title='%s' % label_list[method_ind])
            plt.setp(ax[trial_i_ind, method_ind].get_xticklabels(), visible=False)
            plt.setp(ax[trial_i_ind, method_ind].get_yticklabels(), visible=False)
            ax[trial_i_ind, method_ind].tick_params(axis='both', which='both', length=0)
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    f_name = '../results/kegg/figs/figure_10.pdf'
    plt.savefig(f_name, dpi=600, bbox_inches='tight', pad_inches=0.03, format='pdf')
    plt.close()


def show_figure_12():
    import networkx as nx
    import matplotlib.pyplot as plt
    from pylab import rcParams
    rcParams['figure.figsize'] = 8, 5.
    from matplotlib import rc
    from matplotlib.font_manager import FontProperties
    plt.rc('font', **{'size': 12, 'weight': 'bold'})
    rc('text', usetex=True)
    font0 = FontProperties()
    font0.set_weight('bold')
    all_subgraphs = pickle.load(open('../results/kegg/generated_subgraphs.pkl'))
    method_list = ['rda-l1', 'da-iht', 'adagrad', 'da-gl', 'da-sgl', 'graph-da-iht']
    label_list = [r'$\displaystyle \ell_1$-\textsc{RDA}', r'\textsc{DA-IHT}', r'\textsc{AdaGrad}',
                  r'\textsc{DA-GL}', r'\textsc{DA-SGL}', r'\textsc{GraphDA}']
    fig, ax = plt.subplots(5, 6)
    for trial_i_ind, trial_i in enumerate(range(5, 10)):
        for method_ind, method in enumerate(method_list):
            print(trial_i, method)
            g = nx.Graph()
            detected_graph = all_subgraphs['s1'][method][trial_i]
            detected_nodes = list(detected_graph['nodes'])
            true_nodes = all_subgraphs['hsa05213']['nodes']
            intersect = list(set(true_nodes).intersection(detected_graph['nodes']))
            for edge_ind, edge in enumerate(detected_graph['edges']):
                g.add_edge(edge[0], edge[1], weight=detected_graph['weights'][edge_ind])
            for node in list(detected_graph['nodes']):
                g.add_node(node)
            for node in list(true_nodes):
                g.add_node(node)
            g = nx.minimum_spanning_tree(g)
            print('method: %s pre: %.3f rec: %.3f fm: %.3f' %
                  (method, float(len(intersect)) / float(len(detected_nodes)),
                   float(len(intersect)) / float(len(true_nodes)),
                   float(2.0 * len(intersect)) / float(len(detected_nodes) + len(true_nodes))))
            color_list = []
            for node in detected_nodes:
                if node in intersect:
                    color_list.append('r')
                else:
                    color_list.append('b')
            nx.draw_spring(g, ax=ax[trial_i_ind, method_ind], node_size=10, edge_color='black', edge_width=2.,
                           font_size=4, node_edgecolor='black', node_facecolor='white',
                           node_edgewidth=1., k=10.0, nodelist=detected_nodes, node_color=color_list)
            ax[trial_i_ind, method_ind].axis('on')
            ax[0, method_ind].set(title='%s' % label_list[method_ind])
            plt.setp(ax[trial_i_ind, method_ind].get_xticklabels(), visible=False)
            plt.setp(ax[trial_i_ind, method_ind].get_yticklabels(), visible=False)
            ax[trial_i_ind, method_ind].tick_params(axis='both', which='both', length=0)
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    f_name = '../results/kegg/figs/figure_12.pdf'
    plt.savefig(f_name, dpi=600, bbox_inches='tight', pad_inches=0.03, format='pdf')
    plt.close()


def draw_sub_graphs(graph, pathway_id):
    import matplotlib.pyplot as plt
    # draw graph in inset
    from matplotlib import rc
    from pylab import rcParams
    import networkx as nx
    font = {'size': 15}
    plt.rc('font', **font)
    rc('text', usetex=True)
    rcParams['figure.figsize'] = 18, 10
    color_list = ['r', 'b', 'gray', 'y', 'm', 'c', 'k', 'g']
    fig, ax = plt.subplots(2, 4)
    for i in range(2):
        for j in range(4):
            ax[i, j].axes.get_xaxis().set_visible(False)
            ax[i, j].axes.get_yaxis().set_visible(False)
    sub_graphs = nx.connected_component_subgraphs(graph)
    num_cc = len(list(nx.connected_component_subgraphs(graph)))
    print(graph.number_of_nodes(), num_cc)
    for ind, gcc in enumerate(sorted(sub_graphs, key=len, reverse=True)):
        density = nx.number_of_edges(gcc) * 1. / nx.number_of_nodes(gcc) * 1.
        if ind >= 8:
            break
        i = ind / 4
        j = ind % 4
        pos = nx.spring_layout(gcc)
        nx.draw_networkx_nodes(gcc, pos, node_size=20,
                               node_color=color_list[ind % 8], ax=ax[i, j])
        nx.draw_networkx_edges(gcc, pos, alpha=0.4, ax=ax[i, j])
        num_nodes = nx.number_of_nodes(gcc)
        s1 = 'cc[%d]:%d, density: %.4f' % (ind, num_nodes, density)
        s2 = 'cc[%d-rest], density: %.4f, num-cc:%d' % (ind, density, num_cc)
        if ind != 7:
            ax[i, j].set_title(s1)
        else:
            ax[i, j].set_title(s2)
    plt.savefig('../results/kegg/figs/%s.png' % pathway_id)
    plt.close()


def process_real_data():
    """
    We chosen the dataset from the following website:
    https://github.com/fraenkel-lab/GSLR.git
    Most part of this code is copied from the following github:
    Please cite their poster if you want to use this code:
    LeNail, Alexander, Ludwig Schmidt, Johnathan Li, Tobias
    Ehrenberger, Karen Sachs, Stefanie Jegelka, and Ernest Fraenkel.
    "Graph-Sparse Logistic Regression." arXiv preprint arXiv:1712.05510 (2017)."
    """
    import pandas as pd
    import math
    import pickle as pkl
    import networkx as nx
    from itertools import product
    from numpy.random import normal
    ovarian_file = '../dataset/kegg/raw/real_data_ovarian_inbiomap_exp.tsv'
    ovarian = pd.read_csv(ovarian_file, index_col=0)
    ori_name = 'ovarian'
    # an ordered list of genes (corresponding nodes in the graph.)
    ori_nodes = nodes = list(ovarian.columns.values)
    print('number of genes: %d' % len(ori_nodes), ori_nodes[:5])
    ori_node_indices = {key: ind for ind, key in enumerate(ori_nodes)}
    ori_n = len(list(ovarian.index.values))
    ori_p = len(list(ovarian.columns.values))
    # the name of these samples, totally 206 samples.
    ori_sample_names = list(ovarian.index.values)
    print('sample0: %s' % ori_sample_names[0])
    print('sample1: %s' % ori_sample_names[1])
    print('sample2: %s' % ori_sample_names[2])
    ori_data = ovarian.values  # the dimension is 206x16349
    num_missed_values, zero_features = 0, dict()
    for i in range(ori_p):
        # here is the default values. where these features have none values.
        # we use standard normal to generate values for these absent features.
        if np.mean(ori_data[:, i]) == 0.0 and np.var(ori_data[:, i]) == 0.0:
            ori_data[:, i] = normal(loc=0.0, scale=1.0, size=len(ori_data[:, i]))
            zero_features[ori_nodes[i]] = 0
            num_missed_values += 1
    print('total number of default features: %d' % num_missed_values)

    # explore pathways
    pathways_file = '../dataset/kegg/raw/real_kegg_df.filtered.with_correlates.pkl'
    pathways_df = pd.read_pickle(pathways_file)
    pathways = [(pathway_id,
                 np.unique(pathways_df.loc[pd.IndexSlice[:, :, [pathway_id]], ['genes', 'correlates']].values[0][0]))
                for pathway_id in pathways_df.index.get_level_values(2)]
    print('total number of pathways: %d' % len(pathways))
    # remove some empty pathways
    pathways = [item for item in pathways if len(item[1]) > 0]
    print('total number of nonzero-length pathways: %d' % len(pathways))
    # we notice that there are some genes which are not in the graph.
    # Therefore, we remove these genes
    print('total number of genes in original: %d' % len(set([_ for pathway__ in pathways for _ in pathway__[1]])))
    ori_pathways = dict()
    for pathway_id, pathway in pathways:
        # only keeps the genes that are in the graph
        ori_pathways[pathway_id] = [node for node in pathway if node in ori_node_indices]
    print('total number of genes reduced to: %d' % len(set([_ for key_ in ori_pathways for _ in ori_pathways[key_]])))
    # to explore the graph
    edges_path = '../dataset/kegg/raw/real_inbiomap_temp.tsv'
    header = ['protein1', 'protein2', 'cost']
    edges_exp = pd.read_csv(edges_path, sep='\t', names=header)
    ori_graph_nodes = set()
    ori_edges, ori_weights = [], []
    for item in edges_exp.values:
        u, v, weight = item[0], item[1], item[2]
        ori_graph_nodes.add(u)
        ori_graph_nodes.add(v)
        ori_edges.append((u, v))
        ori_weights.append(weight)
    print('number of edges: %d' % (len(ori_edges)))
    print('number of nodes: %d' % (len(ori_graph_nodes)))
    ori_edge_indices = {edge: _ for _, edge in enumerate(ori_edges)}
    ori_nodes_degree = {gene: 0 for gene in ori_nodes}
    # adjacency matrix and nodes degree
    adj, degree = dict(), np.zeros(len(nodes))
    for edge in ori_edges:
        u, v = edge
        if u > v:
            u, v = v, u
        ori_nodes_degree[u] += 1
        ori_nodes_degree[v] += 1
        if u not in adj:
            adj[u] = set()
        else:
            adj[u].add(v)
            degree[ori_node_indices[u]] += 1
        if v not in adj:
            adj[v] = set()
        else:
            adj[v].add(u)
            degree[ori_node_indices[v]] += 1

    for _ in np.argsort(degree)[::-1][:20]:
        print(degree[_], _)
    for i in range(20):
        print('degree is less than %d: %d' % (i, len([_ for _ in degree if _ <= i])))
    for i in range(2000, 2020):
        print('degree is larger than %d: %d' % (i, len([_ for _ in degree if _ >= i])))
    # sort pathways by number of genes in pathways.
    sorted_pathways = sorted([(_id_, len(ori_pathways[_id_]), ori_pathways[_id_])
                              for _id_ in ori_pathways], key=lambda _: _[1])
    ori_pathways_graph = dict()
    for item in sorted_pathways:
        graph = nx.Graph()
        pathway_id, num_genes, pathway = item[0], item[1], item[2]
        for (u, v) in product(pathway, pathway):
            graph.add_node(u)
            graph.add_node(v)
            if u < v and (u in adj) and (v in adj[u]):
                graph.add_edge(u, v)
        if len(pathway) != 0:
            dense = nx.number_of_edges(graph) * 1. / len(pathway) * 1.
        else:
            dense = 0.0
        density = (len(ori_edges) * 1.) / (len(nodes) * 1.)
        print('density of this graph: %.6f, density of whole graph: %.6f' % (dense, density))
        # try to call this if you want to explore the graph.
        if not graph.number_of_edges() > 1:
            draw_sub_graphs(graph, item[0])
        sub_graph = dict()
        sub_graph['density'] = dense
        sub_graph['graph'] = {'nodes': {_ for _ in nx.nodes(graph)}, 'edges': {_ for _ in nx.edges(graph)}}
        sub_graph['sub_graphs'] = [{'nodes': {_ for _ in nx.nodes(__)}, 'edges': {_ for _ in nx.edges(__)}}
                                   for __ in nx.connected_component_subgraphs(graph)]
        ori_pathways_graph[pathway_id] = sub_graph
        print(pathway_id, 'number of graphs: %d, number of (nodes,edges): %d, %d'
              % (len(sub_graph['sub_graphs']), len(sub_graph['graph']['nodes']), nx.number_of_edges(graph)))
    reduced_data = dict()
    selected_nodes_set, index = dict(), 0
    selected_nodes, selected_indices = [], []
    for pathway_id in sorted(ori_pathways.keys()):
        for gene in ori_pathways[pathway_id]:
            if gene not in selected_nodes_set:
                ori_index = ori_node_indices[gene]
                selected_nodes_set[gene] = (index, ori_index)
                selected_nodes.append(gene)
                selected_indices.append(ori_index)
                index += 1
    print('number of nodes in final graph: %d' % len(selected_nodes))
    print(selected_nodes[:10])
    reduced_data['nodes'] = selected_nodes
    print('number of zero-features: %d' % len([_ for _ in selected_nodes if _ in zero_features]))
    reduced_data['node_indices'] = {key: _ for _, key in enumerate(selected_nodes)}
    reduced_data['data'] = ori_data[:, selected_indices]  # it is ordered.
    reduced_data['n'] = ori_n
    reduced_data['p'] = len(selected_indices)
    reduced_data['name'] = ori_name
    reduced_data['edges'] = []
    reduced_data['weights'] = []
    for edge in ori_edges:
        if (edge[0] in selected_nodes_set) and (edge[1] in selected_nodes_set):
            if edge[0] != edge[1]:
                reduced_data['edges'].append(edge)
                e_index = ori_edge_indices[edge]
                reduced_data['weights'].append(ori_weights[e_index])
            else:
                print('remove self-loop: %s %s' % (edge[0], edge[1]))
    print('number of edges in final graph: %d' % len(reduced_data['edges']))
    print('number of pathways: %d' % len(ori_pathways))
    reduced_data['pathways'] = ori_pathways
    reduced_data['pathways_graph'] = ori_pathways_graph
    reduced_data['sample_names'] = ori_sample_names
    reduced_data['zero_features'] = zero_features
    pkl.dump(reduced_data, open('../dataset/kegg/raw/ovarian_data.pkl', 'wb'))


def sample_cov(m, data):
    import math
    n, d = data.shape
    mu = np.mean(data, axis=0)
    _, s, v = np.linalg.svd((data - mu) / math.sqrt(n), full_matrices=False)
    randomness = np.random.randn(m, min(n, d))
    return np.dot(randomness, np.dot(np.diag(s), v))


def generate_dataset_s1(pathway_id, trial_id, num_trials, num_tr, num_te, num_va):
    import pickle as pkl
    from numpy.random import normal
    ovarian = pkl.load(open('../dataset/kegg/raw/ovarian_data.pkl'))
    posi_label, nega_label, strategy = +1, -1, 's1'
    sub_graph = [ovarian['node_indices'][_] for _ in ovarian['pathways'][pathway_id]]
    n, p, s = num_tr + num_te + num_va, ovarian['p'], len(sub_graph)
    num_posi, num_nega = n / 2, n / 2  # balanced dataset
    variances = np.var(ovarian['data'], axis=0)
    negatives = sample_cov(num_nega, ovarian['data'])
    negatives = np.around(negatives + np.mean(ovarian['data'], axis=0), 6)
    new_pathway_means = np.zeros_like(np.mean(ovarian['data'], axis=0))
    new_pathway_means[sub_graph] = normal(loc=0, scale=variances[sub_graph])
    new_means = np.mean(ovarian['data'], axis=0) + new_pathway_means
    positives = sample_cov(num_posi, ovarian['data'])
    positives = np.around(positives + new_means, 6)
    samples = np.concatenate((negatives, positives))
    labels = np.ones(num_nega + num_posi, dtype=np.float64)
    labels[:num_nega] = -1.
    rand_indices = np.random.permutation(len(labels))
    x_tr, y_tr = samples[rand_indices], labels[rand_indices]
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
            'edges': ovarian['edges'],
            'weights': ovarian['weights'],
            'subgraph': sub_graph,
            'name': ovarian['name'],
            'nodes': ovarian['nodes'],
            'node_indices': ovarian['node_indices'],
            'pathways': ovarian['pathways'],
            'pathways_graph': ovarian['pathways_graph'],
            'trial_i': trial_id,
            'num_trials': num_trials,
            'n_tr': num_tr,
            'n_va': num_va,
            'n_te': num_te,
            'strategy': strategy,
            'pathway_id': pathway_id,
            'p': p}
    print(pathway_id, trial_id, np.sum(data['x_tr']), np.sum(x_tr))
    pkl.dump(data, open('../dataset/kegg/input/data_%s_%s_%d.pkl' % (strategy, pathway_id, trial_id), 'wb'))


def generate_dataset_s2(pathway_id, trial_id, num_trials, num_tr, num_te, num_va):
    import pickle as pkl
    from numpy.random import normal
    ovarian = pkl.load(open('../dataset/kegg/raw/ovarian_data.pkl'))
    posi_label, nega_label, strategy = +1, -1, 's2'
    sub_graph = [ovarian['node_indices'][_] for _ in ovarian['pathways'][pathway_id]]
    n, p, s = num_tr + num_te + num_va, ovarian['p'], len(sub_graph)
    num_posi, num_nega = n / 2, n / 2  # balanced dataset
    negatives = sample_cov(num_nega, ovarian['data'])
    negatives = np.around(negatives + np.mean(ovarian['data'], axis=0), 6)
    means = np.mean(ovarian['data'], axis=0)
    stddev = np.std(ovarian['data'], axis=0)
    variances = np.var(ovarian['data'], axis=0)
    shift_sign = np.random.randint(2, size=means.shape) * 2 - 1
    new_pathway_means = np.zeros_like(np.mean(ovarian['data'], axis=0))
    new_pathway_means[sub_graph] = normal(loc=np.asarray(stddev * shift_sign)[sub_graph], scale=variances[sub_graph])
    new_means = np.mean(ovarian['data'], axis=0) + new_pathway_means
    positives = sample_cov(num_posi, ovarian['data'])
    positives = np.around(positives + new_means, 6)
    samples = np.concatenate((negatives, positives))
    labels = np.ones(num_nega + num_posi, dtype=np.float64)
    labels[:num_nega] = -1.
    rand_indices = np.random.permutation(len(labels))
    x_tr, y_tr = samples[rand_indices], labels[rand_indices]
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
            'edges': ovarian['edges'],
            'weights': ovarian['weights'],
            'subgraph': sub_graph,
            'name': ovarian['name'],
            'nodes': ovarian['nodes'],
            'node_indices': ovarian['node_indices'],
            'pathways': ovarian['pathways'],
            'pathways_graph': ovarian['pathways_graph'],
            'trial_i': trial_id,
            'num_trials': num_trials,
            'n_tr': num_tr,
            'n_va': num_va,
            'n_te': num_te,
            'strategy': strategy,
            'pathway_id': pathway_id,
            'p': p}
    print(pathway_id, trial_id, np.sum(data['x_tr']), np.sum(x_tr))
    pkl.dump(data, open('../dataset/kegg/input/data_%s_%s_%d.pkl' % (strategy, pathway_id, trial_id), 'wb'))


def data_process():
    # generate real data if it is necessary.
    # process_real_data()
    # we do 10 times and take the average.
    pathway_id, num_trials = 'hsa05213', 20
    for trial_i in range(num_trials):
        generate_dataset_s1(pathway_id=pathway_id, trial_id=trial_i, num_trials=num_trials,
                            num_tr=200, num_te=200, num_va=200)
    for trial_i in range(num_trials):
        generate_dataset_s2(pathway_id=pathway_id, trial_id=trial_i, num_trials=num_trials,
                            num_tr=200, num_te=200, num_va=200)


def main():
    command = sys.argv[1]
    if command == 'show_figure_9':
        show_figure_9()
    elif command == 'show_figure_10':
        show_figure_10()
    elif command == 'show_figure_12':
        show_figure_12()
    elif command == 'data_process':
        data_process()
    elif command == 'test_graphda':
        # to run graph_da algorithm
        test_graph_da()
    elif command == 'test_baselines':
        # to run all baseline methods.
        test_baselines()


if __name__ == "__main__":
    main()
