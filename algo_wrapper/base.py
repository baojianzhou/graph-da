# -*- coding: utf-8 -*-
import numpy as np
import os
import sys

__all__ = ['simu_graph', 'simu_graph_rectangle', 'expit', 'logistic_predict',
           'node_pre_rec_fm', 'least_square_predict']


def simu_graph(num_nodes, rand=False, graph_type='grid'):
    """
    To generate a grid graph. Each node has 4-neighbors.
    :param num_nodes: number of nodes in the graph.
    :param rand: if rand True, then generate random weights in (0., 1.)
    :param graph_type: ['grid', 'chain']
    :return: edges and corresponding to unite weights.
    """
    edges, weights = [], []
    if graph_type == 'grid':
        length_ = int(np.sqrt(num_nodes))
        width_, index = length_, 0
        for i in range(length_):
            for j in range(width_):
                if (index % length_) != (length_ - 1):
                    edges.append((index, index + 1))
                    if index + length_ < int(width_ * length_):
                        edges.append((index, index + length_))
                else:
                    if index + length_ < int(width_ * length_):
                        edges.append((index, index + length_))
                index += 1
        edges = np.asarray(edges, dtype=int)
    elif graph_type == 'chain':
        for i in range(num_nodes - 1):
            edges.append((i, i + 1))
    else:
        edges = []
    # generate weights of the graph
    if rand:
        weights = []
        while len(weights) < len(edges):
            rand_x = np.random.random()
            if rand_x > 0.:
                weights.append(rand_x)
        weights = np.asarray(weights, dtype=np.float64)
    else:  # set unit weights for edge costs.
        weights = np.ones(len(edges), dtype=np.float64)
    return edges, weights


def simu_graph_rectangle(width, height, rand=False):
    """
    To generate a grid graph. Each node has 4-neighbors.
    :param width
    :param height
    :param rand: if rand True, then generate random weights in (0., 1.)
    :return: edges and corresponding to unite weights.
    """
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
    # generate weights of the graph
    if rand:
        weights = []
        while len(weights) < len(edges):
            rand_x = np.random.random()
            if rand_x > 0.:
                weights.append(rand_x)
        weights = np.asarray(weights, dtype=np.float64)
    else:  # set unit weights for edge costs.
        weights = np.ones(len(edges), dtype=np.float64)
    return edges, weights


def expit(x):
    """ expit function. 1 /(1+exp(-x)) """
    if type(x) == np.float64:
        if x > 0.:
            return 1. / (1. + np.exp(-x))
        else:
            return np.exp(x) / (1. + np.exp(x))
    out = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] > 0.:
            out[i] = 1. / (1. + np.exp(-x[i]))
        else:
            out[i] = np.exp(x[i]) / (1. + np.exp(x[i]))
    return out


def logistic_predict(x_va, wt):
    """ To predict the probability for sample xi. {+1,-1} """
    pre_prob, y_pred, p = [], [], x_va.shape[1]
    for i in range(len(x_va)):
        pred_posi = expit(np.dot(wt[:p], x_va[i]) + wt[p])
        pred_nega = 1. - pred_posi
        if pred_posi >= pred_nega:
            y_pred.append(1)
        else:
            y_pred.append(-1)
        pre_prob.append(pred_posi)
    return np.asarray(pre_prob), np.asarray(y_pred)


def least_square_predict(x_va, wt):
    """ To predict the probability for sample xi. """
    pred_val, p = [], x_va.shape[1]
    for i in range(len(x_va)):
        pred_val.append(np.dot(wt[:p], x_va[i] + wt[p]))
    return np.asarray(pred_val)


def _expit(x):
    """ expit function. 1 /(1+exp(-x)) """
    if type(x) == np.float64:
        if x > 0.:
            return 1. / (1. + np.exp(-x))
        else:
            return np.exp(x) / (1. + np.exp(x))
    out = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] > 0.:
            out[i] = 1. / (1. + np.exp(-x[i]))
        else:
            out[i] = np.exp(x[i]) / (1. + np.exp(x[i]))
    return out


def _log_logistic(x):
    """ return log( 1/(1+exp(x)) )"""
    out = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] > 0:
            out[i] = -np.log(1 + np.exp(-x[i]))
        else:
            out[i] = x[i] - np.log(1 + np.exp(x[i]))
    return out


def _grad_w(x_tr, y_tr, wt, eta):
    """ return {+1,-1} Logistic (val,grad) on training samples. """
    assert len(wt) == (x_tr.shape[1] + 1)
    c, p = wt[-1], x_tr.shape[1]
    wt = wt[:p]
    yz = y_tr * (np.dot(x_tr, wt) + c)
    z = _expit(yz)
    loss = -np.sum(_log_logistic(yz)) + .5 * eta * np.dot(wt, wt)
    grad = np.zeros(p + 1)
    z0 = (z - 1) * y_tr
    grad[:p] = np.dot(x_tr.T, z0) + eta * wt
    grad[-1] = z0.sum()
    return loss, grad


def node_pre_rec_fm(true_nodes, pred_nodes):
    """ Return the precision, recall and f-measure.
    :param true_nodes:
    :param pred_nodes:
    :return: precision, recall and f-measure """
    true_nodes, pred_nodes = set(true_nodes), set(pred_nodes)
    pre, rec, fm = 0.0, 0.0, 0.0
    if len(pred_nodes) != 0:
        pre = len(true_nodes & pred_nodes) / float(len(pred_nodes))
    if len(true_nodes) != 0:
        rec = len(true_nodes & pred_nodes) / float(len(true_nodes))
    if (pre + rec) > 0.:
        fm = (2. * pre * rec) / (pre + rec)
    return [pre, rec, fm]


def auc_node_fm(auc, node_fm):
    if 0.0 <= auc <= 1.0 and 0.0 <= node_fm <= 1.0:
        return 2.0 * (auc * node_fm) / (auc + node_fm)
    else:
        print('auc and node-fm must be in the range [0.0,1.0]')
        exit(0)


def m_print(result, method, trial_i, n_tr_, fig_i, mu, sub_graph,
            header=False):
    if header:
        print('-' * 165)
        print('method         fig_i  s      tr_id '
              ' n_tr       mu   auc      acc     f1      ' +
              'n_pre   n_rec   n_fm     nega_in  nega_out'
              '  posi_in  posi_out intercept  run_time')
    auc = result['auc'][-1]
    acc = result['acc'][-1]
    f1 = result['f1'][-1]
    node_pre = result['n_pre'][-1]
    node_rec = result['n_rec'][-1]
    node_fm = result['n_fm'][-1]
    num_nega_in = len([_ for ind, _ in enumerate(result['wt'][-1]) if
                       ind in sub_graph and _ < 0.0])
    num_nega_out = len([_ for ind, _ in enumerate(result['wt'][-1]) if
                        ind not in sub_graph and _ < 0.0])
    num_posi_in = len([_ for ind, _ in enumerate(result['wt'][-1]) if
                       ind in sub_graph and _ > 0.0])
    num_posi_out = len([_ for ind, _ in enumerate(result['wt'][-1]) if
                        ind not in sub_graph and _ > 0.0])
    sparsity = np.count_nonzero(result['wt'][-1][:1089])
    intercept = result['intercept'][-1]
    run_time = result['run_time'][-1]
    print('{:14s} {:6s} {:6s} {:6s} {:6s} {:7.1f} '
          '{:7.4f}  {:7.4f} {:7.4f} {:7.4f} {:7.4f} {:7.4f} '
          '{:8d} {:8d} {:8d} {:8d} {:12.4f} {:12.3f}'
          .format(method, fig_i, str(sparsity), str(trial_i), str(n_tr_),
                  mu, auc, acc, f1, node_pre, node_rec, node_fm, num_nega_in,
                  num_nega_out, num_posi_in, num_posi_out, intercept,
                  run_time))


def gen_test_case(x_tr, y_tr, w0, edges, weights):
    f = open('test_case.txt', 'wb')
    f.write('P %d %d %d\n' % (len(x_tr), len(x_tr[0]), len(edges)))
    for i in range(len(x_tr)):
        f.write('x_tr ')
        for j in range(len(x_tr[i])):
            f.write('%.8f' % x_tr[i][j] + ' ')
        f.write(str(y_tr[i]) + '\n')
    for i in range(len(edges)):
        f.write('E ' + str(edges[i][0]) + ' ' +
                str(edges[i][1]) + ' ' + '%.8f' % weights[i] + '\n')
    for i in range(len(w0)):
        f.write('N %d %.8f\n' % (i, w0[i]))
    f.close()
