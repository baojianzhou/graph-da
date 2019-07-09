# -*- coding: utf-8 -*-
"""
This python file is the Python wrapper for all related algorithms.
References:
    [1] Xiao, Lin. "Dual averaging methods for regularized stochastic
        learning and online optimization." Journal of Machine Learning
        Research 11.Oct (2010): 2543-2596.
    [2] Langford, John, Lihong Li, and Tong Zhang. "Sparse online
        learning via truncated gradient." Journal of Machine Learning
        Research 10.Mar (2009): 777-801.
    [3] Ma, Yuting, and Tian Zheng. "Stabilized Sparse Online Learning
        for Sparse Data." arXiv preprint arXiv:1604.06498 (2016).
    [4] Yang, Haiqin, Zenglin Xu, Irwin King, and Michael R. Lyu.
        "Online learning for group lasso." In Proceedings of the 27th
        International Conference on Machine Learning (ICML-10),
        pp. 1191-1198. 2010.
    [5] Duchi, John, Elad Hazan, and Yoram Singer. "Adaptive subgradient
        methods for online learning and stochastic optimization."
        Journal of Machine Learning Research 12.Jul (2011): 2121-2159.
    [6] Fan, J., Gong, W., Li, C. J., & Sun, Q. (2018, March). Statistical
        sparse online regression: A diffusion approximation perspective.
        In International Conference on Artificial Intelligence and
        Statistics (pp. 1017-1026). Please check the long version:
        http://proceedings.mlr.press/v84/fan18b/fan18b-supp.pdf
"""
__all__ = ['algo_test',
           'algo_online_sto_iht',
           'algo_online_da_iht',
           'algo_online_rda_l1',
           'algo_online_ada_grad',
           'algo_online_da_gl',
           'algo_online_adam',
           'algo_online_da_sgl',
           'algo_online_sgd_l1',
           'algo_online_sgd_l2',
           'algo_online_graph_iht',
           'algo_online_graph_da_iht',
           'algo_online_graph_da_iht_2',
           'algo_online_best_subset',
           'algo_online_ghtp',
           'algo_online_graph_ghtp',
           'algo_batch_iht',
           'algo_batch_ghtp',
           'algo_batch_graph_iht',
           'algo_batch_graph_ghtp',
           'algo_batch_graph_posi',
           'algo_iht',
           'algo_sto_iht',
           'algo_graph_iht',
           'algo_graph_sto_iht',
           'algo_graph_da_iht',
           'algo_gradmp',
           'algo_sto_gradmp',
           'algo_graph_gradmp',
           'algo_graph_sto_gradmp',
           'algo_head_tail_binsearch',
           'wrapper_head_tail_binsearch']
import time
import numpy as np

try:
    import sparse_module

    try:
        from sparse_module import test
        from sparse_module import online_sgd_l1_logit
        from sparse_module import online_sgd_l2_logit
        from sparse_module import online_rda_l1
        from sparse_module import online_ada_grad
        from sparse_module import online_da_gl
        from sparse_module import online_adam
        from sparse_module import online_da_sgl
        from sparse_module import online_sto_iht
        from sparse_module import online_da_iht
        from sparse_module import online_graph_iht
        from sparse_module import online_graph_da_iht
        from sparse_module import online_graph_da_iht_2
        from sparse_module import online_best_subset
        from sparse_module import online_ghtp_logit
        from sparse_module import online_graph_ghtp_logit
        from sparse_module import batch_iht_logit
        from sparse_module import batch_ghtp_logit
        from sparse_module import batch_graph_iht_logit
        from sparse_module import batch_graph_ghtp_logit
        from sparse_module import batch_graph_posi_logit
        from sparse_module import wrap_head_tail_binsearch
    except ImportError:
        print('cannot find some function(s) in sparse_module')
        exit(0)
except ImportError:
    print('cannot find the module: sparse_module')


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


def algo_test():
    x = np.arange(1, 13).reshape(3, 4)
    sum_x = test(np.asarray(x, dtype=np.double))
    print('sum: %.2f' % sum_x)


def algo_head_tail_binsearch(
        edges, w, costs, g, root, s_low, s_high, max_num_iter, verbose):
    prizes = w * w
    if s_high >= len(prizes) - 1:  # to avoid problem.
        s_high = len(prizes) - 1
    re_nodes = wrap_head_tail_binsearch(
        edges, prizes, costs, g, root, s_low, s_high, max_num_iter, verbose)
    proj_w = np.zeros_like(w)
    proj_w[re_nodes[0]] = w[re_nodes[0]]
    return re_nodes[0], proj_w


def algo_online_rda_l1(x_tr, y_tr, w0, lambda_, gamma, rho, loss, verbose):
    """
    The enhanced l1-RDA method. Shown in Algorithm 2 of reference [1].
    That is Equation (10) is equivalent to Equation (30) if rho=0.0.

    :param x_tr: training samples (n,p) dimension.
    :param y_tr: training labels (n,1) dimension.
    :param w0: the initial points (default values is zero vector). (p+1,)
    :param lambda_: the l1-regularization parameter, lambda_*||w||_1
    :param gamma: parameter to control the learning rate.
    :param rho: a sparsity-enhancing parameter.
    :param loss: the type of loss to choose. to choose f.
    :param verbose: print out information.
    :return: statistical results
    """
    # TODO check x_tr, y_tr should be double
    if loss == 'logistic':
        re = online_rda_l1(x_tr, y_tr, w0, lambda_, gamma, rho, 0, verbose)
    elif loss == 'least_square':
        re = online_rda_l1(x_tr, y_tr, w0, lambda_, gamma, rho, 1, verbose)
    else:
        re = None
        print('cannot find loss: %s' % loss)
        exit(0)
    return {'wt': np.asarray(re[0]),
            'wt_bar': np.asarray(re[1]),
            'losses': np.asanyarray(re[2]),
            'nonzeros_wt': np.asarray(re[3]),
            'nonzeros_wt_bar': np.asarray(re[4]),
            'pred_prob_wt': np.asarray(re[5]),
            'pred_label_wt': np.asarray(re[6]),
            'pred_prob_wt_bar': np.asarray(re[7]),
            'pred_label_wt_bar': np.asarray(re[8]),
            'missed_wt': np.asarray(re[9]),
            'missed_wt_bar': np.asarray(re[10]),
            'total_time': re[11],
            'y_tr': y_tr}


def algo_online_ada_grad(x_tr, y_tr, w0, lambda_, eta, delta, loss, verbose):
    if loss == 'logistic':
        re = online_ada_grad(x_tr, y_tr, w0, lambda_, eta, delta, 0, verbose)
    elif loss == 'least_square':
        re = online_ada_grad(x_tr, y_tr, w0, lambda_, eta, delta, 1, verbose)
    else:
        re = None
        print('cannot find loss: %s' % loss)
        exit(0)
    return {'wt': np.asarray(re[0]),
            'wt_bar': np.asarray(re[1]),
            'losses': np.asanyarray(re[2]),
            'nonzeros_wt': np.asarray(re[3]),
            'nonzeros_wt_bar': np.asarray(re[4]),
            'pred_prob_wt': np.asarray(re[5]),
            'pred_label_wt': np.asarray(re[6]),
            'pred_prob_wt_bar': np.asarray(re[7]),
            'pred_label_wt_bar': np.asarray(re[8]),
            'missed_wt': np.asarray(re[9]),
            'missed_wt_bar': np.asarray(re[10]),
            'total_time': re[11],
            'y_tr': y_tr}


def algo_online_da_gl(x_tr, y_tr, w0, lambda_, gamma, group_list,
                      group_size_list, num_group, loss, verbose):
    """
    The DA-GL method. Shown in Algorithm 1 of reference [4].
    That is Equation (10).

    :param x_tr: training samples (n,p) dimension.
    :param y_tr: training labels (n,1) dimension.
    :param w0: the initial points (default values is zero vector). (p+1,)
    :param lambda_: regularization parameter for group lasso Equation (2).
    :param gamma: parameter h(x) to control the learning rate.
    :param group_list: list of groups.
    :param group_size_list: i-th element stands for the size of group i
    :param num_group: the total number of groups considered.
    :param loss: the type of the loss function.
    :param verbose:
    :return:
    """
    if loss == 'logistic':
        re = online_da_gl(x_tr, y_tr, w0, lambda_, gamma, group_list,
                          group_size_list, num_group, 0, verbose)
    elif loss == 'least_square':
        re = online_da_gl(x_tr, y_tr, w0, lambda_, gamma, group_list,
                          group_size_list, num_group, 1, verbose)
    else:
        re = None
        print('cannot find loss: %s' % loss)
        exit(0)

    return {'wt': np.asarray(re[0]),
            'wt_bar': np.asarray(re[1]),
            'losses': np.asanyarray(re[2]),
            'nonzeros_wt': np.asarray(re[3]),
            'nonzeros_wt_bar': np.asarray(re[4]),
            'pred_prob_wt': np.asarray(re[5]),
            'pred_label_wt': np.asarray(re[6]),
            'pred_prob_wt_bar': np.asarray(re[7]),
            'pred_label_wt_bar': np.asarray(re[8]),
            'missed_wt': np.asarray(re[9]),
            'missed_wt_bar': np.asarray(re[10]),
            'total_time': re[11],
            'y_tr': y_tr}


def algo_online_adam(x_tr, y_tr, w0, alpha, beta1, beta2,
                     epsilon, loss, verbose):
    """
    :param x_tr:
    :param y_tr:
    :param w0:
    :param alpha:
    :param beta1:
    :param beta2:
    :param epsilon:
    :param loss:
    :param verbose:
    :return:
    """
    if loss == 'logistic':
        re = online_adam(x_tr, y_tr, w0, alpha, beta1, beta2,
                         epsilon, 0, verbose)
    elif loss == 'least_square':
        re = online_adam(x_tr, y_tr, w0, alpha, beta1, beta2,
                         epsilon, 1, verbose)
    else:
        re = None
        print('cannot find loss: %s' % loss)
        exit(0)

    return {'wt': np.asarray(re[0]),
            'wt_bar': np.asarray(re[1]),
            'losses': np.asanyarray(re[2]),
            'nonzeros_wt': np.asarray(re[3]),
            'nonzeros_wt_bar': np.asarray(re[4]),
            'pred_prob_wt': np.asarray(re[5]),
            'pred_label_wt': np.asarray(re[6]),
            'pred_prob_wt_bar': np.asarray(re[7]),
            'pred_label_wt_bar': np.asarray(re[8]),
            'missed_wt': np.asarray(re[9]),
            'missed_wt_bar': np.asarray(re[10]),
            'total_time': re[11],
            'y_tr': y_tr}


def algo_online_da_sgl(x_tr, y_tr, w0, lambda_, gamma, group_list,
                       group_size_list, r, num_group, loss, verbose):
    """
    The DA-SGL method. Shown in Algorithm 1 of reference [4].
    That is Equation (11).

    :param x_tr: training samples (n,p) dimension.
    :param y_tr: training labels (n,1) dimension.
    :param w0: the initial points (default values is zero vector). (p+1,)
    :param lambda_: regularization parameter for group lasso Equation (2).
    :param gamma: parameter h(x) to control the learning rate.
    :param group_list: list of groups.
    :param group_size_list: i-th element stands for the size of group i
    :param r
    :param num_group: the total number of groups considered.
    :param loss: the type of the loss function.
    :param verbose:
    :return:
    """
    if loss == 'logistic':
        re = online_da_sgl(x_tr, y_tr, w0, lambda_, gamma, group_list,
                           group_size_list, r, num_group, 0, verbose)
    elif loss == 'least_square':
        re = online_da_sgl(x_tr, y_tr, w0, lambda_, gamma, group_list,
                           group_size_list, r, num_group, 1, verbose)
    else:
        re = None
        print('cannot find loss: %s' % loss)
        exit(0)
    return {'wt': np.asarray(re[0]),
            'wt_bar': np.asarray(re[1]),
            'losses': np.asanyarray(re[2]),
            'nonzeros_wt': np.asarray(re[3]),
            'nonzeros_wt_bar': np.asarray(re[4]),
            'pred_prob_wt': np.asarray(re[5]),
            'pred_label_wt': np.asarray(re[6]),
            'pred_prob_wt_bar': np.asarray(re[7]),
            'pred_label_wt_bar': np.asarray(re[8]),
            'missed_wt': np.asarray(re[9]),
            'missed_wt_bar': np.asarray(re[10]),
            'total_time': re[11],
            'y_tr': y_tr}


def algo_online_sgd_l1(x_tr, y_tr, w0, gamma, lambda_, verbose):
    re = online_sgd_l1_logit(x_tr, y_tr, w0, gamma, lambda_, verbose)
    return {'wt': np.asarray(re[0]),
            'wt_bar': np.asarray(re[1]),
            'losses': np.asanyarray(re[2]),
            'nonzeros_wt': np.asarray(re[3]),
            'nonzeros_wt_bar': np.asarray(re[4]),
            'pred_prob_wt': np.asarray(re[5]),
            'pred_label_wt': np.asarray(re[6]),
            'pred_prob_wt_bar': np.asarray(re[7]),
            'pred_label_wt_bar': np.asarray(re[8]),
            'missed_wt': np.asarray(re[9]),
            'missed_wt_bar': np.asarray(re[10]),
            'total_time': re[11],
            'y_tr': y_tr}


def algo_online_sgd_l2(x_tr, y_tr, w0, lr, eta, verbose):
    re = online_sgd_l2_logit(x_tr, y_tr, w0, lr, eta, verbose)
    return {'wt': np.asarray(re[0]),
            'wt_bar': np.asarray(re[1]),
            'losses': np.asanyarray(re[2]),
            'nonzeros_wt': np.asarray(re[3]),
            'nonzeros_wt_bar': np.asarray(re[4]),
            'pred_prob_wt': np.asarray(re[5]),
            'pred_label_wt': np.asarray(re[6]),
            'pred_prob_wt_bar': np.asarray(re[7]),
            'pred_label_wt_bar': np.asarray(re[8]),
            'missed_wt': np.asarray(re[9]),
            'missed_wt_bar': np.asarray(re[10]),
            'total_time': re[11],
            'y_tr': y_tr}


def algo_online_sto_iht(x_tr, y_tr, w0, l_rate, l2_lambda, s, loss, verbose):
    if loss == 'logistic':
        re = online_sto_iht(x_tr, y_tr, w0, l_rate, l2_lambda, s, 0, verbose)
    elif loss == 'least_square':
        re = online_sto_iht(x_tr, y_tr, w0, l_rate, l2_lambda, s, 1, verbose)
    else:
        re = None
        print('cannot find loss: %s' % loss)
        exit(0)
    return {'wt': np.asarray(re[0]),
            'wt_bar': np.asarray(re[1]),
            'losses': np.asanyarray(re[2]),
            'nonzeros_wt': np.asarray(re[3]),
            'nonzeros_wt_bar': np.asarray(re[4]),
            'pred_prob_wt': np.asarray(re[5]),
            'pred_label_wt': np.asarray(re[6]),
            'pred_prob_wt_bar': np.asarray(re[7]),
            'pred_label_wt_bar': np.asarray(re[8]),
            'missed_wt': np.asarray(re[9]),
            'missed_wt_bar': np.asarray(re[10]),
            'total_time': re[11],
            'y_tr': y_tr}


def algo_online_da_iht(x_tr, y_tr, w0, gamma, l2_lambda, s, loss, verbose):
    if loss == 'logistic':
        re = online_da_iht(x_tr, y_tr, w0, gamma, l2_lambda, s, 0, verbose)
    elif loss == 'least_square':
        re = online_da_iht(x_tr, y_tr, w0, gamma, l2_lambda, s, 1, verbose)
    else:
        re = None
        print('cannot find loss: %s' % loss)
        exit(0)
    return {'wt': np.asarray(re[0]),
            'wt_bar': np.asarray(re[1]),
            'losses': np.asanyarray(re[2]),
            'nonzeros_wt': np.asarray(re[3]),
            'nonzeros_wt_bar': np.asarray(re[4]),
            'pred_prob_wt': np.asarray(re[5]),
            'pred_label_wt': np.asarray(re[6]),
            'pred_prob_wt_bar': np.asarray(re[7]),
            'pred_label_wt_bar': np.asarray(re[8]),
            'missed_wt': np.asarray(re[9]),
            'missed_wt_bar': np.asarray(re[10]),
            'total_time': re[11],
            'y_tr': y_tr}


def algo_online_graph_iht(
        x_tr, y_tr, w0, lr, l2_lambda, edges, weights, g, s, ratio,
        max_num_iter, loss, verbose):
    sparsity_low, sparsity_high = int(s), int((1. + ratio) * float(s))
    root = -1
    if loss == 'logistic':
        re = online_graph_iht(
            x_tr, y_tr, w0, edges, weights, lr, l2_lambda, g,
            sparsity_low, sparsity_high, root, max_num_iter, 0, verbose)
    elif loss == 'least_square':
        re = online_graph_iht(
            x_tr, y_tr, w0, edges, weights, lr, l2_lambda, g,
            sparsity_low, sparsity_high, root, max_num_iter, 1, verbose)
    else:
        re = None
        print('cannot find loss: %s' % loss)
        exit(0)
    return {'wt': np.asarray(re[0]),
            'wt_bar': np.asarray(re[1]),
            'losses': np.asanyarray(re[2]),
            'nonzeros_wt': np.asarray(re[3]),
            'nonzeros_wt_bar': np.asarray(re[4]),
            'pred_prob_wt': np.asarray(re[5]),
            'pred_label_wt': np.asarray(re[6]),
            'pred_prob_wt_bar': np.asarray(re[7]),
            'pred_label_wt_bar': np.asarray(re[8]),
            'missed_wt': np.asarray(re[9]),
            'missed_wt_bar': np.asarray(re[10]),
            'total_time': re[11],
            're_nodes': np.asarray(re[12]),
            're_edges': np.asarray(re[13]),
            'num_pcst': re[14],
            'run_time_head': re[15],
            'run_time_tail': re[16],
            'y_tr': y_tr}


def algo_online_graph_da_iht(
        x_tr, y_tr, w0, gamma, l2_lambda, edges, weights, s, head_budget,
        tail_budget, loss, verbose):
    head_err_tol, head_delta, head_max_iter = 1e-6, 1. / 169., 50
    tail_err_tol, tail_nu, tail_max_iter = 1e-6, 2.5, 50
    g, pcst_epsilon = 1, 1e-6
    if loss == 'logistic':
        re = online_graph_da_iht(
            x_tr, y_tr, w0, edges, weights, gamma, g, s, l2_lambda,
            head_budget, head_err_tol, head_delta, head_max_iter,
            tail_budget, tail_err_tol, tail_nu, tail_max_iter,
            pcst_epsilon, 0, verbose)
    elif loss == 'logistic_v2':
        re = online_graph_da_iht(
            x_tr, y_tr, w0, edges, weights, gamma, g, s, l2_lambda,
            head_budget, head_err_tol, head_delta, head_max_iter,
            tail_budget, tail_err_tol, tail_nu, tail_max_iter,
            pcst_epsilon, 3, verbose)
    elif loss == 'least_square':
        re = online_graph_da_iht(
            x_tr, y_tr, w0, edges, weights, gamma, g, s, l2_lambda,
            head_budget, head_err_tol, head_delta, head_max_iter,
            tail_budget, tail_err_tol, tail_nu, tail_max_iter,
            pcst_epsilon, 1, verbose)
    elif loss == 'least_square_without_head':
        re = online_graph_da_iht(
            x_tr, y_tr, w0, edges, weights, gamma, g, s, l2_lambda,
            head_budget, head_err_tol, head_delta, head_max_iter,
            tail_budget, tail_err_tol, tail_nu, tail_max_iter,
            pcst_epsilon, 2, verbose)
    else:
        re = None
        print('cannot find loss: %s' % loss)
        exit(0)
    return {'wt': np.asarray(re[0]),
            'wt_bar': np.asarray(re[1]),
            'losses': np.asanyarray(re[2]),
            'nonzeros_wt': np.asarray(re[3]),
            'nonzeros_wt_bar': np.asarray(re[4]),
            'pred_prob_wt': np.asarray(re[5]),
            'pred_label_wt': np.asarray(re[6]),
            'pred_prob_wt_bar': np.asarray(re[7]),
            'pred_label_wt_bar': np.asarray(re[8]),
            'missed_wt': np.asarray(re[9]),
            'missed_wt_bar': np.asarray(re[10]),
            'total_time': re[11],
            're_nodes': np.asarray(re[12]),
            're_edges': np.asarray(re[13]),
            'num_pcst': re[14],
            'run_time_head': re[15],
            'run_time_tail': re[16],
            'y_tr': y_tr}


def algo_online_graph_da_iht_2(
        x_tr, y_tr, w0, gamma, l2_lambda, edges, weights, num_clusters, s,
        ratio, max_num_iter, loss, verbose):
    sparsity_low, sparsity_high = int(s), int((1. + ratio) * float(s))
    root = -1
    if loss == 'logistic':
        re = online_graph_da_iht_2(
            x_tr, y_tr, w0, edges, weights, gamma, num_clusters, root,
            l2_lambda, sparsity_low, sparsity_high, max_num_iter, 0, verbose)
    elif loss == 'least_square':
        re = online_graph_da_iht_2(
            x_tr, y_tr, w0, edges, weights, gamma, num_clusters, root,
            l2_lambda, sparsity_low, sparsity_high, max_num_iter, 1, verbose)
    else:
        re = None
        print('cannot find loss: %s' % loss)
        exit(0)
    return {'wt': np.asarray(re[0]),
            'wt_bar': np.asarray(re[1]),
            'losses': np.asanyarray(re[2]),
            'nonzeros_wt': np.asarray(re[3]),
            'nonzeros_wt_bar': np.asarray(re[4]),
            'pred_prob_wt': np.asarray(re[5]),
            'pred_label_wt': np.asarray(re[6]),
            'pred_prob_wt_bar': np.asarray(re[7]),
            'pred_label_wt_bar': np.asarray(re[8]),
            'missed_wt': np.asarray(re[9]),
            'missed_wt_bar': np.asarray(re[10]),
            'total_time': re[11],
            're_nodes': np.asarray(re[12]),
            're_edges': np.asarray(re[13]),
            'num_pcst': re[14],
            'run_time_head': re[15],
            'run_time_tail': re[16],
            'y_tr': y_tr}


def algo_online_best_subset(
        x_tr, y_tr, w0, best_subset, gamma, l2_lambda, s, loss, verbose):
    if loss == 'logistic':
        re = online_best_subset(
            x_tr, y_tr, w0, best_subset, gamma, l2_lambda, s, 0, verbose)
    elif loss == 'least_square':
        re = online_best_subset(
            x_tr, y_tr, w0, best_subset, gamma, l2_lambda, s, 0, verbose)
    elif loss == 'least_square_without_head':
        re = online_best_subset(
            x_tr, y_tr, w0, best_subset, gamma, l2_lambda, s, 0, verbose)
    else:
        re = None
        print('cannot find loss: %s' % loss)
        exit(0)
    return {'wt': np.asarray(re[0]),
            'wt_bar': np.asarray(re[1]),
            'losses': np.asanyarray(re[2]),
            'nonzeros_wt': np.asarray(re[3]),
            'nonzeros_wt_bar': np.asarray(re[4]),
            'pred_prob_wt': np.asarray(re[5]),
            'pred_label_wt': np.asarray(re[6]),
            'pred_prob_wt_bar': np.asarray(re[7]),
            'pred_label_wt_bar': np.asarray(re[8]),
            'missed_wt': np.asarray(re[9]),
            'missed_wt_bar': np.asarray(re[10]),
            'total_time': re[11],
            're_nodes': np.asarray(re[12]),
            're_edges': np.asarray(re[13]),
            'num_pcst': re[14],
            'run_time_head': re[15],
            'run_time_tail': re[16],
            'y_tr': y_tr}


def algo_online_graph_ghtp(
        x_tr, y_tr, w0, lr, eta, edges, weights, s, verbose):
    (wt, wt_bar, nonzeros_wt, nonzeros_wt_bar, pred_prob_wt, pred_label_wt,
     pred_prob_wt_bar, pred_label_wt_bar, re_nodes, re_edges, num_pcst,
     losses, run_time_head, run_time_tail, missed_wt, missed_wt_bar,
     total_time) = online_graph_ghtp_logit(
        x_tr, y_tr, w0, lr, eta, edges, weights, s, verbose)
    return {'wt': np.asarray(wt),
            'wt_bar': np.asarray(wt_bar),
            'nonzeros_wt': np.asarray(nonzeros_wt),
            'nonzeros_wt_bar': np.asarray(nonzeros_wt_bar),
            'pred_prob_wt': np.asarray(pred_prob_wt),
            'pred_label_wt': np.asarray(pred_label_wt),
            'pred_prob_wt_bar': np.asarray(pred_prob_wt_bar),
            'pred_label_wt_bar': np.asarray(pred_label_wt_bar),
            'losses': np.asarray(losses),
            'missed_wt': missed_wt,
            'missed_wt_bar': missed_wt_bar,
            'total_time': total_time,
            'num_pcst': num_pcst,
            'run_time_head': run_time_head,
            'run_time_tail': run_time_tail,
            're_nodes': np.asarray(re_nodes, dtype=int),
            're_edges': np.asarray(re_edges, dtype=int),
            'y_tr': y_tr}


def algo_online_ghtp(x_tr, y_tr, w0, lr, eta, s, verbose):
    (wt, wt_bar, nonzeros_wt, nonzeros_wt_bar, pred_prob_wt, pred_label_wt,
     pred_prob_wt_bar, pred_label_wt_bar, losses, missed_wt, missed_wt_bar,
     total_time) = online_ghtp_logit(x_tr, y_tr, w0, lr, eta, s, verbose)
    return {'wt': np.asarray(wt),
            'wt_bar': np.asarray(wt_bar),
            'nonzeros_wt': np.asarray(nonzeros_wt),
            'nonzeros_wt_bar': np.asarray(nonzeros_wt_bar),
            'pred_prob_wt': np.asarray(pred_prob_wt),
            'pred_label_wt': np.asarray(pred_label_wt),
            'pred_prob_wt_bar': np.asarray(pred_prob_wt_bar),
            'pred_label_wt_bar': np.asarray(pred_label_wt_bar),
            'losses': np.asarray(losses),
            'missed_wt': missed_wt,
            'missed_wt_bar': missed_wt_bar,
            'total_time': total_time,
            'y_tr': y_tr}


def algo_graph_sto_iht_linsearch(
        x_tr, y_tr, max_epochs, lr, w, w0, tol_algo, b, edges, costs, g,
        root, h_low, h_high, t_low, t_high, max_num_iter, verbose):
    start_time = time.time()
    np.random.seed()  # do not forget it.
    w_hat, num_iter = w0, 0
    (m, p) = x_tr.shape
    x_tr_t = np.transpose(x_tr)
    num_blocks = int(m) / int(b)
    y_error_list = []
    w_error_list = []
    w_error_ratio_list = []
    for _ in range(max_epochs * num_blocks):
        num_iter = _
        # random select a block
        ii = np.random.randint(0, num_blocks)
        block = range(b * ii, b * (ii + 1))
        xtx = np.dot(x_tr_t[:, block], x_tr[block])
        xty = np.dot(x_tr_t[:, block], y_tr[block])
        gradient = -2. * (xty - np.dot(xtx, w_hat))
        fun_val_right = np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) ** 2.
        tmp_num_iter, adaptive_step, beta = 0, 2.0, 0.8
        while tmp_num_iter < 20:
            x_tmp = w_hat - adaptive_step * gradient
            fun_val_left = np.linalg.norm(y_tr - np.dot(x_tr, x_tmp)) ** 2.
            reg_term = adaptive_step / 2. * np.linalg.norm(gradient) ** 2.
            if fun_val_left > fun_val_right - reg_term:
                adaptive_step *= beta
            else:
                break
            tmp_num_iter += 1

        head_nodes, proj_grad = algo_head_tail_binsearch(
            edges, gradient, costs, g, root, h_low, h_high,
            max_num_iter, verbose)
        bt = w_hat - adaptive_step * proj_grad
        tail_nodes, proj_bt = algo_head_tail_binsearch(
            edges, bt, costs, g, root, t_low, t_high, max_num_iter, verbose)
        w_hat = proj_bt
        w_error = np.linalg.norm(w_hat - w)
        y_error = np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) ** 2.
        w_error_ratio = (w_error ** 2.) / (np.linalg.norm(w) ** 2.)
        y_error_list.append(y_error)
        w_error_list.append(w_error)
        w_error_ratio_list.append(w_error_ratio)
        # diverge cases because of the large learning rate: early stopping
        if np.linalg.norm(w_hat) >= 1e3:
            break
        if np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) <= tol_algo:
            break
    run_time = time.time() - start_time
    return {'w_error_list': w_error_list,
            'y_error_list': y_error_list,
            'w_error_ratio_list': w_error_ratio_list,
            'num_iter': num_iter,
            'run_time': run_time}


def algo_graph_da_iht(
        x_tr, y_tr, max_epochs, sigma, w, w0, tol_algo, b, edges, costs, g,
        root, h_low, h_high, t_low, t_high, max_num_iter, verbose):
    start_time = time.time()
    np.random.seed()  # do not forget it.
    w_hat, num_iter = w0, 0
    (m, p) = x_tr.shape
    x_tr_t = np.transpose(x_tr)
    num_blocks = int(m) / int(b)
    y_error_list = []
    w_error_list = []
    w_error_ratio_list = []
    grad, grad_bar = np.zeros_like(w), np.zeros_like(w)
    for _ in range(max_epochs * num_blocks):
        num_iter = _
        # random select a block
        ii = np.random.randint(0, num_blocks)
        block = range(b * ii, b * (ii + 1))
        xtx = np.dot(x_tr_t[:, block], x_tr[block])
        xty = np.dot(x_tr_t[:, block], y_tr[block])
        grad = -2. * (xty - np.dot(xtx, w_hat))
        grad_bar = (_ / (_ + 1.)) * grad_bar + (1. / (_ + 1.)) * grad
        grad_bar = (-1 / sigma) * grad_bar
        head_nodes, proj_grad = algo_head_tail_binsearch(
            edges, grad_bar, costs, g, root, h_low, h_high,
            max_num_iter, verbose)
        tail_nodes, proj_bt = algo_head_tail_binsearch(
            edges, proj_grad, costs, g, root, t_low,
            t_high, max_num_iter, verbose)
        w_hat = proj_bt

        w_error = np.linalg.norm(w_hat - w)
        y_error = np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) ** 2.
        w_error_ratio = (w_error ** 2.) / (np.linalg.norm(w) ** 2.)
        y_error_list.append(y_error)
        w_error_list.append(w_error)
        w_error_ratio_list.append(w_error_ratio)

        # diverge cases because of the large learning rate: early stopping
        if np.linalg.norm(w_hat) >= 1e3:
            break
        if np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) <= tol_algo:
            break
    run_time = time.time() - start_time
    return {'w_error_list': w_error_list,
            'y_error_list': y_error_list,
            'w_error_ratio_list': w_error_ratio_list,
            'num_iter': num_iter,
            'run_time': run_time}


def algo_iht(x_tr, y_tr, max_epochs, lr, s, w, w0, tol_algo):
    start_time = time.time()
    w_hat, num_iter = w0, 0
    (m, p) = x_tr.shape
    x_tr_t = np.transpose(x_tr)
    xtx, xty = np.dot(x_tr_t, x_tr), np.dot(x_tr_t, y_tr)
    y_error_list = []
    w_error_list = []
    w_error_ratio_list = []
    for _ in range(max_epochs):
        num_iter = _
        # we obey the implementation used in their code
        bt = w_hat - lr * (np.dot(xtx, w_hat) - xty)
        bt[np.argsort(np.abs(bt))[0:p - s]] = 0.
        w_hat = bt

        w_error = np.linalg.norm(w_hat - w)
        y_error = np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) ** 2.
        w_error_ratio = (w_error ** 2.) / (np.linalg.norm(w) ** 2.)
        y_error_list.append(y_error)
        w_error_list.append(w_error)
        w_error_ratio_list.append(w_error_ratio)

        # diverge cases because of the large learning rate: early stopping
        if np.linalg.norm(w_hat) >= 1e3:  # diverge cases.
            break
        if np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) <= tol_algo:
            break
    run_time = time.time() - start_time
    return {'w_error_list': w_error_list,
            'y_error_list': y_error_list,
            'w_error_ratio_list': w_error_ratio_list,
            'num_iter': num_iter,
            'run_time': run_time}


def algo_sto_iht(x_tr, y_tr, max_epochs, lr, s, w, w0, tol_algo, b):
    start_time = time.time()
    np.random.seed()  # do not forget it.
    w_hat, num_iter = w0, 0
    (m, p) = x_tr.shape
    x_tr_t = np.transpose(x_tr)
    b = m if m < b else b
    num_blocks = int(m) / int(b)
    prob = [1. / num_blocks] * num_blocks
    y_error_list = []
    w_error_list = []
    w_error_ratio_list = []
    for _ in range(max_epochs * num_blocks):
        num_iter = _
        # random select a block
        ii = np.random.randint(0, num_blocks)
        block = range(b * ii, b * (ii + 1))
        xtx = np.dot(x_tr_t[:, block], x_tr[block])
        xty = np.dot(x_tr_t[:, block], y_tr[block])
        gradient = - 2. * (xty - np.dot(xtx, w_hat))
        bt = w_hat - (lr / (prob[ii] * num_blocks)) * gradient
        bt[np.argsort(np.abs(bt))[0:p - s]] = 0.
        w_hat = bt

        w_error = np.linalg.norm(w_hat - w)
        y_error = np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) ** 2.
        w_error_ratio = (w_error ** 2.) / (np.linalg.norm(w) ** 2.)
        y_error_list.append(y_error)
        w_error_list.append(w_error)
        w_error_ratio_list.append(w_error_ratio)

        # diverge cases because of the large learning rate: early stopping
        if np.linalg.norm(w_hat) >= 1e3:
            break
        if np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) <= tol_algo:
            break
    run_time = time.time() - start_time
    return {'w_error_list': w_error_list,
            'y_error_list': y_error_list,
            'w_error_ratio_list': w_error_ratio_list,
            'num_iter': num_iter,
            'run_time': run_time}


def algo_graph_iht(
        x_tr, y_tr, max_epochs, lr, w, w0, tol_algo, edges, costs, g, root,
        h_low, h_high, t_low, t_high, max_num_iter, verbose):
    start_time = time.time()
    w_hat, num_iter = w0, 0
    x_tr_t = np.transpose(x_tr)
    xtx, xty = np.dot(x_tr_t, x_tr), np.dot(x_tr_t, y_tr)
    y_error_list = []
    w_error_list = []
    w_error_ratio_list = []
    for _ in range(max_epochs):
        num_iter = _
        grad = -1. * (xty - np.dot(xtx, w_hat))
        head_nodes, proj_gradient = algo_head_tail_binsearch(
            edges, grad, costs, g, root, h_low, h_high, max_num_iter, verbose)
        bt = w_hat - lr * proj_gradient
        tail_nodes, proj_bt = algo_head_tail_binsearch(
            edges, bt, costs, g, root, t_low, t_high, max_num_iter, verbose)
        w_hat = proj_bt

        w_error = np.linalg.norm(w_hat - w)
        y_error = np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) ** 2.
        w_error_ratio = (w_error ** 2.) / (np.linalg.norm(w) ** 2.)
        y_error_list.append(y_error)
        w_error_list.append(w_error)
        w_error_ratio_list.append(w_error_ratio)
        # diverge cases because of the large learning rate: early stopping
        if np.linalg.norm(w_hat) >= 1e3:
            break
        if np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) <= tol_algo:
            break
    run_time = time.time() - start_time
    return {'w_error_list': w_error_list,
            'y_error_list': y_error_list,
            'w_error_ratio_list': w_error_ratio_list,
            'num_iter': num_iter,
            'run_time': run_time}


def algo_graph_sto_iht(
        x_tr, y_tr, max_epochs, lr, w, w0, tol_algo, b, edges, costs, g,
        root, h_low, h_high, t_low, t_high, max_num_iter, verbose):
    start_time = time.time()
    np.random.seed()  # do not forget it.
    w_hat, num_iter = w0, 0
    (m, p) = x_tr.shape
    x_tr_t = np.transpose(x_tr)
    b = m if m < b else b
    num_blocks = int(m) / int(b)
    y_error_list = []
    w_error_list = []
    w_error_ratio_list = []
    for _ in range(max_epochs * num_blocks):
        num_iter = _
        # random select a block
        ii = np.random.randint(0, num_blocks)
        block = range(b * ii, b * (ii + 1))
        xtx = np.dot(x_tr_t[:, block], x_tr[block])
        xty = np.dot(x_tr_t[:, block], y_tr[block])
        gradient = -2. * (xty - np.dot(xtx, w_hat))
        head_nodes, proj_grad = algo_head_tail_binsearch(
            edges, gradient, costs, g, root, h_low, h_high,
            max_num_iter, verbose)
        bt = w_hat - lr * proj_grad
        tail_nodes, proj_bt = algo_head_tail_binsearch(
            edges, bt, costs, g, root, t_low, t_high, max_num_iter, verbose)
        w_hat = proj_bt
        w_error = np.linalg.norm(w_hat - w)
        y_error = np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) ** 2.
        w_error_ratio = (w_error ** 2.) / (np.linalg.norm(w) ** 2.)
        y_error_list.append(y_error)
        w_error_list.append(w_error)
        w_error_ratio_list.append(w_error_ratio)
        # diverge cases because of the large learning rate: early stopping
        if np.linalg.norm(w_hat) >= 1e3:
            break
        if np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) <= tol_algo:
            break
    run_time = time.time() - start_time
    return {'w_error_list': w_error_list,
            'y_error_list': y_error_list,
            'w_error_ratio_list': w_error_ratio_list,
            'num_iter': num_iter,
            'run_time': run_time}


def algo_gradmp(x_tr, y_tr, max_epochs, w, w0, tol_algo, s):
    start_time = time.time()
    w_hat, num_iter = np.zeros_like(w0), 0
    x_tr_t = np.transpose(x_tr)
    m, p = x_tr.shape
    y_error_list = []
    w_error_list = []
    w_error_ratio_list = []
    xtx, xty = np.dot(x_tr_t, x_tr), np.dot(x_tr_t, y_tr)
    for _ in range(max_epochs):
        num_iter = _
        grad = -(2. / float(m)) * (np.dot(xtx, w_hat) - xty)  # proxy
        gamma = np.argsort(abs(grad))[-2 * s:]  # identify
        gamma = np.union1d(w_hat.nonzero()[0], gamma)
        bt = np.zeros_like(w_hat)
        bt[gamma] = np.dot(np.linalg.pinv(x_tr[:, gamma]), y_tr)
        gamma = np.argsort(abs(bt))[-s:]
        w_hat = np.zeros_like(w_hat)
        w_hat[gamma] = bt[gamma]

        w_error = np.linalg.norm(w_hat - w)
        y_error = np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) ** 2.
        w_error_ratio = (w_error ** 2.) / (np.linalg.norm(w) ** 2.)
        y_error_list.append(y_error)
        w_error_list.append(w_error)
        w_error_ratio_list.append(w_error_ratio)

        if np.linalg.norm(w_hat) >= 1e3:
            break
        if np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) <= tol_algo:
            break
    run_time = time.time() - start_time
    return {'w_error_list': w_error_list,
            'y_error_list': y_error_list,
            'w_error_ratio_list': w_error_ratio_list,
            'num_iter': num_iter,
            'run_time': run_time}


def algo_sto_gradmp(x_tr, y_tr, max_epochs, w, w0, tol_algo, s, b):
    start_time = time.time()
    w_hat, num_iter = np.zeros_like(w0), 0
    x_tr_t = np.transpose(x_tr)
    m, p = x_tr.shape
    b = m if m < b else b
    num_blocks = int(m) / int(b)
    y_error_list = []
    w_error_list = []
    w_error_ratio_list = []
    for _ in range(max_epochs * num_blocks):
        num_iter = _
        # random select a block
        ii = np.random.randint(0, num_blocks)
        block = range(b * ii, b * (ii + 1))
        xtx = np.dot(x_tr_t[:, block], x_tr[block])
        xty = np.dot(x_tr_t[:, block], y_tr[block])
        # proxy
        grad = -2. * (np.dot(xtx, w_hat) - xty)
        # identify
        gamma = np.argsort(abs(grad))[-2 * s:]
        gamma = np.union1d(w_hat.nonzero()[0], gamma)
        bt = np.zeros_like(w_hat)
        bt[gamma] = np.dot(np.linalg.pinv(x_tr[:, gamma]), y_tr)
        gamma = np.argsort(abs(bt))[-s:]
        w_hat = np.zeros_like(w_hat)
        w_hat[gamma] = bt[gamma]

        w_error = np.linalg.norm(w_hat - w)
        y_error = np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) ** 2.
        w_error_ratio = (w_error ** 2.) / (np.linalg.norm(w) ** 2.)
        y_error_list.append(y_error)
        w_error_list.append(w_error)
        w_error_ratio_list.append(w_error_ratio)

        if np.linalg.norm(w_hat) >= 1e3:
            break
        if np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) <= tol_algo:
            break
    run_time = time.time() - start_time
    return {'w_error_list': w_error_list,
            'y_error_list': y_error_list,
            'w_error_ratio_list': w_error_ratio_list,
            'num_iter': num_iter,
            'run_time': run_time}


def algo_graph_gradmp(
        x_tr, y_tr, max_epochs, w, w0, tol_algo, edges, costs, g, root, h_low,
        h_high, t_low, t_high, max_num_iter, verbose):
    start_time = time.time()
    w_hat, num_iter = np.zeros_like(w0), 0
    x_tr_t = np.transpose(x_tr)
    m, p = x_tr.shape
    xtx, xty = np.dot(x_tr_t, x_tr), np.dot(x_tr_t, y_tr)
    y_error_list = []
    w_error_list = []
    w_error_ratio_list = []
    for _ in range(max_epochs):
        num_iter = _
        grad = -(2. / float(m)) * (np.dot(xtx, w_hat) - xty)  # proxy
        head_nodes, proj_grad = algo_head_tail_binsearch(
            edges, grad, costs, g, root, h_low, h_high, max_num_iter, verbose)
        gamma = np.union1d(w_hat.nonzero()[0], head_nodes)
        bt = np.zeros_like(w_hat)
        bt[gamma] = np.dot(np.linalg.pinv(x_tr[:, gamma]), y_tr)
        tail_nodes, proj_bt = algo_head_tail_binsearch(
            edges, bt, costs, g, root, t_low, t_high, max_num_iter, verbose)
        w_hat = proj_bt

        w_error = np.linalg.norm(w_hat - w)
        y_error = np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) ** 2.
        w_error_ratio = (w_error ** 2.) / (np.linalg.norm(w) ** 2.)
        y_error_list.append(y_error)
        w_error_list.append(w_error)
        w_error_ratio_list.append(w_error_ratio)
        if np.linalg.norm(w_hat) >= 1e3:
            break
        if np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) <= tol_algo:
            break
    run_time = time.time() - start_time
    return {'w_error_list': w_error_list,
            'y_error_list': y_error_list,
            'w_error_ratio_list': w_error_ratio_list,
            'num_iter': num_iter,
            'run_time': run_time}


def algo_graph_sto_gradmp(
        x_tr, y_tr, max_epochs, w, w0, tol_algo, b, edges, costs, g, root,
        h_low, h_high, t_low, t_high, max_num_iter, verbose):
    start_time = time.time()
    w_hat, num_iter = np.zeros_like(w0), 0
    x_tr_t = np.transpose(x_tr)
    m, p = x_tr.shape
    b = m if m < b else b
    num_blocks = int(m) / int(b)
    y_error_list = []
    w_error_list = []
    w_error_ratio_list = []
    for _ in range(max_epochs * num_blocks):
        num_iter = _
        # random select a block
        ii = np.random.randint(0, num_blocks)
        block = range(b * ii, b * (ii + 1))
        xtx = np.dot(x_tr_t[:, block], x_tr[block])
        xty = np.dot(x_tr_t[:, block], y_tr[block])
        # proxy
        grad = -2. * (np.dot(xtx, w_hat) - xty)
        # identify
        head_nodes, proj_grad = algo_head_tail_binsearch(
            edges, grad, costs, g, root, h_low, h_high, max_num_iter, verbose)
        gamma = np.union1d(w_hat.nonzero()[0], head_nodes)
        bt = np.zeros_like(w_hat)
        bt[gamma] = np.dot(np.linalg.pinv(x_tr[:, gamma]), y_tr)
        tail_nodes, proj_bt = algo_head_tail_binsearch(
            edges, bt, costs, g, root, t_low, t_high, max_num_iter, verbose)
        w_hat = proj_bt
        w_error = np.linalg.norm(w_hat - w)
        y_error = np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) ** 2.
        w_error_ratio = (w_error ** 2.) / (np.linalg.norm(w) ** 2.)
        y_error_list.append(y_error)
        w_error_list.append(w_error)
        w_error_ratio_list.append(w_error_ratio)
        if np.linalg.norm(w_hat) >= 1e3:
            break
        if np.linalg.norm(y_tr - np.dot(x_tr, w_hat)) <= tol_algo:
            break
    run_time = time.time() - start_time
    return {'w_error_list': w_error_list,
            'y_error_list': y_error_list,
            'w_error_ratio_list': w_error_ratio_list,
            'num_iter': num_iter,
            'run_time': run_time}


def algo_batch_iht(x_tr, y_tr, w0, lr, s, tol, max_iter, eta, verbose):
    (wt, losses, total_time) = batch_iht_logit(
        x_tr, y_tr, w0, lr, s, tol, max_iter, eta, verbose)
    return {'wt': np.asarray(wt),
            'losses': np.asarray(losses),
            'total_time': total_time}


def algo_batch_ghtp(x_tr, y_tr, w0, lr, s, tol, max_iter, eta, verbose):
    (wt, losses, total_time) = batch_ghtp_logit(
        x_tr, y_tr, w0, lr, s, tol, max_iter, eta, verbose)
    return {'wt': np.asarray(wt),
            'losses': np.asarray(losses),
            'total_time': total_time}


def algo_batch_graph_iht(
        x_tr, y_tr, w0, lr, s, tol, max_iter, eta, edges, weis, verbose):
    (wt, losses, total_time) = batch_graph_iht_logit(
        x_tr, y_tr, w0, lr, s, tol, max_iter, eta, edges, weis, verbose)
    return {'wt': np.asarray(wt),
            'losses': np.asarray(losses),
            'total_time': total_time}


def algo_batch_graph_ghtp(
        x_tr, y_tr, w0, lr, s, tol, max_iter, eta, edges, weis, verbose):
    (wt, losses, total_time) = batch_graph_ghtp_logit(
        x_tr, y_tr, w0, lr, s, tol, max_iter, eta, edges, weis, verbose)
    return {'wt': np.asarray(wt),
            'losses': np.asarray(losses),
            'total_time': total_time}


def algo_batch_graph_posi(
        x_tr, y_tr, w0, lr, s, tol, max_iter, eta, edges, weis, verbose):
    (wt, losses, total_time) = batch_graph_posi_logit(
        x_tr, y_tr, w0, lr, s, tol, max_iter, eta, edges, weis, verbose)
    return {'wt': np.asarray(wt),
            'losses': np.asarray(losses),
            'total_time': total_time}


def wrapper_head_tail_binsearch(
        edges, prizes, costs, g, root, sparsity_low, sparsity_high,
        max_num_iter, verbose):
    return wrap_head_tail_binsearch(
        edges, prizes, costs, g, root, sparsity_low, sparsity_high,
        max_num_iter, verbose)
