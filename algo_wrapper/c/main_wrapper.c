//
//
#include <Python.h>
#include <numpy/arrayobject.h>
#include "sparse_algorithms.h"
#include "algo_online_adam.h"
#include "algo_online_da_gl.h"
#include "algo_online_rda_l1.h"
#include "algo_online_ada_grad.h"
#include "algo_online_sto_iht.h"
#include "algo_online_da_iht.h"
#include "algo_online_best_subset.h"
#include "algo_online_graph_iht.h"
#include "algo_online_graph_da_iht.h"

bool get_data(
        int n, int p, int m, double *x_tr, double *y_tr, double *w0,
        EdgePair *edges, double *weights, PyArrayObject *x_tr_,
        PyArrayObject *y_tr_, PyArrayObject *w0_, PyArrayObject *edges_,
        PyArrayObject *weights_) {
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < p; j++) {
            x_tr[i * p + j] = *(double *) PyArray_GETPTR2(x_tr_, i, j);
        }
        y_tr[i] = *(double *) PyArray_GETPTR1(y_tr_, i);
    }
    for (i = 0; i < (p + 1); i++) {
        w0[i] = *(double *) PyArray_GETPTR1(w0_, i);;
    }
    if (edges != NULL) {
        for (i = 0; i < m; i++) {
            edges[i].first = *(int *) PyArray_GETPTR2(edges_, i, 0);
            edges[i].second = *(int *) PyArray_GETPTR2(edges_, i, 1);
            weights[i] = *(double *) PyArray_GETPTR1(weights_, i);
        }
    }
    return true;
}

PyObject *batch_get_result(
        int p, int max_iter, double total_time, double *wt, double *losses) {
    PyObject *results = PyTuple_New(3);
    PyObject *re_wt = PyList_New(p + 1);
    for (int i = 0; i < p + 1; i++) {
        PyList_SetItem(re_wt, i, PyFloat_FromDouble(wt[i]));
    }
    PyObject *re_losses = PyList_New(max_iter);
    for (int i = 0; i < max_iter; i++) {
        PyList_SetItem(re_losses, i, PyFloat_FromDouble(losses[i]));
    }
    PyObject *re_total_time = PyFloat_FromDouble(total_time);
    PyTuple_SetItem(results, 0, re_wt);
    PyTuple_SetItem(results, 1, re_losses);
    PyTuple_SetItem(results, 2, re_total_time);
    return results;
}

PyObject *online_get_result(int p, int num_tr, OnlineStat *stat) {
    PyObject *results = PyTuple_New(17);

    PyObject *re_wt = PyList_New(p + 1);
    PyObject *re_wt_bar = PyList_New(p + 1);
    PyObject *re_nonzeros_wt = PyList_New(num_tr);
    PyObject *re_nonzeros_wt_bar = PyList_New(num_tr);
    PyObject *re_nodes = PyList_New(stat->re_nodes->size);
    PyObject *re_edges = PyList_New(stat->re_edges->size);
    PyObject *re_pred_prob_wt = PyList_New(num_tr);
    PyObject *re_pred_prob_wt_bar = PyList_New(num_tr);
    PyObject *re_pred_label_wt = PyList_New(num_tr);
    PyObject *re_pred_label_wt_bar = PyList_New(num_tr);
    PyObject *re_num_pcst = PyInt_FromLong(stat->num_pcst);
    PyObject *re_losses = PyList_New(num_tr);
    PyObject *re_run_time_head = PyFloat_FromDouble(stat->run_time_head);
    PyObject *re_run_time_tail = PyFloat_FromDouble(stat->run_time_tail);
    PyObject *re_missed_wt = PyList_New(num_tr);
    PyObject *re_missed_wt_bar = PyList_New(num_tr);
    PyObject *re_total_time = PyFloat_FromDouble(stat->total_time);
    for (int i = 0; i < p + 1; i++) {
        PyList_SetItem(re_wt, i, PyFloat_FromDouble(stat->wt[i]));
        PyList_SetItem(re_wt_bar, i, PyFloat_FromDouble(stat->wt_bar[i]));
    }
    for (int i = 0; i < num_tr; i++) {
        PyList_SetItem(re_nonzeros_wt, i,
                       PyInt_FromLong(stat->nonzeros_wt[i]));
        PyList_SetItem(re_nonzeros_wt_bar, i,
                       PyInt_FromLong(stat->nonzeros_wt_bar[i]));
    }
    for (int i = 0; i < stat->re_nodes->size; i++) {
        PyList_SetItem(re_nodes, i, PyInt_FromLong(stat->re_nodes->array[i]));
    }
    for (int i = 0; i < stat->re_edges->size; i++) {
        PyList_SetItem(re_edges, i, PyInt_FromLong(stat->re_edges->array[i]));
    }
    for (int i = 0; i < num_tr; i++) {
        PyList_SetItem(re_pred_prob_wt, i,
                       PyFloat_FromDouble(stat->p_prob_wt[i]));
        PyList_SetItem(re_pred_label_wt, i,
                       PyFloat_FromDouble(stat->p_label_wt[i]));
        PyList_SetItem(re_pred_prob_wt_bar, i,
                       PyFloat_FromDouble(stat->p_prob_wt_bar[i]));
        PyList_SetItem(re_pred_label_wt_bar, i,
                       PyFloat_FromDouble(stat->p_label_wt_bar[i]));
    }
    for (int i = 0; i < num_tr; i++) {
        PyList_SetItem(re_losses, i, PyFloat_FromDouble(stat->losses[i]));
    }
    for (int i = 0; i < num_tr; i++) {
        PyList_SetItem(re_missed_wt, i, PyInt_FromLong(stat->missed_wt[i]));
        PyList_SetItem(re_missed_wt_bar, i,
                       PyInt_FromLong(stat->missed_wt_bar[i]));
    }

    PyTuple_SetItem(results, 0, re_wt);
    PyTuple_SetItem(results, 1, re_wt_bar);
    PyTuple_SetItem(results, 2, re_losses);
    PyTuple_SetItem(results, 3, re_nonzeros_wt);
    PyTuple_SetItem(results, 4, re_nonzeros_wt_bar);
    PyTuple_SetItem(results, 5, re_pred_prob_wt);
    PyTuple_SetItem(results, 6, re_pred_label_wt);
    PyTuple_SetItem(results, 7, re_pred_prob_wt_bar);
    PyTuple_SetItem(results, 8, re_pred_label_wt_bar);
    PyTuple_SetItem(results, 9, re_missed_wt);
    PyTuple_SetItem(results, 10, re_missed_wt_bar);
    PyTuple_SetItem(results, 11, re_total_time);
    PyTuple_SetItem(results, 12, re_nodes);
    PyTuple_SetItem(results, 13, re_edges);
    PyTuple_SetItem(results, 14, re_num_pcst);
    PyTuple_SetItem(results, 15, re_run_time_head);
    PyTuple_SetItem(results, 16, re_run_time_tail);
    return results;
}

static PyObject *wrap_head_tail_binsearch(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("error: unknown error !!\n");
        return NULL;
    }
    head_tail_binsearch_para *para = malloc(sizeof(head_tail_binsearch_para));
    PyArrayObject *edges_, *costs_, *prizes_;
    if (!PyArg_ParseTuple(args, "O!O!O!iiiiii",
                          &PyArray_Type, &edges_,
                          &PyArray_Type, &prizes_,
                          &PyArray_Type, &costs_,
                          &para->g,
                          &para->root,
                          &para->sparsity_low,
                          &para->sparsity_high,
                          &para->max_num_iter,
                          &para->verbose)) { return NULL; }

    para->p = (int) prizes_->dimensions[0];
    para->m = (int) edges_->dimensions[0];
    para->prizes = (double *) PyArray_DATA(prizes_);
    para->costs = (double *) PyArray_DATA(costs_);
    para->edges = malloc(sizeof(EdgePair) * para->m);
    for (int i = 0; i < para->m; i++) {
        para->edges[i].first = *(int *) PyArray_GETPTR2(edges_, i, 0);
        para->edges[i].second = *(int *) PyArray_GETPTR2(edges_, i, 1);
    }
    GraphStat *graph_stat = make_graph_stat(para->p, para->m);
    head_tail_binsearch(
            para->edges, para->costs, para->prizes, para->p, para->m, para->g,
            para->root, para->sparsity_low, para->sparsity_high,
            para->max_num_iter, GWPruning, para->verbose, graph_stat);
    PyObject *results = PyTuple_New(1);
    PyObject *re_nodes = PyList_New(graph_stat->re_nodes->size);
    for (int i = 0; i < graph_stat->re_nodes->size; i++) {
        int cur_node = graph_stat->re_nodes->array[i];
        PyList_SetItem(re_nodes, i, PyInt_FromLong(cur_node));
    }
    PyTuple_SetItem(results, 0, re_nodes);
    free_graph_stat(graph_stat);
    free(para->edges);
    free(para);
    return results;
}

static PyObject *online_rda_l1(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("error: unknown error !!\n");
        return NULL;
    }
    rda_l1_para *para = malloc(sizeof(rda_l1_para));
    PyArrayObject *x_tr_, *y_tr_, *w0_;
    if (!PyArg_ParseTuple(args, "O!O!O!dddii",
                          &PyArray_Type, &x_tr_,
                          &PyArray_Type, &y_tr_,
                          &PyArray_Type, &w0_,
                          &para->lambda,
                          &para->gamma,
                          &para->rho,
                          &para->loss_func,
                          &para->verbose)) { return NULL; }
    para->num_tr = (int) x_tr_->dimensions[0];
    para->p = (int) x_tr_->dimensions[1];
    para->x_tr = (double *) PyArray_DATA(x_tr_);
    para->y_tr = (double *) PyArray_DATA(y_tr_);
    para->w0 = (double *) PyArray_DATA(w0_);
    OnlineStat *stat = make_online_stat(para->p, para->num_tr);
    if (para->loss_func == 0) {
        algo_online_rda_l1_logit(para, stat);
    } else if (para->loss_func == 1) {
        algo_online_rda_l1_least_square(para, stat);
    }
    PyObject *results = online_get_result(para->p, para->num_tr, stat);
    free(para), free_online_stat(stat);
    return results;
}

// input: x_tr, y_tr, w0, lambda_, lr,
// group_list, group_size_list, num_group, verbose
static PyObject *online_da_gl(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("error: unknown error !!\n");
        return NULL;
    }
    da_gl_para *para = malloc(sizeof(da_gl_para));
    PyArrayObject *x_tr_, *y_tr_, *w0_, *group_size_list_, *group_list_;
    if (!PyArg_ParseTuple(args, "O!O!O!ddO!O!iii", &PyArray_Type, &x_tr_,
                          &PyArray_Type, &y_tr_, &PyArray_Type, &w0_,
                          &para->lambda, &para->gamma, &PyArray_Type,
                          &group_list_, &PyArray_Type, &group_size_list_,
                          &para->num_group, &para->loss_func,
                          &para->verbose)) { return NULL; }
    para->num_tr = (int) x_tr_->dimensions[0];
    para->p = (int) x_tr_->dimensions[1];
    para->x_tr = (double *) PyArray_DATA(x_tr_);
    para->y_tr = (double *) PyArray_DATA(y_tr_);
    para->w0 = (double *) PyArray_DATA(w0_);
    para->group_list = (int *) PyArray_DATA(group_list_);
    para->group_size_list = (int *) PyArray_DATA(group_size_list_);
    OnlineStat *stat = make_online_stat(para->p, para->num_tr);
    if (para->loss_func == 0) {
        algo_online_da_gl_logit(para, stat);
    } else if (para->loss_func == 1) {
        algo_online_da_gl_least_square(para, stat);
    }
    PyObject *results = online_get_result(para->p, para->num_tr, stat);
    free(para), free_online_stat(stat);
    return results;
}

//input: x_tr, y_tr, w0, lambda_, lr, group_list, group_size, verbose
static PyObject *online_da_sgl(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("error: unknown error !!\n");
        return NULL;
    }
    da_sgl_para *para = malloc(sizeof(da_sgl_para));
    PyArrayObject *x_tr_, *y_tr_, *w0_, *group_size_list_, *group_list_, *r_;
    if (!PyArg_ParseTuple(args, "O!O!O!ddO!O!O!iii", &PyArray_Type, &x_tr_,
                          &PyArray_Type, &y_tr_, &PyArray_Type, &w0_,
                          &para->lambda, &para->gamma, &PyArray_Type,
                          &group_list_, &PyArray_Type, &group_size_list_,
                          &PyArray_Type, &r_, &para->num_group,
                          &para->loss_func, &para->verbose)) { return NULL; }
    para->num_tr = (int) x_tr_->dimensions[0];
    para->p = (int) x_tr_->dimensions[1];
    para->x_tr = (double *) PyArray_DATA(x_tr_);
    para->y_tr = (double *) PyArray_DATA(y_tr_);
    para->w0 = (double *) PyArray_DATA(w0_);
    para->group_list = (int *) PyArray_DATA(group_list_);
    para->group_size_list = (int *) PyArray_DATA(group_size_list_);
    para->r = (int *) PyArray_DATA(r_);
    OnlineStat *stat = make_online_stat(para->p, para->num_tr);
    if (para->loss_func == 0) {
        algo_online_da_sgl_logit(para, stat);
    } else if (para->loss_func == 1) {
        algo_online_da_sgl_least_square(para, stat);
    }
    PyObject *results = online_get_result(para->p, para->num_tr, stat);
    free(para), free_online_stat(stat);
    return results;
}


static PyObject *online_ada_grad(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("error: unknown error !!\n");
        return NULL;
    }
    ada_grad_para *para = malloc(sizeof(ada_grad_para));
    PyArrayObject *x_tr_, *y_tr_, *w0_;
    if (!PyArg_ParseTuple(args, "O!O!O!dddii", &PyArray_Type, &x_tr_,
                          &PyArray_Type, &y_tr_, &PyArray_Type, &w0_,
                          &para->lambda, &para->eta, &para->delta,
                          &para->loss_func, &para->verbose)) { return NULL; }
    para->num_tr = (int) x_tr_->dimensions[0];
    para->p = (int) x_tr_->dimensions[1];
    para->x_tr = (double *) PyArray_DATA(x_tr_);
    para->y_tr = (double *) PyArray_DATA(y_tr_);
    para->w0 = (double *) PyArray_DATA(w0_);
    OnlineStat *stat = make_online_stat(para->p, para->num_tr);
    if (para->loss_func == 0) {
        algo_online_ada_grad_logit(para, stat);
    } else if (para->loss_func == 1) {
        algo_online_ada_grad_least_square(para, stat);
    }
    PyObject *results = online_get_result(para->p, para->num_tr, stat);
    free(para), free_online_stat(stat);
    return results;
}

//input: x_tr, y_tr, w0, lr, lambda, verbose
static PyObject *online_sgd_l1_logit(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error for no reason.\n");
        return NULL;
    }
    sgd_l1_para *para = malloc(sizeof(sgd_l1_para));
    PyArrayObject *x_tr_, *y_tr_, *w0_;
    if (!PyArg_ParseTuple(args, "O!O!O!ddi", &PyArray_Type, &x_tr_,
                          &PyArray_Type, &y_tr_, &PyArray_Type, &w0_,
                          &para->gamma, &para->lambda,
                          &para->verbose)) { return NULL; }
    para->num_tr = (int) x_tr_->dimensions[0];      // # of samples
    para->p = (int) x_tr_->dimensions[1];           // # of features/nodes
    para->x_tr = (double *) PyArray_DATA(x_tr_);
    para->y_tr = (double *) PyArray_DATA(y_tr_);
    para->w0 = (double *) PyArray_DATA(w0_);
    OnlineStat *stat = make_online_stat(para->p, para->num_tr);
    algo_online_sgd_l1_logit(para, stat);
    PyObject *results = online_get_result(para->p, para->num_tr, stat);
    free(para), free_online_stat(stat);
    return results;
}


//input: x_tr, y_tr, w0, lr, l2_lambda, verbose
static PyObject *online_sgd_l2_logit(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error for no reason.\n");
        return NULL;
    }
    sgd_l2_para *para = malloc(sizeof(sgd_l2_para));
    PyArrayObject *x_tr_, *y_tr_, *w0_;
    if (!PyArg_ParseTuple(args, "O!O!O!ddi", &PyArray_Type, &x_tr_,
                          &PyArray_Type, &y_tr_, &PyArray_Type, &w0_,
                          &para->lr, &para->eta,
                          &para->verbose)) { return NULL; }
    para->num_tr = (int) (x_tr_->dimensions[0]);     // number of samples
    para->p = (int) (x_tr_->dimensions[1]);     // number of features
    para->x_tr = (double *) PyArray_DATA(x_tr_);
    para->y_tr = (double *) PyArray_DATA(y_tr_);
    para->w0 = (double *) PyArray_DATA(w0_);
    OnlineStat *stat = make_online_stat(para->p, para->num_tr);
    algo_online_sgd_l2_logit(para, stat);
    PyObject *results = online_get_result(para->p, para->num_tr, stat);
    free(para), free_online_stat(stat);
    return results;
}


//input: x_tr, y_tr, w0, lr, l2_lambda, sparsity, verbose
static PyObject *online_sto_iht(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error for no reason.\n");
        return NULL;
    }
    iht_para *para = malloc(sizeof(iht_para));
    PyArrayObject *x_tr_, *y_tr_, *w0_;
    if (!PyArg_ParseTuple(args, "O!O!O!ddiii", &PyArray_Type, &x_tr_,
                          &PyArray_Type, &y_tr_, &PyArray_Type, &w0_,
                          &para->lr, &para->l2_lambda, &para->sparsity,
                          &para->loss_func, &para->verbose)) { return NULL; }
    para->num_tr = (int) x_tr_->dimensions[0];      // # of samples
    para->p = (int) x_tr_->dimensions[1];   // # of features/nodes
    para->x_tr = (double *) PyArray_DATA(x_tr_);
    para->y_tr = (double *) PyArray_DATA(y_tr_);
    para->w0 = (double *) PyArray_DATA(w0_);
    OnlineStat *stat = make_online_stat(para->p, para->num_tr);
    if (para->loss_func == 0) {
        algo_online_iht_logit(para, stat);
    } else if (para->loss_func == 1) {
        algo_online_iht_least_square(para, stat);
    }

    PyObject *results = online_get_result(para->p, para->num_tr, stat);
    free(para), free_online_stat(stat);
    return results;
}


static PyObject *online_da_iht(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error for no reason.\n");
        return NULL;
    }

    da_iht_para *para = malloc(sizeof(da_iht_para));
    PyArrayObject *x_tr_, *y_tr_, *w0_;
    // x_tr, y_tr, w0, gamma, l2_lambda, s, 0, verbose
    if (!PyArg_ParseTuple(args, "O!O!O!ddiii", &PyArray_Type, &x_tr_,
                          &PyArray_Type, &y_tr_, &PyArray_Type, &w0_,
                          &para->gamma, &para->l2_lambda,
                          &para->sparsity, &para->loss_func,
                          &para->verbose)) { return NULL; }
    para->num_tr = (int) x_tr_->dimensions[0];      // # of samples
    para->p = (int) x_tr_->dimensions[1];   // # of features/nodes
    para->x_tr = (double *) PyArray_DATA(x_tr_);
    para->y_tr = (double *) PyArray_DATA(y_tr_);
    para->w0 = (double *) PyArray_DATA(w0_);
    OnlineStat *stat = make_online_stat(para->p, para->num_tr);
    if (para->loss_func == 0) {
        algo_online_da_iht_logit(para, stat);
    } else if (para->loss_func == 1) {
        algo_online_da_iht_least_square(para, stat);
    }
    PyObject *results = online_get_result(para->p, para->num_tr, stat);
    free(para), free_online_stat(stat);
    return results;
}


// input: x_tr, y_tr, w0, alpha, beta1, beta2, epsilon, loss_func, verbose
static PyObject *online_adam(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error for no reason.\n");
        return NULL;
    }
    adam_para *para = malloc(sizeof(da_iht_para));
    PyArrayObject *x_tr_, *y_tr_, *w0_;
    if (!PyArg_ParseTuple(args, "O!O!O!ddddii", &PyArray_Type, &x_tr_,
                          &PyArray_Type, &y_tr_, &PyArray_Type, &w0_,
                          &para->alpha, &para->beta1,
                          &para->beta2, &para->epsilon, &para->loss_func,
                          &para->verbose)) { return NULL; }
    para->num_tr = (int) x_tr_->dimensions[0];      // # of samples
    para->p = (int) x_tr_->dimensions[1];   // # of features/nodes
    para->x_tr = (double *) PyArray_DATA(x_tr_);
    para->y_tr = (double *) PyArray_DATA(y_tr_);
    para->w0 = (double *) PyArray_DATA(w0_);
    OnlineStat *stat = make_online_stat(para->p, para->num_tr);
    if (para->loss_func == 0) {
        algo_online_adam_logit(para, stat);
    } else if (para->loss_func == 1) {
        algo_online_adam_least_square(para, stat);
    }
    PyObject *results = online_get_result(para->p, para->num_tr, stat);
    free(para), free_online_stat(stat);
    return results;
}


//input:x_tr, y_tr, w0, edges, costs, lr, l2_lambda, g,
//            sparsity_low, sparsity_high, root, max_num_iter, 0, verbose
static PyObject *online_graph_iht(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error for no reason.\n");
        return NULL;
    }
    graph_iht_para *para = malloc(sizeof(graph_da_iht_para));
    PyArrayObject *x_tr_, *y_tr_, *w0_, *edges_, *weights_;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!ddiiiiiii",
                          &PyArray_Type, &x_tr_,
                          &PyArray_Type, &y_tr_,
                          &PyArray_Type, &w0_,
                          &PyArray_Type, &edges_,
                          &PyArray_Type, &weights_,
                          &para->lr,
                          &para->l2_lambda,
                          &para->g,
                          &para->sparsity_low,
                          &para->sparsity_high,
                          &para->root,
                          &para->max_num_iter,
                          &para->loss_func,
                          &para->verbose)) { return NULL; }
    para->num_tr = (int) x_tr_->dimensions[0];      // # of samples
    para->p = (int) x_tr_->dimensions[1];   // # of features/nodes
    para->m = (int) edges_->dimensions[0];  // # of edges
    para->x_tr = (double *) PyArray_DATA(x_tr_);
    para->y_tr = (double *) PyArray_DATA(y_tr_);
    para->w0 = (double *) PyArray_DATA(w0_);
    para->weights = (double *) PyArray_DATA(weights_);
    para->edges = malloc(sizeof(EdgePair) * para->m);
    for (int i = 0; i < para->m; i++) {
        para->edges[i].first = *(int *) PyArray_GETPTR2(edges_, i, 0);
        para->edges[i].second = *(int *) PyArray_GETPTR2(edges_, i, 1);
    }
    OnlineStat *stat = make_online_stat(para->p, para->num_tr);
    if (para->loss_func == 0) {
        algo_online_graph_iht_logit(para, stat);
    } else if (para->loss_func == 1) {
        algo_online_graph_iht_least_square(para, stat);
    }
    PyObject *results = online_get_result(para->p, para->num_tr, stat);
    free(para->edges), free(para), free_online_stat(stat);
    return results;
}


static PyObject *online_graph_da_iht(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error for no reason.\n");
        return NULL;
    }
    graph_da_iht_para *para = malloc(sizeof(graph_da_iht_para));
    PyArrayObject *x_tr_, *y_tr_, *w0_, *edges_, *weights_;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!diiddddidddidii",
                          &PyArray_Type, &x_tr_,
                          &PyArray_Type, &y_tr_,
                          &PyArray_Type, &w0_,
                          &PyArray_Type, &edges_,
                          &PyArray_Type, &weights_,
                          &para->gamma,
                          &para->g,
                          &para->sparsity,
                          &para->l2_lambda,
                          &para->head_budget,
                          &para->head_err_tol,
                          &para->head_delta,
                          &para->head_max_iter,
                          &para->tail_budget,
                          &para->tail_err_tol,
                          &para->tail_nu,
                          &para->tail_max_iter,
                          &para->pcst_epsilon,
                          &para->loss_func,
                          &para->verbose)) { return NULL; }
    para->num_tr = (int) x_tr_->dimensions[0];      // # of samples
    para->p = (int) x_tr_->dimensions[1];   // # of features/nodes
    para->m = (int) edges_->dimensions[0];  // # of edges
    para->x_tr = (double *) PyArray_DATA(x_tr_);
    para->y_tr = (double *) PyArray_DATA(y_tr_);
    para->w0 = (double *) PyArray_DATA(w0_);
    para->weights = (double *) PyArray_DATA(weights_);
    para->edges = malloc(sizeof(EdgePair) * para->m);
    for (int i = 0; i < para->m; i++) {
        para->edges[i].first = *(int *) PyArray_GETPTR2(edges_, i, 0);
        para->edges[i].second = *(int *) PyArray_GETPTR2(edges_, i, 1);
    }
    OnlineStat *stat = make_online_stat(para->p, para->num_tr);
    if (para->loss_func == 0) {
        algo_online_graph_da_iht_logit(para, stat);
    } else if (para->loss_func == 1) {
        algo_online_graph_da_iht_least_square(para, stat);
    } else if (para->loss_func == 2) {
        algo_online_graph_da_iht_least_square_without_head(para, stat);
    } else {
        algo_online_graph_da_iht_logit(para, stat);
    }
    PyObject *results = online_get_result(para->p, para->num_tr, stat);
    free(para->edges), free(para), free_online_stat(stat);
    return results;
}


static PyObject *online_graph_da_iht_2(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error for no reason.\n");
        return NULL;
    }
    graph_da_iht_para_2 *para = malloc(sizeof(graph_da_iht_para_2));
    PyArrayObject *x_tr_, *y_tr_, *w0_, *edges_, *weights_;
    // x_tr, y_tr, w0, edges, costs, gamma, num_clusters, root,
    //            l2_lambda, sparsity_low, sparsity_high, max_num_iter, 0, verbose
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!diidiiiii",
                          &PyArray_Type, &x_tr_,
                          &PyArray_Type, &y_tr_,
                          &PyArray_Type, &w0_,
                          &PyArray_Type, &edges_,
                          &PyArray_Type, &weights_,
                          &para->gamma,
                          &para->g,
                          &para->root,
                          &para->l2_lambda,
                          &para->sparsity_low,
                          &para->sparsity_high,
                          &para->max_num_iter,
                          &para->loss_func,
                          &para->verbose)) { return NULL; }
    para->num_tr = (int) x_tr_->dimensions[0];      // # of samples
    para->p = (int) x_tr_->dimensions[1];   // # of features/nodes
    para->m = (int) edges_->dimensions[0];  // # of edges
    para->x_tr = (double *) PyArray_DATA(x_tr_);
    para->y_tr = (double *) PyArray_DATA(y_tr_);
    para->w0 = (double *) PyArray_DATA(w0_);
    para->weights = (double *) PyArray_DATA(weights_);
    para->edges = malloc(sizeof(EdgePair) * para->m);
    for (int i = 0; i < para->m; i++) {
        para->edges[i].first = *(int *) PyArray_GETPTR2(edges_, i, 0);
        para->edges[i].second = *(int *) PyArray_GETPTR2(edges_, i, 1);
    }
    OnlineStat *stat = make_online_stat(para->p, para->num_tr);
    if (para->loss_func == 0) {
        algo_online_graph_da_iht_logit_2(para, stat);
    } else if (para->loss_func == 1) {
        algo_online_graph_da_iht_least_square_2(para, stat);
    }
    PyObject *results = online_get_result(para->p, para->num_tr, stat);
    free(para->edges), free(para), free_online_stat(stat);
    return results;
}


// x_tr, y_tr, w0,best_subset, gamma, l2_lambda_, s, 0, verbose
static PyObject *online_best_subset(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error for no reason.\n");
        return NULL;
    }
    best_subset *para = malloc(sizeof(best_subset));
    PyArrayObject *x_tr_, *y_tr_, *w0_, *best_subset_;
    if (!PyArg_ParseTuple(args, "O!O!O!O!ddiii",
                          &PyArray_Type, &x_tr_,
                          &PyArray_Type, &y_tr_,
                          &PyArray_Type, &w0_,
                          &PyArray_Type, &best_subset_,
                          &para->gamma,
                          &para->l2_lambda,
                          &para->sparsity,
                          &para->loss_func,
                          &para->verbose)) { return NULL; }
    para->num_tr = (int) x_tr_->dimensions[0];      // # of samples
    para->p = (int) x_tr_->dimensions[1];   // # of features/nodes
    para->x_tr = (double *) PyArray_DATA(x_tr_);
    para->y_tr = (double *) PyArray_DATA(y_tr_);
    para->w0 = (double *) PyArray_DATA(w0_);
    para->best_subset_len = (int) best_subset_->dimensions[0];
    para->best_subset = (int *) PyArray_DATA(best_subset_);
    OnlineStat *stat = make_online_stat(para->p, para->num_tr);
    if (para->loss_func == 0) {
        algo_online_best_subset_logit(para, stat);
    } else if (para->loss_func == 1) {
        algo_online_best_subset_logit(para, stat);
    } else {
        algo_online_best_subset_logit(para, stat);
    }
    PyObject *results = online_get_result(para->p, para->num_tr, stat);
    free(para), free_online_stat(stat);
    return results;
}


static PyObject *online_ghtp_logit(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error for no reason.\n");
        return NULL;
    }
    PyArrayObject *x_tr_, *y_tr_, *w0_;
    int n, p, i, sparsity, verbose;
    double lr, eta;
    if (!PyArg_ParseTuple(args, "O!O!O!ddii", &PyArray_Type, &x_tr_,
                          &PyArray_Type, &y_tr_, &PyArray_Type, &w0_,
                          &lr, &eta, &sparsity, &verbose)) { return NULL; }
    n = (int) (x_tr_->dimensions[0]);     // number of samples
    p = (int) (x_tr_->dimensions[1]);     // number of features
    double *x_tr = (double *) malloc(n * p * sizeof(double));
    double *y_tr = (double *) malloc(n * sizeof(double));
    double *w0 = (double *) malloc((p + 1) * sizeof(double));
    get_data(n, p, -1, x_tr, y_tr, w0, NULL, NULL,
             x_tr_, y_tr_, w0_, NULL, NULL);

    double *wt = malloc(sizeof(double) * (p + 1));
    double *wt_bar = malloc(sizeof(double) * (p + 1));
    int *nonzeros_wt = malloc(sizeof(int) * n);
    int *nonzeros_wt_bar = malloc(sizeof(int) * n);
    double *pred_prob_wt = malloc(sizeof(double) * n);
    double *pred_label_wt = malloc(sizeof(double) * n);
    double *pred_prob_wt_bar = malloc(sizeof(double) * n);
    double *pred_label_wt_bar = malloc(sizeof(double) * n);
    double *losses = malloc(sizeof(double) * n);
    double total_time = 0.0;
    int *missed_wt = malloc(sizeof(int) * n);
    int *missed_wt_bar = malloc(sizeof(int) * n);
    if (sparsity > p) {
        printf("the parameter of sparsity is too large!\n");
        printf("sparsity is: %d while p: %d!\n", sparsity, p);
        exit(0);
    }
    algo_online_ghtp_logit(x_tr, y_tr, w0, sparsity, p, n, lr, eta, verbose,
                           wt, wt_bar, nonzeros_wt, nonzeros_wt_bar,
                           pred_prob_wt, pred_label_wt, pred_prob_wt_bar,
                           pred_label_wt_bar, losses, missed_wt, missed_wt_bar,
                           &total_time);
    PyObject *results = PyTuple_New(12);
    PyObject *re_wt = PyList_New(p + 1), *re_wt_bar = PyList_New(p + 1);
    PyObject *re_nonzeros_wt = PyList_New(n);
    PyObject *re_nonzeros_wt_bar = PyList_New(n);
    for (i = 0; i < p + 1; i++) {
        PyList_SetItem(re_wt, i, PyFloat_FromDouble(wt[i]));
        PyList_SetItem(re_wt_bar, i, PyFloat_FromDouble(wt_bar[i]));
    }
    for (i = 0; i < n; i++) {
        PyList_SetItem(re_nonzeros_wt, i, PyInt_FromLong(nonzeros_wt[i]));
        PyList_SetItem(re_nonzeros_wt_bar, i,
                       PyInt_FromLong(nonzeros_wt_bar[i]));
    }
    PyObject *re_pred_prob_wt = PyList_New(n);
    PyObject *re_pred_label_wt = PyList_New(n);
    PyObject *re_pred_prob_wt_bar = PyList_New(n);
    PyObject *re_pred_label_wt_bar = PyList_New(n);
    for (i = 0; i < n; i++) {
        PyList_SetItem(re_pred_prob_wt, i,
                       PyFloat_FromDouble(pred_prob_wt[i]));
        PyList_SetItem(re_pred_label_wt, i,
                       PyFloat_FromDouble(pred_label_wt[i]));
        PyList_SetItem(re_pred_prob_wt_bar, i,
                       PyFloat_FromDouble(pred_prob_wt_bar[i]));
        PyList_SetItem(re_pred_label_wt_bar, i,
                       PyFloat_FromDouble(pred_label_wt_bar[i]));
    }
    PyObject *re_losses = PyList_New(n);
    for (i = 0; i < n; i++) {
        PyList_SetItem(re_losses, i, PyFloat_FromDouble(losses[i]));
    }
    PyObject *re_missed_wt = PyList_New(n);
    PyObject *re_missed_wt_bar = PyList_New(n);
    for (i = 0; i < n; i++) {
        PyList_SetItem(re_missed_wt, i, PyInt_FromLong(missed_wt[i]));
        PyList_SetItem(re_missed_wt_bar, i, PyInt_FromLong(missed_wt_bar[i]));
    }
    PyObject *re_total_time = PyFloat_FromDouble(total_time);
    PyTuple_SetItem(results, 0, re_wt);
    PyTuple_SetItem(results, 1, re_wt_bar);
    PyTuple_SetItem(results, 2, re_nonzeros_wt);
    PyTuple_SetItem(results, 3, re_nonzeros_wt_bar);
    PyTuple_SetItem(results, 4, re_pred_prob_wt);
    PyTuple_SetItem(results, 5, re_pred_label_wt);
    PyTuple_SetItem(results, 6, re_pred_prob_wt_bar);
    PyTuple_SetItem(results, 7, re_pred_label_wt_bar);
    PyTuple_SetItem(results, 8, re_losses);
    PyTuple_SetItem(results, 9, re_missed_wt);
    PyTuple_SetItem(results, 10, re_missed_wt_bar);
    PyTuple_SetItem(results, 11, re_total_time);
    // free all used memory
    free(pred_label_wt), free(pred_prob_wt);
    free(pred_label_wt_bar), free(pred_prob_wt_bar);
    free(w0), free(wt), free(wt_bar);
    free(nonzeros_wt), free(nonzeros_wt_bar);
    free(losses), free(x_tr), free(y_tr);
    free(missed_wt), free(missed_wt_bar);
    return results;
}

//input: x_tr, y_tr, w0, lr, l2_lambda, edges, costs, sparsity
static PyObject *online_graph_ghtp_logit(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error for no reason.\n");
        return NULL;
    }
    int i, n, p, m, sparsity, verbose;
    double lr, eta, *x_tr, *y_tr, *w0, *weights;
    EdgePair *edges;
    PyArrayObject *x_tr_, *y_tr_, *w0_, *edges_, *weights_;
    if (!PyArg_ParseTuple(args, "O!O!O!ddO!O!ii", &PyArray_Type, &x_tr_,
                          &PyArray_Type, &y_tr_, &PyArray_Type, &w0_,
                          &lr, &eta, &PyArray_Type, &edges_, &PyArray_Type,
                          &weights_, &sparsity, &verbose)) { return NULL; }
    n = (int) x_tr_->dimensions[0];         // number of samples
    p = (int) x_tr_->dimensions[1];         // number of features(nodes)
    m = (int) edges_->dimensions[0];        // number of edges
    x_tr = malloc(n * p * sizeof(double));
    y_tr = malloc(n * sizeof(double));
    w0 = malloc((p + 1) * sizeof(double));
    edges = malloc(sizeof(EdgePair) * m);
    weights = malloc(sizeof(double) * m);
    get_data(n, p, m, x_tr, y_tr, w0, edges, weights,
             x_tr_, y_tr_, w0_, edges_, weights_);
    int g = 1;
    int num_pcst = 0;
    int *nonzeros_wt = malloc(sizeof(int) * n);
    int *nonzeros_wt_bar = malloc(sizeof(int) * n);
    int *missed_wt = malloc(sizeof(int) * n);
    int *missed_wt_bar = malloc(sizeof(int) * n);

    double *wt = malloc(sizeof(double) * (p + 1));
    double *wt_bar = malloc(sizeof(double) * (p + 1));
    double *pred_prob_wt = malloc(sizeof(double) * n);
    double *pred_prob_wt_bar = malloc(sizeof(double) * n);
    double *pred_label_wt = malloc(sizeof(double) * n);
    double *pred_label_wt_bar = malloc(sizeof(double) * n);
    double *losses = malloc(sizeof(double) * n);
    double run_time_head = 0.0;
    double run_time_tail = 0.0;
    double total_time = 0.0;
    Array *f_nodes = malloc(sizeof(Array));
    Array *f_edges = malloc(sizeof(Array));
    f_nodes->array = malloc(sizeof(int) * p);
    f_edges->array = malloc(sizeof(int) * p);

    algo_online_graph_ghtp_logit(
            edges, weights, g, sparsity, p, m, x_tr, y_tr, w0, n, lr, eta,
            verbose, NULL);
    // to save results
    PyObject *results = PyTuple_New(17);
    PyObject *re_wt = PyList_New(p + 1);
    PyObject *re_wt_bar = PyList_New(p + 1);
    PyObject *re_nonzeros_wt = PyList_New(n);
    PyObject *re_nonzeros_wt_bar = PyList_New(n);
    for (i = 0; i < p + 1; i++) {
        PyList_SetItem(re_wt, i, PyFloat_FromDouble(wt[i]));
        PyList_SetItem(re_wt_bar, i, PyFloat_FromDouble(wt_bar[i]));
    }
    for (i = 0; i < n; i++) {
        PyList_SetItem(re_nonzeros_wt, i, PyInt_FromLong(nonzeros_wt[i]));
        PyList_SetItem(re_nonzeros_wt_bar, i,
                       PyInt_FromLong(nonzeros_wt_bar[i]));
    }
    PyObject *re_pred_prob_wt = PyList_New(n);
    PyObject *re_pred_label_wt = PyList_New(n);
    PyObject *re_pred_prob_wt_bar = PyList_New(n);
    PyObject *re_pred_label_wt_bar = PyList_New(n);
    for (i = 0; i < n; i++) {
        PyList_SetItem(re_pred_prob_wt, i,
                       PyFloat_FromDouble(pred_prob_wt[i]));
        PyList_SetItem(re_pred_label_wt_bar, i,
                       PyFloat_FromDouble(pred_label_wt_bar[i]));
        PyList_SetItem(re_pred_prob_wt_bar, i,
                       PyFloat_FromDouble(pred_prob_wt_bar[i]));
        PyList_SetItem(re_pred_label_wt, i,
                       PyFloat_FromDouble(pred_label_wt[i]));
    }
    PyObject *re_nodes = PyList_New(f_nodes->size);
    PyObject *re_edges = PyList_New(f_edges->size);
    for (i = 0; i < f_nodes->size; i++) {
        PyList_SetItem(re_nodes, i, PyFloat_FromDouble(f_nodes->array[i]));
    }
    for (i = 0; i < f_edges->size; i++) {
        PyList_SetItem(re_edges, i, PyFloat_FromDouble(f_edges->array[i]));
    }
    PyObject *re_num_pcst = PyInt_FromLong(num_pcst);
    PyObject *re_losses = PyList_New(n);
    for (i = 0; i < n; i++) {
        PyList_SetItem(re_losses, i, PyFloat_FromDouble(losses[i]));
    }
    PyObject *re_run_time_head = PyFloat_FromDouble(run_time_head);
    PyObject *re_run_time_tail = PyFloat_FromDouble(run_time_tail);
    PyObject *re_missed_wt = PyList_New(n);
    PyObject *re_missed_wt_bar = PyList_New(n);
    for (i = 0; i < n; i++) {
        PyList_SetItem(re_missed_wt, i, PyInt_FromLong(missed_wt[i]));
        PyList_SetItem(re_missed_wt_bar, i, PyInt_FromLong(missed_wt_bar[i]));
    }
    PyObject *re_total_time = PyFloat_FromDouble(total_time);
    PyTuple_SetItem(results, 0, re_wt);
    PyTuple_SetItem(results, 1, re_wt_bar);
    PyTuple_SetItem(results, 2, re_nonzeros_wt);
    PyTuple_SetItem(results, 3, re_nonzeros_wt_bar);
    PyTuple_SetItem(results, 4, re_pred_prob_wt);
    PyTuple_SetItem(results, 5, re_pred_label_wt);
    PyTuple_SetItem(results, 6, re_pred_prob_wt_bar);
    PyTuple_SetItem(results, 7, re_pred_label_wt_bar);
    PyTuple_SetItem(results, 8, re_nodes);
    PyTuple_SetItem(results, 9, re_edges);
    PyTuple_SetItem(results, 10, re_num_pcst);
    PyTuple_SetItem(results, 11, re_losses);
    PyTuple_SetItem(results, 12, re_run_time_head);
    PyTuple_SetItem(results, 13, re_run_time_tail);
    PyTuple_SetItem(results, 14, re_missed_wt);
    PyTuple_SetItem(results, 15, re_missed_wt_bar);
    PyTuple_SetItem(results, 16, re_total_time);

    //free all used memory
    free(x_tr), free(y_tr);
    free(w0), free(edges), free(weights);
    free(nonzeros_wt), free(nonzeros_wt_bar);
    free(missed_wt), free(missed_wt_bar);
    free(wt), free(wt_bar);
    free(pred_label_wt), free(pred_prob_wt);
    free(pred_label_wt_bar), free(pred_prob_wt_bar);
    free(f_nodes->array), free(f_edges->array);
    free(f_nodes), free(f_edges);
    free(losses);
    return results;
}


static PyObject *online_ghtp_logit_sparse(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error for no reason.\n");
        return NULL;
    }
    PyArrayObject *x_tr_, *y_tr_, *w0_;
    int n, p, i, sparsity, verbose;
    double lr, eta;
    if (!PyArg_ParseTuple(args, "O!O!O!ddii", &PyArray_Type, &x_tr_,
                          &PyArray_Type, &y_tr_, &PyArray_Type, &w0_,
                          &lr, &eta, &sparsity, &verbose)) { return NULL; }
    n = (int) (x_tr_->dimensions[0]);     // number of samples
    p = (int) (x_tr_->dimensions[1]);     // number of features
    double *x_tr = (double *) malloc(n * p * sizeof(double));
    double *y_tr = (double *) malloc(n * sizeof(double));
    double *w0 = (double *) malloc((p + 1) * sizeof(double));

    get_data(n, p, -1, x_tr, y_tr, w0, NULL, NULL,
             x_tr_, y_tr_, w0_, NULL, NULL);


    double *wt = malloc(sizeof(double) * (p + 1));
    double *wt_bar = malloc(sizeof(double) * (p + 1));
    int *nonzeros_wt = malloc(sizeof(int) * n);
    int *nonzeros_wt_bar = malloc(sizeof(int) * n);
    double *pred_prob_wt = malloc(sizeof(double) * n);
    double *pred_label_wt = malloc(sizeof(double) * n);
    double *pred_prob_wt_bar = malloc(sizeof(double) * n);
    double *pred_label_wt_bar = malloc(sizeof(double) * n);
    double *losses = malloc(sizeof(double) * n);
    double total_time = 0.0;
    int *missed_wt = malloc(sizeof(int) * n);
    int *missed_wt_bar = malloc(sizeof(int) * n);
    if (sparsity > p) {
        printf("the parameter of sparsity is too large!\n");
        printf("sparsity is: %d while p: %d!\n", sparsity, p);
        exit(0);
    }
    algo_online_ghtp_logit_sparse(
            x_tr, y_tr, w0, sparsity, p, n, lr, eta, verbose, wt, wt_bar,
            nonzeros_wt, nonzeros_wt_bar, pred_prob_wt, pred_label_wt,
            pred_prob_wt_bar, pred_label_wt_bar, losses, missed_wt,
            missed_wt_bar, &total_time);
    PyObject *results = PyTuple_New(12);
    PyObject *re_wt = PyList_New(p + 1), *re_wt_bar = PyList_New(p + 1);
    PyObject *re_nonzeros_wt = PyList_New(n);
    PyObject *re_nonzeros_wt_bar = PyList_New(n);
    for (i = 0; i < p + 1; i++) {
        PyList_SetItem(re_wt, i, PyFloat_FromDouble(wt[i]));
        PyList_SetItem(re_wt_bar, i, PyFloat_FromDouble(wt_bar[i]));
    }
    for (i = 0; i < n; i++) {
        PyList_SetItem(re_nonzeros_wt, i, PyInt_FromLong(nonzeros_wt[i]));
        PyList_SetItem(re_nonzeros_wt_bar, i,
                       PyInt_FromLong(nonzeros_wt_bar[i]));
    }
    PyObject *re_pred_prob_wt = PyList_New(n);
    PyObject *re_pred_label_wt = PyList_New(n);
    PyObject *re_pred_prob_wt_bar = PyList_New(n);
    PyObject *re_pred_label_wt_bar = PyList_New(n);
    for (i = 0; i < n; i++) {
        PyList_SetItem(re_pred_prob_wt, i,
                       PyFloat_FromDouble(pred_prob_wt[i]));
        PyList_SetItem(re_pred_label_wt, i,
                       PyFloat_FromDouble(pred_label_wt[i]));
        PyList_SetItem(re_pred_prob_wt_bar, i,
                       PyFloat_FromDouble(pred_prob_wt_bar[i]));
        PyList_SetItem(re_pred_label_wt_bar, i,
                       PyFloat_FromDouble(pred_label_wt_bar[i]));
    }
    PyObject *re_losses = PyList_New(n);
    for (i = 0; i < n; i++) {
        PyList_SetItem(re_losses, i, PyFloat_FromDouble(losses[i]));
    }
    PyObject *re_missed_wt = PyList_New(n);
    PyObject *re_missed_wt_bar = PyList_New(n);
    for (i = 0; i < n; i++) {
        PyList_SetItem(re_missed_wt, i, PyInt_FromLong(missed_wt[i]));
        PyList_SetItem(re_missed_wt_bar, i, PyInt_FromLong(missed_wt_bar[i]));
    }
    PyObject *re_total_time = PyFloat_FromDouble(total_time);
    PyTuple_SetItem(results, 0, re_wt);
    PyTuple_SetItem(results, 1, re_wt_bar);
    PyTuple_SetItem(results, 2, re_nonzeros_wt);
    PyTuple_SetItem(results, 3, re_nonzeros_wt_bar);
    PyTuple_SetItem(results, 4, re_pred_prob_wt);
    PyTuple_SetItem(results, 5, re_pred_label_wt);
    PyTuple_SetItem(results, 6, re_pred_prob_wt_bar);
    PyTuple_SetItem(results, 7, re_pred_label_wt_bar);
    PyTuple_SetItem(results, 8, re_losses);
    PyTuple_SetItem(results, 9, re_missed_wt);
    PyTuple_SetItem(results, 10, re_missed_wt_bar);
    PyTuple_SetItem(results, 11, re_total_time);
    // free all used memory
    free(pred_label_wt), free(pred_prob_wt);
    free(pred_label_wt_bar), free(pred_prob_wt_bar);
    free(w0), free(wt), free(wt_bar);
    free(nonzeros_wt), free(nonzeros_wt_bar);
    free(losses), free(x_tr), free(y_tr);
    free(missed_wt), free(missed_wt_bar);
    return results;
}


static PyObject *batch_iht_logit(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error for no reason.\n");
        return NULL;
    }
    PyArrayObject *x_tr_, *y_tr_, *w0_;
    int n, p, sparsity, verbose, max_iter;
    double lr, eta, tol;
    if (!PyArg_ParseTuple(
            args, "O!O!O!dididi", &PyArray_Type, &x_tr_, &PyArray_Type, &y_tr_,
            &PyArray_Type, &w0_, &lr, &sparsity, &tol, &max_iter, &eta,
            &verbose)) { return NULL; }
    n = (int) (x_tr_->dimensions[0]);     // number of samples
    p = (int) (x_tr_->dimensions[1]);     // number of features
    double *x_tr = malloc(sizeof(double) * (n * p));
    double *y_tr = malloc(sizeof(double) * n);
    double *w0 = malloc(sizeof(double) * (p + 1));
    get_data(n, p, -1, x_tr, y_tr, w0, NULL, NULL,
             x_tr_, y_tr_, w0_, NULL, NULL);
    double *wt = malloc(sizeof(double) * (p + 1));
    double *losses = malloc(sizeof(double) * max_iter);
    double total_time = 0.0;
    if (sparsity > p) {
        printf("the parameter of sparsity is too large!\n");
        printf("sparsity is: %d while p: %d!\n", sparsity, p);
        exit(0);
    }
    algo_batch_iht_logit(x_tr, y_tr, w0, sparsity, p, n, lr, eta, max_iter,
                         tol, verbose, wt, losses, &total_time);
    PyObject *results = batch_get_result(p, max_iter, total_time, wt, losses);
    free(losses), free(x_tr), free(y_tr), free(w0), free(wt);
    return results;
}

static PyObject *batch_ghtp_logit(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error for no reason.\n");
        return NULL;
    }
    PyArrayObject *x_tr_, *y_tr_, *w0_;
    int n, p, sparsity, verbose, max_iter;
    double lr, eta, tol;
    if (!PyArg_ParseTuple(
            args, "O!O!O!dididi", &PyArray_Type, &x_tr_, &PyArray_Type, &y_tr_,
            &PyArray_Type, &w0_, &lr, &sparsity, &tol, &max_iter, &eta,
            &verbose)) { return NULL; }
    n = (int) (x_tr_->dimensions[0]);     // number of samples
    p = (int) (x_tr_->dimensions[1]);     // number of features
    double *x_tr = malloc(n * p * sizeof(double));
    double *y_tr = malloc(n * sizeof(double));
    double *w0 = malloc((p + 1) * sizeof(double));
    get_data(n, p, -1, x_tr, y_tr, w0, NULL, NULL,
             x_tr_, y_tr_, w0_, NULL, NULL);

    double *wt = malloc(sizeof(double) * (p + 1));
    double *losses = malloc(sizeof(double) * max_iter);
    double total_time = 0.0;
    if (sparsity > p) {
        printf("the parameter of sparsity is too large!\n");
        printf("sparsity is: %d while p: %d!\n", sparsity, p);
        exit(0);
    }
    algo_batch_ghtp_logit(x_tr, y_tr, w0, sparsity, p, n, lr, eta, max_iter,
                          tol, verbose, wt, losses, &total_time);
    PyObject *results = batch_get_result(p, max_iter, total_time, wt, losses);
    free(losses), free(x_tr), free(y_tr), free(w0), free(wt);
    return results;
}


static PyObject *batch_graph_iht_logit(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error for no reason.\n");
        return NULL;
    }
    EdgePair *edges;
    int g = 1, n, p, m, sparsity, verbose, max_iter;
    double lr, eta, tol, *weights, *x_tr, *y_tr, *w0, *wt, *losses;
    PyArrayObject *x_tr_, *y_tr_, *w0_, *edges_, *weights_;
    if (!PyArg_ParseTuple(
            args, "O!O!O!dididO!O!i", &PyArray_Type, &x_tr_, &PyArray_Type,
            &y_tr_, &PyArray_Type, &w0_, &lr, &sparsity, &tol, &max_iter, &eta,
            &PyArray_Type, &edges_, &PyArray_Type, &weights_,
            &verbose)) { return NULL; }
    n = (int) (x_tr_->dimensions[0]);       // number of samples
    p = (int) (x_tr_->dimensions[1]);       // number of features
    m = (int) edges_->dimensions[0];        // number of edges
    x_tr = malloc(sizeof(double) * (n * p));
    y_tr = malloc(sizeof(double) * n);
    w0 = malloc(sizeof(double) * (p + 1));
    edges = malloc(sizeof(EdgePair) * m);
    weights = malloc(sizeof(double) * m);
    get_data(n, p, m, x_tr, y_tr, w0, edges, weights,
             x_tr_, y_tr_, w0_, edges_, weights_);
    wt = malloc(sizeof(double) * (p + 1));
    losses = malloc(sizeof(double) * max_iter);
    double run_time_head = 0.0, run_time_tail = 0.0, total_time = 0.0;
    if (sparsity > p) {
        printf("the parameter of sparsity is too large!\n");
        printf("sparsity is: %d while p: %d!\n", sparsity, p);
        exit(0);
    }
    Array *f_nodes = malloc(sizeof(Array));
    Array *f_edges = malloc(sizeof(Array));
    f_nodes->array = malloc(sizeof(int) * p);
    f_edges->array = malloc(sizeof(int) * p);
    algo_batch_graph_iht_logit(
            edges, weights, g, sparsity, p, m, x_tr, y_tr, w0, n, tol,
            max_iter, lr, eta, verbose, wt, f_nodes, losses, &run_time_head,
            &run_time_tail, &total_time);
    PyObject *results = batch_get_result(p, max_iter, total_time, wt, losses);
    free(x_tr), free(y_tr), free(w0), free(edges), free(weights);
    free(wt), free(losses), free(f_nodes->array), free(f_edges->array);
    free(f_nodes), free(f_edges);
    return results;
}


static PyObject *batch_graph_ghtp_logit(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error for no reason.\n");
        return NULL;
    }
    EdgePair *edges;
    int g = 1, n, p, m, sparsity, verbose, max_iter;
    double lr, eta, tol, *weights, *x_tr, *y_tr, *w0, *wt, *losses;
    PyArrayObject *edges_, *weights_, *x_tr_, *y_tr_, *w0_;
    if (!PyArg_ParseTuple(args, "O!O!O!dididO!O!i",
                          &PyArray_Type, &x_tr_, &PyArray_Type, &y_tr_,
                          &PyArray_Type, &w0_, &lr, &sparsity, &tol, &max_iter,
                          &eta, &PyArray_Type, &edges_, &PyArray_Type,
                          &weights_, &verbose)) { return NULL; }
    n = (int) (x_tr_->dimensions[0]);     // number of samples
    p = (int) (x_tr_->dimensions[1]);     // number of features
    m = (int) edges_->dimensions[0];        // number of edges
    x_tr = malloc(n * p * sizeof(double));
    y_tr = malloc(n * sizeof(double));
    w0 = malloc((p + 1) * sizeof(double));
    edges = malloc(sizeof(EdgePair) * m);
    weights = malloc(sizeof(double) * m);
    get_data(n, p, m, x_tr, y_tr, w0, edges, weights,
             x_tr_, y_tr_, w0_, edges_, weights_);
    wt = malloc(sizeof(double) * (p + 1));
    losses = malloc(sizeof(double) * max_iter);
    double run_time_head = 0.0, run_time_tail = 0.0, total_time = 0.0;
    if (sparsity > p) {
        printf("the parameter of sparsity is too large!\n");
        printf("sparsity is: %d while p: %d!\n", sparsity, p);
        exit(0);
    }
    Array *f_nodes = malloc(sizeof(Array));
    Array *f_edges = malloc(sizeof(Array));
    f_nodes->array = malloc(sizeof(int) * p);
    f_edges->array = malloc(sizeof(int) * p);
    algo_batch_graph_ghtp_logit(
            edges, weights, g, sparsity, p, m, x_tr, y_tr, w0, n, tol,
            max_iter, lr, eta, verbose, wt, f_nodes, losses, &run_time_head,
            &run_time_tail, &total_time);
    PyObject *results = batch_get_result(p, max_iter, total_time, wt, losses);
    free(w0), free(wt), free(edges), free(weights), free(x_tr), free(y_tr);
    free(f_nodes->array), free(f_edges->array);
    free(f_nodes), free(f_edges), free(losses);
    return results;
}


static PyObject *batch_graph_posi_logit(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error for no reason.\n");
        return NULL;
    }
    PyArrayObject *x_tr_, *y_tr_, *w0_;
    int n, p, m, sparsity, verbose, max_iter;
    double lr, eta, tol, *weights;
    EdgePair *edges;
    PyArrayObject *edges_, *weights_;
    if (!PyArg_ParseTuple(args, "O!O!O!dididO!O!i",
                          &PyArray_Type, &x_tr_, &PyArray_Type, &y_tr_,
                          &PyArray_Type, &w0_, &lr, &sparsity, &tol, &max_iter,
                          &eta, &PyArray_Type, &edges_, &PyArray_Type,
                          &weights_, &verbose)) { return NULL; }
    int g = 1;
    n = (int) (x_tr_->dimensions[0]);     // number of samples
    p = (int) (x_tr_->dimensions[1]);     // number of features
    m = (int) edges_->dimensions[0];        // number of edges
    double *x_tr = malloc(n * p * sizeof(double));
    double *y_tr = malloc(n * sizeof(double));
    double *w0 = malloc((p + 1) * sizeof(double));
    edges = malloc(sizeof(EdgePair) * m);
    weights = malloc(sizeof(double) * m);
    get_data(n, p, m, x_tr, y_tr, w0, edges, weights,
             x_tr_, y_tr_, w0_, edges_, weights_);
    double *wt = malloc(sizeof(double) * (p + 1));
    double *losses = malloc(sizeof(double) * max_iter);
    double run_time_head;
    double run_time_tail;
    double total_time = 0.0;
    if (sparsity > p) {
        printf("the parameter of sparsity is too large!\n");
        printf("sparsity is: %d while p: %d!\n", sparsity, p);
        exit(0);
    }
    Array *f_nodes = malloc(sizeof(Array));
    Array *f_edges = malloc(sizeof(Array));
    f_nodes->array = malloc(sizeof(int) * p);
    f_edges->array = malloc(sizeof(int) * p);
    algo_batch_graph_posi_logit(
            edges, weights, g, sparsity, p, m, x_tr, y_tr, w0, n, tol,
            max_iter, lr, eta, verbose, wt, f_nodes, losses, &run_time_head,
            &run_time_tail, &total_time);
    PyObject *results = batch_get_result(p, max_iter, total_time, wt, losses);
    free(w0), free(wt), free(edges), free(weights);
    free(f_nodes->array), free(f_edges->array);
    free(f_nodes), free(f_edges), free(losses), free(x_tr), free(y_tr);
    return results;
}

static PyObject *test(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error for no reason.\n");
        return NULL;
    }
    double sum = 0.0;
    PyArrayObject *x_tr_;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &x_tr_)) { return NULL; }
    int n = (int) (x_tr_->dimensions[0]);     // number of samples
    int p = (int) (x_tr_->dimensions[1]);     // number of features
    printf("%d %d\n", n, p);
    double *x_tr = PyArray_DATA(x_tr_);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            printf("%.2f ", x_tr[i * p + j]);
            sum += x_tr[i * p + j];
        }
        printf("\n");
    }
    PyObject *results = PyFloat_FromDouble(sum);
    return results;
}


static PyMethodDef sparse_methods[] = {
        {"test",                     (PyCFunction) test,
                METH_VARARGS, "test docs"},
        {"online_sgd_l1_logit",      (PyCFunction) online_sgd_l1_logit,
                METH_VARARGS, "online_sgd_l1_logit docs"},
        {"online_sgd_l2_logit",      (PyCFunction) online_sgd_l2_logit,
                METH_VARARGS, "online_rda_l1 docs"},
        {"online_rda_l1",            (PyCFunction) online_rda_l1,
                METH_VARARGS, "online_rda_l1 docs"},
        {"online_da_gl",             (PyCFunction) online_da_gl,
                METH_VARARGS, "online_da_gl docs"},
        {"wrap_head_tail_binsearch", (PyCFunction) wrap_head_tail_binsearch,
                METH_VARARGS, "online_da_gl docs"},
        {"online_da_sgl",            (PyCFunction) online_da_sgl,
                METH_VARARGS, "online_da_sgl docs"},
        {"online_ada_grad",          (PyCFunction) online_ada_grad,
                METH_VARARGS, "online_ada_grad docs"},
        {"online_sto_iht",           (PyCFunction) online_sto_iht,
                METH_VARARGS, "online_sto_iht docs"},
        {"online_da_iht",            (PyCFunction) online_da_iht,
                METH_VARARGS, "online_da_iht docs"},
        {"online_adam",              (PyCFunction) online_adam,
                METH_VARARGS, "online_adam docs"},
        {"online_graph_iht",         (PyCFunction) online_graph_iht,
                METH_VARARGS, "online_graph_iht docs"},
        {"online_graph_da_iht",      (PyCFunction) online_graph_da_iht,
                METH_VARARGS, "online_graph_da_iht_logit docs"},
        {"online_graph_da_iht_2",    (PyCFunction) online_graph_da_iht_2,
                METH_VARARGS, "online_graph_da_iht_logit_2 docs"},
        {"online_best_subset",       (PyCFunction) online_best_subset,
                METH_VARARGS, "online_best_subset docs"},
        {"online_ghtp_logit",        (PyCFunction) online_ghtp_logit,
                METH_VARARGS, "online_ghtp_iht docs"},
        {"online_graph_ghtp_logit",  (PyCFunction) online_graph_ghtp_logit,
                METH_VARARGS, "online_graph_ghtp_logit docs"},
        {"online_ghtp_logit_sparse", (PyCFunction) online_ghtp_logit_sparse,
                METH_VARARGS, "online_iht_logit_sparse docs"},
        {"batch_iht_logit",          (PyCFunction) batch_iht_logit,
                METH_VARARGS, "batch_iht_logit docs"},
        {"batch_ghtp_logit",         (PyCFunction) batch_ghtp_logit,
                METH_VARARGS, "batch_ghtp_logit docs"},
        {"batch_graph_iht_logit",    (PyCFunction) batch_graph_iht_logit,
                METH_VARARGS, "batch_graph_iht_logit docs"},
        {"batch_graph_ghtp_logit",   (PyCFunction) batch_graph_ghtp_logit,
                METH_VARARGS, "batch_graph_ghtp_logit docs"},
        {"batch_graph_posi_logit",   (PyCFunction) batch_graph_posi_logit,
                METH_VARARGS, "batch_graph_posi_logit docs"},
        {NULL, NULL, 0, NULL}};

/** Python version 2 for module initialization */
PyMODINIT_FUNC initsparse_module() {
    Py_InitModule3("sparse_module", sparse_methods,
                   "some docs for sparse learning algorithms.");
    import_array();
}

int main() {
    printf("test of main wrapper!\n");
}