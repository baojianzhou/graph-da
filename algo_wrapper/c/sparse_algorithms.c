//
//
#include <cblas.h>
#include "sort.h"
#include "loss.h"
#include "sparse_algorithms.h"

#define sign(x) ((x > 0) -(x < 0))

OnlineStat *make_online_stat(int p, int num_tr) {
    OnlineStat *stat = malloc(sizeof(OnlineStat));
    stat->wt = malloc(sizeof(double) * (p + 1));
    stat->wt_bar = malloc(sizeof(double) * (p + 1));
    stat->nonzeros_wt = malloc(sizeof(int) * num_tr);
    stat->nonzeros_wt_bar = malloc(sizeof(int) * num_tr);
    stat->p_prob_wt = malloc(sizeof(double) * num_tr);
    stat->p_label_wt = malloc(sizeof(double) * num_tr);
    stat->p_prob_wt_bar = malloc(sizeof(double) * num_tr);
    stat->p_label_wt_bar = malloc(sizeof(double) * num_tr);
    stat->losses = malloc(sizeof(double) * num_tr);
    stat->missed_wt = malloc(sizeof(int) * num_tr);
    stat->missed_wt_bar = malloc(sizeof(int) * num_tr);
    for (int i = 0; i < num_tr; i++) {
        stat->nonzeros_wt[i] = 0;
        stat->nonzeros_wt_bar[i] = 0;
        stat->missed_wt[i] = 0;
        stat->missed_wt_bar[i] = 0;
    }
    stat->total_missed_wt = 0;
    stat->total_missed_wt_bar = 0;
    stat->total_time = 0.0;
    stat->num_pcst = 0;
    stat->run_time_head = 0.0;
    stat->run_time_tail = 0.0;
    stat->re_nodes = malloc(sizeof(Array));
    stat->re_nodes->size = 0;
    stat->re_nodes->array = malloc(sizeof(int) * p);
    stat->re_edges = malloc(sizeof(Array));
    stat->re_edges->size = 0;
    stat->re_edges->array = malloc(sizeof(int) * p);
    return stat;
}

bool free_online_stat(OnlineStat *online_stat) {
    free(online_stat->missed_wt_bar);
    free(online_stat->missed_wt);
    free(online_stat->losses);
    free(online_stat->p_prob_wt_bar);
    free(online_stat->p_label_wt_bar);
    free(online_stat->p_label_wt);
    free(online_stat->p_prob_wt);
    free(online_stat->nonzeros_wt);
    free(online_stat->nonzeros_wt_bar);
    free(online_stat->wt);
    free(online_stat->wt_bar);
    free(online_stat->re_nodes->array);
    free(online_stat->re_nodes);
    free(online_stat->re_edges->array);
    free(online_stat->re_edges);
    free(online_stat);
    return true;
}


StochasticStat *make_stochastic_stat(int p, int num_tr) {
    StochasticStat *stat = malloc(sizeof(StochasticStat));
    stat->wt = malloc(sizeof(double) * (p + 1));
    stat->wt_bar = malloc(sizeof(double) * (p + 1));
    stat->nonzeros_wt = malloc(sizeof(int) * num_tr);
    stat->nonzeros_wt_bar = malloc(sizeof(int) * num_tr);
    stat->losses = malloc(sizeof(double) * num_tr);
    for (int i = 0; i < num_tr; i++) {
        stat->nonzeros_wt[i] = 0;
        stat->nonzeros_wt_bar[i] = 0;
    }
    stat->total_time = 0.0;
    stat->num_pcst = 0;
    stat->run_time_head = 0.0;
    stat->run_time_tail = 0.0;
    stat->re_nodes = malloc(sizeof(Array));
    stat->re_nodes->size = 0;
    stat->re_nodes->array = malloc(sizeof(int) * p);
    stat->re_edges = malloc(sizeof(Array));
    stat->re_edges->size = 0;
    stat->re_edges->array = malloc(sizeof(int) * p);
    return stat;
}

bool free_stochastic_stat(StochasticStat *online_stat) {
    free(online_stat->losses);
    free(online_stat->nonzeros_wt);
    free(online_stat->nonzeros_wt_bar);
    free(online_stat->wt);
    free(online_stat->wt_bar);
    free(online_stat->re_nodes->array);
    free(online_stat->re_nodes);
    free(online_stat->re_edges->array);
    free(online_stat->re_edges);
    free(online_stat);
    return true;
}

double l2_norm(const double *x, int x_len) {
    double result = 0.0;
    for (int i = 0; i < x_len; i++) {
        result += x[i] * x[i];
    }
    return sqrt(result);
}

void predict_logistic(OnlineStat *stat, int tt, int p, double y_i,
                      double *x_i, double *wt, double *wt_bar) {
    double *prob;
    double *label;
    prob = &stat->p_prob_wt[tt];
    label = &stat->p_label_wt[tt];
    logistic_predict(x_i, wt, prob, label, 0.5, 1, p);
    prob = &stat->p_prob_wt_bar[tt];
    label = &stat->p_label_wt_bar[tt];
    logistic_predict(x_i, wt_bar, prob, label, 0.5, 1, p);
    if (stat->p_label_wt[tt] != y_i) {
        stat->total_missed_wt++;
    }
    stat->missed_wt[tt] = stat->total_missed_wt;
    if (stat->p_label_wt_bar[tt] != y_i) {
        stat->total_missed_wt_bar++;
    }
    stat->missed_wt_bar[tt] = stat->total_missed_wt_bar;
}


void predict_least_square(OnlineStat *stat, int tt, int p, double y_i,
                          double *x_i, double *wt, double *wt_bar) {
    double *prob;
    double *label;
    prob = &stat->p_prob_wt[tt];
    label = &stat->p_label_wt[tt];
    least_square_predict(x_i, wt, prob, label, 0.5, 1, p);
    prob = &stat->p_prob_wt_bar[tt];
    label = &stat->p_label_wt_bar[tt];
    least_square_predict(x_i, wt_bar, prob, label, 0.5, 1, p);
    if (stat->p_label_wt[tt] != y_i) {
        stat->total_missed_wt++;
    }
    stat->missed_wt[tt] = stat->total_missed_wt;
    if (stat->p_label_wt_bar[tt] != y_i) {
        stat->total_missed_wt_bar++;
    }
    stat->missed_wt_bar[tt] = stat->total_missed_wt_bar;
}


void min_f_posi(const Array *proj_nodes, const double *x_tr,
                const double *y_tr, int max_iter, double eta, double *wt,
                int n, int p) {
    openblas_set_num_threads(1);
    int i;
    double *loss_grad = (double *) malloc((p + 2) * sizeof(double));
    double *tmp_loss_grad = (double *) malloc((p + 2) * sizeof(double));
    double *wt_tmp = (double *) malloc((p + 1) * sizeof(double));
    /**
     * make sure the start point is a feasible point. here we do a trick:
     * we treat wt as an initial point itself. and of course wt is always a
     * feasible point. A Frank-Wolfe style minimization with
     * backtracking line search. Other algorithms can be considered:
     * Newton's method, trust region, etc.
     */
    double beta, lr, grad_sq;
    for (i = 0; i < max_iter; i++) {
        logistic_loss_grad(wt, x_tr, y_tr, loss_grad, eta, n, p);
        beta = 0.8, lr = 1.0;
        grad_sq = 0.5 * cblas_ddot(p + 1, loss_grad + 1, 1, loss_grad + 1, 1);
        while (true) { // using backtracking line search
            cblas_dcopy(p + 1, wt, 1, wt_tmp, 1);
            cblas_daxpy(p + 1, -lr, loss_grad + 1, 1, wt_tmp, 1);
            logistic_loss_grad(wt_tmp, x_tr, y_tr, tmp_loss_grad, eta, n, p);
            if (tmp_loss_grad[0] > (loss_grad[0] - lr * grad_sq)) {
                lr *= beta;
            } else {
                break;
            }
        }
        // projection step
        cblas_dcopy(p + 1, wt_tmp, 1, wt, 1);
        cblas_dscal(p + 1, 0.0, wt_tmp, 1);
        for (int k = 0; k < proj_nodes->size; k++) {
            wt_tmp[proj_nodes->array[k]] = wt[proj_nodes->array[k]];
            // positive constraint.
            if (wt_tmp[proj_nodes->array[k]] < 0.) {
                wt_tmp[proj_nodes->array[k]] = 0.0;
            }
        }
        wt_tmp[p] = wt[p];
        cblas_dcopy(p + 1, wt_tmp, 1, wt, 1);
    }
    free(wt_tmp);
    free(tmp_loss_grad);
    free(loss_grad);
}

void min_f(const Array *proj_nodes, const double *x_tr,
           const double *y_tr, int max_iter, double eta, double *wt,
           int n, int p) {
    openblas_set_num_threads(1);
    double *loss_grad = malloc((p + 2) * sizeof(double));
    double *tmp_loss_grad = malloc((p + 2) * sizeof(double));
    double *wt_tmp = malloc((p + 1) * sizeof(double));
    double beta, lr, grad_sq;
    /**
     * make sure the start point is a feasible point. here we do a trick:
     * we treat wt as an initial point itself. and of course wt is always a
     * feasible point. A Frank-Wolfe style minimization with
     * backtracking line search. Other algorithms can be considered:
     * Newton's method, trust region, etc.
     */
    for (int i = 0; i < max_iter; i++) {
        logistic_loss_grad(wt, x_tr, y_tr, loss_grad, eta, n, p);
        beta = 0.8, lr = 1.0;
        grad_sq = 0.5 * cblas_ddot(p + 1, loss_grad + 1, 1, loss_grad + 1, 1);
        while (true) { // using backtracking line search
            cblas_dcopy(p + 1, wt, 1, wt_tmp, 1);
            cblas_daxpy(p + 1, -lr, loss_grad + 1, 1, wt_tmp, 1);
            logistic_loss_grad(wt_tmp, x_tr, y_tr, tmp_loss_grad, eta, n, p);
            if (tmp_loss_grad[0] > (loss_grad[0] - lr * grad_sq)) {
                lr *= beta;
            } else {
                break;
            }
        }
        // projection step
        cblas_dcopy(p + 1, wt_tmp, 1, wt, 1);
        cblas_dscal(p + 1, 0.0, wt_tmp, 1);
        for (int k = 0; k < proj_nodes->size; k++) {
            wt_tmp[proj_nodes->array[k]] = wt[proj_nodes->array[k]];
        }
        wt_tmp[p] = wt[p];
        cblas_dcopy(p + 1, wt_tmp, 1, wt, 1);
    }
    free(wt_tmp);
    free(tmp_loss_grad);
    free(loss_grad);
}

void min_f_sparse(
        const Array *proj_nodes, const double *x_tr,
        const double *y_tr, int max_iter, double eta, double *wt, int n,
        int p) {
    openblas_set_num_threads(1);
    double *loss_grad = malloc((p + 2) * sizeof(double));
    double *tmp_loss_grad = malloc((p + 2) * sizeof(double));
    double *wt_tmp = malloc((p + 1) * sizeof(double));
    double beta, lr, grad_sq;
    /**
     * make sure the start point is a feasible point. here we do a trick:
     * we treat wt as an initial point itself. and of course wt is always a
     * feasible point. A Frank-Wolfe style minimization with
     * backtracking line search. Other algorithms can be considered:
     * Newton's method, trust region, etc.
     */
    for (int i = 0; i < max_iter; i++) {
        logistic_loss_grad_sparse(
                wt, x_tr, y_tr, loss_grad, eta, n, p);
        beta = 0.8, lr = 1.0;
        grad_sq = 0.5 * cblas_ddot(p + 1, loss_grad + 1, 1, loss_grad + 1, 1);
        while (true) { // using backtracking line search
            cblas_dcopy(p + 1, wt, 1, wt_tmp, 1);
            cblas_daxpy(p + 1, -lr, loss_grad + 1, 1, wt_tmp, 1);
            logistic_loss_grad_sparse(
                    wt_tmp, x_tr, y_tr, tmp_loss_grad, eta, n, p);
            if (tmp_loss_grad[0] > (loss_grad[0] - lr * grad_sq)) {
                lr *= beta;
            } else {
                break;
            }
        }
        // projection step
        cblas_dcopy(p + 1, wt_tmp, 1, wt, 1);
        cblas_dscal(p + 1, 0.0, wt_tmp, 1);
        for (int k = 0; k < proj_nodes->size; k++) {
            wt_tmp[proj_nodes->array[k]] = wt[proj_nodes->array[k]];
        }
        wt_tmp[p] = wt[p];
        cblas_dcopy(p + 1, wt_tmp, 1, wt, 1);
    }
    free(wt_tmp);
    free(tmp_loss_grad);
    free(loss_grad);
}


bool algo_online_sgd_l1_logit(sgd_l1_para *para, OnlineStat *stat) {
    clock_t start_total = clock();
    openblas_set_num_threads(1);
    int p = para->p, num_tr = para->num_tr;
    double *w0 = para->w0, *wt, *wt_bar, *loss_grad;
    double *x_tr = para->x_tr, *y_tr = para->y_tr, *p_prob, *p_label;
    double gamma = para->gamma, lambda = para->lambda;

    wt = malloc((p + 1) * sizeof(double));
    wt_bar = malloc((p + 1) * sizeof(double));
    loss_grad = malloc((p + 2) * sizeof(double));

    cblas_dcopy(p + 1, w0, 1, wt, 1); // w0 --> wt
    cblas_dcopy(p + 1, w0, 1, wt_bar, 1); // w0 --> wt_bar

    double alpha_t = sqrt(2. / num_tr) / gamma;
    for (int tt = 0; tt < num_tr; tt++) {
        //1. receive a question x_t and predict a value.
        double *x_i = x_tr + tt * p, *y_i = y_tr + tt;
        p_prob = &stat->p_prob_wt[tt], p_label = &stat->p_label_wt[tt];
        logistic_predict(x_i, wt, p_prob, p_label, 0.5, 1, p);
        p_prob = &stat->p_prob_wt_bar[tt], p_label = &stat->p_label_wt_bar[tt];
        logistic_predict(x_i, wt_bar, p_prob, p_label, 0.5, 1, p);
        // 2. observe y_i = y_tr + tt and have some loss
        logistic_loss_grad(wt, x_i, y_i, loss_grad, 0.0, 1, p);
        if (stat->p_label_wt[tt] != y_tr[tt]) {
            stat->total_missed_wt++;
        }
        stat->missed_wt[tt] = stat->total_missed_wt;
        if (stat->p_label_wt_bar[tt] != y_tr[tt]) {
            stat->total_missed_wt_bar++;
        }
        stat->missed_wt_bar[tt] = stat->total_missed_wt_bar;
        stat->losses[tt] = loss_grad[0];
        //3. update the model:
        for (int i = 0; i < p + 1; i++) {
            wt[i] += -alpha_t * (loss_grad[i + 1] + lambda * sign(wt[i]));
        }
        //4. online to batch conversion.
        cblas_dscal(p + 1, tt / (tt + 1.), wt_bar, 1);
        cblas_daxpy(p + 1, 1 / (tt + 1.), wt, 1, wt_bar, 1);
        for (int i = 0; i < p; i++) {
            if (wt[i] != 0.0) {
                stat->nonzeros_wt[tt] += 1;
            }
            if (wt_bar[i] != 0.0) {
                stat->nonzeros_wt_bar[tt] += 1;
            }
        }
    }
    for (int i = 0; i < p + 1; i++) {
        stat->wt[i] = wt[i];
        stat->wt_bar[i] = wt_bar[i];
    }
    stat->total_time = (double) (clock() - start_total) / CLOCKS_PER_SEC;
    free(loss_grad), free(wt), free(wt_bar);
    return true;
}

bool algo_online_sgd_l2_logit(sgd_l2_para *para, OnlineStat *stat) {
    clock_t start_time = clock();
    openblas_set_num_threads(1);
    int p = para->p, num_tr = para->num_tr;
    double *wt, *wt_bar, *loss_grad, lr = para->lr, eta = para->eta;
    double *p_prob, *p_label, *x_tr = para->x_tr, *y_tr = para->y_tr;
    wt = malloc(sizeof(double) * (p + 1));
    wt_bar = malloc(sizeof(double) * (p + 1));
    cblas_dcopy(p + 1, para->w0, 1, wt, 1); // w0 --> wt
    cblas_dcopy(p + 1, para->w0, 1, wt_bar, 1); // w0 --> wt_bar
    loss_grad = malloc((p + 2) * sizeof(double));

    for (int tt = 0; tt < num_tr; tt++) {
        // 1.   unlabeled example x_i = x_tr + tt*p arrives and
        //      make prediction based on existing costs wt, wt_bar
        double *x_i = x_tr + tt * p, *y_i = y_tr + tt;
        p_prob = &stat->p_prob_wt[tt], p_label = &stat->p_label_wt[tt];
        logistic_predict(x_i, wt, p_prob, p_label, 0.5, 1, p);
        p_prob = &stat->p_prob_wt_bar[tt], p_label = &stat->p_label_wt_bar[tt];
        logistic_predict(x_i, wt_bar, p_prob, p_label, 0.5, 1, p);
        // 2. observe y_i = y_tr + tt and have some loss
        logistic_loss_grad(wt, x_i, y_i, loss_grad, eta, 1, p);
        if (stat->p_label_wt[tt] != y_tr[tt]) {
            stat->total_missed_wt++;
        }
        stat->missed_wt[tt] = stat->total_missed_wt;
        if (stat->p_label_wt_bar[tt] != y_tr[tt]) {
            stat->total_missed_wt_bar++;
        }
        stat->missed_wt_bar[tt] = stat->total_missed_wt_bar;
        stat->losses[tt] = loss_grad[0];
        cblas_daxpy(p + 1, -lr / sqrt(tt + 1.), loss_grad + 1, 1, wt, 1);
        cblas_dscal(p + 1, tt / (tt + 1.), wt_bar, 1);
        cblas_daxpy(p + 1, 1 / (tt + 1.), wt, 1, wt_bar, 1);
        for (int i = 0; i < p; i++) {
            if (wt[i] != 0.0) {
                stat->nonzeros_wt[tt] += 1;
            }
            if (wt_bar[i] != 0.0) {
                stat->nonzeros_wt_bar[tt] += 1;
            }
        }
    }
    for (int i = 0; i < p + 1; i++) {
        stat->wt[i] = wt[i];
        stat->wt_bar[i] = wt_bar[i];
    }
    stat->total_time = (double) (clock() - start_time) / CLOCKS_PER_SEC;
    free(loss_grad);
    return true;
}


bool algo_online_ghtp_logit(
        const double *x_tr, const double *y_tr, const double *w0, int s, int p,
        int num_tr, double lr, double eta, int verbose, double *wt,
        double *wt_bar, int *nonzeros_wt, int *nonzeros_wt_bar,
        double *pred_prob_wt, double *pred_label_wt, double *pred_prob_wt_bar,
        double *pred_label_wt_bar, double *losses, int *missed_wt,
        int *missed_wt_bar, double *total_time) {
    openblas_set_num_threads(1);
    int i;
    cblas_dcopy(p + 1, w0, 1, wt, 1); // w0 --> wt
    cblas_dcopy(p + 1, w0, 1, wt_bar, 1); // w0 --> wt_bar
    double *loss_grad = malloc((p + 2) * sizeof(double));
    double *wt_tmp = malloc((p + 1) * sizeof(double));
    int *sorted_ind = malloc(sizeof(double) * p);
    int total_missed_wt = 0, total_missed_wt_bar = 0, max_iter = 20;
    Array *re_nodes = malloc(sizeof(Array));
    re_nodes->array = malloc(sizeof(int) * p);
    clock_t start_total = clock();
    if (verbose > 0) {
        printf("learning rate: %lf, num_tr: %d ,p: %d\n", lr, num_tr, p);
    }
    for (i = 0; i < num_tr; i++) {
        nonzeros_wt[i] = 0;
        nonzeros_wt_bar[i] = 0;
    }
    for (int tt = 0; tt < num_tr; tt++) {
        // each time has one sample.
        logistic_predict(x_tr + tt * p, wt, &pred_prob_wt[tt],
                         &pred_label_wt[tt], 0.5, 1, p);
        logistic_predict(x_tr + tt * p, wt_bar, &pred_prob_wt_bar[tt],
                         &pred_label_wt_bar[tt], 0.5, 1, p);
        logistic_loss_grad(wt, x_tr + tt * p, y_tr + tt, loss_grad, eta, 1, p);
        if (pred_label_wt[tt] != y_tr[tt]) {
            total_missed_wt++;
        }
        if (pred_label_wt_bar[tt] != y_tr[tt]) {
            total_missed_wt_bar++;
        }
        missed_wt[tt] = total_missed_wt;
        missed_wt_bar[tt] = total_missed_wt_bar;
        losses[tt] = loss_grad[0];
        if (verbose > 0) {
            printf("losses[%d]:%.6f sparsity:%d ", tt, losses[tt], s);
        }
        cblas_dcopy(p + 1, wt, 1, wt_tmp, 1); // wt --> wt_tmp
        cblas_daxpy(p + 1, -lr / sqrt(tt + 1.), loss_grad + 1, 1, wt_tmp, 1);
        arg_magnitude_sort_descend(wt_tmp, sorted_ind, p);
        re_nodes->size = 0;
        for (i = 0; i < s; i++) {
            re_nodes->array[re_nodes->size++] = sorted_ind[i];
        }
        min_f(re_nodes, x_tr, y_tr, max_iter, eta, wt, tt + 1, p);
        if (verbose > 0) {
            printf("pred_prob:%.4f total_missed:%d/%d\n",
                   pred_prob_wt[tt], total_missed_wt, tt + 1);
        }
        cblas_dscal(p + 1, tt / (tt + 1.), wt_bar, 1);
        cblas_daxpy(p + 1, 1 / (tt + 1.), wt, 1, wt_bar, 1);
        for (i = 0; i < p; i++) {
            if (wt[i] != 0.0) {
                nonzeros_wt[tt] += 1;
            }
            if (wt_bar[i] != 0.0) {
                nonzeros_wt_bar[tt] += 1;
            }
        }
    }
    *total_time = (double) (clock() - start_total) / CLOCKS_PER_SEC;
    free(loss_grad), free(wt_tmp), free(sorted_ind);
    return true;
}


bool algo_online_graph_ghtp_logit(
        const EdgePair *edges, const double *costs, int g, int s, int p, int m,
        const double *x_tr, const double *y_tr, const double *w0, int num_tr,
        double lr, double eta, int verbose, OnlineStat *stat) {
    openblas_set_num_threads(1);
    int i, fmin_max_iter = 20, max_iter = 50;
    double *wt, *wt_bar, *loss_grad, *wt_tmp, *proj_prizes, *proj_costs;
    double *p_prob, *p_label;
    GraphStat *head_stat = make_graph_stat(p, m);
    GraphStat *tail_stat = make_graph_stat(p, m);
    wt = malloc(sizeof(double) * (p + 1));
    wt_bar = malloc(sizeof(double) * (p + 1));
    loss_grad = malloc(sizeof(double) * (p + 2));
    wt_tmp = malloc(sizeof(double) * (p + 1));
    proj_prizes = malloc(sizeof(double) * p);
    proj_costs = malloc(sizeof(double) * m);
    cblas_dcopy(p + 1, w0, 1, wt, 1);      // w0 --> wt
    cblas_dcopy(p + 1, w0, 1, wt_bar, 1);  // w0 --> wt_bar

    clock_t start_total = clock();
    double C = 2. * (s - 1.), delta = 1. / 169., nu = 2.5;
    int total_missed_wt = 0, total_missed_wt_bar = 0;
    enum PruningMethod pruning = GWPruning;
    for (i = 0; i < num_tr; i++) {
        stat->nonzeros_wt[i] = 0;
        stat->nonzeros_wt_bar[i] = 0;
    }
    for (i = 0; i < m; i++) {
        proj_costs[i] = costs[i] + (s - 1.) / (double) s;
    }
    Array *tmp_nodes = malloc(sizeof(Array));
    tmp_nodes->size = 0;
    tmp_nodes->array = malloc(sizeof(int) * (p + 1));
    for (int tt = 0; tt < num_tr; tt++) {
        p_prob = &stat->p_prob_wt[tt];
        p_label = &stat->p_label_wt[tt];
        logistic_predict(x_tr + tt * p, wt, p_prob, p_label, 0.5, 1, p);
        p_prob = &stat->p_prob_wt_bar[tt];
        p_label = &stat->p_label_wt_bar[tt];
        logistic_predict(x_tr + tt * p, wt_bar, p_prob, p_label, 0.5, 1, p);
        logistic_loss_grad(wt, x_tr + tt * p, y_tr + tt, loss_grad, eta, 1, p);
        if (stat->p_label_wt[tt] != y_tr[tt]) {
            total_missed_wt += 1;
        }
        if (stat->p_label_wt_bar[tt] != y_tr[tt]) {
            total_missed_wt_bar += 1;
        }
        stat->missed_wt[tt] = total_missed_wt;
        stat->missed_wt_bar[tt] = total_missed_wt_bar;
        stat->losses[tt] = loss_grad[0];
        for (i = 0; i < p; i++) {
            proj_prizes[i] = loss_grad[i] * loss_grad[i];
        }
        head_proj_exact(edges, proj_costs, proj_prizes, g, C, delta, max_iter,
                        1e-6, -1, pruning, 1e-6, p, m, verbose, head_stat);
        cblas_dscal(p + 1, 0.0, wt_tmp, 1);
        for (i = 0; i < head_stat->re_nodes->size; i++) {
            int cur_node = head_stat->re_nodes->array[i];
            wt_tmp[cur_node] = loss_grad[cur_node + 1];
        }
        cblas_dcopy(p + 1, wt_tmp, 1, loss_grad + 1, 1);
        cblas_dcopy(p + 1, wt, 1, wt_tmp, 1); // wt --> wt_tmp
        cblas_daxpy(p + 1, -lr / sqrt(tt + 1.), loss_grad + 1, 1, wt_tmp, 1);
        tmp_nodes->size = 0;
        for (i = 0; i < p; i++) {
            if (wt_tmp[i] != 0.0) {
                tmp_nodes->array[tmp_nodes->size++] = i;
            }
        }
        for (i = 0; i < head_stat->re_nodes->size; i++) {
            int cur_node = head_stat->re_nodes->array[i];
            wt[cur_node] = wt_tmp[cur_node];
        }
        stat->run_time_head += head_stat->run_time;
        min_f(tmp_nodes, x_tr, y_tr, fmin_max_iter, eta, wt, tt + 1, p);
        for (i = 0; i < p; i++) {
            proj_prizes[i] = wt[i] * wt[i];
        }
        tail_proj_exact(edges, proj_costs, proj_prizes, g, C, nu, max_iter,
                        1e-6, -1, pruning, 1e-6, p, m, verbose, tail_stat);
        stat->run_time_tail += tail_stat->run_time;
        cblas_dcopy(p + 1, wt, 1, wt_tmp, 1);
        cblas_dscal(p + 1, 0.0, wt, 1);
        for (i = 0; i < tail_stat->re_nodes->size; i++) {
            int cur_node = tail_stat->re_nodes->array[i];
            wt[cur_node] = wt_tmp[cur_node];
        }
        wt[p] = wt_tmp[p];
        cblas_dscal(p + 1, tt / (tt + 1.), wt_bar, 1);
        cblas_daxpy(p + 1, 1 / (tt + 1.), wt, 1, wt_bar, 1);
        for (i = 0; i < p; i++) {
            if (wt[i] != 0.0) {
                stat->nonzeros_wt[tt] += 1;
            }
            if (wt_bar[i] != 0.0) {
                stat->nonzeros_wt_bar[tt] += 1;
            }
        }
    }
    stat->total_time = (double) (clock() - start_total) / CLOCKS_PER_SEC;
    free(proj_costs), free(proj_prizes), free(loss_grad);
    free(wt_tmp), free(tmp_nodes->array), free(tmp_nodes);
    return true;
}


bool algo_online_ghtp_logit_sparse(
        const double *x_tr, const double *y_tr, const double *w0, int s, int p,
        int num_tr, double lr, double eta, int verbose, double *wt,
        double *wt_bar, int *nonzeros_wt, int *nonzeros_wt_bar,
        double *pred_prob_wt, double *pred_label_wt, double *pred_prob_wt_bar,
        double *pred_label_wt_bar, double *losses, int *missed_wt,
        int *missed_wt_bar, double *total_time) {
    openblas_set_num_threads(1);
    int i;
    cblas_dcopy(p + 1, w0, 1, wt, 1); // w0 --> wt
    cblas_dcopy(p + 1, w0, 1, wt_bar, 1); // w0 --> wt_bar
    double *loss_grad = malloc((p + 2) * sizeof(double));
    double *wt_tmp = malloc((p + 1) * sizeof(double));
    int *sorted_ind = malloc(sizeof(double) * p);
    int total_missed_wt = 0, total_missed_wt_bar = 0, max_iter = 50;
    Array *re_nodes = malloc(sizeof(Array));
    re_nodes->array = malloc(sizeof(int) * p);
    clock_t start_total = clock();
    if (verbose > 0) {
        printf("learning rate: %lf, num_tr: %d ,p: %d\n", lr, num_tr, p);
    }
    for (i = 0; i < num_tr; i++) {
        nonzeros_wt[i] = 0;
        nonzeros_wt_bar[i] = 0;
    }
    for (int tt = 0; tt < num_tr; tt++) {
        // each time has one sample.
        logistic_predict(x_tr + tt * p, wt, &pred_prob_wt[tt],
                         &pred_label_wt[tt], 0.5, 1, p);
        logistic_predict(x_tr + tt * p, wt_bar, &pred_prob_wt_bar[tt],
                         &pred_label_wt_bar[tt], 0.5, 1, p);
        logistic_loss_grad_sparse(
                wt, x_tr + tt * p, y_tr + tt, loss_grad, eta, 1, p);
        if (pred_label_wt[tt] != y_tr[tt]) {
            total_missed_wt++;
        }
        if (pred_label_wt_bar[tt] != y_tr[tt]) {
            total_missed_wt_bar++;
        }
        missed_wt[tt] = total_missed_wt;
        missed_wt_bar[tt] = total_missed_wt_bar;
        losses[tt] = loss_grad[0];
        if (verbose > 0) {
            printf("losses[%d]:%.6f sparsity:%d ", tt, losses[tt], s);
        }
        cblas_dcopy(p + 1, wt, 1, wt_tmp, 1); // wt --> wt_tmp
        cblas_daxpy(p + 1, -lr / sqrt(tt + 1.), loss_grad + 1, 1, wt_tmp, 1);
        arg_magnitude_sort_descend(wt_tmp, sorted_ind, p);
        re_nodes->size = 0;
        for (i = 0; i < s; i++) {
            re_nodes->array[re_nodes->size++] = sorted_ind[i];
        }
        min_f_sparse(re_nodes, x_tr, y_tr, max_iter, eta, wt, tt + 1, p);
        if (verbose > 0) {
            printf("pred_prob:%.4f total_missed:%d/%d\n",
                   pred_prob_wt[tt], total_missed_wt, tt + 1);
        }
        cblas_dscal(p + 1, tt / (tt + 1.), wt_bar, 1);
        cblas_daxpy(p + 1, 1 / (tt + 1.), wt, 1, wt_bar, 1);
        for (i = 0; i < p; i++) {
            if (wt[i] != 0.0) {
                nonzeros_wt[tt] += 1;
            }
            if (wt_bar[i] != 0.0) {
                nonzeros_wt_bar[tt] += 1;
            }
        }
    }
    *total_time = (double) (clock() - start_total) / CLOCKS_PER_SEC;
    free(loss_grad), free(wt_tmp), free(sorted_ind);
    return true;
}

bool algo_batch_iht_logit(
        const double *x_tr, const double *y_tr, const double *w0, int s,
        int p, int num_tr, double lr, double eta, int max_iter, double tol,
        int verbose, double *wt, double *losses, double *total_time) {
    openblas_set_num_threads(1);
    int i;
    cblas_dcopy(p + 1, w0, 1, wt, 1); // w0 --> wt
    double *loss_grad = malloc((p + 2) * sizeof(double));
    double *wt_tmp = malloc((p + 1) * sizeof(double));
    int *sorted_ind = malloc(sizeof(double) * p);
    Array *re_nodes = malloc(sizeof(Array));
    re_nodes->array = malloc(sizeof(int) * p);
    clock_t start_total = clock();
    if (verbose > 0) {
        printf("learning rate: %lf, num_tr: %d ,p: %d\n", lr, num_tr, p);
    }
    for (int tt = 0; tt < max_iter; tt++) {
        // each time has one sample.
        logistic_loss_grad(wt, x_tr, y_tr, loss_grad, eta, num_tr, p);
        losses[tt] = loss_grad[0];
        if (verbose > 0) {
            printf("losses[%d]:%.6f sparsity:%d ", tt, losses[tt], s);
        }
        cblas_dcopy(p + 1, wt, 1, wt_tmp, 1); // wt --> wt_tmp
        cblas_daxpy(p + 1, -lr, loss_grad + 1, 1, wt_tmp, 1);
        arg_magnitude_sort_descend(wt_tmp, sorted_ind, p);
        for (i = 0; i < s; i++) {
            wt[sorted_ind[i]] = wt_tmp[sorted_ind[i]];
        }
        wt[p] = wt_tmp[p];
        if (tt >= 1 && (fabs(losses[tt] - losses[tt - 1]) < tol)) {
            break; // stop earlier when it almost stops decreasing the loss
        }
    }
    *total_time = (double) (clock() - start_total) / CLOCKS_PER_SEC;
    free(loss_grad), free(wt_tmp), free(sorted_ind);
    free(re_nodes->array), free(re_nodes);
    return true;
}

bool algo_batch_ghtp_logit(
        const double *x_tr, const double *y_tr, const double *w0, int s,
        int p, int num_tr, double lr, double eta, int max_iter, double tol,
        int verbose, double *wt, double *losses, double *total_time) {
    openblas_set_num_threads(1);
    int i;
    cblas_dcopy(p + 1, w0, 1, wt, 1); // w0 --> wt
    double *loss_grad = malloc((p + 2) * sizeof(double));
    double *wt_tmp = malloc((p + 1) * sizeof(double));
    int *sorted_ind = malloc(sizeof(double) * p);
    Array *re_nodes = malloc(sizeof(Array));
    re_nodes->array = malloc(sizeof(int) * p);
    clock_t start_total = clock();
    if (verbose > 0) {
        printf("learning rate: %lf, num_tr: %d ,p: %d\n", lr, num_tr, p);
    }
    for (int tt = 0; tt < max_iter; tt++) {
        // each time has one sample.
        logistic_loss_grad(wt, x_tr, y_tr, loss_grad, eta, num_tr, p);
        losses[tt] = loss_grad[0];
        if (verbose > 0) {
            printf("losses[%d]:%.6f sparsity:%d ", tt, losses[tt], s);
        }
        cblas_dcopy(p + 1, wt, 1, wt_tmp, 1); // wt --> wt_tmp
        cblas_daxpy(p + 1, -lr, loss_grad + 1, 1, wt_tmp, 1);
        arg_magnitude_sort_descend(wt_tmp, sorted_ind, p);
        re_nodes->size = 0;
        for (i = 0; i < s; i++) {
            re_nodes->array[re_nodes->size++] = sorted_ind[i];
        }
        min_f(re_nodes, x_tr, y_tr, max_iter, eta, wt, num_tr, p);
        if (tt >= 1 && (fabs(losses[tt] - losses[tt - 1]) < tol)) {
            break; // stop earlier when it almost stops decreasing the loss
        }
    }
    *total_time = (double) (clock() - start_total) / CLOCKS_PER_SEC;
    free(loss_grad), free(wt_tmp), free(sorted_ind);
    free(re_nodes->array), free(re_nodes);
    return true;
}

bool algo_batch_graph_iht_logit(
        const EdgePair *edges, const double *costs, int g, int s, int p, int m,
        const double *x_tr, const double *y_tr, const double *w0, int num_tr,
        double tol, int max_iter, double lr, double eta, int verbose,
        double *wt, Array *re_nodes, double *losses, double *run_time_head,
        double *run_time_tail, double *total_time) {
    openblas_set_num_threads(1);
    GraphStat *head_stat = make_graph_stat(p, m);
    GraphStat *tail_stat = make_graph_stat(p, m);
    cblas_dcopy(p + 1, w0, 1, wt, 1); // w0 --> wt
    double *loss_grad = malloc(sizeof(double) * (p + 2));
    double *wt_tmp = malloc(sizeof(double) * (p + 1));
    double *tmp_prizes = malloc(sizeof(double) * p);
    double *tmp_costs = malloc(sizeof(double) * m);
    clock_t start_head, start_tail, start_total = clock();
    double C = 2. * (s - 1.), delta = 1. / 169., nu = 2.5, err_tol = 1e-6;
    double budget = (s - 1.), eps = 1e-6;
    int i, root = -1;
    enum PruningMethod pruning = GWPruning;
    for (i = 0; i < m; i++) { tmp_costs[i] = costs[i] + budget / (double) s; }
    for (int tt = 0; tt < max_iter; tt++) {
        // each time has one sample.
        logistic_loss_grad(wt, x_tr, y_tr, loss_grad, eta, num_tr, p);
        losses[tt] = loss_grad[0];
        for (i = 0; i < p; i++) {
            tmp_prizes[i] = loss_grad[i + 1] * loss_grad[i + 1];
        }
        start_head = clock();
        head_proj_exact(edges, tmp_costs, tmp_prizes, g, C, delta, max_iter,
                        err_tol, root, pruning, eps, p, m, verbose, head_stat);
        cblas_dcopy(p + 1, loss_grad + 1, 1, wt_tmp, 1); // wt --> wt_tmp
        cblas_dscal(p + 1, 0.0, loss_grad + 1, 1);
        for (i = 0; i < re_nodes->size; i++) {
            int cur_node = re_nodes->array[i];
            loss_grad[cur_node + 1] = wt_tmp[cur_node];
        }
        *run_time_head += ((double) (clock() - start_head)) / CLOCKS_PER_SEC;
        cblas_dcopy(p + 1, wt, 1, wt_tmp, 1); // wt --> wt_tmp
        cblas_daxpy(p + 1, -lr, loss_grad + 1, 1, wt_tmp, 1);
        for (i = 0; i < p; i++) {
            tmp_prizes[i] = wt_tmp[i] * wt_tmp[i];
        }
        start_tail = clock();
        tail_proj_exact(edges, tmp_costs, tmp_prizes, g, C, nu, max_iter,
                        err_tol, root, pruning, eps, p, m, verbose, tail_stat);
        *run_time_tail += ((double) (clock() - start_tail)) / CLOCKS_PER_SEC;
        cblas_dscal(p + 1, 0.0, wt, 1);
        for (i = 0; i < re_nodes->size; i++) {
            int cur_node = re_nodes->array[i];
            wt[cur_node] = wt_tmp[cur_node];
        }
        wt[p] = wt_tmp[p];
        if ((tt >= 1 && (fabs(losses[tt] - losses[tt - 1]) < tol)) ||
            (tt >= 1 && (losses[tt] >= losses[tt - 1]))) {
            break; // stop earlier when it almost stops decreasing the loss
        }
    }
    *total_time = (double) (clock() - start_total) / CLOCKS_PER_SEC;
    free(tmp_costs), free(tmp_prizes), free(loss_grad), free(wt_tmp);
    return true;
}


bool algo_batch_graph_ghtp_logit(
        const EdgePair *edges, const double *costs, int g, int s, int p, int m,
        const double *x_tr, const double *y_tr, const double *w0, int num_tr,
        double tol, int max_iter, double lr, double eta, int verbose,
        double *wt, Array *re_nodes, double *losses, double *run_time_head,
        double *run_time_tail, double *total_time) {
    openblas_set_num_threads(1);
    GraphStat *head_stat = make_graph_stat(p, m);
    GraphStat *tail_stat = make_graph_stat(p, m);
    int i, root = -1, fmin_max_iter = 20;
    cblas_dcopy(p + 1, w0, 1, wt, 1); // w0 --> wt
    double *loss_grad = malloc((p + 2) * sizeof(double));
    double *wt_tmp = malloc((p + 1) * sizeof(double));
    double *tmp_prizes = malloc(sizeof(double) * p);
    double *tmp_costs = malloc(sizeof(double) * m);
    clock_t start_head, start_tail, start_total = clock();
    double C = 2. * (s - 1.), delta = 1. / 169., nu = 2.5;
    double err_tol = 1e-6, eps = 1e-6;
    enum PruningMethod pruning = GWPruning;
    for (i = 0; i < m; i++) {
        tmp_costs[i] = costs[i] + (s - 1.) / (double) s;
    }
    Array *tmp_nodes = malloc(sizeof(Array));
    tmp_nodes->array = malloc(sizeof(int) * p);

    for (int tt = 0; tt < max_iter; tt++) {
        // each time has one sample.
        logistic_loss_grad(wt, x_tr, y_tr, loss_grad, eta, num_tr, p);
        losses[tt] = loss_grad[0];
        for (i = 0; i < p; i++) {
            tmp_prizes[i] = loss_grad[i + 1] * loss_grad[i + 1];
        }
        start_head = clock();
        head_proj_exact(edges, tmp_costs, tmp_prizes, g, C, delta, max_iter,
                        err_tol, root, pruning, eps, p, m, verbose, head_stat);
        *run_time_head += ((double) (clock() - start_head)) / CLOCKS_PER_SEC;

        cblas_dscal(p + 1, 0.0, wt_tmp, 1);
        for (i = 0; i < re_nodes->size; i++) {
            wt_tmp[re_nodes->array[i]] = loss_grad[re_nodes->array[i] + 1];
        }
        cblas_dcopy(p + 1, wt_tmp, 1, loss_grad + 1, 1);
        cblas_dcopy(p + 1, wt, 1, wt_tmp, 1); // wt --> wt_tmp
        cblas_daxpy(p + 1, -lr, loss_grad + 1, 1, wt_tmp, 1);
        tmp_nodes->size = 0;
        for (i = 0; i < p; i++) {
            if (wt_tmp[i] != 0.0) {
                tmp_nodes->array[tmp_nodes->size++] = i;
            }
        }
        min_f(tmp_nodes, x_tr, y_tr, fmin_max_iter, eta, wt, num_tr, p);
        for (i = 0; i < p; i++) {
            tmp_prizes[i] = wt[i] * wt[i];
        }
        start_tail = clock();
        tail_proj_exact(edges, tmp_costs, tmp_prizes, g, C, nu, max_iter,
                        err_tol, root, pruning, eps, p, m, verbose, tail_stat);
        *run_time_tail += ((double) (clock() - start_tail)) / CLOCKS_PER_SEC;
        cblas_dcopy(p + 1, wt, 1, wt_tmp, 1);
        cblas_dscal(p + 1, 0.0, wt, 1);
        for (i = 0; i < re_nodes->size; i++) {
            wt[re_nodes->array[i]] = wt_tmp[re_nodes->array[i]];
        }
        wt[p] = wt_tmp[p];
        if ((tt >= 1 && (fabs(losses[tt] - losses[tt - 1]) < tol)) ||
            (tt >= 1 && (losses[tt] >= losses[tt - 1]))) {
            break; // stop earlier when it almost stops decreasing the loss
        }
    }
    *total_time = (double) (clock() - start_total) / CLOCKS_PER_SEC;
    free(tmp_costs), free(tmp_prizes), free(loss_grad);
    free(wt_tmp), free(tmp_nodes->array), free(tmp_nodes);
    return true;
}


bool algo_batch_graph_posi_logit(
        const EdgePair *edges, const double *costs, int g, int s, int p, int m,
        const double *x_tr, const double *y_tr, const double *w0, int num_tr,
        double tol, int max_iter, double lr, double eta, int verbose,
        double *wt, Array *re_nodes, double *losses, double *run_time_head,
        double *run_time_tail, double *total_time) {
    openblas_set_num_threads(1);
    GraphStat *head_stat = make_graph_stat(p, m);
    GraphStat *tail_stat = make_graph_stat(p, m);
    int i, root = -1, fmin_max_iter = 20;
    cblas_dcopy(p + 1, w0, 1, wt, 1); // w0 --> wt
    double *loss_grad = malloc((p + 2) * sizeof(double));
    double *wt_tmp = malloc((p + 1) * sizeof(double));
    double *tmp_prizes = malloc(sizeof(double) * p);
    double *tmp_costs = malloc(sizeof(double) * m);
    clock_t start_head, start_tail, start_total = clock();
    double C = 2. * (s - 1.), delta = 1. / 169., nu = 2.5;
    double err_tol = 1e-6, budget = (s - 1.), eps = 1e-6;
    enum PruningMethod pruning = GWPruning;
    for (i = 0; i < m; i++) {
        tmp_costs[i] = costs[i] + budget / (double) s;
    }
    Array *tmp_nodes = malloc(sizeof(Array));
    tmp_nodes->array = malloc(sizeof(int) * p);

    for (int tt = 0; tt < max_iter; tt++) {
        // each time has one sample.
        logistic_loss_grad(wt, x_tr, y_tr, loss_grad, eta, num_tr, p);
        losses[tt] = loss_grad[0];
        for (i = 0; i < p; i++) {
            tmp_prizes[i] = loss_grad[i + 1] * loss_grad[i + 1];
        }
        start_head = clock();
        head_proj_exact(edges, tmp_costs, tmp_prizes, g, C, delta, max_iter,
                        err_tol, root, pruning, eps, p, m, verbose, head_stat);
        *run_time_head += ((double) (clock() - start_head)) / CLOCKS_PER_SEC;

        cblas_dscal(p + 1, 0.0, wt_tmp, 1);
        for (i = 0; i < re_nodes->size; i++) {
            wt_tmp[re_nodes->array[i]] = loss_grad[re_nodes->array[i] + 1];
        }
        cblas_dcopy(p + 1, wt_tmp, 1, loss_grad + 1, 1);
        cblas_dcopy(p + 1, wt, 1, wt_tmp, 1); // wt --> wt_tmp
        cblas_daxpy(p + 1, -lr, loss_grad + 1, 1, wt_tmp, 1);
        tmp_nodes->size = 0;
        for (i = 0; i < p; i++) {
            if (wt_tmp[i] != 0.0) {
                tmp_nodes->array[tmp_nodes->size++] = i;
            }
        }
        min_f_posi(tmp_nodes, x_tr, y_tr, fmin_max_iter, eta, wt, num_tr, p);
        for (i = 0; i < p; i++) {
            tmp_prizes[i] = wt[i] * wt[i];
        }
        start_tail = clock();
        tail_proj_exact(edges, tmp_costs, tmp_prizes, g, C, nu, max_iter,
                        err_tol, root, pruning, eps, p, m, verbose, tail_stat);
        *run_time_tail += ((double) (clock() - start_tail)) / CLOCKS_PER_SEC;
        cblas_dcopy(p + 1, wt, 1, wt_tmp, 1);
        cblas_dscal(p + 1, 0.0, wt, 1);
        for (i = 0; i < re_nodes->size; i++) {
            wt[re_nodes->array[i]] = wt_tmp[re_nodes->array[i]];
        }
        wt[p] = wt_tmp[p];
        if ((tt >= 1 && (fabs(losses[tt] - losses[tt - 1]) < tol)) ||
            (tt >= 1 && (losses[tt] >= losses[tt - 1]))) {
            break; // stop earlier when it almost stops decreasing the loss
        }
    }
    *total_time = (double) (clock() - start_total) / CLOCKS_PER_SEC;
    free(tmp_costs), free(tmp_prizes), free(loss_grad);
    free(wt_tmp), free(tmp_nodes->array), free(tmp_nodes);
    return true;
}