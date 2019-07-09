//
//
#include <cblas.h>
#include "algo_online_graph_da_iht.h"
#include "loss.h"

#define sign(x) ((x > 0) -(x < 0))

bool algo_online_graph_da_iht_logit(
        graph_da_iht_para *para, OnlineStat *stat) {
    clock_t start_time = clock();
    openblas_set_num_threads(1);

    int p = para->p;                    // number of features.
    int m = para->m;                    // number of edges.
    int verbose = para->verbose;        // verbose to print some information.
    int s = para->sparsity;             // sparsity parameter.
    int g = para->g;                    // number of connected components
    int num_tr = para->num_tr;          // number of training samples.

    double gamma = para->gamma;                   // parameter gamma.
    double l2_lambda = para->l2_lambda;         // ell-2 norma regularization.
    double head_err_tol = para->head_err_tol;   // error tolerance of head.
    double tail_err_tol = para->tail_err_tol;   // error tolerance of tail.
    double pcst_epsilon = para->pcst_epsilon;   // epsilon tolerance of pcst.
    int head_max_iter = para->head_max_iter;    // maximal iterations allowed.
    int tail_max_iter = para->tail_max_iter;    // maximal iterations allowed.
    double *x_tr = para->x_tr;          // training dataset.
    double *y_tr = para->y_tr;          // training labels.
    EdgePair *edges = para->edges;      // edges of the graph
    double *costs = para->weights;        // edge costs.
    double tail_budget = para->tail_budget; // tail_budget
    double head_budget = para->head_budget; // head_budget
    double tail_nu = para->tail_nu;         // tail parameter nu
    double head_delta = para->head_delta;   // head parameter delta

    double *proj_prizes = malloc(sizeof(double) * p);   // projected prizes.
    double *proj_costs = malloc(sizeof(double) * m);    // projected costs.
    double *wt = malloc(sizeof(double) * (p + 1));      // learned model
    double *wt_bar = malloc(sizeof(double) * (p + 1));  // take average of wt
    double *gt_bar = malloc(sizeof(double) * (p + 1));  // average of gradient
    double *loss_grad = malloc(sizeof(double) * (p + 2)); // loss+gradient
    double *wt_tmp = malloc(sizeof(double) * (p + 1));  // temp variable.

    GraphStat *head_stat = make_graph_stat(p, m);   // head projection paras
    GraphStat *tail_stat = make_graph_stat(p, m);   // tail projection paras

    cblas_dcopy(p + 1, para->w0, 1, wt, 1);     // initialize: w0 --> wt
    cblas_dcopy(p + 1, para->w0, 1, wt_bar, 1); // initialize: w0 --> wt_bar
    cblas_dscal(p + 1, 0.0, gt_bar, 1);         // initialize:  0 --> gt_bar

    for (int tt = 0; tt < num_tr; tt++) {
        // 1.   unlabeled example x_i = x_tr + tt*p arrives and
        //      make prediction based on existing costs wt, wt_bar.
        double *x_i = x_tr + tt * p;    //i-th data sample.
        double *y_i = y_tr + tt;        //i-th data label.
        predict_logistic(stat, tt, p, *y_i, x_i, wt, wt_bar);
        // 2.   observe y_i = y_tr + tt and have some loss
        logistic_loss_grad(wt, x_i, y_i, loss_grad, l2_lambda, 1, p);
        stat->losses[tt] = loss_grad[0];
        if (verbose > 0) {
            printf("tt: %3d missed_wt: %3d loss: %.2e norm_grad:%.2e "
                   "gamma: %.1e l2_lambda: %.1e sparsity: %d\n",
                   tt, stat->missed_wt[tt], stat->losses[tt],
                   l2_norm(loss_grad + 1, p), gamma, l2_lambda, s);
        }
        // 3.   take average of gradient --> gt_bar.
        cblas_dscal(p + 1, tt / (tt + 1.), gt_bar, 1);
        cblas_daxpy(p + 1, 1. / (tt + 1.), loss_grad + 1, 1, gt_bar, 1);

        // 4.   head projection on current sample
        for (int i = 0; i < p; i++) {
            proj_prizes[i] = gt_bar[i] * gt_bar[i];
        }
        head_tail_binsearch(
                edges, costs, proj_prizes, p, m, g, -1, (int)(p/2),
                (int)(p/2*(1.1)), head_max_iter, GWPruning, verbose,
                head_stat);
        stat->run_time_head += head_stat->run_time;
        cblas_dscal(p + 1, 0.0, wt_tmp, 1);
        for (int i = 0; i < head_stat->re_nodes->size; i++) {
            int cur_node = head_stat->re_nodes->array[i];
            wt_tmp[cur_node] = gt_bar[cur_node];
        }
        wt_tmp[p] = gt_bar[p];
        cblas_dscal(p + 1, -sqrt(tt + 1.) / gamma, wt_tmp, 1);
        for (int i = 0; i < p; i++) {
            proj_prizes[i] = wt_tmp[i] * wt_tmp[i];
        }
        head_tail_binsearch(
                edges, costs, proj_prizes, p, m, g, -1, (int)(p/2),
                (int)(p/2*(1.1)), tail_max_iter, GWPruning, verbose,
                tail_stat);
        stat->run_time_tail += tail_stat->run_time;
        // 5.   update model
        cblas_dscal(p, 0.0, wt, 1);
        for (int i = 0; i < tail_stat->re_nodes->size; i++) {
            int cur_node = tail_stat->re_nodes->array[i];
            wt[cur_node] = wt_tmp[cur_node];
        }
        wt[p] = wt_tmp[p]; // keep the intercept.
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
    // save results
    stat->re_nodes->size = tail_stat->re_nodes->size;
    stat->re_edges->size = tail_stat->re_edges->size;
    for (int i = 0; i < tail_stat->re_nodes->size; i++) {
        stat->re_nodes->array[i] = tail_stat->re_nodes->array[i];
    }
    for (int i = 0; i < tail_stat->re_edges->size; i++) {
        stat->re_edges->array[i] = tail_stat->re_edges->array[i];
    }
    for (int i = 0; i < p + 1; i++) {
        stat->wt[i] = wt[i];
        stat->wt_bar[i] = wt_bar[i];
    }
    stat->total_time = (double) (clock() - start_time) / CLOCKS_PER_SEC;
    free(proj_costs), free(proj_prizes), free(loss_grad);
    free(wt), free(wt_bar), free(wt_tmp), free(gt_bar);
    free_graph_stat(head_stat), free_graph_stat(tail_stat);
    return true;
}


bool algo_online_graph_da_iht_logit_2(
        graph_da_iht_para_2 *para, OnlineStat *stat) {
    clock_t start_time = clock();
    openblas_set_num_threads(1);

    int p = para->p;                    // number of features.
    int m = para->m;                    // number of edges.
    int verbose = para->verbose;        // verbose to print some information.
    int g = para->g;                    // number of connected components
    int num_tr = para->num_tr;          // number of training samples.

    double gamma = para->gamma;                 // parameter gamma.
    double l2_lambda = para->l2_lambda;         // ell-2 norma regularization.
    int sparsity_low = para->sparsity_low;   // error tolerance of head.
    int sparsity_high = para->sparsity_high;   // error tolerance of tail.
    double *costs = para->weights;
    int root = para->root;
    int max_num_iter = para->max_num_iter;
    double *x_tr = para->x_tr;          // training dataset.
    double *y_tr = para->y_tr;          // training labels.
    EdgePair *edges = para->edges;      // edges of the graph

    double *proj_prizes = malloc(sizeof(double) * p);   // projected prizes.
    double *proj_costs = malloc(sizeof(double) * m);    // projected costs.
    double *wt = malloc(sizeof(double) * (p + 1));      // learned model
    double *wt_bar = malloc(sizeof(double) * (p + 1));  // take average of wt
    double *gt_bar = malloc(sizeof(double) * (p + 1));  // average of gradient
    double *loss_grad = malloc(sizeof(double) * (p + 2)); // loss+gradient
    double *wt_tmp = malloc(sizeof(double) * (p + 1));  // temp variable.

    GraphStat *graph_stat = make_graph_stat(p, m);   // head projection paras

    cblas_dcopy(p + 1, para->w0, 1, wt, 1);     // initialize: w0 --> wt
    cblas_dcopy(p + 1, para->w0, 1, wt_bar, 1); // initialize: w0 --> wt_bar
    cblas_dscal(p + 1, 0.0, gt_bar, 1);         // initialize:  0 --> gt_bar

    for (int tt = 0; tt < num_tr; tt++) {
        // 1.   unlabeled example x_i = x_tr + tt*p arrives and
        //      make prediction based on existing costs wt, wt_bar.
        double *x_i = x_tr + tt * p;    //i-th data sample.
        double *y_i = y_tr + tt;        //i-th data label.
        predict_logistic(stat, tt, p, *y_i, x_i, wt, wt_bar);
        // 2.   observe y_i = y_tr + tt and have some loss
        logistic_loss_grad(wt, x_i, y_i, loss_grad, l2_lambda, 1, p);
        stat->losses[tt] = loss_grad[0];
        if (verbose > 0) {
            printf("tt: %3d missed_wt: %3d loss: %.2e norm_grad:%.2e "
                   "gamma: %.1e l2_lambda: %.1e sparsity: %d\n",
                   tt, stat->missed_wt[tt], stat->losses[tt],
                   l2_norm(loss_grad + 1, p), gamma, l2_lambda, sparsity_low);
        }
        // 3.   take average of gradient --> gt_bar.
        cblas_dscal(p + 1, tt / (tt + 1.), gt_bar, 1);
        cblas_daxpy(p + 1, 1. / (tt + 1.), loss_grad + 1, 1, gt_bar, 1);

        // 4.   head projection on current sample
        for (int i = 0; i < p; i++) {
            proj_prizes[i] = gt_bar[i] * gt_bar[i];
        }
        head_tail_binsearch(
                edges, costs, proj_prizes, p, m, g, root, (int)(p/2),
                (int)(p/2*(1.1)), max_num_iter, GWPruning, verbose,
                graph_stat);
        stat->run_time_head += graph_stat->run_time;
        cblas_dscal(p + 1, 0.0, wt_tmp, 1);
        for (int i = 0; i < graph_stat->re_nodes->size; i++) {
            int cur_node = graph_stat->re_nodes->array[i];
            wt_tmp[cur_node] = gt_bar[cur_node];
        }
        wt_tmp[p] = gt_bar[p];
        cblas_dscal(p + 1, -sqrt(tt + 1.) / gamma, wt_tmp, 1);
        for (int i = 0; i < p; i++) {
            proj_prizes[i] = wt_tmp[i] * wt_tmp[i];
        }
        head_tail_binsearch(
                edges, costs, proj_prizes, p, m, g, root, sparsity_low,
                sparsity_high, max_num_iter, GWPruning, verbose, graph_stat);
        stat->run_time_tail += graph_stat->run_time;
        // 5.   update model
        cblas_dscal(p, 0.0, wt, 1);
        for (int i = 0; i < graph_stat->re_nodes->size; i++) {
            int cur_node = graph_stat->re_nodes->array[i];
            wt[cur_node] = wt_tmp[cur_node];
        }
        wt[p] = wt_tmp[p]; // keep the intercept.
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
    // save results
    stat->re_nodes->size = graph_stat->re_nodes->size;
    stat->re_edges->size = graph_stat->re_edges->size;
    for (int i = 0; i < graph_stat->re_nodes->size; i++) {
        stat->re_nodes->array[i] = graph_stat->re_nodes->array[i];
    }
    for (int i = 0; i < graph_stat->re_edges->size; i++) {
        stat->re_edges->array[i] = graph_stat->re_edges->array[i];
    }
    for (int i = 0; i < p + 1; i++) {
        stat->wt[i] = wt[i];
        stat->wt_bar[i] = wt_bar[i];
    }
    stat->total_time = (double) (clock() - start_time) / CLOCKS_PER_SEC;
    free(proj_costs), free(proj_prizes), free(loss_grad);
    free(wt), free(wt_bar), free(wt_tmp), free(gt_bar);
    free_graph_stat(graph_stat);
    return true;
}


bool algo_online_graph_da_iht_least_square_2(
        graph_da_iht_para_2 *para, OnlineStat *stat) {
    clock_t start_time = clock();
    openblas_set_num_threads(1);

    int p = para->p;                    // number of features.
    int m = para->m;                    // number of edges.
    int verbose = para->verbose;        // verbose to print some information.
    int g = para->g;                    // number of connected components
    int num_tr = para->num_tr;          // number of training samples.

    double gamma = para->gamma;                 // parameter gamma.
    double l2_lambda = para->l2_lambda;         // ell-2 norma regularization.
    int sparsity_low = para->sparsity_low;   // error tolerance of head.
    int sparsity_high = para->sparsity_high;   // error tolerance of tail.
    double *costs = para->weights;
    int root = para->root;
    int max_num_iter = para->max_num_iter;
    double *x_tr = para->x_tr;          // training dataset.
    double *y_tr = para->y_tr;          // training labels.
    EdgePair *edges = para->edges;      // edges of the graph

    double *proj_prizes = malloc(sizeof(double) * p);   // projected prizes.
    double *proj_costs = malloc(sizeof(double) * m);    // projected costs.
    double *wt = malloc(sizeof(double) * (p + 1));      // learned model
    double *wt_bar = malloc(sizeof(double) * (p + 1));  // take average of wt
    double *gt_bar = malloc(sizeof(double) * (p + 1));  // average of gradient
    double *loss_grad = malloc(sizeof(double) * (p + 2)); // loss+gradient
    double *wt_tmp = malloc(sizeof(double) * (p + 1));  // temp variable.

    GraphStat *graph_stat = make_graph_stat(p, m);   // head projection paras

    cblas_dcopy(p + 1, para->w0, 1, wt, 1);     // initialize: w0 --> wt
    cblas_dcopy(p + 1, para->w0, 1, wt_bar, 1); // initialize: w0 --> wt_bar
    cblas_dscal(p + 1, 0.0, gt_bar, 1);         // initialize:  0 --> gt_bar

    for (int tt = 0; tt < num_tr; tt++) {
        // 1.   unlabeled example x_i = x_tr + tt*p arrives and
        //      make prediction based on existing costs wt, wt_bar.
        double *x_i = x_tr + tt * p;    //i-th data sample.
        double *y_i = y_tr + tt;        //i-th data label.
        predict_least_square(stat, tt, p, *y_i, x_i, wt, wt_bar);
        // 2.   observe y_i = y_tr + tt and have some loss
        least_square_loss_grad(wt, x_i, y_i, loss_grad, l2_lambda, 1, p);
        stat->losses[tt] = loss_grad[0];
        if (verbose > 0) {
            printf("tt: %3d missed_wt: %3d loss: %.2e norm_grad:%.2e "
                   "gamma: %.1e l2_lambda: %.1e sparsity: %d\n",
                   tt, stat->missed_wt[tt], stat->losses[tt],
                   l2_norm(loss_grad + 1, p), gamma, l2_lambda, sparsity_low);
        }
        // 3.   take average of gradient --> gt_bar.
        cblas_dscal(p + 1, tt / (tt + 1.), gt_bar, 1);
        cblas_daxpy(p + 1, 1. / (tt + 1.), loss_grad + 1, 1, gt_bar, 1);

        // 4.   head projection on current sample
        for (int i = 0; i < p; i++) {
            proj_prizes[i] = gt_bar[i] * gt_bar[i];
        }
        head_tail_binsearch(
                edges, costs, proj_prizes, p, m, g, root, (int)(p/2),
                (int)(p/2*(1.1)), max_num_iter, GWPruning, verbose,
                graph_stat);
        // printf("tt:%d head_size: %d num_iter: %d\n",
        //       tt, graph_stat->re_nodes->size, graph_stat->num_iter);
        stat->run_time_head += graph_stat->run_time;
        cblas_dscal(p + 1, 0.0, wt_tmp, 1);
        for (int i = 0; i < graph_stat->re_nodes->size; i++) {
            int cur_node = graph_stat->re_nodes->array[i];
            wt_tmp[cur_node] = gt_bar[cur_node];
        }
        wt_tmp[p] = gt_bar[p];
        cblas_dscal(p + 1, -sqrt(tt + 1.) / gamma, wt_tmp, 1);
        for (int i = 0; i < p; i++) {
            proj_prizes[i] = wt_tmp[i] * wt_tmp[i];
        }
        head_tail_binsearch(
                edges, costs, proj_prizes, p, m, g, root, sparsity_low,
                sparsity_high, max_num_iter, GWPruning, verbose,
                graph_stat);
        // printf("tt:%d tail_size: %d num_iter: %d\n",
        //       tt, graph_stat->re_nodes->size, graph_stat->num_iter);
        stat->run_time_tail += graph_stat->run_time;
        // 5.   update model
        cblas_dscal(p, 0.0, wt, 1);
        for (int i = 0; i < graph_stat->re_nodes->size; i++) {
            int cur_node = graph_stat->re_nodes->array[i];
            wt[cur_node] = wt_tmp[cur_node];
        }
        wt[p] = wt_tmp[p]; // keep the intercept.
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
    // save results
    stat->re_nodes->size = graph_stat->re_nodes->size;
    stat->re_edges->size = graph_stat->re_edges->size;
    for (int i = 0; i < graph_stat->re_nodes->size; i++) {
        stat->re_nodes->array[i] = graph_stat->re_nodes->array[i];
    }
    for (int i = 0; i < graph_stat->re_edges->size; i++) {
        stat->re_edges->array[i] = graph_stat->re_edges->array[i];
    }
    for (int i = 0; i < p + 1; i++) {
        stat->wt[i] = wt[i];
        stat->wt_bar[i] = wt_bar[i];
    }
    stat->total_time = (double) (clock() - start_time) / CLOCKS_PER_SEC;
    free(proj_costs), free(proj_prizes), free(loss_grad);
    free(wt), free(wt_bar), free(wt_tmp), free(gt_bar);
    free_graph_stat(graph_stat);
    return true;
}


bool algo_online_graph_da_iht_least_square(
        graph_da_iht_para *para, OnlineStat *stat) {
    clock_t start_time = clock();
    openblas_set_num_threads(1);
    int p = para->p;
    int m = para->m;
    int verbose = para->verbose;
    int s = para->sparsity;
    int num_tr = para->num_tr;
    double gamma = para->gamma;
    double l2_lambda = para->l2_lambda;
    double *x_tr = para->x_tr;
    double *y_tr = para->y_tr;
    double *weights = para->weights;
    EdgePair *edges = para->edges;
    double *proj_prizes;
    double *proj_costs;
    GraphStat *head_stat = make_graph_stat(p, m);
    GraphStat *tail_stat = make_graph_stat(p, m);
    double *wt, *wt_bar, *gt_bar, *loss_grad, *wt_tmp, *prob, *label;

    wt = malloc(sizeof(double) * (p + 1));
    wt_bar = malloc(sizeof(double) * (p + 1));
    gt_bar = malloc(sizeof(double) * (p + 1));
    loss_grad = malloc(sizeof(double) * (p + 2));
    wt_tmp = malloc(sizeof(double) * (p + 1));
    proj_prizes = malloc(sizeof(double) * p);
    proj_costs = malloc(sizeof(double) * m);

    cblas_dcopy(p + 1, para->w0, 1, wt, 1);         // w0 --> wt
    cblas_dcopy(p + 1, para->w0, 1, wt_bar, 1);     // w0 --> wt_bar
    cblas_dscal(p + 1, 0.0, gt_bar, 1);             //  0 --> gt_bar

    // projection costs
    for (int i = 0; i < m; i++) {
        proj_costs[i] = weights[i] + (s - 1.) / (double) (s);
    }
    double nu = 2.5;
    double delta = 1. / 169.;
    double budget = 2. * (s - 1.);
    for (int tt = 0; tt < num_tr; tt++) {
        // 1.   unlabeled example x_i = x_tr + tt*p arrives and
        //      make prediction based on existing costs wt, wt_bar
        double *x_i = x_tr + tt * p;
        double *y_i = y_tr + tt;
        prob = &stat->p_prob_wt[tt];
        label = &stat->p_label_wt[tt];
        least_square_predict(x_i, wt, prob, label, 0.5, 1, p);
        prob = &stat->p_prob_wt_bar[tt];
        label = &stat->p_label_wt_bar[tt];
        least_square_predict(x_i, wt_bar, prob, label, 0.5, 1, p);
        if (stat->p_label_wt[tt] != y_tr[tt]) {
            stat->total_missed_wt++;
        }
        stat->missed_wt[tt] = stat->total_missed_wt;
        if (stat->p_label_wt_bar[tt] != y_tr[tt]) {
            stat->total_missed_wt_bar++;
        }
        stat->missed_wt_bar[tt] = stat->total_missed_wt_bar;
        // 2. observe y_i = y_tr + tt and have some loss
        least_square_loss_grad(wt, x_i, y_i, loss_grad, l2_lambda, 1, p);
        stat->losses[tt] = loss_grad[0];

        cblas_dscal(p + 1, tt / (tt + 1.), gt_bar, 1);
        cblas_daxpy(p + 1, 1. / (tt + 1.), loss_grad + 1, 1, gt_bar, 1);

        // head projection on current sample
        for (int i = 0; i < p; i++) {
            proj_prizes[i] = gt_bar[i] * gt_bar[i];
        }
        head_proj_exact(
                para->edges, proj_costs, proj_prizes, 1, 10. * budget, delta,
                100, 1e-6, -1, GWPruning, 1e-6, p, m, verbose, head_stat);
        stat->run_time_head += head_stat->run_time;
        // tail projection update wt= wt - lr/sqrt(tt) * head(f_xi, wt)
        cblas_dscal(p, 0.0, wt_tmp, 1);
        for (int i = 0; i < head_stat->re_nodes->size; i++) {
            int cur_node = head_stat->re_nodes->array[i];
            wt_tmp[cur_node] = gt_bar[cur_node];
        }
        wt_tmp[p] = gt_bar[p];
        cblas_dscal(p + 1, -sqrt(tt + 1.) / gamma, wt_tmp, 1);
        for (int i = 0; i < p; i++) {
            proj_prizes[i] = wt_tmp[i] * wt_tmp[i];
        }
        tail_proj_exact(
                edges, proj_costs, proj_prizes, 1, 1. * budget, nu,
                100, 1e-6, -1, GWPruning, 1e-6, p, m, verbose, tail_stat);
        stat->run_time_tail += tail_stat->run_time;
        // update model
        cblas_dscal(p, 0.0, wt, 1);
        for (int i = 0; i < tail_stat->re_nodes->size; i++) {
            int cur_node = tail_stat->re_nodes->array[i];
            wt[cur_node] = wt_tmp[cur_node];
        }
        wt[p] = wt_tmp[p]; // keep the intercept.
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
    // save results
    stat->re_nodes->size = tail_stat->re_nodes->size;
    stat->re_edges->size = tail_stat->re_edges->size;
    for (int i = 0; i < tail_stat->re_nodes->size; i++) {
        stat->re_nodes->array[i] = tail_stat->re_nodes->array[i];
    }
    for (int i = 0; i < tail_stat->re_edges->size; i++) {
        stat->re_edges->array[i] = tail_stat->re_edges->array[i];
    }
    for (int i = 0; i < p + 1; i++) {
        stat->wt[i] = wt[i];
        stat->wt_bar[i] = wt_bar[i];
    }
    stat->total_time = (double) (clock() - start_time) / CLOCKS_PER_SEC;
    free(proj_costs);
    free(proj_prizes);
    free(loss_grad);
    free(wt);
    free(wt_bar);
    free(wt_tmp);
    free(gt_bar);
    free_graph_stat(head_stat);
    free_graph_stat(tail_stat);
    return true;
}


bool algo_online_graph_da_iht_least_square_without_head(
        graph_da_iht_para *para, OnlineStat *stat) {

    clock_t start_time = clock();
    openblas_set_num_threads(1);

    // input parameters
    int p = para->p;
    int m = para->m;
    int verbose = para->verbose;
    int s = para->sparsity;
    int num_tr = para->num_tr;
    double gamma = para->gamma;
    double l2_lambda = para->l2_lambda;
    double *x_tr = para->x_tr;
    double *y_tr = para->y_tr;
    double *weights = para->weights;
    EdgePair *edges = para->edges;

    // temp parameters
    double *proj_prizes;
    double *proj_costs;
    GraphStat *tail_stat;
    double *wt;
    double *wt_bar;
    double *gt_bar;
    double *loss_grad;
    double *wt_tmp;
    double *prob;
    double *label;

    tail_stat = make_graph_stat(p, m);
    wt = malloc(sizeof(double) * (p + 1));
    wt_bar = malloc(sizeof(double) * (p + 1));
    gt_bar = malloc(sizeof(double) * (p + 1));
    loss_grad = malloc(sizeof(double) * (p + 2));
    wt_tmp = malloc(sizeof(double) * (p + 1));
    proj_prizes = malloc(sizeof(double) * p);
    proj_costs = malloc(sizeof(double) * m);

    cblas_dcopy(p + 1, para->w0, 1, wt, 1);         // w0 --> wt
    cblas_dcopy(p + 1, para->w0, 1, wt_bar, 1);     // w0 --> wt_bar
    cblas_dscal(p + 1, 0.0, gt_bar, 1);             //  0 --> gt_bar

    // projection costs
    for (int i = 0; i < m; i++) {
        proj_costs[i] = weights[i] + (s - 1.) / (double) (s);
    }

    double nu = 2.5;
    double cost_budget = 2. * (s - 1.);
    for (int tt = 0; tt < num_tr; tt++) {
        // 1.   unlabeled example x_i = x_tr + tt*p arrives and
        //      make prediction based on existing costs wt, wt_bar
        double *x_i = x_tr + tt * p;
        double *y_i = y_tr + tt;
        prob = &stat->p_prob_wt[tt];
        label = &stat->p_label_wt[tt];
        least_square_predict(x_i, wt, prob, label, 0.5, 1, p);
        prob = &stat->p_prob_wt_bar[tt];
        label = &stat->p_label_wt_bar[tt];
        least_square_predict(x_i, wt_bar, prob, label, 0.5, 1, p);
        if (stat->p_label_wt[tt] != y_tr[tt]) {
            stat->total_missed_wt++;
        }
        stat->missed_wt[tt] = stat->total_missed_wt;
        if (stat->p_label_wt_bar[tt] != y_tr[tt]) {
            stat->total_missed_wt_bar++;
        }
        stat->missed_wt_bar[tt] = stat->total_missed_wt_bar;
        // 2. observe y_i = y_tr + tt and have some loss
        least_square_loss_grad(wt, x_i, y_i, loss_grad, l2_lambda, 1, p);
        stat->losses[tt] = loss_grad[0];

        cblas_dscal(p + 1, tt / (tt + 1.), gt_bar, 1);
        cblas_daxpy(p + 1, 1. / (tt + 1.), loss_grad + 1, 1, gt_bar, 1);

        cblas_ccopy(p + 1, gt_bar, 1, wt_tmp, 1);
        cblas_dscal(p + 1, -sqrt(tt + 1.) / gamma, wt_tmp, 1);
        for (int i = 0; i < p; i++) {
            proj_prizes[i] = wt_tmp[i] * wt_tmp[i];
        }
        tail_proj_exact(
                edges, proj_costs, proj_prizes, 1, cost_budget, nu,
                100, 1e-6, -1, GWPruning, 1e-6, p, m, verbose, tail_stat);
        stat->run_time_tail += tail_stat->run_time;
        // update model
        cblas_dscal(p, 0.0, wt, 1);
        for (int i = 0; i < tail_stat->re_nodes->size; i++) {
            int cur_node = tail_stat->re_nodes->array[i];
            wt[cur_node] = wt_tmp[cur_node];
        }
        wt[p] = wt_tmp[p]; // keep the intercept.
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
    // save results
    stat->re_nodes->size = tail_stat->re_nodes->size;
    stat->re_edges->size = tail_stat->re_edges->size;
    for (int i = 0; i < tail_stat->re_nodes->size; i++) {
        stat->re_nodes->array[i] = tail_stat->re_nodes->array[i];
    }
    for (int i = 0; i < tail_stat->re_edges->size; i++) {
        stat->re_edges->array[i] = tail_stat->re_edges->array[i];
    }
    for (int i = 0; i < p + 1; i++) {
        stat->wt[i] = wt[i];
        stat->wt_bar[i] = wt_bar[i];
    }
    stat->total_time = (double) (clock() - start_time) / CLOCKS_PER_SEC;
    free(proj_costs);
    free(proj_prizes);
    free(loss_grad);
    free(wt);
    free(wt_bar);
    free(wt_tmp);
    free(gt_bar);
    free_graph_stat(tail_stat);
    return true;
}