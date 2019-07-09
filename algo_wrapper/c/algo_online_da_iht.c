//
//

#include <cblas.h>
#include "algo_online_da_iht.h"
#include "loss.h"
#include "sort.h"

#define sign(x) ((x > 0) -(x < 0))



bool algo_online_da_iht_logit(da_iht_para *para, OnlineStat *stat) {
    clock_t start_total = clock();
    openblas_set_num_threads(1);
    int p = para->p;
    int s = para->sparsity;
    int *sorted_ind;
    double gamma = para->gamma;
    double l2_lambda = para->l2_lambda;
    double *x_tr = para->x_tr;
    double *y_tr = para->y_tr;

    double *wt, *wt_bar, *loss_grad, *wt_tmp, *gt_bar, *p_prob, *p_label;


    wt = malloc(sizeof(double) * (p + 1));
    wt_bar = malloc(sizeof(double) * (p + 1));
    gt_bar = malloc(sizeof(double) * (p + 1));
    loss_grad = malloc(sizeof(double) * (p + 2));
    wt_tmp = malloc(sizeof(double) * (p + 1));
    sorted_ind = malloc(sizeof(int) * p);

    cblas_dcopy(p + 1, para->w0, 1, wt, 1);     // w0 --> wt
    cblas_dcopy(p + 1, para->w0, 1, wt_bar, 1); // w0 --> wt_bar
    cblas_dscal(p + 1, 0.0, gt_bar, 1);         // 0  --> gt_bar

    for (int tt = 0; tt < para->num_tr; tt++) {
        // 1.   unlabeled example x_i = x_tr + tt*p arrives and
        //      make prediction based on existing costs wt, wt_bar
        double *x_i = x_tr + tt * p;
        double *y_i = y_tr + tt;
        p_prob = &stat->p_prob_wt[tt];
        p_label = &stat->p_label_wt[tt];
        logistic_predict(x_i, wt, p_prob, p_label, 0.5, 1, p);
        p_prob = &stat->p_prob_wt_bar[tt];
        p_label = &stat->p_label_wt_bar[tt];
        logistic_predict(x_i, wt_bar, p_prob, p_label, 0.5, 1, p);
        // 2. observe y_i = y_tr + tt and have some loss
        logistic_loss_grad(wt, x_i, y_i, loss_grad, l2_lambda, 1, p);
        stat->losses[tt] = loss_grad[0];
        cblas_dscal(p + 1, tt / (tt + 1.), gt_bar, 1);
        cblas_daxpy(p + 1, 1. / (tt + 1.), loss_grad + 1, 1, gt_bar, 1);
        if (stat->p_label_wt[tt] != y_tr[tt]) {
            stat->total_missed_wt++;
        }
        stat->missed_wt[tt] = stat->total_missed_wt;
        if (stat->p_label_wt_bar[tt] != y_tr[tt]) {
            stat->total_missed_wt_bar++;
        }
        stat->missed_wt_bar[tt] = stat->total_missed_wt_bar;
        // 3. update model
        double learning_rate = -1. / gamma * sqrt(tt + 1.);
        arg_magnitude_sort_descend(gt_bar, sorted_ind, p);
        cblas_dcopy(p + 1, gt_bar, 1, wt_tmp, 1); // gt_bar --> wt_tmp
        cblas_dscal(p + 1, learning_rate, wt_tmp, 1);
        cblas_dscal(p + 1, 0.0, wt, 1);
        for (int i = 0; i < s; i++) {
            wt[sorted_ind[i]] = wt_tmp[sorted_ind[i]];
        }
        wt[p] = wt_tmp[p];
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
    free(wt), free(wt_bar), free(wt_tmp);
    free(loss_grad), free(sorted_ind), free(gt_bar);
    return true;
}


bool algo_online_da_iht_least_square(da_iht_para *para, OnlineStat *stat) {
    clock_t start_total = clock();
    openblas_set_num_threads(1);
    int p = para->p;
    int s = para->sparsity;
    int *sorted_ind;
    double gamma = para->gamma;
    double l2_lambda = para->l2_lambda;
    double *x_tr = para->x_tr;
    double *y_tr = para->y_tr;

    double *wt, *wt_bar, *loss_grad, *wt_tmp, *gt_bar, *p_prob, *p_label;


    wt = malloc(sizeof(double) * (p + 1));
    wt_bar = malloc(sizeof(double) * (p + 1));
    gt_bar = malloc(sizeof(double) * (p + 1));
    loss_grad = malloc(sizeof(double) * (p + 2));
    wt_tmp = malloc(sizeof(double) * (p + 1));
    sorted_ind = malloc(sizeof(int) * p);

    cblas_dcopy(p + 1, para->w0, 1, wt, 1);     // w0 --> wt
    cblas_dcopy(p + 1, para->w0, 1, wt_bar, 1); // w0 --> wt_bar
    cblas_dscal(p + 1, 0.0, gt_bar, 1);         // 0  --> gt_bar

    for (int tt = 0; tt < para->num_tr; tt++) {
        // 1.   unlabeled example x_i = x_tr + tt*p arrives and
        //      make prediction based on existing costs wt, wt_bar
        double *x_i = x_tr + tt * p;
        double *y_i = y_tr + tt;
        p_prob = &stat->p_prob_wt[tt];
        p_label = &stat->p_label_wt[tt];
        least_square_predict(x_i, wt, p_prob, p_label, 0.5, 1, p);
        p_prob = &stat->p_prob_wt_bar[tt];
        p_label = &stat->p_label_wt_bar[tt];
        least_square_predict(x_i, wt_bar, p_prob, p_label, 0.5, 1, p);
        // 2. observe y_i = y_tr + tt and have some loss
        least_square_loss_grad(wt, x_i, y_i, loss_grad, l2_lambda, 1, p);
        stat->losses[tt] = loss_grad[0];
        cblas_dscal(p + 1, tt / (tt + 1.), gt_bar, 1);
        cblas_daxpy(p + 1, 1. / (tt + 1.), loss_grad + 1, 1, gt_bar, 1);
        if (stat->p_label_wt[tt] != y_tr[tt]) {
            stat->total_missed_wt++;
        }
        stat->missed_wt[tt] = stat->total_missed_wt;
        if (stat->p_label_wt_bar[tt] != y_tr[tt]) {
            stat->total_missed_wt_bar++;
        }
        stat->missed_wt_bar[tt] = stat->total_missed_wt_bar;
        // 3. update model
        double learning_rate = -1. / gamma * sqrt(tt + 1.);
        arg_magnitude_sort_descend(gt_bar, sorted_ind, p);
        cblas_dcopy(p + 1, gt_bar, 1, wt_tmp, 1); // gt_bar --> wt_tmp
        cblas_dscal(p + 1, learning_rate, wt_tmp, 1);
        cblas_dscal(p + 1, 0.0, wt, 1);
        for (int i = 0; i < s; i++) {
            wt[sorted_ind[i]] = wt_tmp[sorted_ind[i]];
        }
        wt[p] = wt_tmp[p];
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
    free(wt), free(wt_bar), free(wt_tmp);
    free(loss_grad), free(sorted_ind), free(gt_bar);
    return true;
}
