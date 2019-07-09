//
//

#include <cblas.h>
#include "algo_online_rda_l1.h"
#include "loss.h"

#define sign(x) ((x > 0) -(x < 0))


double l2_norm_(const double *x, int x_len) {
    double result = 0.0;
    for (int i = 0; i < x_len; i++) {
        result += x[i] * x[i];
    }
    return sqrt(result);
}

bool algo_online_rda_l1_least_square(rda_l1_para *para, OnlineStat *stat) {
    openblas_set_num_threads(1);
    clock_t start_time = clock();
    int p = para->p, num_tr = para->num_tr;
    double *gt_bar, *wt, *wt_bar, *loss_grad;
    double *x_tr = para->x_tr, *y_tr = para->y_tr, *w0 = para->w0;
    double *p_prob, *p_label, wei, lambda_t_rda;
    double rho = para->rho, gamma = para->gamma, lambda = para->lambda;

    gt_bar = malloc(sizeof(double) * (p + 1));
    wt = malloc(sizeof(double) * (p + 1));
    wt_bar = malloc(sizeof(double) * (p + 1));
    loss_grad = malloc(sizeof(double) * (p + 2));

    cblas_dcopy(p + 1, w0, 1, wt, 1);         // wt --> wt_bar
    cblas_dcopy(p + 1, w0, 1, wt_bar, 1);     // wt --> wt_bar
    cblas_dscal(p + 1, 0.0, gt_bar, 1);       // zero vector

    for (int tt = 0; tt < num_tr; tt++) {
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
        // 2.   observe y_i = y_tr + tt and have some loss to calculate
        //      the subgradient of f.
        least_square_loss_grad(wt, x_i, y_i, loss_grad, 0.0, 1, p);
        stat->losses[tt] = loss_grad[0];
        if (stat->p_label_wt[tt] != y_tr[tt]) {
            stat->total_missed_wt++;
        }
        stat->missed_wt[tt] = stat->total_missed_wt;
        if (stat->p_label_wt_bar[tt] != y_tr[tt]) {
            stat->total_missed_wt_bar++;
        }
        stat->missed_wt_bar[tt] = stat->total_missed_wt_bar;
        // 3.   compute the dual average.
        cblas_dscal(p + 1, tt / (tt + 1.), gt_bar, 1);
        cblas_daxpy(p + 1, 1. / (tt + 1.), loss_grad + 1, 1, gt_bar, 1);
        // 4.   update the model: enhanced l1-rda method. Equation (30)
        wei = -sqrt(tt + 1.) / gamma;
        lambda_t_rda = lambda + (gamma * rho) / sqrt(tt + 1.);
        for (int i = 0; i < p; i++) {
            if (fabs(gt_bar[i]) <= lambda_t_rda) {
                wt[i] = 0.0; //thresholding entries
            } else {
                wt[i] = wei * (gt_bar[i] - lambda_t_rda * sign(gt_bar[i]));
            }
        }
        // Notice: the bias term do not need to add regularization.
        wt[p] = wei * gt_bar[p];
        // 5.   online to batch conversion
        cblas_dscal(p + 1, tt / (tt + 1.), wt_bar, 1);
        cblas_daxpy(p + 1, 1. / (tt + 1.), wt, 1, wt_bar, 1);
        for (int i = 0; i < p; i++) {
            if (wt[i] != 0.0) {
                stat->nonzeros_wt[tt] += 1;
            }
            if (wt_bar[i] != 0.0) {
                stat->nonzeros_wt_bar[tt] += 1;
            }
        }
    }// finish all training examples.
    for (int i = 0; i < p + 1; i++) {
        stat->wt[i] = wt[i];
        stat->wt_bar[i] = wt_bar[i];
    }
    stat->total_time = (double) (clock() - start_time) / CLOCKS_PER_SEC;
    free(loss_grad), free(gt_bar), free(wt), free(wt_bar);
    return true;
}


bool algo_online_rda_l1_logit(rda_l1_para *para, OnlineStat *stat) {
    clock_t start_time = clock();
    openblas_set_num_threads(1);
    int p = para->p;
    int num_tr = para->num_tr;
    double *x_tr = para->x_tr;
    double *y_tr = para->y_tr;
    double *w0 = para->w0;
    double rho = para->rho;
    double gamma = para->gamma;
    double lambda = para->lambda;
    double *gt_bar = malloc(sizeof(double) * (p + 1));
    double *wt = malloc(sizeof(double) * (p + 1));
    double *wt_bar = malloc(sizeof(double) * (p + 1));
    double *loss_grad = malloc(sizeof(double) * (p + 2));
    double *p_prob, *p_label, wei, lambda_t_rda;

    cblas_dcopy(p + 1, w0, 1, wt, 1);         // wt --> wt_bar
    cblas_dcopy(p + 1, w0, 1, wt_bar, 1);     // wt --> wt_bar
    cblas_dscal(p + 1, 0.0, gt_bar, 1);

    for (int tt = 0; tt < num_tr; tt++) {
        // 1. unlabeled example x_i = x_tr + tt*p arrives and
        //    make prediction based on existing costs wt, wt_bar
        double *x_i = x_tr + tt * p;
        double *y_i = y_tr + tt;
        p_prob = &stat->p_prob_wt[tt];
        p_label = &stat->p_label_wt[tt];
        logistic_predict(x_i, wt, p_prob, p_label, 0.5, 1, p);
        p_prob = &stat->p_prob_wt_bar[tt];
        p_label = &stat->p_label_wt_bar[tt];
        logistic_predict(x_i, wt_bar, p_prob, p_label, 0.5, 1, p);
        // 2.   observe y_i = y_tr + tt and have some loss to calculate
        //      the subgradient of f.
        // printf("l2(xi):%.6e, yi:%.1e\n", l2_norm_(x_i, p), *y_i);
        // printf("l2(wt):%.6e, loss:%.6e\n", l2_norm_(wt, p), loss_grad[p + 1]);
        logistic_loss_grad(wt, x_i, y_i, loss_grad, 0.0, 1, p);
        // printf("l2(wt):%.6e, loss:%.6e\n", l2_norm_(wt, p), loss_grad[p + 1]);
        stat->losses[tt] = loss_grad[0];
        if (stat->p_label_wt[tt] != y_tr[tt]) {
            stat->total_missed_wt++;
        }
        stat->missed_wt[tt] = stat->total_missed_wt;
        if (stat->p_label_wt_bar[tt] != y_tr[tt]) {
            stat->total_missed_wt_bar++;
        }
        stat->missed_wt_bar[tt] = stat->total_missed_wt_bar;
        // 3.   compute the dual average.
        cblas_dscal(p + 1, tt / (tt + 1.), gt_bar, 1);
        cblas_daxpy(p + 1, 1. / (tt + 1.), loss_grad + 1, 1, gt_bar, 1);
        // 4.   update the model: enhanced l1-rda method. Equation (30)
        wei = -sqrt(tt + 1.) / gamma;
        lambda_t_rda = lambda + (gamma * rho) / sqrt(tt + 1.);
        for (int i = 0; i < p; i++) {
            if (fabs(gt_bar[i]) <= lambda_t_rda) {
                wt[i] = 0.0;    // thresholding entries
            } else {
                wt[i] = wei * (gt_bar[i] - lambda_t_rda * sign(gt_bar[i]));
            }
        }
        // Notice: the bias term do not need to add regularization.
        wt[p] = wei * gt_bar[p];
        // 5.   online to batch conversion
        cblas_dscal(p + 1, tt / (tt + 1.), wt_bar, 1);
        cblas_daxpy(p + 1, 1. / (tt + 1.), wt, 1, wt_bar, 1);
        for (int i = 0; i < p; i++) {
            if (wt[i] != 0.0) {
                stat->nonzeros_wt[tt] += 1;
            }
            if (wt_bar[i] != 0.0) {
                stat->nonzeros_wt_bar[tt] += 1;
            }
        }
    }// finish all training examples.
    for (int i = 0; i < p + 1; i++) {
        stat->wt[i] = wt[i];
        stat->wt_bar[i] = wt_bar[i];
    }
    stat->total_time = (double) (clock() - start_time) / CLOCKS_PER_SEC;
    free(loss_grad), free(gt_bar), free(wt), free(wt_bar);
    return true;
}
