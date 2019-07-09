//
//
#include <cblas.h>
#include "loss.h"
#include "sort.h"
#include "algo_online_sto_iht.h"


bool algo_online_iht_logit(iht_para *para, OnlineStat *stat) {

    clock_t start_total = clock();
    openblas_set_num_threads(1);
    int p = para->p, s = para->sparsity, num_tr = para->num_tr, *sorted_ind;
    double *wt, *wt_bar, *loss_grad, *wt_tmp, eta = para->l2_lambda,
            gamma = para->lr;
    double *p_prob, *p_label, *x_tr = para->x_tr, *y_tr = para->y_tr;

    wt = malloc(sizeof(double) * (p + 1));
    wt_bar = malloc(sizeof(double) * (p + 1));
    loss_grad = malloc(sizeof(double) * (p + 2));
    wt_tmp = malloc(sizeof(double) * (p + 1));
    sorted_ind = malloc(sizeof(int) * p);

    cblas_dcopy(p + 1, para->w0, 1, wt, 1);      // w0 --> wt
    cblas_dcopy(p + 1, para->w0, 1, wt_bar, 1);  // w0 --> wt_bar

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
        // 3. update wt= wt - lr/sqrt(tt) * grad(f_xi, wt)
        cblas_dcopy(p + 1, wt, 1, wt_tmp, 1); // wt --> wt_tmp
        cblas_daxpy(p + 1, -gamma / sqrt(tt + 1.), loss_grad + 1, 1, wt_tmp,
                    1);
        // 4. projection step: select largest k entries.
        arg_magnitude_sort_descend(wt_tmp, sorted_ind, p);
        cblas_dscal(p + 1, 0.0, wt, 1);
        for (int i = 0; i < s; i++) {
            wt[sorted_ind[i]] = wt_tmp[sorted_ind[i]];
        }
        // 5. intercept is not a feature. keep the intercept in entry p.
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
    free(wt), free(wt_bar), free(loss_grad), free(wt_tmp), free(sorted_ind);
    return true;
}


bool algo_online_iht_least_square(iht_para *para, OnlineStat *stat) {

    clock_t start_total = clock();
    openblas_set_num_threads(1);
    int p = para->p;        // number of parameters.
    int s = para->sparsity; // sparsity parameter.
    int *sorted_ind;
    double l2_lambda = para->l2_lambda; // l2-regularization parameter.
    double *x_tr = para->x_tr;  // training dataset.
    double *y_tr = para->y_tr;  // training values.
    double *wt, *wt_bar, *loss_grad, *wt_tmp, *p_prob, *p_label;

    wt = malloc(sizeof(double) * (p + 1));
    wt_bar = malloc(sizeof(double) * (p + 1));
    loss_grad = malloc(sizeof(double) * (p + 2));
    wt_tmp = malloc(sizeof(double) * (p + 1));
    sorted_ind = malloc(sizeof(int) * p);

    cblas_dcopy(p + 1, para->w0, 1, wt, 1);      // w0 --> wt
    cblas_dcopy(p + 1, para->w0, 1, wt_bar, 1);  // w0 --> wt_bar

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

        if (stat->p_label_wt[tt] != y_tr[tt]) {
            stat->total_missed_wt++;
        }
        stat->missed_wt[tt] = stat->total_missed_wt;
        if (stat->p_label_wt_bar[tt] != y_tr[tt]) {
            stat->total_missed_wt_bar++;
        }
        stat->missed_wt_bar[tt] = stat->total_missed_wt_bar;

        // 3. update wt= wt - lr/sqrt(tt) * grad(f_xi, wt)
        cblas_dcopy(p + 1, wt, 1, wt_tmp, 1); // wt --> wt_tmp
        double wei = -para->lr / sqrt(tt + 1.);
        cblas_daxpy(p + 1, wei, loss_grad + 1, 1, wt_tmp, 1);
        // 4. projection step: select largest k entries.
        arg_magnitude_sort_descend(wt_tmp, sorted_ind, p);
        cblas_dscal(p + 1, 0.0, wt, 1);
        for (int i = 0; i < s; i++) {
            wt[sorted_ind[i]] = wt_tmp[sorted_ind[i]];
        }
        // 5. intercept is not a feature. keep the intercept in entry p.
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
    free(wt), free(wt_bar), free(loss_grad), free(wt_tmp), free(sorted_ind);
    return true;
}
