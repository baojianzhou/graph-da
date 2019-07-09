//
//

#include <cblas.h>
#include "loss.h"
#include "algo_online_ada_grad.h"

#define sign(x) ((x > 0) -(x < 0))  // to determine : sign(x)
#define math_abs(x) (x > 0 ? x:-x)  // to calculate : abs(x)
#define max_posi(x) (x > 0 ? x:0)   // to calculate : max(x,0)

/**
 * Please do not call this method directly. Try to use Python wrapper instead.
 * @param para
 * @param stat
 * @return
 */
bool algo_online_ada_grad_logit(ada_grad_para *para, OnlineStat *stat) {
    clock_t start_total = clock();
    openblas_set_num_threads(1);
    int p = para->p, num_tr = para->num_tr;
    double *w0 = para->w0, *wt, *wt_bar, *loss_grad;
    double *x_tr = para->x_tr, *y_tr = para->y_tr, *p_prob, *p_label;

    wt = malloc((p + 1) * sizeof(double));
    wt_bar = malloc((p + 1) * sizeof(double));
    loss_grad = malloc((p + 2) * sizeof(double));

    double *gt = malloc((p + 1) * sizeof(double));
    double *gt_bar = malloc((p + 1) * sizeof(double));
    cblas_dcopy(p + 1, w0, 1, wt, 1); // w0 --> wt
    cblas_dcopy(p + 1, w0, 1, gt, 1); // w0 --> wt
    cblas_dcopy(p + 1, w0, 1, gt_bar, 1); // w0 --> wt
    cblas_dcopy(p + 1, w0, 1, wt_bar, 1); // w0 --> wt_bar

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
            gt[i] += loss_grad[i + 1] * loss_grad[i + 1];
        }
        cblas_dscal(p + 1, tt / (tt + 1.), gt_bar, 1);
        cblas_daxpy(p + 1, 1 / (tt + 1.), loss_grad + 1, 1, gt_bar, 1);
        for (int i = 0; i < p; i++) {
            double wei = sign(-gt_bar[i]) * para->eta * (tt + 1.) /
                         (para->delta + sqrt(gt[i]));
            double truncate = max_posi(math_abs(gt_bar[i]) - para->lambda);
            wt[i] = wei * truncate;
        }
        double wei = sign(-gt_bar[p]) * para->eta * (tt + 1.) /
                     (para->delta + sqrt(gt[p]));
        double truncate = max_posi(math_abs(gt_bar[p]));
        wt[p] = wei * truncate;
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
    free(loss_grad);
    free(wt);
    free(wt_bar);
    free(gt);
    free(gt_bar);
    return true;
}


/**
 * Please do not call this method directly. Try to use Python wrapper instead.
 * @param para
 * @param stat
 * @return
 */
bool algo_online_ada_grad_least_square(ada_grad_para *para, OnlineStat *stat) {
    clock_t start_total = clock();
    openblas_set_num_threads(1);
    int p = para->p, num_tr = para->num_tr;
    double *w0 = para->w0, *wt, *wt_bar, *loss_grad, *p_prob, *p_label;
    double *x_tr = para->x_tr;
    double *y_tr = para->y_tr;
    double eta = para->eta;
    double delta = para->delta;
    double lambda = para->lambda;

    wt = malloc((p + 1) * sizeof(double));
    wt_bar = malloc((p + 1) * sizeof(double));
    loss_grad = malloc((p + 2) * sizeof(double));

    double *gt = malloc((p + 1) * sizeof(double));
    double *gt_bar = malloc((p + 1) * sizeof(double));
    cblas_dcopy(p + 1, w0, 1, wt, 1);       // w0 --> wt
    cblas_dcopy(p + 1, w0, 1, gt, 1);       // w0 --> gt
    cblas_dcopy(p + 1, w0, 1, gt_bar, 1);   // w0 --> gt_bar
    cblas_dcopy(p + 1, w0, 1, wt_bar, 1);   // w0 --> wt_bar

    for (int tt = 0; tt < num_tr; tt++) {
        //1. receive a question x_t and predict a value.
        double *x_i = x_tr + tt * p, *y_i = y_tr + tt;
        p_prob = &stat->p_prob_wt[tt], p_label = &stat->p_label_wt[tt];
        least_square_predict(x_i, wt, p_prob, p_label, 0.5, 1, p);
        p_prob = &stat->p_prob_wt_bar[tt], p_label = &stat->p_label_wt_bar[tt];
        least_square_predict(x_i, wt_bar, p_prob, p_label, 0.5, 1, p);
        // 2. observe y_i = y_tr + tt and have some loss
        least_square_loss_grad(wt, x_i, y_i, loss_grad, 0.0, 1, p);
        if (stat->p_label_wt[tt] != y_tr[tt]) {
            stat->total_missed_wt++;
        }
        stat->missed_wt[tt] = stat->total_missed_wt;
        if (stat->p_label_wt_bar[tt] != y_tr[tt]) {
            stat->total_missed_wt_bar++;
        }
        stat->missed_wt_bar[tt] = stat->total_missed_wt_bar;
        stat->losses[tt] = loss_grad[0];
        //3.    update the model:
        for (int i = 0; i < p + 1; i++) {
            gt[i] += loss_grad[i + 1] * loss_grad[i + 1];
        }
        cblas_dscal(p + 1, tt / (tt + 1.), gt_bar, 1);
        cblas_daxpy(p + 1, 1 / (tt + 1.), loss_grad + 1, 1, gt_bar, 1);
        for (int i = 0; i < p; i++) {
            double wei = sign(-gt_bar[i]) * eta * (tt + 1.) /
                         (delta + sqrt(gt[i]));
            double truncate = max_posi(math_abs(gt_bar[i]) - lambda);
            wt[i] = wei * truncate;
        }
        double wei = sign(-gt_bar[p]) * eta * (tt + 1.) /
                     (delta + sqrt(gt[p]));
        double truncate = max_posi(math_abs(gt_bar[p]));
        wt[p] = wei * truncate;
        //4.    online to batch conversion.
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
