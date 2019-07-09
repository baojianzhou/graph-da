//
//
#include "loss.h"
#include<cblas.h>
#include "algo_online_adam.h"

bool algo_online_adam_logit(adam_para *para, OnlineStat *stat) {

    clock_t start_total = clock();
    openblas_set_num_threads(1);
    int p = para->p;
    int num_tr = para->num_tr;
    double *w0 = para->w0;
    double *x_tr = para->x_tr;
    double *y_tr = para->y_tr;
    double beta1 = para->beta1;
    double beta2 = para->beta2;
    double alpha = para->alpha;
    double epsilon = para->epsilon;
    double beta1_t = beta1;
    double beta2_t = beta2;
    double *p_prob, *p_label;

    double *wt = malloc((p + 1) * sizeof(double));
    double *wt_bar = malloc((p + 1) * sizeof(double));
    double *loss_grad = malloc((p + 2) * sizeof(double));
    double *gt_square = malloc((p + 1) * sizeof(double));
    double *m_vec = malloc((p + 1) * sizeof(double));
    double *v_vec = malloc((p + 1) * sizeof(double));

    cblas_dcopy(p + 1, w0, 1, wt, 1); // w0 --> wt
    cblas_dcopy(p + 1, w0, 1, gt_square, 1); // w0 --> wt
    cblas_dcopy(p + 1, w0, 1, wt_bar, 1); // w0 --> wt_bar
    cblas_dscal(p + 1, 0.0, m_vec, 1);
    cblas_dscal(p + 1, 0.0, v_vec, 1);

    for (int tt = 0; tt < num_tr; tt++) {
        //1. receive a question x_t and predict a value.
        double *x_i = x_tr + tt * p;
        double *y_i = y_tr + tt;
        p_prob = &stat->p_prob_wt[tt], p_label = &stat->p_label_wt[tt];
        logistic_predict(x_i, wt, p_prob, p_label, 0.5, 1, p);
        p_prob = &stat->p_prob_wt_bar[tt], p_label = &stat->p_label_wt_bar[tt];
        logistic_predict(x_i, wt_bar, p_prob, p_label, 0.5, 1, p);

        // 2. observe y_i = y_tr + tt and have some loss
        logistic_loss_grad(wt, x_i, y_i, loss_grad, 0.0, 1, p);
        for (int i = 0; i < p + 1; i++) {
            gt_square[i] = loss_grad[i + 1] * loss_grad[i + 1];
        }
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
        cblas_dscal(p + 1, beta1, m_vec, 1);
        cblas_daxpy(p + 1, 1. - beta1, loss_grad + 1, 1, m_vec, 1);
        cblas_dscal(p + 1, beta2, v_vec, 1);
        cblas_daxpy(p + 1, 1. - beta2, gt_square, 1, v_vec, 1);

        for (int i = 0; i < p + 1; i++) {
            double numerator = alpha * m_vec[i];
            double denominator = (1. - beta1_t);
            denominator *= (sqrt(v_vec[i] / (1. - beta2_t)) + epsilon);
            wt[i] = wt[i] - numerator / denominator;
        }
        beta1_t = beta1_t * beta1;
        beta2_t = beta2_t * beta2;

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
    free(gt_square);
    free(v_vec);
    free(m_vec);
    return true;

}

bool algo_online_adam_least_square(adam_para *para, OnlineStat *stat) {
    clock_t start_total = clock();
    openblas_set_num_threads(1);
    int p = para->p;
    int num_tr = para->num_tr;
    double *w0 = para->w0;
    double *x_tr = para->x_tr;
    double *y_tr = para->y_tr;
    double beta1 = para->beta1;
    double beta2 = para->beta2;
    double alpha = para->alpha;
    double epsilon = para->epsilon;
    double beta1_t = beta1;
    double beta2_t = beta2;
    double *p_prob, *p_label;

    double *wt = malloc((p + 1) * sizeof(double));
    double *wt_bar = malloc((p + 1) * sizeof(double));
    double *loss_grad = malloc((p + 2) * sizeof(double));
    double *gt_square = malloc((p + 1) * sizeof(double));
    double *m_vec = malloc((p + 1) * sizeof(double));
    double *v_vec = malloc((p + 1) * sizeof(double));

    cblas_dcopy(p + 1, w0, 1, wt, 1); // w0 --> wt
    cblas_dcopy(p + 1, w0, 1, gt_square, 1); // w0 --> wt
    cblas_dcopy(p + 1, w0, 1, wt_bar, 1); // w0 --> wt_bar
    cblas_dscal(p + 1, 0.0, m_vec, 1);
    cblas_dscal(p + 1, 0.0, v_vec, 1);

    for (int tt = 0; tt < num_tr; tt++) {
        //1. receive a question x_t and predict a value.
        double *x_i = x_tr + tt * p;
        double *y_i = y_tr + tt;
        p_prob = &stat->p_prob_wt[tt], p_label = &stat->p_label_wt[tt];
        least_square_predict(x_i, wt, p_prob, p_label, 0.5, 1, p);
        p_prob = &stat->p_prob_wt_bar[tt], p_label = &stat->p_label_wt_bar[tt];
        least_square_predict(x_i, wt_bar, p_prob, p_label, 0.5, 1, p);

        // 2. observe y_i = y_tr + tt and have some loss
        least_square_loss_grad(wt, x_i, y_i, loss_grad, 0.0, 1, p);
        for (int i = 0; i < p + 1; i++) {
            gt_square[i] = loss_grad[i + 1] * loss_grad[i + 1];
        }
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
        cblas_dscal(p + 1, beta1, m_vec, 1);
        cblas_daxpy(p + 1, 1. - beta1, loss_grad + 1, 1, m_vec, 1);
        cblas_dscal(p + 1, beta2, v_vec, 1);
        cblas_daxpy(p + 1, 1. - beta2, gt_square, 1, v_vec, 1);

        for (int i = 0; i < p + 1; i++) {
            double numerator = alpha * m_vec[i];
            double denominator = (1. - beta1_t);
            denominator *= (sqrt(v_vec[i] / (1. - beta2_t)) + epsilon);
            wt[i] = wt[i] - numerator / denominator;
        }
        beta1_t = beta1_t * beta1;
        beta2_t = beta2_t * beta2;

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
    free(gt_square);
    free(v_vec);
    free(m_vec);
    return true;
}