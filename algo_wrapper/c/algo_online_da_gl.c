//
//
#include <cblas.h>
#include "loss.h"
#include "algo_online_da_gl.h"

#define sign(x) ((x > 0) -(x < 0))  // to determine : sign(x)
#define math_abs(x) (x > 0 ? x:-x)  // to calculate : abs(x)
#define max_posi(x) (x > 0 ? x:0)   // to calculate : max(x,0)

/**
 * Please do not call this method directly. Try to use Python wrapper instead.
 * @param para
 * @param stat
 * @return
 */
bool algo_online_da_gl_logit(da_gl_para *para, OnlineStat *stat) {
    clock_t start_total = clock();
    openblas_set_num_threads(1);
    int p = para->p, num_tr = para->num_tr;
    double gamma = para->gamma, lambda = para->lambda;
    double *w0 = para->w0, *wt, *wt_bar, *loss_grad, *ut_bar;
    double *x_tr = para->x_tr, *y_tr = para->y_tr, *p_prob, *p_label;

    wt = malloc((p + 1) * sizeof(double));
    wt_bar = malloc((p + 1) * sizeof(double));
    ut_bar = malloc((p + 1) * sizeof(double));
    loss_grad = malloc((p + 2) * sizeof(double));

    cblas_dcopy(p + 1, w0, 1, wt, 1); // w0 --> wt
    cblas_dcopy(p + 1, w0, 1, ut_bar, 1); // w0 --> wt
    cblas_dcopy(p + 1, w0, 1, wt_bar, 1); // w0 --> wt_bar

    for (int tt = 0; tt < num_tr; tt++) {
        //1.    receive a question x_t and predict a value.
        double *x_i = x_tr + tt * p;
        double *y_i = y_tr + tt;
        p_prob = &stat->p_prob_wt[tt];
        p_label = &stat->p_label_wt[tt];
        logistic_predict(x_i, wt, p_prob, p_label, 0.5, 1, p);
        p_prob = &stat->p_prob_wt_bar[tt];
        p_label = &stat->p_label_wt_bar[tt];
        logistic_predict(x_i, wt_bar, p_prob, p_label, 0.5, 1, p);
        // 2.   observe y_i = y_tr + tt and have some loss
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
        //3.    update the dual averaging:
        cblas_dscal(p + 1, tt / (tt + 1.), ut_bar, 1);
        cblas_daxpy(p + 1, 1 / (tt + 1.), loss_grad + 1, 1, ut_bar, 1);
        //4.    update the model: da-group lasso method. Equation (10)
        int g_index = 0; // feature starts from 0
        double wei = -(sqrt(tt + 1) / gamma);
        for (int g = 0; g < para->num_group; g++) {
            double norm_ut_bar = 0.0;
            int d_g = para->group_size_list[g];
            for (int j = 0; j < d_g; j++) {
                int var_i = para->group_list[g_index + j];
                norm_ut_bar += ut_bar[var_i] * ut_bar[var_i];
            }
            norm_ut_bar = sqrt(norm_ut_bar);
            double wei_2 = 1. - lambda * sqrt(d_g) / norm_ut_bar;
            for (int j = 0; j < d_g; j++) {
                int var_i = para->group_list[g_index + j];
                wt[var_i] = wei * (max_posi(wei_2)) * ut_bar[var_i];
            }
            g_index += d_g;
        }
        // Notice: the bias term do not need to add regularization.
        wt[p] = wei * ut_bar[p];
        //5.    online to batch conversion.
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
    }// finish all training examples.
    for (int i = 0; i < p + 1; i++) {
        stat->wt[i] = wt[i];
        stat->wt_bar[i] = wt_bar[i];
    }
    stat->total_time = (double) (clock() - start_total) / CLOCKS_PER_SEC;
    free(loss_grad), free(wt), free(wt_bar), free(ut_bar);
    return true;
}

bool algo_online_da_gl_least_square(da_gl_para *para, OnlineStat *stat) {
    clock_t start_total = clock();
    openblas_set_num_threads(1);
    int p = para->p, num_tr = para->num_tr;
    double gamma = para->gamma, lambda = para->lambda;
    double *w0 = para->w0, *wt, *wt_bar, *loss_grad, *ut_bar;
    double *x_tr = para->x_tr, *y_tr = para->y_tr, *p_prob, *p_label;

    wt = malloc((p + 1) * sizeof(double));
    wt_bar = malloc((p + 1) * sizeof(double));
    ut_bar = malloc((p + 1) * sizeof(double));
    loss_grad = malloc((p + 2) * sizeof(double));

    cblas_dcopy(p + 1, w0, 1, wt, 1);       // w0 --> wt
    cblas_dcopy(p + 1, w0, 1, ut_bar, 1);   // w0 --> ut_bar
    cblas_dcopy(p + 1, w0, 1, wt_bar, 1);   // w0 --> wt_bar

    for (int tt = 0; tt < num_tr; tt++) {
        //1.    receive a question x_t and predict a value.
        double *x_i = x_tr + tt * p;
        double *y_i = y_tr + tt;
        p_prob = &stat->p_prob_wt[tt];
        p_label = &stat->p_label_wt[tt];
        least_square_predict(x_i, wt, p_prob, p_label, 0.5, 1, p);
        p_prob = &stat->p_prob_wt_bar[tt];
        p_label = &stat->p_label_wt_bar[tt];
        least_square_predict(x_i, wt_bar, p_prob, p_label, 0.5, 1, p);
        // 2.   observe y_i = y_tr + tt and have some loss
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
        //3.    update the dual averaging:
        cblas_dscal(p + 1, tt / (tt + 1.), ut_bar, 1);
        cblas_daxpy(p + 1, 1 / (tt + 1.), loss_grad + 1, 1, ut_bar, 1);
        //4.    update the model: da-group lasso method. Equation (10)
        int g_index = 0; // feature starts from 0
        double wei = -(sqrt(tt + 1) / gamma);
        for (int g = 0; g < para->num_group; g++) {
            double norm_ut_bar = 0.0;
            int d_g = para->group_size_list[g];
            for (int j = 0; j < d_g; j++) {
                int var_i = para->group_list[g_index + j];
                norm_ut_bar += ut_bar[var_i] * ut_bar[var_i];
            }
            norm_ut_bar = sqrt(norm_ut_bar);
            double wei_2 = 1. - lambda * sqrt(d_g) / norm_ut_bar;
            for (int j = 0; j < d_g; j++) {
                int var_i = para->group_list[g_index + j];
                wt[var_i] = wei * (max_posi(wei_2)) * ut_bar[var_i];
            }
            g_index += d_g;
        }
        // Notice: the bias term do not need to add into regularization.
        wt[p] = wei * ut_bar[p];
        //5.    online to batch conversion.
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
    }// finish all training examples.
    for (int i = 0; i < p + 1; i++) {
        stat->wt[i] = wt[i];
        stat->wt_bar[i] = wt_bar[i];
    }
    stat->total_time = (double) (clock() - start_total) / CLOCKS_PER_SEC;
    free(loss_grad), free(wt), free(wt_bar), free(ut_bar);
    return true;
}

/**
 * Please do not call this method directly. Try to use Python wrapper instead.
 * @param para
 * @param stat
 * @return
 */
bool algo_online_da_sgl_logit(da_sgl_para *para, OnlineStat *stat) {
    openblas_set_num_threads(1);
    clock_t start_total = clock();
    int p = para->p, num_tr = para->num_tr;
    double gamma = para->gamma, lambda = para->lambda;
    double *w0 = para->w0, *wt, *wt_bar, *loss_grad, *ut_bar, *ct_bar;
    double *x_tr = para->x_tr, *y_tr = para->y_tr, *p_prob, *p_label;

    wt = malloc((p + 1) * sizeof(double));
    wt_bar = malloc((p + 1) * sizeof(double));
    ut_bar = malloc((p + 1) * sizeof(double));
    ct_bar = malloc((p + 1) * sizeof(double));
    loss_grad = malloc((p + 2) * sizeof(double));

    cblas_dcopy(p + 1, w0, 1, wt, 1); // w0 --> wt
    cblas_dcopy(p + 1, w0, 1, ut_bar, 1); // w0 --> wt
    cblas_dcopy(p + 1, w0, 1, ct_bar, 1); // w0 --> wt
    cblas_dcopy(p + 1, w0, 1, wt_bar, 1); // w0 --> wt_bar

    for (int tt = 0; tt < num_tr; tt++) {
        //1. receive a question x_t and predict a value.
        double *x_i = x_tr + tt * p;
        double *y_i = y_tr + tt;
        p_prob = &stat->p_prob_wt[tt];
        p_label = &stat->p_label_wt[tt];
        logistic_predict(x_i, wt, p_prob, p_label, 0.5, 1, p);
        p_prob = &stat->p_prob_wt_bar[tt];
        p_label = &stat->p_label_wt_bar[tt];
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
        // 3.   compute the dual average.
        cblas_dscal(p + 1, tt / (tt + 1.), ut_bar, 1);
        cblas_daxpy(p + 1, 1 / (tt + 1.), loss_grad + 1, 1, ut_bar, 1);
        // 4.   update the model: enhanced l1-rda method. Equation (11)
        int g_index = 0; // feature starts from 0
        double wei = -sqrt(tt + 1.) / gamma;
        for (int i = 0; i < para->num_group; i++) {
            double norm_ct_bar = 0.0;
            int d_g = para->group_size_list[i];
            for (int j = 0; j < d_g; j++) {
                int var_i = para->group_list[g_index + j];
                double wei1 = math_abs(ut_bar[var_i]) - lambda * para->r[i];
                ct_bar[var_i] = max_posi(wei1) * sign(ut_bar[var_i]);
                norm_ct_bar += ct_bar[var_i] * ct_bar[var_i];
            }
            norm_ct_bar = sqrt(norm_ct_bar);
            double wei_2 = 1. - lambda * sqrt(d_g) / norm_ct_bar;
            for (int j = 0; j < d_g; j++) {
                int var_i = para->group_list[g_index + j];
                wt[var_i] = wei * (max_posi(wei_2)) * ct_bar[var_i];
            }
            g_index += d_g;
        }
        wt[p] = wei * ut_bar[p];
        //5.    online to batch conversion.
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
    }// finish all training examples.
    for (int i = 0; i < p + 1; i++) {
        stat->wt[i] = wt[i];
        stat->wt_bar[i] = wt_bar[i];
    }
    stat->total_time = (double) (clock() - start_total) / CLOCKS_PER_SEC;
    free(loss_grad), free(wt), free(wt_bar), free(ut_bar), free(ct_bar);
    return true;
}

bool algo_online_da_sgl_least_square(da_sgl_para *para, OnlineStat *stat) {
    openblas_set_num_threads(1);
    clock_t start_total = clock();
    int p = para->p, num_tr = para->num_tr;
    double gamma = para->gamma, lambda = para->lambda;
    double *w0 = para->w0, *wt, *wt_bar, *loss_grad, *ut_bar, *ct_bar;
    double *x_tr = para->x_tr, *y_tr = para->y_tr, *p_prob, *p_label;

    wt = malloc((p + 1) * sizeof(double));
    wt_bar = malloc((p + 1) * sizeof(double));
    ut_bar = malloc((p + 1) * sizeof(double));
    ct_bar = malloc((p + 1) * sizeof(double));
    loss_grad = malloc((p + 2) * sizeof(double));

    cblas_dcopy(p + 1, w0, 1, wt, 1); // w0 --> wt
    cblas_dcopy(p + 1, w0, 1, ut_bar, 1); // w0 --> wt
    cblas_dcopy(p + 1, w0, 1, ct_bar, 1); // w0 --> wt
    cblas_dcopy(p + 1, w0, 1, wt_bar, 1); // w0 --> wt_bar

    for (int tt = 0; tt < num_tr; tt++) {
        //1. receive a question x_t and predict a value.
        double *x_i = x_tr + tt * p;
        double *y_i = y_tr + tt;
        p_prob = &stat->p_prob_wt[tt];
        p_label = &stat->p_label_wt[tt];
        least_square_predict(x_i, wt, p_prob, p_label, 0.5, 1, p);
        p_prob = &stat->p_prob_wt_bar[tt];
        p_label = &stat->p_label_wt_bar[tt];
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
        // 3.   compute the dual average.
        cblas_dscal(p + 1, tt / (tt + 1.), ut_bar, 1);
        cblas_daxpy(p + 1, 1 / (tt + 1.), loss_grad + 1, 1, ut_bar, 1);
        // 4.   update the model: enhanced l1-rda method. Equation (11)
        int g_index = 0; // feature starts from 0
        double wei = -sqrt(tt + 1.) / gamma;
        for (int i = 0; i < para->num_group; i++) {
            double norm_ct_bar = 0.0;
            int d_g = para->group_size_list[i];
            for (int j = 0; j < d_g; j++) {
                int var_i = para->group_list[g_index + j];
                double wei1 = math_abs(ut_bar[var_i]) - lambda * para->r[i];
                ct_bar[var_i] = max_posi(wei1) * sign(ut_bar[var_i]);
                norm_ct_bar += ct_bar[var_i] * ct_bar[var_i];
            }
            norm_ct_bar = sqrt(norm_ct_bar);
            double wei_2 = 1. - lambda * sqrt(d_g) / norm_ct_bar;
            for (int j = 0; j < d_g; j++) {
                int var_i = para->group_list[g_index + j];
                wt[var_i] = wei * (max_posi(wei_2)) * ct_bar[var_i];
            }
            g_index += d_g;
        }
        wt[p] = wei * ut_bar[p];
        //5.    online to batch conversion.
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
    }// finish all training examples.
    for (int i = 0; i < p + 1; i++) {
        stat->wt[i] = wt[i];
        stat->wt_bar[i] = wt_bar[i];
    }
    stat->total_time = (double) (clock() - start_total) / CLOCKS_PER_SEC;
    free(loss_grad), free(wt), free(wt_bar), free(ut_bar), free(ct_bar);
    return true;
}