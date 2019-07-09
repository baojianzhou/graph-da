//
//
#include <cblas.h>
#include "loss.h"
#include "sort.h"
#include "algo_sto_iht.h"

void loss_grad(const double *w, const double *x_tr, const double *y_tr,
               double *loss_grad, double l2_reg, int n_samples,
               int n_features, LossFunc loss) {
    if (loss == Logistic) {
        logistic_loss_grad(w, x_tr, y_tr, loss_grad, l2_reg, n_samples,
                           n_features);
    } else if (loss == LeastSquare) {
        logistic_loss_grad(w, x_tr, y_tr, loss_grad, l2_reg, n_samples,
                           n_features);
    }
}

bool algo_sto_iht(sto_iht_para *para, StochasticStat *stat) {
    clock_t start_total = clock();
    openblas_set_num_threads(1);
    int p = para->p;
    int s = para->sparsity;
    int num_tr = para->num_tr;
    int *sorted_ind;
    double *wt;
    double *wt_bar;
    double *loss_grad;
    double *wt_tmp;
    double eta = para->l2_lambda;
    double lr = para->lr;
    double *x_tr = para->x_tr; // data matrix
    double *y_tr = para->y_tr; // measurements(training labels)

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
        // 2. observe y_i = y_tr + tt and have some loss
        logistic_loss_grad(wt, x_i, y_i, loss_grad, eta, 1, p);
        stat->losses[tt] = loss_grad[0];
        // 3. update wt= wt - lr/sqrt(tt) * grad(f_xi, wt)
        cblas_dcopy(p + 1, wt, 1, wt_tmp, 1); // wt --> wt_tmp
        cblas_daxpy(p + 1, -lr, loss_grad + 1, 1, wt_tmp,
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
