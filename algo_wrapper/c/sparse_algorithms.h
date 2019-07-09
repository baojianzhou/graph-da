//
//

#ifndef FAST_PCST_SPARSE_ALGORITHMS_H
#define FAST_PCST_SPARSE_ALGORITHMS_H

#include "head_tail_proj.h"


typedef enum LossFunc {
    LeastSquare,
    Logistic
} LossFunc;

typedef struct {
    double *x_tr;
    double *y_tr;
    double *w0;
    int p;
    int num_tr;
    double lambda;
    double gamma;
    double rho;
    int verbose;
    int loss_func;
} rda_l1_para;

typedef struct {
    double *x_tr;
    double *y_tr;
    double *w0;
    int p;
    int num_tr;
    double lambda;
    double gamma;
    int *group_list;
    int num_group;
    int *group_size_list;
    int verbose;
    int loss_func;
} da_gl_para;


typedef struct {
    double *x_tr;
    double *y_tr;
    double *w0;
    int p;
    int num_tr;
    double lambda;
    double gamma;
    int *group_list;
    int num_group;
    int *group_size_list;
    int *r;
    int verbose;
    int loss_func;
} da_sgl_para;

typedef struct {
    double *x_tr;
    double *y_tr;
    double *w0;
    int p;
    int num_tr;
    double eta;
    double delta;
    double lambda;
    int verbose;
    int loss_func;
} ada_grad_para;

typedef struct {
    double *x_tr;
    double *y_tr;
    double *w0;
    int p;
    int num_tr;
    double alpha;
    double beta1;
    double beta2;
    double epsilon;
    int loss_func;
    int verbose;
} adam_para;


//x_tr, y_tr, w0, p, n, lr, l2_lambda, verbose
typedef struct {
    double *x_tr;
    double *y_tr;
    double *w0;
    int p;
    int num_tr;
    double lr;
    double eta;
    int verbose;
} sgd_l2_para;

typedef struct {
    double *x_tr;
    double *y_tr;
    double *w0;
    int p;
    int num_tr;
    double gamma;
    double lambda;
    int verbose;
} sgd_l1_para;

typedef struct {
    double *x_tr;
    double *y_tr;
    EdgePair *edges;
    double *weights;
    double head_budget;
    double head_err_tol;
    int head_max_iter;
    int tail_max_iter;
    double tail_budget;
    double tail_nu;
    double tail_err_tol;
    double head_delta;
    double pcst_epsilon;
    int g;
    double *w0;
    int m;
    int p;
    int num_tr;
    int sparsity;
    int verbose;
    double gamma;
    double l2_lambda;
    int loss_func;
} graph_da_iht_para;


typedef struct {
    double *x_tr;
    double *y_tr;
    EdgePair *edges;
    double *w0;
    double *weights;
    int sparsity_low;
    int sparsity_high;
    int max_num_iter;
    int g;
    int root;
    int m;
    int p;
    int num_tr;
    int verbose;
    double gamma;
    double l2_lambda;
    int loss_func;
} graph_da_iht_para_2;


typedef struct {
    double *x_tr;
    double *y_tr;
    double *w0;
    int p;
    int num_tr;
    int sparsity;
    int verbose;
    double gamma;
    double l2_lambda;
    int loss_func;
    int *best_subset;
    int best_subset_len;
} best_subset;


typedef struct {
    double *x_tr;
    double *y_tr;
    EdgePair *edges;
    double *weights;
    double *w0;
    int m;
    int p;
    int num_tr;
    int sparsity_low;
    int sparsity_high;
    int max_num_iter;
    int g;
    int root;
    int verbose;
    double lr;
    double l2_lambda;
    int loss_func;
} graph_iht_para;

typedef struct {
    EdgePair *edges;
    double *costs;
    double *prizes;
    int m;
    int p;
    int num_tr;
    int sparsity_low;
    int sparsity_high;
    int max_num_iter;
    int g;
    int root;
    int verbose;
} head_tail_binsearch_para;

typedef struct {
    double *x_tr;
    double *y_tr;
    double *w0;
    int p;
    int num_tr;
    int sparsity;
    int verbose;
    double lr;
    double l2_lambda;
    int loss_func;
} iht_para;

typedef struct {
    double *x_tr;
    double *y_tr;
    double *w0;
    int p;
    int num_tr;
    int sparsity;
    int verbose;
    double lr;
    double l2_lambda;
    LossFunc loss_func;
} sto_iht_para;

typedef struct {
    double *x_tr;
    double *y_tr;
    double *w0;
    int p;
    int num_tr;
    int sparsity;
    int verbose;
    double gamma; // the learning rate para, where beta_t = gamma * \sqrt(tt)
    double l2_lambda; // regularization of l2-norm.
    int loss_func;
} da_iht_para;


typedef struct {
    double *wt;
    double *wt_bar;
    int *nonzeros_wt;
    int *nonzeros_wt_bar;
    double *p_prob_wt;
    double *p_label_wt;
    double *p_prob_wt_bar;
    double *p_label_wt_bar;
    double *losses;
    int *missed_wt;
    int *missed_wt_bar;
    int total_missed_wt;
    int total_missed_wt_bar;
    double total_time;
    Array *re_nodes;
    Array *re_edges;
    int num_pcst;
    double run_time_head;
    double run_time_tail;
} OnlineStat;

typedef struct {
    double *wt;
    double *wt_bar;
    int *nonzeros_wt;
    int *nonzeros_wt_bar;
    double *losses;
    double total_time;
    Array *re_nodes;
    Array *re_edges;
    int num_pcst;
    double run_time_head;
    double run_time_tail;
} StochasticStat;


OnlineStat *make_online_stat(int p, int num_tr);

bool free_online_stat(OnlineStat *online_stat);

StochasticStat *make_stochastic_stat(int p, int num_tr);

bool free_stochastic_stat(StochasticStat *online_stat);

/**
 *
 * @param para related parameters
 * @param stat results
 * @return
 */

double l2_norm(const double *x, int x_len);


void predict_logistic(OnlineStat *stat, int tt, int p, double y_i,
                      double *x_i, double *wt, double *wt_bar);

void predict_least_square(OnlineStat *stat, int tt, int p, double y_i,
                          double *x_i, double *wt, double *wt_bar);

bool algo_online_sgd_l1_logit(sgd_l1_para *para, OnlineStat *stat);

bool algo_online_sgd_l2_logit(sgd_l2_para *para, OnlineStat *stat);

bool algo_online_graph_ghtp_logit(
        const EdgePair *edges, const double *costs, int g, int s, int p, int m,
        const double *x_tr, const double *y_tr, const double *w0, int num_tr,
        double lr, double eta, int verbose, OnlineStat *stat);

bool algo_online_ghtp_logit(
        const double *x_tr, const double *y_tr, const double *w0, int s, int p,
        int num_tr, double lr, double eta, int verbose, double *wt,
        double *wt_bar, int *nonzeros_wt, int *nonzeros_wt_bar,
        double *pred_prob_wt, double *pred_label_wt, double *pred_prob_wt_bar,
        double *pred_label_wt_bar, double *losses, int *missed_wt,
        int *missed_wt_bar, double *total_time);

bool algo_online_ghtp_logit_sparse(
        const double *x_tr, const double *y_tr, const double *w0, int s, int p,
        int num_tr, double lr, double eta, int verbose, double *wt,
        double *wt_bar, int *nonzeros_wt, int *nonzeros_wt_bar,
        double *pred_prob_wt, double *pred_label_wt, double *pred_prob_wt_bar,
        double *pred_label_wt_bar, double *losses, int *missed_wt,
        int *missed_wt_bar, double *total_time);

bool algo_batch_ghtp_logit(
        const double *x_tr, const double *y_tr, const double *w0, int s, int p,
        int num_tr, double lr, double eta, int max_iter, double tol,
        int verbose, double *wt, double *losses, double *total_time);

bool algo_batch_iht_logit(
        const double *x_tr, const double *y_tr, const double *w0, int s,
        int p, int num_tr, double lr, double eta, int max_iter, double tol,
        int verbose, double *wt, double *losses, double *total_time);

bool algo_batch_graph_ghtp_logit(
        const EdgePair *edges, const double *costs, int g, int s, int p, int m,
        const double *x_tr, const double *y_tr, const double *w0, int num_tr,
        double tol, int max_iter, double lr, double eta, int verbose,
        double *wt, Array *re_nodes, double *losses, double *run_time_head,
        double *run_time_tail, double *total_time);

bool algo_batch_graph_iht_logit(
        const EdgePair *edges, const double *costs, int g, int s, int p, int m,
        const double *x_tr, const double *y_tr, const double *w0, int num_tr,
        double tol, int max_iter, double lr, double eta, int verbose,
        double *wt, Array *re_nodes, double *losses, double *run_time_head,
        double *run_time_tail, double *total_time);

bool algo_batch_graph_posi_logit(
        const EdgePair *edges, const double *costs, int g, int s, int p, int m,
        const double *x_tr, const double *y_tr, const double *w0, int num_tr,
        double tol, int max_iter, double lr, double eta, int verbose,
        double *wt, Array *re_nodes, double *losses, double *run_time_head,
        double *run_time_tail, double *total_time);

#endif //FAST_PCST_SPARSE_ALGORITHMS_H
