//
//

//
//
#include <Python.h>
#include <numpy/arrayobject.h>
#include "fast_pcst_v2.h"
#include "sort.h"

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
    Array *re_nodes;
    Array *re_edges;
    double *prizes;
    double *costs;
    int num_pcst;
    double run_time;
    int num_iter;
} GraphStat;


GraphStat *make_graph_stat(int p, int m) {
    GraphStat *stat = malloc(sizeof(GraphStat));
    stat->num_pcst = 0;
    stat->re_edges = malloc(sizeof(Array));
    stat->re_edges->size = 0;
    stat->re_edges->array = malloc(sizeof(int) * p);
    stat->re_nodes = malloc(sizeof(Array));
    stat->re_nodes->size = 0;
    stat->re_nodes->array = malloc(sizeof(int) * p);
    stat->run_time = 0;
    stat->costs = malloc(sizeof(double) * m);
    stat->prizes = malloc(sizeof(double) * p);
    return stat;
}

bool free_graph_stat(GraphStat *graph_stat) {
    free(graph_stat->re_nodes->array);
    free(graph_stat->re_nodes);
    free(graph_stat->re_edges->array);
    free(graph_stat->re_edges);
    free(graph_stat->costs);
    free(graph_stat->prizes);
    free(graph_stat);
    return true;
}


bool head_tail_binsearch(
        const EdgePair *edges, const double *costs, const double *prizes,
        int n, int m, int target_num_clusters, int root, int sparsity_low,
        int sparsity_high, int max_num_iter, PruningMethod pruning,
        int verbose, GraphStat *stat) {

    // malloc: cur_costs, sorted_prizes, and sorted_indices
    // free: cur_costs, sorted_prizes, and sorted_indices
    double *cur_costs = malloc(sizeof(double) * m);
    double *sorted_prizes = malloc(sizeof(double) * n);
    int *sorted_indices = malloc(sizeof(int) * n);
    for (int ii = 0; ii < m; ii++) {
        cur_costs[ii] = costs[ii];
    }
    for (int ii = 0; ii < n; ii++) {
        sorted_prizes[ii] = prizes[ii];
    }
    int guess_pos = n - sparsity_high;
    arg_sort_descend(sorted_prizes, sorted_indices, n);
    double lambda_low = 0.0;
    double lambda_high = 2.0 * sorted_prizes[sorted_indices[guess_pos]];
    bool using_sparsity_low = false;
    bool using_max_value = false;
    if (lambda_high == 0.0) {
        guess_pos = n - sparsity_low;
        lambda_high = 2.0 * sorted_prizes[sorted_indices[guess_pos]];
        if (lambda_high != 0.0) {
            using_sparsity_low = true;
        } else {
            using_max_value = true;
            lambda_high = prizes[0];
            for (int ii = 1; ii < n; ii++) {
                lambda_high = fmax(lambda_high, prizes[ii]);
            }
            lambda_high *= 2.0;
        }
    }
    if (verbose >= 1) {
        const char *sparsity_low_text = "k_low";
        const char *sparsity_high_text = "k_high";
        const char *max_value_text = "max value";
        const char *guess_text = sparsity_high_text;
        if (using_sparsity_low) {
            guess_text = sparsity_low_text;
        } else if (using_max_value) {
            guess_text = max_value_text;
        }
        printf("n = %d  c: %d  k_low: %d  k_high: %d  l_low: %e  l_high: %e  "
               "max_num_iter: %d  (using %s for initial guess).\n",
               n, target_num_clusters, sparsity_low, sparsity_high,
               lambda_low, lambda_high, max_num_iter, guess_text);
    }
    stat->num_iter = 0;
    lambda_high /= 2.0;
    int cur_k;
    do {
        stat->num_iter += 1;
        lambda_high *= 2.0;
        for (int ii = 0; ii < m; ii++) {
            cur_costs[ii] = lambda_high * costs[ii];
        }
        if (verbose >= 1) {
            for (int ii = 0; ii < m; ii++) {
                printf("E %d %d %.15f\n", edges[ii].first, edges[ii].second,
                       cur_costs[ii]);
            }
            for (int ii = 0; ii < n; ii++) {
                printf("N %d %.15f\n", ii, prizes[ii]);
            }
            printf("\n");
            printf("lambda_high: %f\n", lambda_high);
            printf("target_num_clusters: %d\n", target_num_clusters);
        }
        pcst(edges, prizes, cur_costs, root, target_num_clusters,
             1e-10, pruning, n, m, verbose, stat->re_nodes, stat->re_edges);
        cur_k = stat->re_nodes->size;
        if (verbose >= 1) {
            printf("increase:   l_high: %e  k: %d\n", lambda_high, cur_k);
        }
    } while (cur_k > sparsity_high && stat->num_iter < max_num_iter);

    if (stat->num_iter < max_num_iter && cur_k >= sparsity_low) {
        if (verbose >= 1) {
            printf("Found good lambda in exponential "
                   "increase phase, returning.\n");
        }
        free(cur_costs);
        free(sorted_prizes);
        free(sorted_indices);
        return true;
    }
    double lambda_mid;
    while (stat->num_iter < max_num_iter) {
        stat->num_iter += 1;
        lambda_mid = (lambda_low + lambda_high) / 2.0;
        for (int ii = 0; ii < m; ii++) {
            cur_costs[ii] = lambda_mid * costs[ii];
        }
        pcst(edges, prizes, cur_costs, root, target_num_clusters, 1e-10,
             pruning, n, m, verbose, stat->re_nodes, stat->re_edges);
        cur_k = stat->re_nodes->size;
        if (verbose >= 1) {
            for (int ii = 0; ii < m; ii++) {
                printf("E %d %d %.15f\n", edges[ii].first, edges[ii].second,
                       cur_costs[ii]);
            }
            for (int ii = 0; ii < n; ii++) {
                printf("N %d %.15f\n", ii, prizes[ii]);
            }
            printf("bin_search: l_mid:  %e  k: %d  "
                   "(lambda_low: %e  lambda_high: %e)\n", lambda_mid, cur_k,
                   lambda_low, lambda_high);
        }
        if (sparsity_low <= cur_k && cur_k <= sparsity_high) {
            if (verbose >= 1) {
                printf("Found good lambda in binary "
                       "search phase, returning.\n");
            }
            free(cur_costs);
            free(sorted_prizes);
            free(sorted_indices);
            return true;
        }
        if (cur_k > sparsity_high) {
            lambda_low = lambda_mid;
        } else {
            lambda_high = lambda_mid;
        }
    }
    for (int ii = 0; ii < m; ++ii) {
        cur_costs[ii] = lambda_high * costs[ii];
    }
    pcst(edges, prizes, cur_costs, root, target_num_clusters,
         1e-10, pruning, n, m, verbose, stat->re_nodes, stat->re_edges);
    if (verbose >= 1) {
        for (int ii = 0; ii < m; ii++) {
            printf("E %d %d %.15f\n", edges[ii].first, edges[ii].second,
                   cur_costs[ii]);
        }
        printf("\n");
        for (int ii = 0; ii < n; ii++) {
            printf("N %d %.15f\n", ii, prizes[ii]);
        }
        printf("\n");
        printf("Reached the maximum number of "
               "iterations, using the last l_high: %e  k: %d\n",
               lambda_high, stat->re_nodes->size);
    }
    free(cur_costs);
    free(sorted_prizes);
    free(sorted_indices);
    return true;
}

static PyObject *wrap_head_tail_binsearch(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("error: unknown error !!\n");
        return NULL;
    }
    head_tail_binsearch_para *para = malloc(sizeof(head_tail_binsearch_para));
    PyArrayObject *edges_, *costs_, *prizes_;
    if (!PyArg_ParseTuple(args, "O!O!O!iiiiii",
                          &PyArray_Type, &edges_,
                          &PyArray_Type, &prizes_,
                          &PyArray_Type, &costs_,
                          &para->g,
                          &para->root,
                          &para->sparsity_low,
                          &para->sparsity_high,
                          &para->max_num_iter,
                          &para->verbose)) { return NULL; }

    para->p = (int) prizes_->dimensions[0];
    para->m = (int) edges_->dimensions[0];
    para->prizes = (double *) PyArray_DATA(prizes_);
    para->costs = (double *) PyArray_DATA(costs_);
    para->edges = malloc(sizeof(EdgePair) * para->m);
    for (int i = 0; i < para->m; i++) {
        para->edges[i].first = *(int *) PyArray_GETPTR2(edges_, i, 0);
        para->edges[i].second = *(int *) PyArray_GETPTR2(edges_, i, 1);
    }
    GraphStat *graph_stat = make_graph_stat(para->p, para->m);
    head_tail_binsearch(
            para->edges, para->costs, para->prizes, para->p, para->m, para->g,
            para->root, para->sparsity_low, para->sparsity_high,
            para->max_num_iter, GWPruning, para->verbose, graph_stat);
    PyObject *results = PyTuple_New(1);
    PyObject *re_nodes = PyList_New(graph_stat->re_nodes->size);
    for (int i = 0; i < graph_stat->re_nodes->size; i++) {
        int cur_node = graph_stat->re_nodes->array[i];
        PyList_SetItem(re_nodes, i, PyInt_FromLong(cur_node));
    }
    PyTuple_SetItem(results, 0, re_nodes);
    free(para->edges);
    free(para);
    free_graph_stat(graph_stat);
    return results;
}

static PyMethodDef sparse_methods[] = {
        {"wrap_head_tail_binsearch", (PyCFunction) wrap_head_tail_binsearch,
                METH_VARARGS, "online_da_gl docs"},
        {NULL, NULL, 0, NULL}};

/** Python version 2 for module initialization */
PyMODINIT_FUNC initsparse_module() {
    Py_InitModule3("sparse_module", sparse_methods,
                   "some docs for sparse learning algorithms.");
    import_array();
}

int main() {
    printf("test of main wrapper!\n");
}