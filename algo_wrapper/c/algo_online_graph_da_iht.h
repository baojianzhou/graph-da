//
//

#ifndef ONLINE_OPT_ALGO_ONLINE_GRAPH_DA_IHT_H
#define ONLINE_OPT_ALGO_ONLINE_GRAPH_DA_IHT_H

#include "sparse_algorithms.h"

bool algo_online_graph_da_iht_logit(
        graph_da_iht_para *para, OnlineStat *stat);

bool algo_online_graph_da_iht_logit_2(
        graph_da_iht_para_2 *para, OnlineStat *stat);

bool algo_online_graph_da_iht_least_square_2(
        graph_da_iht_para_2 *para, OnlineStat *stat);

bool algo_online_graph_da_iht_least_square(
        graph_da_iht_para *para, OnlineStat *stat);

bool algo_online_graph_da_iht_least_square_without_head(
        graph_da_iht_para *para, OnlineStat *stat);

#endif //ONLINE_OPT_ALGO_ONLINE_GRAPH_DA_IHT_H
