//
//

#ifndef ONLINE_OPT_ALGO_ONLINE_GRAPH_IHT_H
#define ONLINE_OPT_ALGO_ONLINE_GRAPH_IHT_H

#include "sparse_algorithms.h"

bool algo_online_graph_iht_logit(
        graph_iht_para *para, OnlineStat *stat);

bool algo_online_graph_iht_least_square(
        graph_iht_para *para, OnlineStat *stat);

#endif //ONLINE_OPT_ALGO_ONLINE_GRAPH_IHT_H
