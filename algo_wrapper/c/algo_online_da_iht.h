//
//

#ifndef ONLINE_OPT_ALGO_ONLINE_DA_IHT_H
#define ONLINE_OPT_ALGO_ONLINE_DA_IHT_H

#include "sparse_algorithms.h"

bool algo_online_da_iht_logit(
        da_iht_para *para, OnlineStat *stat);

bool algo_online_da_iht_least_square(
        da_iht_para *para, OnlineStat *stat);

#endif //ONLINE_OPT_ALGO_ONLINE_DA_IHT_H
