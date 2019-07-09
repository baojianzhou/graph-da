//
//

#ifndef ONLINE_OPT_ALGO_ONLINE_ADAM_H
#define ONLINE_OPT_ALGO_ONLINE_ADAM_H

#include "sparse_algorithms.h"

bool algo_online_adam_logit(adam_para *para, OnlineStat *stat);

bool algo_online_adam_least_square(adam_para *para, OnlineStat *stat);

#endif //ONLINE_OPT_ALGO_ONLINE_ADAM_H
