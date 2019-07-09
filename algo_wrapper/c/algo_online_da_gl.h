//
//

#ifndef FAST_PCST_ONLINE_RDA_GL_H
#define FAST_PCST_ONLINE_RDA_GL_H

#include "sparse_algorithms.h"

bool algo_online_da_gl_logit(da_gl_para *para, OnlineStat *stat);

bool algo_online_da_gl_least_square(da_gl_para *para, OnlineStat *stat);

bool algo_online_da_sgl_logit(da_sgl_para *para, OnlineStat *stat);

bool algo_online_da_sgl_least_square(da_sgl_para *para, OnlineStat *stat);

#endif //FAST_PCST_ONLINE_RDA_GL_H
