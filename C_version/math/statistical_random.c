//
// Created by kosenko on 07.04.23.
//
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include "statistical_random.h"

#define rand_plain ((double) random() / INT_MAX)

double randn(){
    double f = rand_plain, r = rand_plain;
    return cos(2*M_PI*f)* sqrt(-2* log(r));
}

double randu_range(double start, double end) {
    return rand_plain * (end - start) + start;
}