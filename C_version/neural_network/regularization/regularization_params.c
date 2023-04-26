//
// Created by maximus on 10.04.23.
//

#include "regularization_params.h"
#include "losses.h"
#include <math.h>

double zero_placeholder(int step) {
    return 0;
}

regularization_params init_regularization_params() {
    regularization_params result;
    result.l1 = zero_placeholder;
    result.l2 = zero_placeholder;
    return result;
}

general_regularization_params init_general_regularization_params() {
    general_regularization_params result;
//    result.batch_size = 1;
    return result;
}


void paste_cost(general_regularization_params *params, costs_names cost_name) {
    switch (cost_name) {
        case MSE:
            params->cost_function = mse;
            params->nablaC = mse_derived;
            break;
        case cross_entropy:
            params->cost_function = crossentropy;
            params->nablaC = crossentropy_loss;
    }
}