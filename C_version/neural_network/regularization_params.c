//
// Created by maximus on 10.04.23.
//

#include "regularization_params.h"
#include <math.h>

double zero_placeholder(int step) {
    return 0;
}

regularization_params init_regularization_params() {
    regularization_params result;
    result.l1 = zero_placeholder;
    result.l2 = zero_placeholder;
}

general_regularization_params init_general_regularization_params() {
    general_regularization_params result;
    result.batch_size = 1;
    return result;
}

double mse(matrix prediction, matrix expected) {
    double sum = 0;
    for (int i = 0; i < prediction.i; i++) {
        double temp = matrix_get_element(prediction, i, 0) - matrix_get_element(expected, i, 0);
        sum += temp * temp;
    }
    return sum / prediction.i;
}

matrix mse_derived(matrix prediction, matrix expected) {
    return matrix_substact(prediction, expected);
}

double crossentropy(matrix prediction, matrix expected) {
    double sum = 0;
    for (int i = 0; i < prediction.i; i++) {
        sum -= expected.table[i][0] * log(prediction.table[i][0] + 0.000000000000001);
    }
    return sum;
}

matrix crossentropy_loss(matrix prediction, matrix expected) {
    matrix result = matrix_creation(prediction.i, 1);
    for (int i = 0; i < prediction.i; i++) {
        result.table[i][0] = (prediction.table[i][0] - expected.table[i][0]) /
                             (prediction.table[i][0] * (1 - prediction.table[i][0]) + 0.000000000001);
    }
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