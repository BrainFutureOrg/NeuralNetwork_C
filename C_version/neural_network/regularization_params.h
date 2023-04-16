//
// Created by maximus on 10.04.23.
//
#include "../matrix_operations.h"

#ifndef C_VERSION_REGULARIZATION_PARAMS_H
#define C_VERSION_REGULARIZATION_PARAMS_H

typedef struct regularization_params {
    double (*l1)(int);

    double (*l2)(int);

} regularization_params;

typedef struct general_regularization_params {
    int batch_size;

    matrix (*nablaC)(matrix, matrix);

    double (*cost_function)(matrix, matrix);
} general_regularization_params;

typedef enum costs_names {
    MSE,
    cross_entropy
} costs_names;

regularization_params init_regularization_params();

void paste_cost(general_regularization_params *params, costs_names cost_name);

general_regularization_params init_general_regularization_params();

#endif //C_VERSION_REGULARIZATION_PARAMS_H
