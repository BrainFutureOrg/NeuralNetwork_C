//
// Created by maximus on 18.04.23.
//

#ifndef C_VERSION_NEURAL_STRUCTS_H
#define C_VERSION_NEURAL_STRUCTS_H

#include "../math/matrix_operations.h"

typedef struct network_start_layer network_start_layer;
typedef struct regularization_params regularization_params;
typedef struct general_regularization_params general_regularization_params;
typedef struct neural_network neural_network;

struct regularization_params {
    double (*l1)(int);

    double (*l2)(int);

    matrix (*weight_initializ)(network_start_layer *, int);
};

struct general_regularization_params {
    int batch_size;

    matrix (*nablaC)(matrix, matrix);

    double (*cost_function)(matrix, matrix);
};

struct network_start_layer {
    int i;

    struct neural_network *next_layer;
};


struct neural_network {
    matrix weights;

    matrix bias;

    void (*activation_function)(matrix *);

    void (*activation_function_derivative)(matrix *);

    regularization_params regularization_params;

    struct neural_network *next_layer;

    struct neural_network *previous_layer;
};

#endif //C_VERSION_NEURAL_STRUCTS_H
