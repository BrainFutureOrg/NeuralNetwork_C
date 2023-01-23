//
// Created by maximus on 22.01.23.
//

#ifndef C_VERSION_NEURAL_NETWORK_H
#define C_VERSION_NEURAL_NETWORK_H

#include "../matrix_operations.h"

typedef struct network_start_layer {
    int i;

    struct neural_network *next_layer;
} network_start_layer;

typedef struct neural_network {
    matrix weights;

    void (*activation_function)(matrix *);

    void (*activation_function_derivative)(matrix *);

    struct neural_network *next_layer;

    struct neural_network *previous_layer;
} neural_network;

network_start_layer create_network(int neuron_numbers);

void add_layer(neural_network network, int neuron_numbers, void (*activation_function)(matrix));

void learn_step(neural_network network, double learning_rate, matrix start_layer, matrix result_layer);

matrix predict(neural_network network, matrix start_layer);

#endif //C_VERSION_NEURAL_NETWORK_H
