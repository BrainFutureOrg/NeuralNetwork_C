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

    matrix bias;

    void (*activation_function)(matrix *);

    void (*activation_function_derivative)(matrix *);

    struct neural_network *next_layer;

    struct neural_network *previous_layer;
} neural_network;

network_start_layer create_network(int neuron_numbers);

void add_layer(network_start_layer *network, int neuron_numbers, char *activation_function_name);

void learn_step(network_start_layer network, double learning_rate, matrix start_layer, matrix result_layer);

matrix predict(network_start_layer network, matrix start_layer);

double accuracy(network_start_layer network, matrix *start_layers, matrix *answers, int len_of_accuracy);

double small_accuracy(network_start_layer network, matrix start_layer, matrix answers);

void print_network(network_start_layer network);

void free_network(network_start_layer startLayer);

void learn_step_optimizerless(network_start_layer network, double learning_rate, matrix start_layer,
                              matrix result_layer);

#endif //C_VERSION_NEURAL_NETWORK_H
