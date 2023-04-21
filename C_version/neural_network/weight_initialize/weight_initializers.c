//
// Created by maximus on 17.04.23.
//

#include "weight_initializers.h"
#include "stdlib.h"
#include "../../math/statistical_random.h"
#include "math.h"

int find_n(network_start_layer *network) {
    int n;
    if (network->next_layer != NULL) {
        neural_network *current = network->next_layer;
        while (current->next_layer != NULL) {
            current = current->next_layer;
        }
        n = current->weights.i;
    } else {
        n = network->i;
    }
    return n;
}

matrix gaussian_weight_initialization(network_start_layer *network, int neuron_numbers) {
    int n = find_n(network);

    matrix weighs = matrix_creation(neuron_numbers, n);
    for (int i = 0; i < weighs.i; i++) {
        for (int j = 0; j < weighs.j; j++) {
            weighs.table[i][j] = randn();
        }
    }
    return weighs;
}

matrix xavier_weight_initialization(network_start_layer *network, int neuron_numbers) {

    int n = find_n(network);

    double param = 1.0 / sqrt(n);

    matrix weighs = matrix_creation(neuron_numbers, n);
    for (int i = 0; i < weighs.i; i++) {
        for (int j = 0; j < weighs.j; j++) {
            weighs.table[i][j] = randu_range(-param, param);
        }
    }

    return weighs;
}

matrix null_weight_initialization(network_start_layer *network, int neuron_numbers) {

    int n = find_n(network);

    matrix weighs = matrix_creation(neuron_numbers, n);
    for (int i = 0; i < weighs.i; i++) {
        for (int j = 0; j < weighs.j; j++) {
            weighs.table[i][j] = 0;
        }
    }

    return weighs;
}

matrix xavier_normalized_weight_initialization(network_start_layer *network, int neuron_numbers) {
    int n = find_n(network);

    double param = sqrt(6) / sqrt(n + neuron_numbers);

    matrix weighs = matrix_creation(neuron_numbers, n);
    for (int i = 0; i < weighs.i; i++) {
        for (int j = 0; j < weighs.j; j++) {
            weighs.table[i][j] = randu_range(-param, param);
        }
    }

    return weighs;
}

matrix he_weight_initialization(network_start_layer *network, int neuron_numbers) {
    int n = find_n(network);
    double std = sqrt(2.0 / n);

    matrix weighs = matrix_creation(neuron_numbers, n);
    for (int i = 0; i < weighs.i; i++) {
        for (int j = 0; j < weighs.j; j++) {
            weighs.table[i][j] = randn() * std;
        }
    }
    return weighs;
}

void set_weights(regularization_params *params, enum weight_init weight_name) {
    switch (weight_name) {

        case GAUSSIAN:
            params->weight_initializ = gaussian_weight_initialization;
            break;
        case XAVIER:
            params->weight_initializ = xavier_weight_initialization;
            break;
        case XAVIER_NORMALIZED:
            params->weight_initializ = xavier_normalized_weight_initialization;
            break;
        case HE_WEIGHT_INITIALIZATION:
            params->weight_initializ = he_weight_initialization;
            break;
        case NULL_WEIGHT_INITIALIZATION:
            params->weight_initializ = null_weight_initialization;
            break;
    }
}