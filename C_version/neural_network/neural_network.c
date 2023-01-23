//
// Created by maximus on 22.01.23.
//

#include "neural_network.h"
#include "stdlib.h"
#include "math.h"
#include "../matrix_operations.h"
#include <limits.h>
#include <string.h>
#include "network_activation_functions.h"

void add_function_with_derivative(neural_network *network_layer, char *activation_function_name) {
    if (strcmp(activation_function_name, "Sigmoid") == 0) {
        network_layer->activation_function = network_sigmoid;
        network_layer->activation_function_derivative = network_sigmoid_derivative;
        return;
    }

    if (strcmp(activation_function_name, "Softmax") == 0) {
        network_layer->activation_function = network_softmax;
        network_layer->activation_function_derivative = network_softmax_derivative;
        return;
    }

    if (strcmp(activation_function_name, "Tanh") == 0) {
        network_layer->activation_function = network_tangential;
        network_layer->activation_function_derivative = network_tangential_derivative;
        return;
    }

    if (strcmp(activation_function_name, "ReLu") == 0) {
        network_layer->activation_function = network_ReLU;
        network_layer->activation_function_derivative = network_ReLU_derivative;
        return;
    }
}

neural_network* last_layer(network_start_layer network){
    neural_network *current = network.next_layer;
    if(current==NULL) return current;
    while (current->next_layer != NULL) {
        current = current->next_layer;
    }
    return current;
}

network_start_layer create_network(int neuron_numbers) {
    network_start_layer result;
    result.i = neuron_numbers;
    return result;
}

void add_after_start_layer(network_start_layer network, int neuron_numbers, char *activation_function_name) {
    network.next_layer = calloc(1, sizeof(neural_network));
    matrix weighs = matrix_creation(neuron_numbers, network.i);
    for (int i = 0; i < weighs.i; i++) {
        for (int j = 0; weighs.j; j++) {
            weighs.table[i][j] = rand() / INT_MAX + 0.001;
        }
    }
    network.next_layer->weights = weighs;
    add_function_with_derivative(network.next_layer, activation_function_name);
}

void add_after_layer(network_start_layer network, int neuron_numbers, char *activation_function_name) {
    neural_network *current = network.next_layer;
    while (current->next_layer != NULL) {
        current = current->next_layer;
    }
    current->next_layer = calloc(1, sizeof(neural_network));
    matrix weighs = matrix_creation(neuron_numbers, current->weights.i);
    for (int i = 0; i < weighs.i; i++) {
        for (int j = 0; weighs.j; j++) {
            weighs.table[i][j] = rand() / INT_MAX + 0.001;
        }
    }
    current->next_layer->weights = weighs;
    add_function_with_derivative(current->next_layer, activation_function_name);
}

void add_layer(network_start_layer network, int neuron_numbers, char *activation_function_name) {
    if (network.next_layer == NULL) {
        add_after_start_layer(network, neuron_numbers, activation_function_name);
    } else {
        add_after_layer(network, neuron_numbers, activation_function_name);
    }
}

int count_hidden_layers(network_start_layer network) {
    neural_network *current = network.next_layer;
    int result = 0;
    while (current != NULL) {
        current = current->next_layer;
        result++;
    }
    return result;
}

matrix *predict_all_layers(network_start_layer network, matrix start_layer) {
    int layers_number = count_hidden_layers(network);
    neural_network *current = network.next_layer;
    matrix *current_results = calloc(layers_number + 1, sizeof(matrix));
    current_results[0] = start_layer;
    for (int i = 1; i < layers_number + 1; i++) {
        current_results[i] = matrix_multiplication(current->weights, current_results[i - 1]);
        current->activation_function(&current_results[i]);
        current = current->next_layer;
    }
    return current_results;
}

void learn_step(network_start_layer network, double learning_rate, matrix start_layer, matrix result_layer) {
    //matrix errors= matrix_creation(result_layer.i, 1)
    matrix errors = matrix_substact(result_layer, predict(network, start_layer));
    int layer_number = count_hidden_layers(network);
    matrix *distributed_errors = calloc(layer_number, sizeof(matrix *));
    distributed_errors[layer_number - 1] = errors;
    for (int i = layer_number - 1; i > 0; i--) {
        distributed_errors[i-1]= matrix_multiplication(matrix_transposition())
    }
}

matrix predict(network_start_layer network, matrix start_layer) {
    neural_network *current = network.next_layer;
    matrix current_results = start_layer;
    while (current != NULL) {
        current_results = matrix_multiplication(current->weights, current_results);
        current->activation_function(&current_results);
        current = current->next_layer;
    }
    return current_results;
}

double small_accuracy(network_start_layer network, matrix start_layer, matrix answers) {
    double accuracy;
    matrix prediction = predict(network, start_layer);
    for (int i = 0; i < answers.i; i++) {
        accuracy += fabs(answers.table[i][0] - prediction.table[i][0]) / prediction.table[i][0] / answers.i;
    }
    return 1 - accuracy;
}

double accuracy(network_start_layer network, matrix *start_layers, matrix *answers, int len_of_accuracy) {
    double accuracy;
    for (int i = 0; i < len_of_accuracy; i++) {
        accuracy += small_accuracy(network, start_layers[i], answers[i]) / len_of_accuracy;
    }
    return accuracy;
}