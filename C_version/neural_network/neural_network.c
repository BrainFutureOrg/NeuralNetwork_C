//
// Created by maximus on 22.01.23.
//

#include "neural_network.h"
#include "activation_functions/network_activation_functions.h"
#include "../math/matrix_operations.h"
#include "../math/statistical_random.h"
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>

#define EPSILON 0.00005

void add_function_with_derivative(neural_network *network_layer, activation_function_names activation_function_name) {
    if (activation_function_name == Sigmoid) {
        network_layer->activation_function = network_sigmoid;
        network_layer->activation_function_derivative = network_sigmoid_derivative;
        return;
    }

    if (activation_function_name == Softmax) {
        network_layer->activation_function = network_softmax;
        network_layer->activation_function_derivative = network_softmax_derivative;
        return;
    }

    if (activation_function_name == Tangential) {
        network_layer->activation_function = network_tangential;
        network_layer->activation_function_derivative = network_tangential_derivative;
        return;
    }

    if (activation_function_name == ReLu) {
        network_layer->activation_function = network_ReLU;
        network_layer->activation_function_derivative = network_ReLU_derivative;
        return;
    }
}


network_start_layer create_network(int neuron_numbers) {
    network_start_layer result;
    result.i = neuron_numbers;
    result.next_layer = NULL;
    return result;
}

void add_after_start_layer(network_start_layer *network, int neuron_numbers,
                           activation_function_names activation_function_name, regularization_params regularization) {
    neural_network *new_layer = calloc(1, sizeof(neural_network));
    network->next_layer = NULL;
//    matrix weighs = matrix_creation(neuron_numbers, network->i);
//    for (int i = 0; i < weighs.i; i++) {
//        for (int j = 0; j < weighs.j; j++) {
//            weighs.table[i][j] = randn();
//        }
//    }

    matrix bias = matrix_creation(neuron_numbers, 1);
    for (int i = 0; i < bias.i; i++) {
        bias.table[i][0] = randn();
    }

    new_layer->weights = regularization.weight_initializ(network, neuron_numbers);
//    matrix_print(new_layer->weights);
//    network->next_layer->weights = weighs;
    new_layer->bias = bias;
    new_layer->next_layer = NULL;
    new_layer->previous_layer = NULL;
    new_layer->regularization_params = regularization;
    add_function_with_derivative(new_layer, activation_function_name);
    network->next_layer = new_layer;
}

void
add_after_layer(network_start_layer *network, int neuron_numbers, activation_function_names activation_function_name,
                regularization_params regularization) {
    neural_network *new_layer = calloc(1, sizeof(neural_network));
    neural_network *current = network->next_layer;
    while (current->next_layer != NULL) {
        current = current->next_layer;
    }
//    matrix weighs = matrix_creation(neuron_numbers, current->weights.i);
//    for (int i = 0; i < weighs.i; i++) {
//        for (int j = 0; j < weighs.j; j++) {
//            weighs.table[i][j] = randn();
//        }
//    }

    matrix bias = matrix_creation(neuron_numbers, 1);
    for (int i = 0; i < bias.i; i++) {
        bias.table[i][0] = randn();
    }
    new_layer->bias = bias;
    new_layer->weights = regularization.weight_initializ(network, neuron_numbers);
//    matrix_print(current->next_layer->weights);
//    current->next_layer->weights = weighs;
    new_layer->next_layer = NULL;
    new_layer->previous_layer = current;
    new_layer->regularization_params = regularization;
    add_function_with_derivative(new_layer, activation_function_name);
    current->next_layer = new_layer;
}

void add_layer(network_start_layer *network, int neuron_numbers, activation_function_names activation_function_name,
               regularization_params regularization) {
    if (network->next_layer == NULL) {
        add_after_start_layer(network, neuron_numbers, activation_function_name, regularization);
    } else {
        add_after_layer(network, neuron_numbers, activation_function_name, regularization);
    }
}


void print_network(network_start_layer network) {
    printf("startlayer\n");
    int i = 0;
    neural_network *current = network.next_layer;
    while (current != NULL) {
        i++;
        printf("layer %d exists\n", i);
        current = current->next_layer;
    }
}

matrix predict(network_start_layer network, matrix start_layer) {
    neural_network *current = network.next_layer;
    int i = 0;
    matrix current_results = start_layer;
    while (current != NULL) {
        matrix multiplication = matrix_multiplication(current->weights, current_results);
        if (i != 0) {
            matrix_free(current_results);
        } else {
            i++;
        }
        current_results = matrix_addition(multiplication, current->bias);
        matrix_free(multiplication);
        current->activation_function(&current_results);
        current = current->next_layer;
    }
    return current_results;
}

int predict_number(network_start_layer network, matrix start_layer) {
    matrix prediction = predict(network, start_layer);
    coordinates max_pred = matrix_argmax(prediction);

    matrix_free(prediction);
    return max_pred.i;
}

double small_accuracy(network_start_layer network, matrix start_layer, matrix answers) {
    double accuracy;

    matrix prediction = predict(network, start_layer);
    coordinates max_pred = matrix_argmax(prediction);
    coordinates max_answ = matrix_argmax(answers);

    matrix_free(prediction);
    return coordinates_equals(max_answ, max_pred);
}

double accuracy(network_start_layer network, matrix *start_layers, matrix *answers, int len_of_accuracy) {
    double accuracy = 0;
    for (int i = 0; i < len_of_accuracy; i++) {
        accuracy += small_accuracy(network, start_layers[i], answers[i]) / len_of_accuracy;
    }
    return accuracy;
}

void confusion_matrix(network_start_layer network, matrix *start_layers, matrix *answers, int len_of_data) {
    int size = answers[0].i;
    matrix confusion = matrix_creation(size, size);
    printf("predict\\true\n");
    for (int i = 0; i < len_of_data; i++) {
        matrix prediction = predict(network, start_layers[i]);
        coordinates max_pred = matrix_argmax(prediction);
        coordinates max_answ = matrix_argmax(answers[i]);
        confusion.table[max_pred.i][max_answ.i] += 1;
        matrix_free(prediction);

    }
    matrix_print_with_indexation(confusion, 4, 0);
    matrix_free(confusion);
}

void confusion_matrix_paired(network_start_layer network, matrix **start_result_layers, int len_of_data) {
    confusion_matrix(network, start_result_layers[0], start_result_layers[1], len_of_data);
}

void free_network(network_start_layer network) {
    neural_network *current = network.next_layer;
    while (current != NULL) {
        neural_network *current2 = current->next_layer;
        matrix_free(current->weights);
        matrix_free(current->bias);
        free(current);
        current = current2;
    }
    network.next_layer = NULL;
}

double mono_mse_loss(network_start_layer network, matrix start_layer, matrix expected_results) {
    matrix real_results = predict(network, start_layer);
    matrix subtracted = matrix_substact(real_results, expected_results);
    double mse = 0;
    for (int i = 0; i < subtracted.i; i++) {
        double add = matrix_get_element(subtracted, i, 0);
        mse += add * add;
    }
    matrix_free(subtracted);
    matrix_free(real_results);
    return mse / expected_results.i;
}

double mse_loss(network_start_layer network, matrix *start_layers, int sample_number, matrix *expected_results) {
    double sum = 0;
    for (int i = 0; i < sample_number; i++) {
        sum += mono_mse_loss(network, start_layers[i], expected_results[i]);
    }
    return sum / sample_number;
}

double many_loss(network_start_layer network, matrix *start_layers, int sample_number, matrix *expected_results,
                 general_regularization_params general_regularization) {
    double sum = 0;
    for (int i = 0; i < sample_number; i++) {
        matrix real_results = predict(network, start_layers[i]);
        sum += general_regularization.cost_function(real_results, expected_results[i]) / sample_number;
        matrix_free(real_results);
    }
    return sum;
}

void test_network(network_start_layer network, matrix *start_layers, int start_layer_number, matrix *expected_results,
                  general_regularization_params general_regularization) {
    double accuracy_num = accuracy(network, start_layers, expected_results, start_layer_number);
    //double loss_num = mse_loss(network, start_layers, start_layer_number, expected_results);
    double loss_num = many_loss(network, start_layers, start_layer_number, expected_results, general_regularization);
    printf("accuracy: %f, loss: %f\n", accuracy_num, loss_num);
}

void test_network_paired(network_start_layer network, matrix **start_result_layers, int sample_number,
                         general_regularization_params general_regularization) {
    test_network(network, start_result_layers[0], sample_number, start_result_layers[1], general_regularization);
}

neural_network *copy_neural_network_layer(neural_network *layer) {
    neural_network *copy = calloc(1, sizeof(neural_network));
    copy->next_layer = copy->previous_layer = NULL;
    copy->activation_function_derivative = layer->activation_function_derivative;
    copy->activation_function = layer->activation_function;
    copy->weights = matrix_copy(layer->weights);
    copy->bias = matrix_copy(layer->bias);
    copy->regularization_params = layer->regularization_params;
    return copy;
}

network_start_layer neural_network_copy(network_start_layer network) {
    network_start_layer network_copy = network;
    neural_network *network_layers = network.next_layer;
    neural_network *last_copied_layer = copy_neural_network_layer(network_layers);
    network_copy.next_layer = last_copied_layer;
    while ((network_layers = network_layers->next_layer) != NULL) {
//        printf("copy_");
        neural_network *layer_copy = copy_neural_network_layer(network_layers);
        layer_copy->previous_layer = last_copied_layer;
        last_copied_layer->next_layer = layer_copy;
        last_copied_layer = layer_copy;
    }
//    printf("\n");
    return network_copy;
}