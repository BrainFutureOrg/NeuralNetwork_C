//
// Created by maximus on 22.01.23.
//

#ifndef C_VERSION_NEURAL_NETWORK_H
#define C_VERSION_NEURAL_NETWORK_H

#include "../matrix_operations.h"

typedef enum {
    ReLu,
    Softmax,
    Sigmoid,
    Tangential
}    activation_function_names;

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

void add_layer(network_start_layer *network, int neuron_numbers, activation_function_names activation_function_name);

void learn_step(network_start_layer network, double learning_rate, matrix start_layer, matrix result_layer);

matrix predict(network_start_layer network, matrix start_layer);
int predict_number(network_start_layer network, matrix start_layer);

double accuracy(network_start_layer network, matrix *start_layers, matrix *answers, int len_of_accuracy);

double small_accuracy(network_start_layer network, matrix start_layer, matrix answers);

void print_network(network_start_layer network);

void free_network(network_start_layer startLayer);

void learn_step_optimizerless(network_start_layer network, double learning_rate, matrix start_layer,
                              matrix result_layer, double l1, double l2);

void learn_step_optimizerless_array(network_start_layer network, double learning_rate, matrix* start_layer,
                              matrix* result_layer, int sample_number, double l1,double l2);

void learn_step_optimizerless_paired_array(network_start_layer network, double learning_rate, matrix** start_result_layer, int sample_number,
                                     double l1, double l2);

void learn_step_optimizerless_batch(network_start_layer network, double learning_rate, matrix* start_layers,
                              matrix* result_layers,int batch_size, double l1, double l2);

void learn_step_optimizerless_array_batch(network_start_layer network, double learning_rate, matrix* start_layer,
                                    matrix* result_layer, int sample_number, int batch_size, double l1,double l2);

void learn_step_optimizerless_paired_array_batch(network_start_layer network, double learning_rate, matrix** start_result_layer, int sample_number, int batch_size,
                                           double l1, double l2);

double mse_loss(network_start_layer network, matrix* start_layers, int sample_number, matrix* expected_results);
void test_network(network_start_layer network, matrix* start_layers, int start_layer_number, matrix* expected_results);
void test_network_paired(network_start_layer network, matrix** start_result_layers, int sample_number);
void confusion_matrix(network_start_layer network, matrix *start_layers, matrix *answers, int len_of_data);
void confusion_matrix_paired(network_start_layer network, matrix** start_result_layers, int len_of_data);
network_start_layer neural_network_copy(network_start_layer network);

#endif //C_VERSION_NEURAL_NETWORK_H
