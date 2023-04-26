//
// Created by maximus on 17.04.23.
//
#include "../neural_network.h"

#ifndef C_VERSION_SGD_H
#define C_VERSION_SGD_H


/*void learn_step_optimizerless_array_batch(network_start_layer network, double learning_rate, matrix *start_layer,
                                          matrix *result_layer, int sample_number,
                                          general_regularization_params general_regularization,
                                          int epoch);

void learn_step_optimizerless_paired_array_batch(network_start_layer network, double learning_rate,
                                                 matrix **start_result_layer, int sample_number,
                                                 general_regularization_params general_regularization,
                                                 int epoch);*/

void gradient_descent_batch(neural_network *layer, matrix *error, int batch_size, double learning_rate,
                            matrix **previous_values, int number_of_current_layer, int epoch);

//void gradient_descent(neural_network *layer, matrix error, double learning_rate, matrix previous_values, int epoch);

void learn_step_sgd_batch(network_start_layer network, double learning_rate, batch *start_result_layers,
                          int epoch, general_regularization_params general_regularizations);

void learn_step_sgd_reader_batch(network_start_layer network, double learning_rate, data_reader *reader,
                                 general_regularization_params general_regularization,
                                 int epoch);

void gradient_descent_batch(neural_network *layer, matrix *error, int batch_size, double learning_rate,
                            matrix **previous_values, int number_of_current_layer, int epoch);

#endif //C_VERSION_SGD_H
