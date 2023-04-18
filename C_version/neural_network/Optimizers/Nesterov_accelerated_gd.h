//
// Created by maximus on 18.04.23.
//
#include "../neural_structs.h"

#ifndef C_VERSION_NESTEROV_ACCELERATED_GD_H
#define C_VERSION_NESTEROV_ACCELERATED_GD_H

void learn_step_nesterov_array_batch(network_start_layer network, double learning_rate, matrix *start_layer,
                                     matrix *result_layer, int sample_number,
                                     general_regularization_params general_regularization,
                                     int epoch, double friction);

void learn_step_nesterov_paired_array_batch(network_start_layer network, double learning_rate,
                                            matrix **start_result_layer, int sample_number,
                                            general_regularization_params general_regularization,
                                            int epoch, double friction);

void gradient_descent_nesterov_batch(neural_network *layer, matrix *error, int batch_size, double learning_rate,
                                     matrix **previous_values, int number_of_current_layer, int epoch,
                                     matrix momentum_weights, matrix momentum_bias);

#endif //C_VERSION_NESTEROV_ACCELERATED_GD_H
