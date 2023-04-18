//
// Created by maximus on 17.04.23.
//
#include "../neural_network.h"


#ifndef C_VERSION_MOMENTUMOPTIMIZER_H
#define C_VERSION_MOMENTUMOPTIMIZER_H

void learn_step_momentum_array_batch(network_start_layer network, double learning_rate, matrix *start_layer,
                                     matrix *result_layer, int sample_number,
                                     general_regularization_params general_regularization,
                                     int epoch, double friction);

void learn_step_momentum_paired_array_batch(network_start_layer network, double learning_rate,
                                            matrix **start_result_layer, int sample_number,
                                            general_regularization_params general_regularization,
                                            int epoch, double friction);

void gradient_descent_momentum_batch(neural_network *layer, matrix *error, int batch_size, double learning_rate,
                                     matrix **previous_values, int number_of_current_layer, int epoch,
                                     matrix momentum_weights, matrix momentum_bias, double friction);

#endif //C_VERSION_MOMENTUMOPTIMIZER_H
