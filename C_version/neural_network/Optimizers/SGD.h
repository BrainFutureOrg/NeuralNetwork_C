//
// Created by maximus on 17.04.23.
//
#include "../neural_network.h"

#ifndef C_VERSION_SGD_H
#define C_VERSION_SGD_H

void gradient_descent_sgd_batch(neural_network *layer, matrix *error, int batch_size, double learning_rate,
                                matrix **previous_values, int number_of_current_layer, int epoch, void *grid_params);

void learn_step_sgd_reader(network_start_layer network, double learning_rate, data_reader *reader, int epoch);

#endif //C_VERSION_SGD_H
