//
// Created by maximus on 01.05.23.
//
#include "../neural_network.h"

#ifndef C_VERSION_ADAM_FUTURE_VIEW_OPTIMIZER_H
#define C_VERSION_ADAM_FUTURE_VIEW_OPTIMIZER_H

#endif //C_VERSION_ADAM_FUTURE_VIEW_OPTIMIZER_H

typedef struct Adam_future_params {
    double b1;
    double b2;
} Adam_future_params;

void gradient_descent_adam_future_batch(neural_network *layer, matrix *error, int batch_size, double learning_rate,
                                        matrix **previous_values, int number_of_current_layer, int epoch,
                                        matrix momentum_weights, matrix momentum_bias, matrix s_weights, matrix s_bias,
                                        Adam_future_params params);

void learn_step_adam_future_reader_batch(network_start_layer network, double learning_rate, data_reader *reader,
                                         general_regularization_params general_regularization,
                                         int epoch, Adam_future_params params);