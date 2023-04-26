//
// Created by maximus on 19.04.23.
//
#include "../neural_network.h"

#ifndef C_VERSION_ADAM_OPTIMIZER_H
#define C_VERSION_ADAM_OPTIMIZER_H

typedef struct Adam_params {
    double b1;
    double b2;
} Adam_params;

void gradient_descent_adam_batch(neural_network *layer, matrix *error, int batch_size, double learning_rate,
                                 matrix **previous_values, int number_of_current_layer, int epoch,
                                 matrix momentum_weights, matrix momentum_bias, matrix s_weights, matrix s_bias,
                                 Adam_params params);

void learn_step_adam_reader_batch(network_start_layer network, double learning_rate, data_reader *reader,
                                  general_regularization_params general_regularization,
                                  int epoch, Adam_params params);
/*void learn_step_adam_array_batch(network_start_layer network, double learning_rate, matrix *start_layer,
                                 matrix *result_layer, int sample_number,
                                 general_regularization_params general_regularization,
                                 int epoch, Adam_params params);*/

/*void learn_step_adam_paired_array_batch(network_start_layer network, double learning_rate,
                                        matrix **start_result_layer, int sample_number,
                                        general_regularization_params general_regularization,
                                        int epoch, Adam_params params);*/

#endif //C_VERSION_ADAM_OPTIMIZER_H
