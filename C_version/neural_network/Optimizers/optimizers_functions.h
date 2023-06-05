//
// Created by maximus on 17.04.23.
//
#include "../neural_network.h"

#ifndef C_VERSION_OPTIMIZERS_FUNCTIONS_H
#define C_VERSION_OPTIMIZERS_FUNCTIONS_H

double signum(double a);

neural_network *last_layer(network_start_layer network);

int count_hidden_layers(network_start_layer network);

matrix *predict_all_layers(network_start_layer network, matrix start_layer);

matrix **predict_all_layers_batch(network_start_layer network, matrix *start_layers, int batch_size);

void learn_step_batch(network_start_layer network, double learning_rate, batch *start_result_layers, int epoch,
                      void (*gradient_descent)(neural_network *, matrix *, int, double, matrix **, int, int, void *),
                      void *gradient_params);

void learn_step_reader(network_start_layer network, double learning_rate, data_reader *reader,
                       int epoch, void *(*create_gradient_params)(network_start_layer),
                       void *(*free_gradient_params)(network_start_layer, void *),
                       void (*gradient_descent)(neural_network *, matrix *, int, double, matrix **, int, int, void *));

void finish_gd(neural_network *layer, matrix new_weights, matrix new_bias, int epoch);

matrix multiplied_for_weights(neural_network *layer, matrix *error, matrix **previous_values, int number_of_current_layer, int i);
#endif //C_VERSION_OPTIMIZERS_FUNCTIONS_H
