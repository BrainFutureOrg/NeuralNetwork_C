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

#endif //C_VERSION_OPTIMIZERS_FUNCTIONS_H
