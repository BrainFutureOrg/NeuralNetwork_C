//
// Created by kosenko on 04.04.23.
//
#include "neural_network.h"
#ifndef C_VERSION_OPTIMIZERS_H
#define C_VERSION_OPTIMIZERS_H

void gradient_descent(neural_network* layer, matrix error, double learning_rate, matrix previous_values);

#endif //C_VERSION_OPTIMIZERS_H
