//
// Created by kosenko on 04.04.23.
//
#include "neural_network.h"

#ifndef C_VERSION_OPTIMIZERS_H
#define C_VERSION_OPTIMIZERS_H

typedef struct weight_bias {
    matrix weight;
    matrix bias;
} weight_bias;

void gradient_descent(neural_network *layer, matrix error, double learning_rate, matrix previous_values, double l1,
                      double l2);

weight_bias
gradient_descent_delta(neural_network *layer, matrix error, double learning_rate, matrix previous_values, double l1,
                       double l2);

void
gradient_descent_dual(neural_network *original_layer, neural_network *changed_layer, matrix error, double learning_rate,
                      matrix previous_values, double l1, double l2);

//void gradient_descent_batch(neural_network* layer, matrix* error, double learning_rate, matrix previous_values, double l1, double l2);
void gradient_descent_batch(neural_network *layer, matrix *error, int batch_size, double learning_rate,
                            matrix **previous_values, int number_of_current_layer, double l1, double l2);
//int signum(double a);

#endif //C_VERSION_OPTIMIZERS_H
