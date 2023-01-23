//
// Created by maximus on 23.01.23.
//

#ifndef C_VERSION_NETWORK_ACTIVATION_FUNCTIONS_H
#define C_VERSION_NETWORK_ACTIVATION_FUNCTIONS_H

#include "../matrix_operations.h"

void network_sigmoid(matrix *M);

void network_sigmoid_derivative(matrix *M);

void network_softmax(matrix *M);

void network_softmax_derivative(matrix *M);

void network_tangential(matrix *M);

void network_tangential_derivative(matrix *M);

//void network_leakyReLU(double x, double slope);

//void network_leakyReLU_derivative(double x, double slope);

void network_ReLU(matrix *M);

void network_ReLU_derivative(matrix *M);

#endif //C_VERSION_NETWORK_ACTIVATION_FUNCTIONS_H
