//
// Created by maximus on 22.01.23.
//

#ifndef C_VERSION_ACTIVATION_FUNCTIONS_H
#define C_VERSION_ACTIVATION_FUNCTIONS_H

#include "../matrix_operations.h"
#include <math.h>

double sigmoid(double x);

double sigmoid_derivative(double x);

void softmax(matrix *M);

void softmax_derivative(matrix *M);

double tangential(double x);

double tangential_derivative(double x);

double leakyReLU(double x, double slope);

double leakyReLU_derivative(double x, double slope);

double ReLU(double x);

double ReLU_derivative(double x);

#endif //C_VERSION_ACTIVATION_FUNCTIONS_H

