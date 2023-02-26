//
// Created by maximus on 23.01.23.
//

#include "network_activation_functions.h"
#include "activation_functions.h"
#include "../matrix_operations.h"

void network_sigmoid(matrix *M) {
    matrix_function_to_elements(*M, sigmoid);
}

void network_sigmoid_derivative(matrix *M) {
    matrix_function_to_elements(*M, sigmoid_derivative);
}

void network_softmax(matrix *M) {//stable
    softmax_stable(M);
}

void network_softmax_derivative(matrix *M) {//stable
    softmax_derivative_stable(M);
}

void network_tangential(matrix *M) {
    matrix_function_to_elements(*M, tangential);
}

void network_tangential_derivative(matrix *M) {
    matrix_function_to_elements(*M, tangential_derivative);
}

//void network_leakyReLU(double x, double slope);

//void network_leakyReLU_derivative(double x, double slope);

void network_ReLU(matrix *M) {
    matrix_function_to_elements(*M, ReLU);
}

void network_ReLU_derivative(matrix *M) {
    matrix_function_to_elements(*M, ReLU_derivative);
}