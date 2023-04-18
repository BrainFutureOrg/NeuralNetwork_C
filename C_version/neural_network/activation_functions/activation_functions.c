//
// Created by maximus on 22.01.23.
//

#include "activation_functions.h"
#include <math.h>
#include "../../math/matrix_operations.h"
#include <limits.h>
#include <float.h>

#define RELU_PARAM 0.01

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double sigmoid_derivative(double x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

void softmax(matrix *M) {
    matrix result = matrix_creation(M->i, M->j);
    double sum = 0;
    for (int i = 0; i < M->i; i++) {
        for (int j = 0; j < M->j; j++) {
            sum += exp(M->table[i][j]);
        }
    }
    for (int i = 0; i < M->i; i++) {
        for (int j = 0; j < M->j; j++) {
            result.table[i][j] = exp(M->table[i][j]) / sum;
        }
    }
    *M = result;
}

void softmax_derivative(matrix *M) {
    softmax(M);
    for (int i = 0; i < M->i; i++) {
        for (int j = 0; j < M->j; j++) {
            M->table[i][j] -= pow(M->table[i][j], 2);
        }
    }
}

double tangential(double x) {
    return tanh(x);
}

void softmax_stable(matrix *M) {
    matrix result = matrix_creation(M->i, M->j);
    double sum = 0;
    double max = M->table[0][0];
    for (int i = 0; i < M->i; i++) {
        for (int j = 0; j < M->j; j++) {
            if (max < M->table[i][j])max = M->table[i][j];
        }
    }
    for (int i = 0; i < M->i; i++) {
        for (int j = 0; j < M->j; j++) {
            sum += exp(M->table[i][j] / max);
        }
    }
    for (int i = 0; i < M->i; i++) {
        for (int j = 0; j < M->j; j++) {
            result.table[i][j] = exp(M->table[i][j] / max) / sum;
        }
    }
    *M = result;
}

void softmax_derivative_stable(matrix *M) {
    softmax_stable(M);
    for (int i = 0; i < M->i; i++) {
        for (int j = 0; j < M->j; j++) {
            M->table[i][j] -= pow(M->table[i][j], 2);
        }
    }
}

double tangential_derivative(double x) {
    return 1 / pow(cosh(x), 2);
}

double leakyReLU(double x, double slope) {
    return x > 0 ? x : x * slope;
}

double leakyReLU_derivative(double x, double slope) {
    return x > 0 ? 1 : x < 0 ? slope : (1 + slope) / 2;
}

double ReLU(double x) {
    return x > 0 ? x : x * RELU_PARAM;
}

double ReLU_derivative(double x) {
    return x > 0 ? 1 : x < 0 ? RELU_PARAM : 0.5 + RELU_PARAM / 2;
}