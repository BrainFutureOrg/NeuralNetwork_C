//
// Created by maximus on 10.04.23.
//

#include <math.h>
#include "losses.h"

double mse(matrix prediction, matrix expected) {
    double sum = 0;
    for (int i = 0; i < prediction.i; i++) {
        double temp = matrix_get_element(prediction, i, 0) - matrix_get_element(expected, i, 0);
        sum += temp * temp;
    }
    return sum / prediction.i;
}

matrix mse_derived(matrix prediction, matrix expected) {
    return matrix_substact(prediction, expected);
}

double crossentropy(matrix prediction, matrix expected) {
    double sum = 0;
    for (int i = 0; i < prediction.i; i++) {
        sum -= expected.table[i][0] * log(prediction.table[i][0] + 0.000000000000001);
    }
    return sum;
}

matrix crossentropy_loss(matrix prediction, matrix expected) {
    matrix result = matrix_creation(prediction.i, 1);
    for (int i = 0; i < prediction.i; i++) {
        result.table[i][0] = (prediction.table[i][0] - expected.table[i][0]) /
                             (prediction.table[i][0] * (1 - prediction.table[i][0]) + 0.000000000001);
    }
    return result;
}