//
// Created by maximus on 21.01.23.
//
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include "matrix_operations.h"

matrix matrix_multiplication (matrix first_matrix, matrix second_matrix) {
    matrix result;
    if (first_matrix.j != second_matrix.i) {
        errno = ERANGE;
        //result.table=(double**)malloc(0);
        result.i = result.j = 0;
        return result;
    }
    result.table = calloc(first_matrix.i, sizeof(double *));
    for (int i = 0; i < first_matrix.i; i++) {
        result.table[i] = calloc(second_matrix.j, sizeof(double));

    }
    for (int i = 0; i < first_matrix.i; i++) {
        for (int j = 0; j < second_matrix.j; j++) {
            result.table[i][j] = 0;
            for (int i2 = 0; i2 < first_matrix.j; i2++) {
                result.table[i][j] += first_matrix.table[i][i2] * second_matrix.table[i2][j];
            }
        }
    }
    result.i = first_matrix.i;
    result.j = second_matrix.j;

    return result;
}

matrix matrix_transposition (matrix matrix_to_transpose) {
    double **result;
    matrix result_matrix;
    result = calloc(matrix_to_transpose.j, sizeof(double *));
    for (int i = 0; i < matrix_to_transpose.j; i++) {
        result[i] = calloc(matrix_to_transpose.i, sizeof(double));
    }
    for (int i = 0; i < matrix_to_transpose.i; i++) {
        for (int j = 0; j < matrix_to_transpose.j; j++) {
            result[j][i] = matrix_to_transpose.table[i][j];
        }
    }
    result_matrix.table = result;
    result_matrix.i = matrix_to_transpose.j;
    result_matrix.j = matrix_to_transpose.i;
    return result_matrix;
}

void matrix_print (matrix matrix_to_print) {
    for (int i = 0; i < matrix_to_print.i; i++) {
        for (int j = 0; j < matrix_to_print.j; j++) {
            printf("%.2f ", matrix_to_print.table[i][j]);
        }
        printf("\n");
    }
}

void matrix_free (matrix matrix_to_free) {
    for (int i = 0; i < matrix_to_free.i; ++i) {
        free(matrix_to_free.table[i]);
    }
    free(matrix_to_free.table);
}