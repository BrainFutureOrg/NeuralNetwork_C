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
    result_matrix = matrix_creation(matrix_to_transpose.j, matrix_to_transpose.i);
    for (int i = 0; i < matrix_to_transpose.i; i++) {
        for (int j = 0; j < matrix_to_transpose.j; j++) {
            result_matrix.table[j][i] = matrix_to_transpose.table[i][j];
        }
    }
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

void matrix_function_to_elements(matrix matrix_for_operation, double (*func)(double)) {
    for (int i = 0; i < matrix_for_operation.i; i++) {
        for (int j = 0; j < matrix_for_operation.j; j++) {
            matrix_for_operation.table[j][i] = func(matrix_for_operation.table[j][i]);
        }
    }
}

void matrix_multiply_by_constant(matrix matrix_for_operation, double number) {
    for (int i = 0; i < matrix_for_operation.i; i++) {
        for (int j = 0; j < matrix_for_operation.j; j++) {
            matrix_for_operation.table[j][i] = number * matrix_for_operation.table[j][i];
        }
    }
}

matrix make_matrix_from_array(double **double_array, int i, int j) {
    matrix result;
    result = matrix_creation(i, j);
    for (int iterator = 0; iterator < i; iterator++) {
        for (int iterator1 = 0; iterator1 < j; iterator1++) {
            result.table[iterator][iterator1] = double_array[iterator][iterator1];
        }
    }
    return result;
}