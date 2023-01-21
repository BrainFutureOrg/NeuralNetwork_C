//
// Created by maximus on 21.01.23.
//

#ifndef C_VERSION_MATRIX_OPERATIONS_H
#define C_VERSION_MATRIX_OPERATIONS_H
typedef struct matrix{
    double **table;
    int i;
    int j;
} matrix;

matrix matrix_multiplication(matrix first_matrix, matrix second_matrix);

matrix matrix_transposition(matrix matrix_to_transpose);

void matrix_print(matrix matrix_to_print);

void matrix_free(matrix matrix_to_free);

#endif //C_VERSION_MATRIX_OPERATIONS_H
