//
// Created by maximus on 21.01.23.
//

#ifndef C_VERSION_MATRIX_OPERATIONS_H
#define C_VERSION_MATRIX_OPERATIONS_H
typedef struct matrix {
    double **table;
    int i;
    int j;
} matrix;

matrix matrix_multiplication(matrix first_matrix, matrix second_matrix);

matrix matrix_transposition(matrix matrix_to_transpose);

matrix matrix_addition(matrix first_matrix, matrix second_matrix);

matrix matrix_creation(int i, int j);

void matrix_print(matrix matrix_to_print);

void matrix_free(matrix matrix_to_free);

matrix matrix_copy(matrix matrix_to_copy);

void matrix_function_to_elements(matrix matrix_for_operation, double (*func)(double));

void matrix_multiply_by_constant(matrix matrix_for_operation, double number);

matrix make_matrix_from_array(double **double_array, int i, int j);

#endif //C_VERSION_MATRIX_OPERATIONS_H
