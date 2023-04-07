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
typedef struct coordinates{
    int i;
    int j;
}coordinates;
matrix matrix_multiplication(matrix first_matrix, matrix second_matrix);

matrix matrix_transposition(matrix matrix_to_transpose);

matrix matrix_addition(matrix first_matrix, matrix second_matrix);

matrix matrix_substact(matrix first_matrix, matrix second_matrix);

matrix matrix_multiplication_elements(matrix first_matrix, matrix second_matrix);

matrix matrix_creation(int i, int j);

void matrix_print(matrix matrix_to_print);

void matrix_free(matrix matrix_to_free);

matrix matrix_copy(matrix matrix_to_copy);
matrix matrix_copy_activated(matrix matrix_to_copy, void (*active_func)(matrix *));

void matrix_function_to_elements(matrix matrix_for_operation, double (*func)(double));

void matrix_multiply_by_constant(matrix matrix_for_operation, double number);

void matrix_add_scalar(matrix matrix_for_operation, double scalar);

matrix make_matrix_from_array(const double *double_array, int i, int j);

matrix create_vector(int feach_num, int this);

void matrix_restrict(matrix matrix_for_operation, double limit);

double frobenius_norm(matrix matrix_for_operation);

void frobenius_normalize(matrix matrix_for_operation);

double matrix_max_absolute(matrix matrix_for_operation);

void max_abs_normalize(matrix matrix_for_operation);

coordinates matrix_argmax(matrix matrix_for_operation);

char coordinates_equals(coordinates coordinates1, coordinates coordinates2);

#endif //C_VERSION_MATRIX_OPERATIONS_H
