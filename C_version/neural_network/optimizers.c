//
// Created by kosenko on 04.04.23.
//
#include "../matrix_operations.h"
#include "optimizers.h"
#include "neural_network.h"
#include <stdlib.h>
#include <math.h>

double signum(double a) {
    return a > 0 ? 1 : a < 0 ? -1 : 0;
}

void gradient_descent(neural_network *layer, matrix error, double learning_rate, matrix previous_values, double l1,
                      double l2) {
    matrix multiplied = matrix_copy(error);
    matrix_multiply_by_constant(multiplied, learning_rate);
    matrix new_bias = matrix_substact(layer->bias, multiplied);//+-
    matrix_free(layer->bias);
    layer->bias = new_bias;
    matrix_free(multiplied);

    matrix a = matrix_copy(previous_values);
    if (layer->previous_layer != NULL)
        layer->previous_layer->activation_function(&a);
    matrix transpozed = matrix_transposition(a);
    matrix_free(a);
    multiplied = matrix_multiplication(error, transpozed);
    matrix_free(transpozed);
    matrix_multiply_by_constant(multiplied, learning_rate);
    matrix new_weights = matrix_substact(layer->weights, multiplied);//+-

    matrix l1_mtrx = matrix_copy(layer->weights);
    matrix_function_to_elements(l1_mtrx, signum);
    matrix_multiply_by_constant(l1_mtrx, l1);

    matrix l2_mtrx = matrix_copy(layer->weights);
    matrix_multiply_by_constant(l2_mtrx, l2);

    matrix_subtract_inplace(new_weights, l1_mtrx);
    matrix_subtract_inplace(new_weights, l2_mtrx);

    matrix_free(layer->weights);
    layer->weights = new_weights;
    matrix_free(multiplied);
    matrix_free(l1_mtrx);
    matrix_free(l2_mtrx);
}

void gradient_descent_batch(neural_network *layer, matrix *error, int batch_size, double learning_rate,
                            matrix **previous_values, int number_of_current_layer, double l1, double l2) {
    matrix new_weights = matrix_copy(layer->weights);
    matrix new_bias = matrix_copy(layer->bias);
    for (int i = 0; i < batch_size; i++) {
        matrix multiplied = matrix_copy(error[i]);
        matrix_multiply_by_constant(multiplied, learning_rate);
        matrix_subtract_inplace(layer->bias, multiplied);//+-

        matrix_free(multiplied);

        matrix a = matrix_copy(previous_values[i][number_of_current_layer]);
        if (layer->previous_layer != NULL)
            layer->previous_layer->activation_function(&a);
        matrix transpozed = matrix_transposition(a);
        matrix_free(a);
        multiplied = matrix_multiplication(error[i], transpozed);
        matrix_free(transpozed);
        matrix_multiply_by_constant(multiplied, learning_rate);
        matrix_subtract_inplace(new_weights, multiplied);//+-
        matrix_free(multiplied);
    }

    matrix l1_mtrx = matrix_copy(layer->weights);
    matrix_function_to_elements(l1_mtrx, signum);
    matrix_multiply_by_constant(l1_mtrx, l1);

    matrix l2_mtrx = matrix_copy(layer->weights);
    matrix_multiply_by_constant(l2_mtrx, l2);

    matrix_subtract_inplace(new_weights, l1_mtrx);
    matrix_subtract_inplace(new_weights, l2_mtrx);

    matrix_free(layer->bias);
    matrix_free(layer->weights);
    layer->weights = new_weights;
    layer->bias = new_bias;

    matrix_free(l1_mtrx);
    matrix_free(l2_mtrx);
}

weight_bias
gradient_descent_delta(neural_network *layer, matrix error, double learning_rate, matrix previous_values, double l1,
                       double l2) {
    weight_bias result;
    matrix multiplied = matrix_copy(error);
    matrix_multiply_by_constant(multiplied, learning_rate);
    result.bias = multiplied;

    matrix a = matrix_copy(previous_values);
    if (layer->previous_layer != NULL)
        layer->previous_layer->activation_function(&a);
    matrix transpozed = matrix_transposition(a);
    matrix_free(a);
    multiplied = matrix_multiplication(error, transpozed);
    matrix_free(transpozed);
    matrix_multiply_by_constant(multiplied, learning_rate);

    matrix l1_mtrx = matrix_copy(layer->weights);
    matrix_function_to_elements(l1_mtrx, signum);
    matrix_multiply_by_constant(l1_mtrx, l1);

    matrix l2_mtrx = matrix_copy(layer->weights);
    matrix_multiply_by_constant(l2_mtrx, l2);

    matrix_addition_inplace(multiplied, l1_mtrx);
    matrix_addition_inplace(multiplied, l2_mtrx);

    matrix_free(l1_mtrx);
    matrix_free(l2_mtrx);
    result.weight = multiplied;
    return result;
}

void
gradient_descent_dual(neural_network *original_layer, neural_network *changed_layer, matrix error, double learning_rate,
                      matrix previous_values, double l1, double l2) {
    matrix multiplied = matrix_copy(error);
    matrix_multiply_by_constant(multiplied, learning_rate);
    matrix new_bias = matrix_substact(original_layer->bias, multiplied);//+-
    matrix_free(changed_layer->bias);
    changed_layer->bias = new_bias;
    matrix_free(multiplied);

    matrix a = matrix_copy(previous_values);
    if (original_layer->previous_layer != NULL)
        original_layer->previous_layer->activation_function(&a);
    matrix transpozed = matrix_transposition(a);
    matrix_free(a);
    multiplied = matrix_multiplication(error, transpozed);
    matrix_free(transpozed);
    matrix_multiply_by_constant(multiplied, learning_rate);
    matrix new_weights = matrix_substact(original_layer->weights, multiplied);//+-

    matrix l1_mtrx = matrix_copy(new_weights);
    matrix_function_to_elements(l1_mtrx, signum);
    matrix_multiply_by_constant(l1_mtrx, l1);

    matrix l2_mtrx = matrix_copy(new_weights);
    matrix_multiply_by_constant(l2_mtrx, l2);

    matrix_subtract_inplace(new_weights, l1_mtrx);
    matrix_subtract_inplace(new_weights, l2_mtrx);

    matrix_free(changed_layer->weights);
    changed_layer->weights = new_weights;
    matrix_free(multiplied);
    matrix_free(l1_mtrx);
    matrix_free(l2_mtrx);
}