//
// Created by kosenko on 04.04.23.
//
#include "../matrix_operations.h"
#include "optimizers.h"
#include "neural_network.h"
#include <stdlib.h>
void gradient_descent(neural_network* layer, matrix error, double learning_rate, matrix previous_values){
    matrix multiplied = matrix_copy(error);
    matrix_multiply_by_constant(multiplied, learning_rate);
    matrix new_bias = matrix_substact(layer->bias, multiplied);//+-
    matrix_free(layer->bias);
    layer->bias=new_bias;
    matrix_free(multiplied);

    matrix a=previous_values;
    if(layer->previous_layer!=NULL)layer->previous_layer->activation_function(&a);
    matrix transpozed= matrix_transposition(a);
    matrix_free(a);
    multiplied = matrix_multiplication(error, transpozed);
    matrix_free(transpozed);
    matrix_multiply_by_constant(multiplied, learning_rate);
    matrix new_weights= matrix_substact(layer->weights, multiplied);//+-
    matrix_add_scalar(new_weights,-l2);
    matrix_free(layer->weights);
    layer->weights=new_weights;
//    matrix_free(multiplied);
}