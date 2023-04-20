//
// Created by maximus on 19.04.23.
//

#include "Adam_optimizer.h"
#include "../neural_network.h"
#include "optimizers_functions.h"
#include "../../terminal_output/progres_bar.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <errno.h>

#define EPSILON 0.000000001

void gradient_descent_adam_batch(neural_network *layer, matrix *error, int batch_size, double learning_rate,
                                 matrix **previous_values, int number_of_current_layer, int epoch,
                                 matrix momentum_weights, matrix momentum_bias, matrix s_weights, matrix s_bias,
                                 double b1, double b2) {
    matrix new_weights = matrix_copy(layer->weights);
    matrix new_bias = matrix_copy(layer->bias);

    double b1t = b1;
    double b2t = b2;
    for (int t = 0; t < epoch; t++) {
        b1t *= b1;
        b2t *= b2;
    }

    for (int i = 0; i < batch_size; i++) {
        matrix multiplied = matrix_copy(error[i]);
        matrix_multiply_by_constant(multiplied, (1 - b1));
        //matrix_subtract_inplace(new_bias, multiplied);
        matrix_multiply_by_constant(momentum_bias, b1);
        matrix_subtract_inplace(momentum_bias, multiplied);

        matrix squared_error = matrix_multiplication_elements(error[i], error[i]);
        matrix_multiply_by_constant(squared_error, (1 - b2));

        matrix_multiply_by_constant(s_bias, b2);
        matrix_addition_inplace(s_bias, squared_error);

        matrix momentum_bias_hat = matrix_copy(momentum_bias);

        matrix_multiply_by_constant(momentum_bias_hat, 1 / (1 - b1t));

        matrix s_bias_hat = matrix_copy(s_bias);
        matrix_multiply_by_constant(s_bias_hat, 1 / (1 - b2t));

        for (int k = 0; k < new_bias.i; k++) {
            new_bias.table[k][0] +=
                    learning_rate * momentum_bias_hat.table[k][0] / sqrt(s_bias_hat.table[k][0] + EPSILON);
        }
        //matrix_addition_inplace(new_bias, momentum_bias);
        matrix_free(s_bias_hat);
        matrix_free(momentum_bias_hat);
        matrix_free(squared_error);

        matrix_free(multiplied);

        matrix a = matrix_copy(previous_values[i][number_of_current_layer]);
        if (layer->previous_layer != NULL)
            layer->previous_layer->activation_function(&a);
        matrix transpozed = matrix_transposition(a);
        matrix_free(a);
        multiplied = matrix_multiplication(error[i], transpozed);
        matrix_free(transpozed);
//        printf("0) %d\n", errno);

        //squared_error = matrix_copy(multiplied);
        squared_error = matrix_multiplication_elements(multiplied, multiplied);//new cringe
        matrix_multiply_by_constant(squared_error, 1 - b2);//
        matrix_multiply_by_constant(s_weights, b2);//new cringe
        matrix_addition_inplace(s_weights, squared_error);

        matrix_multiply_by_constant(multiplied, 1 - b1);
        matrix_multiply_by_constant(momentum_weights, b1);
        matrix_subtract_inplace(momentum_weights, multiplied);

        //squared_error =;


        matrix momentum_weights_hat = matrix_copy(momentum_weights);
        matrix_multiply_by_constant(momentum_weights_hat, 1 / (1 - b1t));

        matrix s_weights_hat = matrix_copy(s_weights);
        matrix_multiply_by_constant(s_weights_hat, 1 / (1 - b2t));

        for (int k = 0; k < new_weights.i; k++) {
            for (int j = 0; j < new_weights.j; j++) {
                new_weights.table[k][j] +=
                        learning_rate * momentum_weights_hat.table[k][j] / sqrt(s_weights_hat.table[k][j] + EPSILON);
            }
        }

        matrix_free(s_weights_hat);
        matrix_free(momentum_weights_hat);
        matrix_free(squared_error);

        matrix_free(multiplied);
    }

    matrix l1_mtrx = matrix_copy(layer->weights);
    matrix_function_to_elements(l1_mtrx, signum);
    matrix_multiply_by_constant(l1_mtrx, layer->regularization_params.l1(epoch));

    matrix l2_mtrx = matrix_copy(layer->weights);
    matrix_multiply_by_constant(l2_mtrx, layer->regularization_params.l2(epoch));

    matrix_subtract_inplace(new_weights, l1_mtrx);
    matrix_subtract_inplace(new_weights, l2_mtrx);

    matrix_free(layer->bias);
    matrix_free(layer->weights);
    layer->weights = new_weights;
    layer->bias = new_bias;

    matrix_free(l1_mtrx);
    matrix_free(l2_mtrx);
}

void learn_step_adam_batch(network_start_layer network, double learning_rate, matrix *start_layers,
                           matrix *result_layers, int batch_size,
                           int epoch, general_regularization_params general_regularizations,
                           matrix *momentum_weights, matrix *momentum_bias, matrix *s_weights, matrix *s_bias,
                           double b1, double b2) {
    matrix **prediction = predict_all_layers_batch(network, start_layers, batch_size);
    neural_network *current = last_layer(network);
    int network_layer_number = count_hidden_layers(network);
    matrix *dl = calloc(batch_size, sizeof(matrix));
    for (int i = 0; i < batch_size; i++) {
        matrix derived_results = matrix_copy_activated(prediction[i][network_layer_number],
                                                       current->activation_function);
        //matrix nablaC = matrix_substact(derived_results, result_layers[i]);//TODO cross entropy
        matrix nablaC = general_regularizations.nablaC(derived_results, result_layers[i]);
        matrix_free(derived_results);
        derived_results = matrix_copy_activated(prediction[i][network_layer_number],
                                                current->activation_function_derivative);

        dl[i] = matrix_multiplication_elements(nablaC, derived_results);
        matrix_free(nablaC);
        matrix_free(derived_results);
    }
    for (int i = network_layer_number - 1; i >= 0; i--) {
        gradient_descent_adam_batch(current, dl, batch_size, learning_rate, prediction, i, epoch,
                                    momentum_weights[i],
                                    momentum_bias[i], s_weights[i], s_bias[i], b1, b2);
        matrix transposed = matrix_transposition(current->weights);
        current = current->previous_layer;
        for (int j = 0; j < batch_size; j++) {
            matrix multiplied = matrix_multiplication(transposed, dl[j]);
            matrix derived_results = matrix_copy(prediction[j][i]);
            if (current != NULL) {
                current->activation_function_derivative(&derived_results);
            }
            matrix_free(dl[j]);
            dl[j] = matrix_multiplication_elements(multiplied, derived_results);
            matrix_free(derived_results);
            matrix_free(multiplied);
        }
        matrix_free(transposed);
    }
    matrix_free_arrayed(dl, batch_size);
    for (int i = 0; i < batch_size; i++)
        matrix_free_arrayed(prediction[i], network_layer_number + 1);
    free(prediction);
}

void learn_step_adam_array_batch(network_start_layer network, double learning_rate, matrix *start_layer,
                                 matrix *result_layer, int sample_number,
                                 general_regularization_params general_regularization,
                                 int epoch, double b1, double b2) {
    int n = count_hidden_layers(network);
    matrix *momentum_weights = calloc(n, sizeof(matrix));
    matrix *momentum_bias = calloc(n, sizeof(matrix));
    matrix *s_weights = calloc(n, sizeof(matrix));
    matrix *s_bias = calloc(n, sizeof(matrix));
    neural_network *current_layer = network.next_layer;
    for (int k = 0; k < n; k++) {
        momentum_weights[k] = matrix_creation(current_layer->weights.i, current_layer->weights.j);
        momentum_bias[k] = matrix_creation(current_layer->bias.i, 1);
        s_weights[k] = matrix_creation(current_layer->weights.i, current_layer->weights.j);
        s_bias[k] = matrix_creation(current_layer->bias.i, 1);
        for (int i = 0; i < current_layer->weights.i; i++) {
            momentum_bias[k].table[i][0] = 0;
            s_bias[k].table[i][0] = 0;
            for (int j = 0; j < current_layer->weights.j; j++) {
                momentum_weights[k].table[i][j] = 0;
                s_weights[k].table[i][j] = 0;
            }
        }
//        if (current_layer->previous_layer != NULL)
        current_layer = current_layer->next_layer;
    }
    int iter_number = sample_number / general_regularization.batch_size;
    progress_bar bar = create_bar(iter_number);

    for (int i = 0; i < iter_number; i++) {
        matrix *start_layers = calloc(general_regularization.batch_size, sizeof(matrix));
        matrix *result_layers = calloc(general_regularization.batch_size, sizeof(matrix));
        for (int j = 0; j < general_regularization.batch_size; j++) {
            start_layers[j] = start_layer[i * general_regularization.batch_size + j];
            result_layers[j] = result_layer[i * general_regularization.batch_size + j];
        }
        learn_step_adam_batch(network, learning_rate, start_layers, result_layers,
                              general_regularization.batch_size, epoch, general_regularization, momentum_weights,
                              momentum_bias, s_weights, s_bias, b1, b2);
        bar_step(&bar);
        free(start_layers);
        free(result_layers);
    }
    matrix *start_layers = calloc(sample_number % general_regularization.batch_size, sizeof(matrix));
    matrix *result_layers = calloc(sample_number % general_regularization.batch_size, sizeof(matrix));
    for (int j = 0; j < sample_number % general_regularization.batch_size; j++) {
        start_layers[j] = start_layer[
                (sample_number / general_regularization.batch_size) * general_regularization.batch_size + j];
        result_layers[j] = result_layer[
                (sample_number / general_regularization.batch_size) * general_regularization.batch_size + j];
    }
    learn_step_adam_batch(network, learning_rate, start_layers, result_layers,
                          sample_number % general_regularization.batch_size, epoch, general_regularization,
                          momentum_weights, momentum_bias, s_weights, s_bias, b1, b2);
    delete_bar(&bar);
    free(start_layers);
    free(result_layers);
    for (int i = 0; i < n; i++) {
        matrix_free(momentum_weights[i]);
        matrix_free(momentum_bias[i]);
    }
    free(momentum_weights);
    free(momentum_bias);
    free(s_weights);
    free(s_bias);
}

void learn_step_adam_paired_array_batch(network_start_layer network, double learning_rate,
                                        matrix **start_result_layer, int sample_number,
                                        general_regularization_params general_regularization,
                                        int epoch, double b1, double b2) {
    learn_step_adam_array_batch(network, learning_rate, start_result_layer[0], start_result_layer[1],
                                sample_number, general_regularization, epoch, b1, b2);
}