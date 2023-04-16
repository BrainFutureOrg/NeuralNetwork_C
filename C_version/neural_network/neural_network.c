//
// Created by maximus on 22.01.23.
//

#include "neural_network.h"
#include "network_activation_functions.h"
#include "optimizers.h"
#include "../matrix_operations.h"
#include "../statistical_random.h"
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>

#define EPSILON 0.00005

void add_function_with_derivative(neural_network *network_layer, activation_function_names activation_function_name) {
    if (activation_function_name == Sigmoid) {
        network_layer->activation_function = network_sigmoid;
        network_layer->activation_function_derivative = network_sigmoid_derivative;
        return;
    }

    if (activation_function_name == Softmax) {
        network_layer->activation_function = network_softmax;
        network_layer->activation_function_derivative = network_softmax_derivative;
        return;
    }

    if (activation_function_name == Tangential) {
        network_layer->activation_function = network_tangential;
        network_layer->activation_function_derivative = network_tangential_derivative;
        return;
    }

    if (activation_function_name == ReLu) {
        network_layer->activation_function = network_ReLU;
        network_layer->activation_function_derivative = network_ReLU_derivative;
        return;
    }
}

neural_network *last_layer(network_start_layer network) {
    neural_network *current = network.next_layer;
    if (current == NULL) return current;
    while (current->next_layer != NULL) {
        current = current->next_layer;
    }
    return current;
}

network_start_layer create_network(int neuron_numbers) {
    network_start_layer result;
    result.i = neuron_numbers;
    result.next_layer = NULL;
    return result;
}

void add_after_start_layer(network_start_layer *network, int neuron_numbers,
                           activation_function_names activation_function_name, regularization_params regularization) {
    network->next_layer = calloc(1, sizeof(neural_network));
    matrix weighs = matrix_creation(neuron_numbers, network->i);
    for (int i = 0; i < weighs.i; i++) {
        for (int j = 0; j < weighs.j; j++) {
            weighs.table[i][j] = randn();
        }
    }

    matrix bias = matrix_creation(neuron_numbers, 1);
    for (int i = 0; i < bias.i; i++) {
        bias.table[i][0] = randn();
    }

    network->next_layer->weights = weighs;
    network->next_layer->bias = bias;
    network->next_layer->next_layer = NULL;
    network->next_layer->previous_layer = NULL;
    network->next_layer->regularization_params = regularization;
    add_function_with_derivative(network->next_layer, activation_function_name);
}

void
add_after_layer(network_start_layer *network, int neuron_numbers, activation_function_names activation_function_name,
                regularization_params regularization) {
    neural_network *current = network->next_layer;
    while (current->next_layer != NULL) {
        current = current->next_layer;
    }
    current->next_layer = calloc(1, sizeof(neural_network));
    matrix weighs = matrix_creation(neuron_numbers, current->weights.i);
    for (int i = 0; i < weighs.i; i++) {
        for (int j = 0; j < weighs.j; j++) {
            weighs.table[i][j] = randn();
        }
    }

    matrix bias = matrix_creation(neuron_numbers, 1);
    for (int i = 0; i < bias.i; i++) {
        bias.table[i][0] = randn();
    }
    current->next_layer->bias = bias;

    current->next_layer->weights = weighs;
    current->next_layer->next_layer = NULL;
    current->next_layer->previous_layer = current;
    current->next_layer->regularization_params = regularization;
    add_function_with_derivative(current->next_layer, activation_function_name);
}

void add_layer(network_start_layer *network, int neuron_numbers, activation_function_names activation_function_name,
               regularization_params regularization) {
    if (network->next_layer == NULL) {
        add_after_start_layer(network, neuron_numbers, activation_function_name, regularization);
    } else {
        add_after_layer(network, neuron_numbers, activation_function_name, regularization);
    }
}

int count_hidden_layers(network_start_layer network) {
    neural_network *current = network.next_layer;
    int result = 0;
    while (current != NULL) {
        current = current->next_layer;
        result++;
    }
    return result;
}

matrix *predict_all_layers(network_start_layer network, matrix start_layer) {
    int layers_number = count_hidden_layers(network);
    neural_network *current = network.next_layer;
    matrix *current_results = calloc(layers_number + 1, sizeof(matrix));
    matrix activated_results = matrix_copy(start_layer);
    current_results[0] = matrix_copy(start_layer);
    for (int i = 1; i < layers_number + 1; i++) {
        current_results[i] = matrix_multiplication(current->weights, activated_results);
        matrix_free(activated_results);
        activated_results = matrix_addition(current_results[i], current->bias);
        current->activation_function(&activated_results);
        current = current->next_layer;
    }
    matrix_free(activated_results);
    return current_results;
}

matrix **predict_all_layers_batch(network_start_layer network, matrix *start_layers, int batch_size) {
    matrix **predictions_batched = calloc(batch_size, sizeof(matrix *));
    for (int i = 0; i < batch_size; i++) {
        predictions_batched[i] = predict_all_layers(network, start_layers[i]);
    }
    return predictions_batched;
}

/*void learn_step_optimizerless(network_start_layer network, double learning_rate, matrix start_layer,
                              matrix result_layer, int epoch) {
    matrix *prediction = predict_all_layers(network, start_layer);
    neural_network *current = last_layer(network);
    int network_layer_number = count_hidden_layers(network);

    matrix derived_results = matrix_copy_activated(prediction[network_layer_number], current->activation_function);
    matrix nablaC = matrix_substact(derived_results, result_layer);
    matrix_free(derived_results);

    derived_results = matrix_copy_activated(prediction[network_layer_number], current->activation_function_derivative);

    matrix dL = matrix_multiplication_elements(nablaC, derived_results);
    matrix_free(nablaC);
    matrix_free(derived_results);
    matrix dl = dL;
    for (int i = network_layer_number - 1; i >= 0; i--) {
        gradient_descent(current, dl, learning_rate, prediction[i], epoch);
        matrix transposed = matrix_transposition(current->weights);
        current = current->previous_layer;
        matrix multiplied = matrix_multiplication(transposed, dl);
        matrix_free(transposed);
        derived_results = matrix_copy(prediction[i]);//invalid read
        if (current != NULL) {
            current->activation_function_derivative(&derived_results);
        }
        matrix_free(dl);
        dl = matrix_multiplication_elements(multiplied, derived_results);
        matrix_free(derived_results);
        matrix_free(multiplied);
    }
    matrix_free(dl);
    for (int i = network_layer_number; i >= 0; i--)
        matrix_free(prediction[i]);
    free(prediction);
}*/

/*void learn_step_optimizerless_array(network_start_layer network, double learning_rate, matrix *start_layer,
                                    matrix *result_layer, int sample_number, double l1, double l2) {
    for (int i = 0; i < sample_number; i++) {
        learn_step_optimizerless(network, learning_rate, start_layer[i], result_layer[i], l1, l2);
    }
}

void
learn_step_optimizerless_paired_array(network_start_layer network, double learning_rate, matrix **start_result_layer,
                                      int sample_number,
                                      double l1, double l2) {
    learn_step_optimizerless_array(network, learning_rate, start_result_layer[0], start_result_layer[1], sample_number,
                                   l1, l2);
}*/

void learn_step_optimizerless_batch(network_start_layer network, double learning_rate, matrix *start_layers,
                                    matrix *result_layers, int batch_size,
                                    int epoch, general_regularization_params general_regularizations) {
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
        gradient_descent_batch(current, dl, batch_size, learning_rate, prediction, i, epoch);
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

void learn_step_optimizerless_array_batch(network_start_layer network, double learning_rate, matrix *start_layer,
                                          matrix *result_layer, int sample_number,
                                          general_regularization_params general_regularization,
                                          int epoch) {
    for (int i = 0; i < sample_number / general_regularization.batch_size; i++) {
        matrix *start_layers = calloc(general_regularization.batch_size, sizeof(matrix));
        matrix *result_layers = calloc(general_regularization.batch_size, sizeof(matrix));
        for (int j = 0; j < general_regularization.batch_size; j++) {
            start_layers[j] = start_layer[i * general_regularization.batch_size + j];
            result_layers[j] = result_layer[i * general_regularization.batch_size + j];
        }
        learn_step_optimizerless_batch(network, learning_rate, start_layers, result_layers,
                                       general_regularization.batch_size, epoch, general_regularization);
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
    learn_step_optimizerless_batch(network, learning_rate, start_layers, result_layers,
                                   sample_number % general_regularization.batch_size, epoch, general_regularization);
    free(start_layers);
    free(result_layers);
}

void learn_step_optimizerless_paired_array_batch(network_start_layer network, double learning_rate,
                                                 matrix **start_result_layer, int sample_number,
                                                 general_regularization_params general_regularization,
                                                 int epoch) {
    learn_step_optimizerless_array_batch(network, learning_rate, start_result_layer[0], start_result_layer[1],
                                         sample_number, general_regularization, epoch);
}

void print_network(network_start_layer network) {
    printf("startlayer\n");
    int i = 0;
    neural_network *current = network.next_layer;
    while (current != NULL) {
        i++;
        printf("layer %d exists\n", i);
        current = current->next_layer;
    }
}

matrix predict(network_start_layer network, matrix start_layer) {
    neural_network *current = network.next_layer;
    int i = 0;
    matrix current_results = start_layer;
    while (current != NULL) {
        matrix multiplication = matrix_multiplication(current->weights, current_results);
        if (i != 0) {
            matrix_free(current_results);
        } else {
            i++;
        }
        current_results = matrix_addition(multiplication, current->bias);
        matrix_free(multiplication);
        current->activation_function(&current_results);
        current = current->next_layer;
    }
    return current_results;
}

int predict_number(network_start_layer network, matrix start_layer) {
    matrix prediction = predict(network, start_layer);
    coordinates max_pred = matrix_argmax(prediction);

    matrix_free(prediction);
    return max_pred.i;
}

double small_accuracy(network_start_layer network, matrix start_layer, matrix answers) {
    double accuracy;

    matrix prediction = predict(network, start_layer);
    coordinates max_pred = matrix_argmax(prediction);
    coordinates max_answ = matrix_argmax(answers);

    matrix_free(prediction);
    return coordinates_equals(max_answ, max_pred);
}

double accuracy(network_start_layer network, matrix *start_layers, matrix *answers, int len_of_accuracy) {
    double accuracy = 0;
    for (int i = 0; i < len_of_accuracy; i++) {
        accuracy += small_accuracy(network, start_layers[i], answers[i]) / len_of_accuracy;
    }
    return accuracy;
}

void confusion_matrix(network_start_layer network, matrix *start_layers, matrix *answers, int len_of_data) {
    int size = answers[0].i;
    matrix confusion = matrix_creation(size, size);
    printf("predict\\true\n");
    for (int i = 0; i < len_of_data; i++) {
        matrix prediction = predict(network, start_layers[i]);
        coordinates max_pred = matrix_argmax(prediction);
        coordinates max_answ = matrix_argmax(answers[i]);
        confusion.table[max_pred.i][max_answ.i] += 1;
        matrix_free(prediction);

    }
    matrix_print_with_indexation(confusion, 4, 0);
    matrix_free(confusion);
}

void confusion_matrix_paired(network_start_layer network, matrix **start_result_layers, int len_of_data) {
    confusion_matrix(network, start_result_layers[0], start_result_layers[1], len_of_data);
}

void free_network(network_start_layer network) {
    neural_network *current = network.next_layer;
    while (current != NULL) {
        neural_network *current2 = current->next_layer;
        matrix_free(current->weights);
        matrix_free(current->bias);
        free(current);
        current = current2;
    }
    network.next_layer = NULL;
}

double mono_mse_loss(network_start_layer network, matrix start_layer, matrix expected_results) {
    matrix real_results = predict(network, start_layer);
    matrix subtracted = matrix_substact(real_results, expected_results);
    double mse = 0;
    for (int i = 0; i < subtracted.i; i++) {
        double add = matrix_get_element(subtracted, i, 0);
        mse += add * add;
    }
    matrix_free(subtracted);
    matrix_free(real_results);
    return mse / expected_results.i;
}

double mse_loss(network_start_layer network, matrix *start_layers, int sample_number, matrix *expected_results) {
    double sum = 0;
    for (int i = 0; i < sample_number; i++) {
        sum += mono_mse_loss(network, start_layers[i], expected_results[i]);
    }
    return sum / sample_number;
}

double many_loss(network_start_layer network, matrix *start_layers, int sample_number, matrix *expected_results,
                 general_regularization_params general_regularization) {
    double sum = 0;
    for (int i = 0; i < sample_number; i++) {
        matrix real_results = predict(network, start_layers[i]);
        sum += general_regularization.cost_function(real_results, expected_results[i]) / sample_number;
        matrix_free(real_results);
    }
    return sum;
}

void test_network(network_start_layer network, matrix *start_layers, int start_layer_number, matrix *expected_results,
                  general_regularization_params general_regularization) {
    double accuracy_num = accuracy(network, start_layers, expected_results, start_layer_number);
    //double loss_num = mse_loss(network, start_layers, start_layer_number, expected_results);
    double loss_num = many_loss(network, start_layers, start_layer_number, expected_results, general_regularization);
    printf("accuracy: %f, loss: %f\n", accuracy_num, loss_num);
}

void test_network_paired(network_start_layer network, matrix **start_result_layers, int sample_number,
                         general_regularization_params general_regularization) {
    test_network(network, start_result_layers[0], sample_number, start_result_layers[1], general_regularization);
}

neural_network *copy_neural_network_layer(neural_network *layer) {
    neural_network *copy = calloc(1, sizeof(neural_network));
    copy->next_layer = copy->previous_layer = NULL;
    copy->activation_function_derivative = layer->activation_function_derivative;
    copy->activation_function = layer->activation_function;
    copy->weights = matrix_copy(layer->weights);
    copy->bias = matrix_copy(layer->bias);
    copy->regularization_params = layer->regularization_params;
    return copy;
}

network_start_layer neural_network_copy(network_start_layer network) {
    network_start_layer network_copy = network;
    neural_network *network_layers = network.next_layer;
    neural_network *last_copied_layer = copy_neural_network_layer(network_layers);
    network_copy.next_layer = last_copied_layer;
    while ((network_layers = network_layers->next_layer) != NULL) {
//        printf("copy_");
        neural_network *layer_copy = copy_neural_network_layer(network_layers);
        layer_copy->previous_layer = last_copied_layer;
        last_copied_layer->next_layer = layer_copy;
        last_copied_layer = layer_copy;
    }
//    printf("\n");
    return network_copy;
}