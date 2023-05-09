//
// Created by maximus on 17.04.23.
//

#include <stdlib.h>
#include "optimizers_functions.h"
#include "../neural_network.h"
#include "../../terminal_output/progres_bar.h"

double signum(double a) {
    return a > 0 ? 1 : a < 0 ? -1 : 0;
}

neural_network *last_layer(network_start_layer network) {
    neural_network *current = network.next_layer;
    if (current == NULL) return current;
    while (current->next_layer != NULL) {
        current = current->next_layer;
    }
    return current;
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


void learn_step_batch(network_start_layer network, double learning_rate, batch *start_result_layers,
                      int epoch, general_regularization_params general_regularizations,
                      void (*gradient_descent)(neural_network *, matrix *, int, double, matrix **, int, int, void *),
                      void *gradient_params) {
    matrix **prediction = predict_all_layers_batch(network, start_result_layers[0].batch_elements,
                                                   start_result_layers[0].size);
    neural_network *current = last_layer(network);
    int network_layer_number = count_hidden_layers(network);
    matrix *dl = calloc(start_result_layers[0].size, sizeof(matrix));

    for (int i = 0; i < start_result_layers[0].size; i++) {
        matrix derived_results = matrix_copy_activated(prediction[i][network_layer_number],
                                                       current->activation_function);
        matrix nablaC = general_regularizations.nablaC(*current, derived_results,
                                                       start_result_layers[1].batch_elements[i]);
        matrix_free(derived_results);

        derived_results = matrix_copy_activated(prediction[i][network_layer_number],
                                                current->activation_function_derivative);

        dl[i] = matrix_multiplication_elements(nablaC, derived_results);
        matrix_free(nablaC);
        matrix_free(derived_results);
    }
    for (int i = network_layer_number - 1; i >= 0; i--) {

        gradient_descent(current, dl, start_result_layers[0].size, learning_rate, prediction, i, epoch,
                         gradient_params);
        matrix transposed = matrix_transposition(current->weights);
        current = current->previous_layer;
        for (int j = 0; j < start_result_layers[0].size; j++) {
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
    matrix_free_arrayed(dl, start_result_layers[0].size);
    for (int i = 0; i < start_result_layers[0].size; i++)
        matrix_free_arrayed(prediction[i], network_layer_number + 1);
    free(prediction);
}

void learn_step_reader(network_start_layer network, double learning_rate, data_reader *reader,
                       general_regularization_params general_regularization,
                       int epoch, void *(*create_gradient_params)(network_start_layer),
                       void *(*free_gradient_params)(network_start_layer, void *),
                       void (*gradient_descent)(neural_network *, matrix *, int, double, matrix **, int, int, void *)) {
    int iter_number =
            reader->sample_number / reader->batch_size + (reader->sample_number % reader->batch_size == 0 ? 0 : 1);
    progress_bar bar = create_bar(iter_number);
    void *gradient_params = create_gradient_params(network);
    for (int i = 0; i < iter_number; i++) {
        batch *batch_pair = read_batch_from_data_nn(reader);
        learn_step_batch(network, learning_rate, batch_pair, epoch, general_regularization, gradient_descent,
                         NULL);
        batch_free(batch_pair[0]);
        batch_free(batch_pair[1]);
        free(batch_pair);
        bar_step(&bar);
    }
    delete_bar(&bar);
    free_gradient_params(network, gradient_params);
    data_reader_rollback(reader);
}
