//
// Created by maximus on 17.04.23.
//

#include <stdlib.h>
#include "optimizers_functions.h"
#include "../neural_network.h"

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