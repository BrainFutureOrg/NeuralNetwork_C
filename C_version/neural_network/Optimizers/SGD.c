#include "SGD.h"
#include "optimizers_functions.h"
#include "../../terminal_output/progres_bar.h"

void gradient_descent_sgd_batch(neural_network *layer, matrix *error, int batch_size, double learning_rate,
                            matrix **previous_values, int number_of_current_layer, int epoch, void *grid_params) {
    matrix new_weights = matrix_copy(layer->weights);
    matrix new_bias = matrix_copy(layer->bias);
    for (int i = 0; i < batch_size; i++) {
        matrix multiplied = matrix_copy(error[i]);
        matrix_multiply_by_constant(multiplied, learning_rate);
        matrix_subtract_inplace(new_bias, multiplied);
        matrix_free(multiplied);

        multiplied= multiplied_for_weights(layer, error, previous_values, number_of_current_layer, i);
        matrix_multiply_by_constant(multiplied, learning_rate);
        matrix_subtract_inplace(new_weights, multiplied);
        matrix_free(multiplied);
    }
    finish_gd(layer, new_weights, new_bias, epoch);
}

void learn_step_sgd_reader(network_start_layer network, double learning_rate, data_reader *reader, int epoch){
    learn_step_reader(network, learning_rate, reader, epoch, NULL, NULL, gradient_descent_sgd_batch);
}



