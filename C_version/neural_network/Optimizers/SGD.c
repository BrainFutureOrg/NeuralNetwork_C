
#include "SGD.h"
#include "../neural_network.h"
#include "optimizers_functions.h"
#include <stdlib.h>

void learn_step_sgd_batch(network_start_layer network, double learning_rate, batch *start_result_layers,
                          int epoch, general_regularization_params general_regularizations) {
    matrix **prediction = predict_all_layers_batch(network, start_result_layers[0].batch_elements,
                                                   start_result_layers[0].size);
    neural_network *current = last_layer(network);
    int network_layer_number = count_hidden_layers(network);
    matrix *dl = calloc(start_result_layers[0].size, sizeof(matrix));

    for (int i = 0; i < start_result_layers[0].size; i++) {
        matrix derived_results = matrix_copy_activated(prediction[i][network_layer_number],
                                                       current->activation_function);
        //matrix nablaC = matrix_substact(derived_results, result_layers[i]);//TODO cross entropy
        matrix nablaC = general_regularizations.nablaC(derived_results, start_result_layers[1].batch_elements[i]);
        matrix_free(derived_results);

        derived_results = matrix_copy_activated(prediction[i][network_layer_number],
                                                current->activation_function_derivative);

        dl[i] = matrix_multiplication_elements(nablaC, derived_results);
        matrix_free(nablaC);
        matrix_free(derived_results);
    }
    for (int i = network_layer_number - 1; i >= 0; i--) {
        gradient_descent_batch(current, dl, start_result_layers[0].size, learning_rate, prediction, i, epoch);
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

/*void learn_step_optimizerless_array_batch(network_start_layer network, double learning_rate, matrix *start_layer,
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
*/
void learn_step_sgd_reader_batch(network_start_layer network, double learning_rate, data_reader *reader,
                                 general_regularization_params general_regularization,
                                 int epoch) {
    int iter_number =
            reader->sample_number / reader->batch_size + (reader->sample_number % reader->batch_size == 0 ? 0 : 1);

    for (int i = 0; i < iter_number; i++) {
        //matrix *start_layers = calloc(general_regularization.batch_size, sizeof(matrix));
        //matrix *result_layers = calloc(general_regularization.batch_size, sizeof(matrix));
        /*for (int j = 0; j < general_regularization.batch_size; j++) {
            start_layers[j] = start_layer[i * general_regularization.batch_size + j];
            result_layers[j] = result_layer[i * general_regularization.batch_size + j];
        }*/
        batch *batch_pair = read_batch_from_data_nn(reader);
        learn_step_sgd_batch(network, learning_rate, batch_pair, epoch, general_regularization);
        //free(start_layers);
        //free(result_layers);
        batch_free(batch_pair[0]);
        batch_free(batch_pair[1]);
        free(batch_pair);
    }
    /*matrix *start_layers = calloc(sample_number % general_regularization.batch_size, sizeof(matrix));
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
    free(result_layers);*/
}

/*void learn_step_optimizerless_paired_array_batch(network_start_layer network, double learning_rate,
                                                 matrix **start_result_layer, int sample_number,
                                                 general_regularization_params general_regularization,
                                                 int epoch) {
    learn_step_optimizerless_array_batch(network, learning_rate, start_result_layer[0], start_result_layer[1],
                                         sample_number, general_regularization, epoch);
}*/


/*void gradient_descent(neural_network *layer, matrix error, double learning_rate, matrix previous_values, int epoch) {
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
    matrix_multiply_by_constant(l1_mtrx, layer->regularization_params.l1(epoch));

    matrix l2_mtrx = matrix_copy(layer->weights);
    matrix_multiply_by_constant(l2_mtrx, layer->regularization_params.l2(epoch));

    matrix_subtract_inplace(new_weights, l1_mtrx);
    matrix_subtract_inplace(new_weights, l2_mtrx);

    matrix_free(layer->weights);
    layer->weights = new_weights;
    matrix_free(multiplied);
    matrix_free(l1_mtrx);
    matrix_free(l2_mtrx);
}*/

void gradient_descent_batch(neural_network *layer, matrix *error, int batch_size, double learning_rate,
                            matrix **previous_values, int number_of_current_layer, int epoch) {
    matrix new_weights = matrix_copy(layer->weights);
    matrix new_bias = matrix_copy(layer->bias);
    for (int i = 0; i < batch_size; i++) {
        matrix multiplied = matrix_copy(error[i]);
        matrix_multiply_by_constant(multiplied, learning_rate);
        matrix_subtract_inplace(new_bias, multiplied);//changed (cringe possible)

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
