//
// Created by maximus on 22.01.23.
//

#ifndef C_VERSION_NEURAL_NETWORK_H
#define C_VERSION_NEURAL_NETWORK_H

#include "../math/matrix_operations.h"
#include "neural_structs.h"
#include "../math/batch_operations.h"
#include "../data/DAO.h"


/*typedef union grid_param {
    double d;
    int i;
} grid_param;*/
typedef enum grid_param_type {
    DOUBLE, INT
} grid_param_type;
typedef struct grid_param {
    grid_param_type type;
    union {
        double d;
        int i;
    };
} grid_param;

network_start_layer create_network(int neuron_numbers);

void add_function_with_derivative(neural_network *network_layer, activation_function_names activation_function_name);

void add_layer(network_start_layer *network, int neuron_numbers, activation_function_names activation_function_name,
               regularization_params regularization);

//void learn_step(network_start_layer network, double learning_rate, matrix start_layer, matrix result_layer);

matrix predict_average(network_start_layer *networks, int network_number, matrix start_layer);

matrix predict(network_start_layer network, matrix start_layer);

int predict_number(network_start_layer network, matrix start_layer);

double accuracy(network_start_layer network, batch start_layers, batch answers);

double small_accuracy(network_start_layer network, matrix start_layer, matrix answers);

void print_network(network_start_layer network);

void free_network(network_start_layer startLayer);

/*void learn_step_optimizerless(network_start_layer network, double learning_rate, matrix start_layer,
                              matrix result_layer, double l1, double l2);

void learn_step_optimizerless_array(network_start_layer network, double learning_rate, matrix *start_layer,
                                    matrix *result_layer, int sample_number, double l1, double l2);

void
learn_step_optimizerless_paired_array(network_start_layer network, double learning_rate, matrix **start_result_layer,
                                      int sample_number,
                                      double l1, double l2);*/

/*void learn_step_optimizerless_batch(network_start_layer network, double learning_rate, matrix *start_layers,
                                    matrix *result_layers, int batch_size,
                                    int epoch);
*/


double mse_loss(network_start_layer network, matrix *start_layers, int sample_number, matrix *expected_results);

//double* test_network(network_start_layer network, batch start_layers, batch expected_results,
//                  general_regularization_params general_regularization);

void test_network_paired(network_start_layer network, data_reader *reader,
                         general_regularization_params general_regularization);

matrix confusion_matrix(network_start_layer network, batch start_layers, batch answers);

void confusion_matrix_paired(network_start_layer network, data_reader *reader);

network_start_layer neural_network_copy(network_start_layer network);

#endif //C_VERSION_NEURAL_NETWORK_H
