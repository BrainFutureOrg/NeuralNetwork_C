//
// Created by maximus on 18.04.23.
//
#include "../neural_structs.h"
#include "../../data/DAO.h"

#ifndef C_VERSION_NESTEROV_ACCELERATED_GD_H
#define C_VERSION_NESTEROV_ACCELERATED_GD_H

typedef struct Nesterov_params {
    double friction;
} Nesterov_params;
/*
void learn_step_nesterov_array_batch(network_start_layer network, double learning_rate, matrix *start_layer,
                                     matrix *result_layer, int sample_number,
                                     general_regularization_params general_regularization,
                                     int epoch, Nesterov_params params);
*/
/*void learn_step_nesterov_paired_array_batch(network_start_layer network, double learning_rate,
                                            matrix **start_result_layer, int sample_number,
                                            general_regularization_params general_regularization,
                                            int epoch, Nesterov_params params);*/

void learn_step_nesterov_reader_batch(network_start_layer network, double learning_rate, data_reader *reader,
                                      general_regularization_params general_regularization,
                                      int epoch, Nesterov_params params);

#endif //C_VERSION_NESTEROV_ACCELERATED_GD_H
