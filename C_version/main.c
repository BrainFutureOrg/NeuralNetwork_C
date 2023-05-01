#include <stdio.h>
#include <stdlib.h>
#include "math/matrix_operations.h"
#include <errno.h>
#include "neural_network/weight_initialize/weight_initializers.h"
#include <time.h>
#include "neural_network/neural_network.h"
#include "data/DAO.h"
#include "neural_network/regularization/regularization_params.h"
#include <math.h>
#include "neural_network/Optimizers/SGD.h"
#include "neural_network/Optimizers/momentum_optimizer.h"
#include "neural_network/Optimizers/Nesterov_accelerated_gd.h"
#include "neural_network/Optimizers/Adam_optimizer.h"
#include "neural_network/Optimizers/Adam_future_view_optimizer.h"
#include "data/save_nn.h"
#include "test/main_tests.h"

#define check_error_void if(errno!=0) return;
#define check_error_main if(errno!=0) { print_error(); return 0; }


void print_error();


void train_network();


double l1l2(int epoch) {
    if (epoch < 1)
        return 1e-4;
    if (epoch < 3)
        return 5e-5;
    if (epoch < 7)
        return 1e-5;
    return 5e-8;
}

double lr(int epoch_number) {
    if (epoch_number < 1)
        return 3e-4;
    if (epoch_number < 6)
        return 9e-5;
    if (epoch_number < 7)
        return 5e-6;
    if (epoch_number < 9)
        return 1e-6;
    return 1e-7;
}

double exponential_learning_rate(double start_value, double decay_rate, int epoch) {
    for (int i = 0; i < epoch; i++) {
        start_value *= decay_rate;
    }
    return start_value;
}

double decay_learning_rate(double start_value, double decay_rate, int epoch) {
    return start_value / (1 + decay_rate * epoch);
}

double cosine_learning_rate(double start_value, double final_value, int max_steps, int epoch) {
    return epoch <= max_steps ? final_value + (start_value - final_value) * (1 + cos(M_PI * epoch / max_steps)) / 2
                              : final_value;
}

double cosine_restart_learning_rate(double start_value, double final_value, int steps_till_restart, int epoch) {
    return final_value +
           (start_value - final_value) * (1 + cos(M_PI * (epoch % steps_till_restart) / steps_till_restart)) / 2;
}

network_start_layer initialise_network() {
    network_start_layer network = create_network(28 * 28);
    regularization_params regularization;
    regularization.l1 = l1l2;
    regularization.l2 = l1l2;
    set_weights(&regularization, XAVIER_NORMALIZED);

    add_layer(&network, 150, Sigmoid, regularization);
    add_layer(&network, 10, Sigmoid, regularization);
    return network;
}

int main() {
    srandom(time(NULL));

    train_network();
    check_error_main
    return 0;
}

#define NN_FILE "network.bin"


double func_for_matrix(double num) {
    return num + 0.05;
}


void data_prepear(matrix data) {
    matrix_multiply_by_constant(data, 1. / 256);
    matrix_function_to_elements(data, func_for_matrix);
}

void train_network() {

    network_start_layer MNIST_network = initialise_network();
    int train_numbers = 30000;
    int validation_numbers = 10000;
    int test_number = 10000;

    int epoch = 3;
    int epoch2 = 7;
    int batch_size = 32;
    double b1 = 0.9;
    double b2 = 0.95;

    Adam_params adam_params;
    adam_params.b1 = b1;
    adam_params.b2 = b2;

    Nesterov_params nesterov_params;
    nesterov_params.friction = 0.9;

    general_regularization_params gereral_regularization;
    paste_cost(&gereral_regularization, cross_entropy);

    data_reader train_reader = create_data_reader("mnist_train.csv", 0, train_numbers, batch_size, data_prepear);
    data_reader validation_reader = create_data_reader("mnist_train.csv", train_numbers + 1, validation_numbers,
                                                       batch_size, data_prepear);
    for (int p = 0; p < epoch; ++p) {
        printf("EPOCH %d\n", p + 1);
        learn_step_adam_future_reader_batch(MNIST_network, decay_learning_rate(lr(p), 2e-4, p), &train_reader,
                                            gereral_regularization, p, adam_params);

        test_network_paired(MNIST_network, &train_reader, gereral_regularization);
        test_network_paired(MNIST_network, &validation_reader, gereral_regularization);
    }
    for (int p = epoch; p < epoch + epoch2; ++p) {
        printf("EPOCH %d\n", p + 1);

        learn_step_nesterov_reader_batch(MNIST_network, exponential_learning_rate(2e-4, 0.6, p),
                                         &train_reader, gereral_regularization, p,
                                         nesterov_params);
        test_network_paired(MNIST_network, &train_reader, gereral_regularization);
        test_network_paired(MNIST_network, &validation_reader, gereral_regularization);
    }

    close_data_reader(train_reader);
    close_data_reader(validation_reader);

    printf("\nTEST\n");

    data_reader test_reader = create_data_reader("mnist_test.csv", 0, test_number,
                                                 batch_size, data_prepear);
    test_network_paired(MNIST_network, &test_reader, gereral_regularization);
    confusion_matrix_paired(MNIST_network, &test_reader);
    close_data_reader(test_reader);
    save_neural_network(NN_FILE, MNIST_network);
    free_network(MNIST_network);
}

void print_error() {
    switch (errno) {
        case ERANGE:
            printf("ERANGE");
            break;
        default:
            printf("ERROR %d\n", errno);
            break;
    }
}
