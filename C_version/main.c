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
#include "neural_network/learning_rates.h"

#define check_error_void if(errno!=0) return;
#define check_error_only_print print_error();
#define check_error_main if(errno!=0) { print_error(); return 0; }

#define ADAM_NETWORK_FILE "adam_network"
#define NN_FILE "network.bin"


void print_error();


void train_network();


double l1l2(int epoch) {
    if (epoch < 1)
        return 2e-4;
    if (epoch < 3)
        return 6e-5;
    if (epoch < 7)
        return 5e-7;
    return 1e-8;
}

double l1l2_0(int epoch) {
    return 0;
}

double lr(int epoch_number) {
    if (epoch_number < 1)
        return 2e-4;
    if (epoch_number < 3)
        return 9e-5;
    if (epoch_number < 7)
        return 8e-6;
    if (epoch_number < 9)
        return 8e-7;
    return 3e-7;
}

regularization_params init_reg_params() {
    regularization_params regularization;
    regularization.l1 = l1l2;
    regularization.l2 = l1l2;
    set_weights(&regularization, XAVIER_NORMALIZED);
    return regularization;
}

network_start_layer initialise_network() {
    network_start_layer network = create_network(28 * 28);
    regularization_params regularization = init_reg_params();

    add_layer(&network, 150, Sigmoid, regularization);

    set_weights(&regularization, XAVIER_NORMALIZED);
    add_layer(&network, 10, Sigmoid, regularization);
    return network;
}

network_start_layer import_network() {
    network_start_layer network = read_neural_network(ADAM_NETWORK_FILE);
    regularization_params regularization = init_reg_params();

    neural_network *current = network.next_layer;
    while (current != NULL) {
        current->regularization_params = regularization;
        current = current->next_layer;
    }
    return network;

}

void train_saved_network();

int main() {
    srandom(time(NULL));

    train_network();
//    train_saved_network();
    check_error_main
    return 0;
}


double func_for_matrix(double num) {
    return num - 0.05;
}


void data_prepear(matrix data) {
    matrix_multiply_by_constant(data, 1. / 256);
    matrix_function_to_elements(data, func_for_matrix);
}

void train_saved_network() {
    network_start_layer MNIST_network = import_network();
    int train_numbers = 30000;
    int validation_numbers = 10000;
    int test_number = 10000;

    int epoch = 4;
    int epoch2 = 6;
    int batch_size = 32;

    Nesterov_params nesterov_params;
    nesterov_params.friction = 0.9;

    general_regularization_params gereral_regularization;
    paste_cost(&gereral_regularization, cross_entropy);

    data_reader train_reader = create_data_reader("mnist_train.csv", 0, train_numbers, batch_size, data_prepear);
    data_reader validation_reader = create_data_reader("mnist_train.csv", train_numbers + 1, validation_numbers,
                                                       batch_size, data_prepear);
//    printf("IMPORTED:\n");
//    test_network_paired(MNIST_network, &train_reader, gereral_regularization);

    for (int p = epoch; p < epoch + epoch2; ++p) {
        printf("EPOCH %d\n", p + 1);

        learn_step_nesterov_reader_batch(MNIST_network, decay_learning_rate(1e-5, 5e-4, p - epoch),
                                         &train_reader, gereral_regularization, p,
                                         nesterov_params);
        printf("train:      ");
        test_network_paired(MNIST_network, &train_reader, gereral_regularization);
        printf("validation: ");
        test_network_paired(MNIST_network, &validation_reader, gereral_regularization);
    }

    close_data_reader(train_reader);
    close_data_reader(validation_reader);

//    printf("\nTEST\n");
//
//    data_reader test_reader = create_data_reader("mnist_test.csv", 0, test_number,
//                                                 batch_size, data_prepear);
//    test_network_paired(MNIST_network, &test_reader, gereral_regularization);
//    confusion_matrix_paired(MNIST_network, &test_reader);
//    close_data_reader(test_reader);
//    save_neural_network(NN_FILE, MNIST_network);
    free_network(MNIST_network);
}

void train_network() {

    network_start_layer MNIST_network = initialise_network();
    int train_numbers = 10000;
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
        learn_step_adam_reader_batch(MNIST_network, lr(p), &train_reader,
                                     gereral_regularization, p, adam_params);

        printf("train:      ");
        test_network_paired(MNIST_network, &train_reader, gereral_regularization);
        printf("validation: ");
        test_network_paired(MNIST_network, &validation_reader, gereral_regularization);
    }

    save_neural_network(ADAM_NETWORK_FILE, MNIST_network);

    for (int p = epoch; p < epoch + epoch2; ++p) {
        printf("EPOCH %d\n", p + 1);

        learn_step_nesterov_reader_batch(MNIST_network, lr(p),
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

double train_network_for_grid(network_start_layer *network, grid_param *params) {

    //network_start_layer MNIST_network = initialise_network();
    int train_numbers = 10000;
    int validation_numbers = 10000;
    int test_number = 10000;

    int epoch = 5;
    int epoch2 = 0;
    int batch_size = 32;
    double b1 = params[0].d;
    double b2 = params[1].d;

    Adam_future_params adam_params;
    adam_params.b1 = b1;
    adam_params.b2 = b2;

    Nesterov_params nesterov_params;
    nesterov_params.friction = params[2].d;

    double decay_start = params[3].d;

    general_regularization_params gereral_regularization;
    paste_cost(&gereral_regularization, MSE);//TODO replace with crossentropy

    data_reader train_reader = create_data_reader("mnist_train.csv", 0, train_numbers, batch_size, data_prepear);
    data_reader validation_reader = create_data_reader("mnist_train.csv", train_numbers + 1, validation_numbers,
                                                       batch_size, data_prepear);
    for (int p = 0; p < epoch; ++p) {
        printf("EPOCH %d\n", p + 1);
        learn_step_adam_future_reader_batch(*network, decay_learning_rate(decay_start, 1e-3, p), &train_reader,
                                            gereral_regularization, p, adam_params);

        test_network_paired(*network, &train_reader, gereral_regularization);
        test_network_paired(*network, &validation_reader, gereral_regularization);
    }

    save_neural_network(ADAM_NETWORK_FILE, *network);

    for (int p = epoch; p < epoch + epoch2; ++p) {
        printf("EPOCH %d\n", p + 1);

        learn_step_nesterov_reader_batch(*network, cosine_learning_rate(1e-4, 1e-7, epoch, p),
                                         &train_reader, gereral_regularization, p,
                                         nesterov_params);
        test_network_paired(*network, &train_reader, gereral_regularization);
        test_network_paired(*network, &validation_reader, gereral_regularization);
    }

    close_data_reader(train_reader);
    close_data_reader(validation_reader);

    printf("\nTEST\n");

    data_reader test_reader = create_data_reader("mnist_test.csv", 0, test_number,
                                                 batch_size, data_prepear);
    test_network_paired(*network, &test_reader, gereral_regularization);
    confusion_matrix_paired(*network, &test_reader);
    close_data_reader(test_reader);
    save_neural_network(NN_FILE, *network);
    free_network(*network);
    double *accuracy_loss = test_network_paired_double(*network, &validation_reader, gereral_regularization);
    double result = accuracy_loss[1];
    free(accuracy_loss);
    return result;
}

double grid_search_test() {
    network_start_layer MNIST_network = initialise_network();
    grid_param **params = calloc(2, sizeof(grid_param *));
    for (int i = 0; i < 2; i++) {
        params[i] = calloc(3, sizeof(grid_param));
    }
    params[0][0].type = DOUBLE;//b1
    params[0][0].d = 0.7;
    params[1][0].type = DOUBLE;
    params[1][0].d = 0.99;
    params[0][1].type = DOUBLE;//b2
    params[0][1].d = 0.7;
    params[1][1].type = DOUBLE;
    params[1][1].d = 0.99;
    params[0][2].type = DOUBLE;//friction
    params[0][2].d = 0.7;
    params[1][2].type = DOUBLE;
    params[1][2].d = 0.99;
    params[0][3].type = DOUBLE;//decay_start_value
    params[0][3].d = 6e-4;
    params[1][3].type = DOUBLE;
    params[1][3].d = 1e-4;
    stochastic_grid_search("grid_search_network.bin", MNIST_network, train_network_for_grid, params, 3, 5);
    for (int i = 0; i < 2; i++) {
        free(params[i]);
    }
    free(params);
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
