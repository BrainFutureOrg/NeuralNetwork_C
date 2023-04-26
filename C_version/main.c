#include <stdio.h>
#include <stdlib.h>
#include "math/matrix_operations.h"
#include <errno.h>
#include <stdlib.h>
#include "neural_network/weight_initialize/weight_initializers.h"
#include <time.h>
//#include <math.h>
#include "neural_network/neural_network.h"
#include "data/DAO.h"
#include "neural_network/regularization/regularization_params.h"

#include "neural_network/Optimizers/SGD.h"
#include "neural_network/Optimizers/momentum_optimizer.h"
#include "neural_network/Optimizers/Nesterov_accelerated_gd.h"
#include "neural_network/Optimizers/Adam_optimizer.h"
#include "test/main_tests.h"

#define check_error_void if(errno!=0) return;
#define check_error_main if(errno!=0) { print_error(); return 0; }


void print_error();


void try_train_network();

void try_train_momentum();

void try_train_nesterov();

void try_train_adam();

double l1l2(int epoch) {
    if (epoch < 1)
        return 1e-4;
    if (epoch < 3)
        return 1e-4;
    if (epoch < 7)
        return 5e-5;
    return 5e-7;
}

double lr(int epoch_number) {
    if (epoch_number < 1)
        return 5e-3;
    if (epoch_number < 3)
        return 5e-4;
    if (epoch_number < 7)
        return 1e-4;
    if (epoch_number < 9)
        return 6e-5;
    return 1e-5;
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
//    check_matrix_print();
    //check_learning();
//    check_DAO_reader();
//    try_train_network();
    //try_train_momentum();
    //try_train_nesterov();
    try_train_adam();
    check_error_main
    return 0;
}


double func_for_matrix(double num) {
    return num + 0.05;
}


void data_prepear(matrix data) {
    matrix_multiply_by_constant(data, 1. / 256);
    matrix_function_to_elements(data, func_for_matrix);
}

matrix **get_data(char *File_name, int line_number) {
    matrix **data = calloc(2, sizeof(matrix *));
    matrix *answers = calloc(line_number, sizeof(matrix));
    matrix *for_predict = calloc(line_number, sizeof(matrix));
    data[1] = answers;
    data[0] = for_predict;
    FILE *file;
    file = open_file(File_name);
    pass_line(file);
    for (int w = 0; w < line_number; w++) {
        double *numbers = get_line_matrix(file);
        for_predict[w] = make_matrix_from_array(&numbers[1], 28 * 28, 1);
        data_prepear(for_predict[w]);
        answers[w] = create_vector(10, (int) numbers[0]);
        free(numbers);
    }
    fclose(file);
    return data;
}

void free_data(matrix **data, int data_num) {
    for (int i = 0; i < data_num; i++) {
        for (int j = 0; j < 2; j++) {
            matrix_free(data[j][i]);
        }
    }
    for (int j = 0; j < 2; j++) {
        free(data[j]);
    }
    free(data);
}

/*
void try_train_network() {

    network_start_layer MNIST_network = initialise_network();
//    matrix_print(MNIST_network.next_layer->bias);
//    matrix_print(MNIST_network.next_layer->weights);
    int train_numbers = 50;
    int validation_numbers = 50;
    int test_number = 50;

    int epoch = 3;
    //double l1 = 0.0005;
    //double l2 = 0.0005;
//    double lr = 0.05;
    int batch_size = 32;

    general_regularization_params gereral_regularization;
    gereral_regularization.batch_size = batch_size;
    paste_cost(&gereral_regularization, cross_entropy);
    matrix **train_full_data = get_data("mnist_train.csv", train_numbers);
    matrix **validation_full_data = get_data("mnist_train.csv", validation_numbers);
//    pass_line(file);
    for (int p = 0; p < epoch; ++p) {
        learn_step_optimizerless_paired_array_batch(MNIST_network, lr(p), train_full_data, train_numbers,
                                                    gereral_regularization, p);
        test_network_paired(MNIST_network, validation_full_data, validation_numbers, gereral_regularization);
    }
    free_data(train_full_data, train_numbers);
    free_data(validation_full_data, validation_numbers);

    printf("\nTEST\n");

    matrix **test_full_data = get_data("mnist_test.csv", test_number);
    test_network_paired(MNIST_network, test_full_data, test_number, gereral_regularization);
    confusion_matrix_paired(MNIST_network, test_full_data, test_number);
    free_data(test_full_data, test_number);
    free_network(MNIST_network);
}

void try_train_momentum() {

    network_start_layer MNIST_network = initialise_network();
//    matrix_print(MNIST_network.next_layer->bias);
//    matrix_print(MNIST_network.next_layer->weights);
    int train_numbers = 500;
    int validation_numbers = 500;
    int test_number = 5000;

    int epoch = 10;
    //double l1 = 0.0005;
    //double l2 = 0.0005;
//    double lr = 0.05;
    int batch_size = 32;
    double friction = 0.9;
    momentum_params params;
    params.friction = friction;

    general_regularization_params gereral_regularization;
    gereral_regularization.batch_size = batch_size;
    paste_cost(&gereral_regularization, cross_entropy);
    matrix **train_full_data = get_data("mnist_train.csv", train_numbers);
    matrix **validation_full_data = get_data("mnist_train.csv", validation_numbers);
//    pass_line(file);
    for (int p = 0; p < epoch; ++p) {
        learn_step_momentum_paired_array_batch(MNIST_network, lr(p), train_full_data, train_numbers,
                                               gereral_regularization, p, params);
        test_network_paired(MNIST_network, validation_full_data, validation_numbers, gereral_regularization);
    }
    free_data(train_full_data, train_numbers);
    free_data(validation_full_data, validation_numbers);

    printf("\nTEST\n");

    matrix **test_full_data = get_data("mnist_test.csv", test_number);
    test_network_paired(MNIST_network, test_full_data, test_number, gereral_regularization);
    confusion_matrix_paired(MNIST_network, test_full_data, test_number);
    free_data(test_full_data, test_number);
    free_network(MNIST_network);
}

void try_train_nesterov() {

    network_start_layer MNIST_network = initialise_network();
//    matrix_print(MNIST_network.next_layer->bias);
//    matrix_print(MNIST_network.next_layer->weights);
    int train_numbers = 5000;
    int validation_numbers = 500;
    int test_number = 5000;

    int epoch = 10;
    int batch_size = 32;
    double friction = 0.9;
    Nesterov_params params;
    params.friction = friction;

    general_regularization_params gereral_regularization;
    gereral_regularization.batch_size = batch_size;
    paste_cost(&gereral_regularization, cross_entropy);
    matrix **train_full_data = get_data("mnist_train.csv", train_numbers);
    matrix **validation_full_data = get_data("mnist_train.csv", validation_numbers);
//    pass_line(file);
    for (int p = 0; p < epoch; ++p) {
        learn_step_nesterov_paired_array_batch(MNIST_network, lr(p), train_full_data, train_numbers,
                                               gereral_regularization, p, params);
        test_network_paired(MNIST_network, validation_full_data, validation_numbers, gereral_regularization);
    }
    free_data(train_full_data, train_numbers);
    free_data(validation_full_data, validation_numbers);

    printf("\nTEST\n");

    matrix **test_full_data = get_data("mnist_test.csv", test_number);
    test_network_paired(MNIST_network, test_full_data, test_number, gereral_regularization);
    confusion_matrix_paired(MNIST_network, test_full_data, test_number);
    free_data(test_full_data, test_number);
    free_network(MNIST_network);
}

void try_train_adam() {

    network_start_layer MNIST_network = initialise_network();
    int train_numbers = 40000;
    int validation_numbers = 5000;
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
        //learn_step_nesterov_reader_batch(MNIST_network, lr(p), &train_reader, gereral_regularization, p,
        //                                 nesterov_params);
        test_network_paired(MNIST_network, &train_reader, gereral_regularization);
        test_network_paired(MNIST_network, &validation_reader, gereral_regularization);
    }
    for (int p = epoch; p < epoch + epoch2; ++p) {
        printf("EPOCH %d\n", p + 1);
        //learn_step_adam_reader_batch(MNIST_network, lr(p), &train_reader,
        //                             gereral_regularization, p, adam_params);
        learn_step_nesterov_reader_batch(MNIST_network, lr(p), &train_reader, gereral_regularization, p,
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
    free_network(MNIST_network);
}

void print_error() {
    switch (errno) {
        case ERANGE:
            printf("ERANGE");
            break;
        default:
            printf("ERROR\n");
            break;
    }
}
