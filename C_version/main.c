#include <stdio.h>
#include <stdlib.h>
#include "matrix_operations.h"
#include <errno.h>
#include <stdlib.h>
#include <time.h>
//#include <math.h>
#include "neural_network/neural_network.h"
#include "data/DAO.h"
#include "neural_network/regularization_params.h"

#define check_error_void if(errno!=0) return;
#define check_error_main if(errno!=0) { print_error(); return 0; }


void print_error();


void try_train_network();

double l1l2(int epoch) {
    if (epoch < 5)
        return 0.0001;
    if (epoch < 8)
        return 0.00001;
    if (epoch < 9)
        return 0.000005;
    return 0.00005;
}

double lr(int epoch_number) {
    if (epoch_number < 7)
        return 0.0005;
    if (epoch_number < 8)
        return 0.0001;
    if (epoch_number < 9)
        return 0.00001;
    return 0.00005;
}

network_start_layer initialise_network() {
    network_start_layer network = create_network(28 * 28);
    regularization_params regularization;
    regularization.l1 = l1l2;
    regularization.l2 = l1l2;
    add_layer(&network, 300, Sigmoid, regularization);
    add_layer(&network, 10, Sigmoid, regularization);
    return network;
}

int main() {
    srandom(time(NULL));
//    check_matrix_print();
    //check_learning();
//    check_DAO();
    try_train_network();
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


void try_train_network() {

    network_start_layer MNIST_network = initialise_network();
//    matrix_print(MNIST_network.next_layer->bias);
//    matrix_print(MNIST_network.next_layer->weights);
    FILE *file;

    int train_numbers = 5000;
    int validation_numbers = 5000;
    int test_number = 500;

    int epoch = 10;
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
