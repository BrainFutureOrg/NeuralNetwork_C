#include <stdio.h>
#include <stdlib.h>
#include "matrix_operations.h"
#include <errno.h>
#include <stdlib.h>
#include <time.h>
//#include <math.h>
#include "neural_network/neural_network.h"
#include "data/DAO.h"

#define check_error_void if(errno!=0) return;
#define check_error_main if(errno!=0) { print_error(); return 0; }

void check_matrix_print();

void check_matrix_multiplication();

void check_learning();

void print_error();

void check_DAO();

void try_train_network();

network_start_layer initialise_network() {
    network_start_layer network = create_network(28 * 28);
    //add_layer(&network, 500, "ReLu");
//    add_layer(&network, 200, "ReLu");
    //add_layer(&network, 5, "ReLu");
    add_layer(&network, 200, "ReLu");
    add_layer(&network, 10, "ReLu");
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

void check_DAO() {
    readline("mnist_train.csv");
    FILE *file = open_file("mnist_train.csv");
    pass_line(file);
    for (int p = 0; p < 4; ++p) {
        double *numbers = get_line_matrix(file);
        printf("%.0lf\n", numbers[0]);
        for (int i = 0; i < 28; ++i) {
            for (int j = 0; j < 28; ++j) {
                printf("%3.0lf ", numbers[i * 28 + j + 1]);
            }
            printf("\n");

        }
        printf("\n");
    }

}

double func_for_matrix(double num) {
    return num + 0.05;
}

void try_train_network() {

    network_start_layer MNIST_network = initialise_network();
    matrix_print(MNIST_network.next_layer->bias);
    matrix_print(MNIST_network.next_layer->weights);
    FILE *file;

    int test_numbers = 2;
//    pass_line(file);
    for (int p = 0; p < 10000; ++p) {
        file = open_file("mnist_train.csv");
        pass_line(file);
        for (int w = 0; w < test_numbers; w++) {
            double *numbers = get_line_matrix(file);
//        matrix_print(matrix_numbers);
            matrix matrix_numbers = make_matrix_from_array(&numbers[1], 28 * 28, 1);
            matrix answer_vector = create_vector(10, (int) numbers[0]);
            matrix_multiply_by_constant(matrix_numbers, 1. / 256);
            matrix_function_to_elements(matrix_numbers, func_for_matrix);
//        print_network()
            learn_step(MNIST_network, 0.00003, matrix_numbers, answer_vector);
//        if (errno != 0) {
//        matrix_print(matrix_numbers);
//            printf("\n");
//            matrix_print(answer_vector);
//            printf("%d i_answer %d j_answer %d i_matrix %d j_matrix %d\n", p, answer_vector.i, answer_vector.j,
//                   matrix_numbers.i, matrix_numbers.j);
//            return;
//        }
            free(numbers);
            matrix_free(matrix_numbers);
            matrix_free(answer_vector);
            printf("%d ", w);
        }
        printf("\nended epoch %d\n", p);
        fclose(file);
    }


    file = open_file("mnist_train.csv");
    pass_line(file);
    int has_result = 1;
    double result;
    int test_number = 2;
    for (int p = 0; p < test_number; ++p) {
        double *numbers = get_line_matrix(file);
        matrix matrix_numbers = make_matrix_from_array(&numbers[1], 28 * 28, 1);
        //new
        matrix_multiply_by_constant(matrix_numbers, 1. / 256);
        matrix_function_to_elements(matrix_numbers, func_for_matrix);
        //end new
        matrix answer_vector = create_vector(10, (int) numbers[0]);
        if (has_result != 0) {
            printf("right - %f predicted - \n", numbers[0]);
            matrix_print(predict(MNIST_network, matrix_numbers));
            result += small_accuracy(MNIST_network, matrix_numbers, answer_vector) / test_number;
        } else {
            result = small_accuracy(MNIST_network, matrix_numbers, answer_vector) / test_number;
            has_result++;
        }
        free(numbers);
        matrix_free(matrix_numbers);
        matrix_free(answer_vector);
        printf("\n\n\n\n\n");
    }
    printf("accuracy = %lf\n", result);
    fclose(file);

    matrix_print(MNIST_network.next_layer->bias);
    matrix_print(MNIST_network.next_layer->weights);

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

void check_matrix_print() {

    double matrix1[3][2] = {{1, 2},
                            {3, 4},
                            {5, 6}};
    int size1 = sizeof(matrix1) / sizeof(matrix1[0]);
    int size2 = sizeof(matrix1[0]) / sizeof(matrix1[0][0]);
    matrix checking;
    checking = make_matrix_from_array(&matrix1[0][0], size1, size2);
    matrix_print(checking);
//    matrix_free(checking);
    matrix checkingT = matrix_transposition(checking);
    matrix_print(checkingT);
    matrix_free(checking);
    matrix_free(checkingT);
}

void check_matrix_multiplication() {
    double m[2][3] = {{1, 2, 5},
                      {3, 4, 6}};
    int size1 = sizeof(m) / sizeof(m[0]);
    int size2 = sizeof(m[0]) / sizeof(m[0][0]);
    double **matrix_pointer;
    matrix_pointer = calloc(size1, sizeof(double *));
    for (int i = 0; i < size1; ++i) {
        matrix_pointer[i] = calloc(size2, sizeof(double));
    }
    for (int i = 0; i < size1; ++i) {
        for (int j = 0; j < size2; ++j) {
            matrix_pointer[i][j] = m[i][j];
        }
    }
    matrix checking;
    checking.table = matrix_pointer;
    checking.i = size1;
    checking.j = size2;
    matrix matr = matrix_multiplication(checking, checking);
    if (errno == 0) {
        matrix_print(matr);
    } else {
        printf("error");
        matrix_free(checking);
        return;
    }
    matrix_free(matr);
    matrix_free(checking);
}


void check_learning() {
    network_start_layer network = create_network(4);
    //printf("start creating network\n");
    add_layer(&network, 5, "Sigmoid");
    add_layer(&network, 6, "Sigmoid");
    //add_layer(&network, 5, "Tanh");
    //add_layer(&network, 5, "Sigmoid");
    add_layer(&network, 4, "Sigmoid");
    //printf("end creating network\n");
    //print_network(network);
    matrix inhuman_experiment;
    matrix inhuman_experiment2;
    inhuman_experiment.i = 4;
    inhuman_experiment.j = 1;
    inhuman_experiment2.i = 4;
    inhuman_experiment2.j = 1;
    double **table = calloc(4, sizeof(double *));
    for (int i = 0; i < 4; i++) {
        table[i] = calloc(1, sizeof(double));
        table[i][0] = (i + 1) / 4.0;
    }
    double **table2 = calloc(4, sizeof(double *));
    for (int i = 0; i < 4; i++) {
        table2[i] = calloc(1, sizeof(double));
        table2[i][0] = (i + 1) / 8.0;
    }
    inhuman_experiment.table = table;
    inhuman_experiment2.table = table2;
    for (int i = 0; i < 40000; i++) {
        //printf("start learning\n");
        learn_step(network, 0.05, inhuman_experiment, inhuman_experiment);
        learn_step(network, 0.05, inhuman_experiment2, inhuman_experiment2);
        printf("ended learning step %d\n", i);
        //just accuracy
        printf("\nepoch %d\n", i);
        printf("\n%f\n", small_accuracy(network, inhuman_experiment, inhuman_experiment));
        printf("\n%f\n", small_accuracy(network, inhuman_experiment2, inhuman_experiment2));
    }
    matrix prediction = predict(network, inhuman_experiment);
    matrix_print(prediction);
    matrix_free(prediction);

    matrix prediction2 = predict(network, inhuman_experiment2);
    matrix_print(prediction2);
    matrix_free(prediction2);


    printf("\n\n");

    printf("%f\n", small_accuracy(network, inhuman_experiment, inhuman_experiment));
    //printf("%f\n", small_accuracy(network, inhuman_experiment2, inhuman_experiment2));
    //print_network(network);




    free_network(network);
    matrix_free(inhuman_experiment);
}

//void check_network_exists(){
//    network_start_layer network = create_network(4);
//}
