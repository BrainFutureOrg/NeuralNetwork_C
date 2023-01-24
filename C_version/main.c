#include <stdio.h>
#include <stdlib.h>
#include "matrix_operations.h"
#include <errno.h>
#include <stdlib.h>
#include <time.h>
//#include <math.h>
#include "neural_network/neural_network.h"

void check_matrix_print();

void check_matrix_multiplication();

void check_learning();

int main() {
    srand(time(NULL));
//    check_matrix_print();
    check_learning();
    return 0;
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
void check_learning(){
    network_start_layer network = create_network(4);
    printf("start creating network\n");
    add_layer(&network, 5, "Sigmoid");
    add_layer(&network, 6, "ReLu");
    add_layer(&network, 5, "Tanh");
    add_layer(&network, 5, "Sigmoid");
    add_layer(&network, 4, "Softmax");
    printf("end creating network\n");
    //print_network(network);
    matrix inhuman_experiment= matrix_creation(4, 1);
    double** table=calloc(4, sizeof (double *));
    for(int i=0; i<4; i++){
        table[i]= calloc(1, sizeof(double ));
        table[i][0]=(i+1)/4.0;
    }
    inhuman_experiment.table=table;
    for(int i=0; i<100; i++){
        learn_step(network, 0.05, inhuman_experiment, inhuman_experiment);
        printf("ended learning step %d\n", i);
    }
    matrix_print(predict(network, inhuman_experiment));
    //print_network(network);
}

//void check_network_exists(){
//    network_start_layer network = create_network(4);
//}
