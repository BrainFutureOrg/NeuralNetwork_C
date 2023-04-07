//
// Created by maximus on 22.01.23.
//

#include "neural_network.h"
#include "stdlib.h"
#include "math.h"
#include "../matrix_operations.h"
#include <limits.h>
#include <string.h>
#include <stdio.h>
#include "network_activation_functions.h"
#include <errno.h>
#include "optimizers.h"

#define EPSILON 0.00005

void add_function_with_derivative(neural_network *network_layer, activation_function_names activation_function_name) {
    if (activation_function_name==Sigmoid) {
        network_layer->activation_function = network_sigmoid;
        network_layer->activation_function_derivative = network_sigmoid_derivative;
        return;
    }

    if (activation_function_name==Softmax) {
        network_layer->activation_function = network_softmax;
        network_layer->activation_function_derivative = network_softmax_derivative;
        return;
    }

    if (activation_function_name==Tangential) {
        network_layer->activation_function = network_tangential;
        network_layer->activation_function_derivative = network_tangential_derivative;
        return;
    }

    if (activation_function_name==ReLu) {
        network_layer->activation_function = network_ReLU;
        network_layer->activation_function_derivative = network_ReLU_derivative;
        return;
    }
}

neural_network *last_layer(network_start_layer network) {
    neural_network *current = network.next_layer;
    if (current == NULL) return current;
    while (current->next_layer != NULL) {
        current = current->next_layer;
    }
    return current;
}

network_start_layer create_network(int neuron_numbers) {
    network_start_layer result;
    result.i = neuron_numbers;
    result.next_layer = NULL;
    return result;
}

void add_after_start_layer(network_start_layer *network, int neuron_numbers, activation_function_names activation_function_name) {
    network->next_layer = calloc(1, sizeof(neural_network));
    matrix weighs = matrix_creation(neuron_numbers, network->i);
    for (int i = 0; i < weighs.i; i++) {
        for (int j = 0; j < weighs.j; j++) {
            weighs.table[i][j] = (double) random() / INT_MAX + 0.001;
        }
    }

    matrix bias = matrix_creation(neuron_numbers, 1);
    for (int i = 0; i < bias.i; i++) {
        bias.table[i][0] = (double) random() / INT_MAX + 0.001;
    }

    network->next_layer->weights = weighs;
    network->next_layer->bias = bias;
    printf("%d\n",network->next_layer->bias.i);
    //matrix_free(weighs);
    network->next_layer->next_layer = NULL;
    network->next_layer->previous_layer = NULL;
    add_function_with_derivative(network->next_layer, activation_function_name);
}

void add_after_layer(network_start_layer *network, int neuron_numbers, char *activation_function_name) {
    neural_network *current = network->next_layer;
    while (current->next_layer != NULL) {
        current = current->next_layer;
    }
    current->next_layer = calloc(1, sizeof(neural_network));
    matrix weighs = matrix_creation(neuron_numbers, current->weights.i);
    for (int i = 0; i < weighs.i; i++) {
        for (int j = 0; j < weighs.j; j++) {
            weighs.table[i][j] = (double) random() / INT_MAX + 0.001;
        }
    }

    matrix bias = matrix_creation(neuron_numbers, 1);
    for (int i = 0; i < bias.i; i++) {
        bias.table[i][0] = (double) random() / INT_MAX + 0.001;
    }
    //network->next_layer->bias = bias;
    current->next_layer->bias=bias;
    printf("%d %d %f\n",network->next_layer->bias.i, network->next_layer->bias.j, network->next_layer->bias.table[0][0]);

    current->next_layer->weights = weighs;
    //matrix_free(weighs);
    current->next_layer->next_layer = NULL;
    current->next_layer->previous_layer = current;
    add_function_with_derivative(current->next_layer, activation_function_name);
}

void add_layer(network_start_layer *network, int neuron_numbers, char *activation_function_name) {
//    printf("add layer start\n");
    if (network->next_layer == NULL) {
        add_after_start_layer(network, neuron_numbers, activation_function_name);
    } else {
        add_after_layer(network, neuron_numbers, activation_function_name);
    }
//    printf("add layer end\n");
}

int count_hidden_layers(network_start_layer network) {
    neural_network *current = network.next_layer;
    int result = 0;
    while (current != NULL) {
        current = current->next_layer;
        result++;
    }
    return result;
}

matrix *predict_all_layers(network_start_layer network, matrix start_layer) {
    int layers_number = count_hidden_layers(network);
    neural_network *current = network.next_layer;
    matrix *current_results = calloc(layers_number + 1, sizeof(matrix));
    matrix activated_results = start_layer;
    current_results[0] = start_layer;
    for (int i = 1; i < layers_number + 1; i++) {
        current_results[i] = matrix_multiplication(current->weights, activated_results);

//        matrix_free(activated_results);
        activated_results = matrix_addition(matrix_copy(current_results[i]), current->bias);
//        printf("%d %d\n", current->bias.i, current->bias.j);
        current->activation_function(&activated_results);
        current = current->next_layer;
    }
    return current_results;
}

/*void learn_step(network_start_layer network, double learning_rate, matrix start_layer,
                matrix result_layer) { //UNSURE
    int layer_number = count_hidden_layers(network);
    matrix *prediction = predict_all_layers(network, start_layer);
    neural_network *current = last_layer(network);
    matrix last_layer_prediction = matrix_copy(prediction[layer_number]);
    current->activation_function(&last_layer_prediction);

    matrix distributed_error = matrix_substact(result_layer, last_layer_prediction);

    for (int i = layer_number; i > 0; i--) {
        //matrix derived_results = matrix_multiplication(current->weights, prediction[i - 1]);
        matrix derived_results = matrix_copy(prediction[i]);
        //current->activation_function_derivative(&derived_results);//NO DELETE
        current->activation_function_derivative(&prediction[i]);
        matrix tmatrix = matrix_transposition(prediction[i - 1]);
        //new
        if (i > 1)current->previous_layer->activation_function(&tmatrix);
        //end new
        //matrix delta_weights = matrix_multiplication(matrix_multiplication_elements(distributed_error, derived_results),
        //                                     tmatrix);//NO DDELETE
        matrix delta = matrix_multiplication_elements(distributed_error, prediction[i]);
        matrix delta_copy = matrix_copy(delta);
        matrix_multiply_by_constant(delta_copy, learning_rate);
        matrix bias = matrix_addition(current->bias, delta_copy);//CRINGE?
        matrix_free(current->bias);
        current->bias = bias;
        matrix_free(delta_copy);
        //matrix_free(bias);
        matrix delta_weights = matrix_multiplication(delta,
                                                     tmatrix);
        matrix_free(tmatrix);
        //l2 normalisation
        //
        matrix_multiply_by_constant(delta_weights, learning_rate);
        matrix weights = current->weights;
        current->weights = matrix_addition(weights, delta_weights);
        matrix_free(weights);
        //matrix_restrict(current->weights, restriction);
        tmatrix = matrix_transposition(current->weights);
        matrix distributed_error2 = matrix_multiplication(tmatrix,
                                                          distributed_error);
        matrix_free(tmatrix);
        matrix_free(distributed_error);
        distributed_error = distributed_error2;
        current = current->previous_layer;
        matrix_free(derived_results);
        matrix_free(delta_weights);
        //matrix_free(prediction[i]);
    }
    for (int i = 0; i < layer_number; i++) {
        //matrix_free(prediction[i]);
    }
    matrix_free(distributed_error);
    free(prediction);
}*/

void learn_step_optimizerless(network_start_layer network, double learning_rate, matrix start_layer,
                matrix result_layer) {
    matrix *prediction = predict_all_layers(network, start_layer);
    neural_network* current = last_layer(network);
    int network_layer_number = count_hidden_layers(network);

    matrix derived_results = matrix_copy_activated(prediction[network_layer_number], current->activation_function);
    matrix nablaC = matrix_substact( derived_results,result_layer);
    matrix_free(derived_results);

    derived_results = matrix_copy_activated(prediction[network_layer_number], current->activation_function_derivative);
    matrix dL=matrix_multiplication_elements(nablaC, derived_results);
    matrix_free(derived_results);
    matrix dl=dL;
    for(int i=network_layer_number-1; i>=0; i--){
        //temporary
        gradient_descent(current, dl, learning_rate, prediction[i]);
        //end temporary
        //if(current->previous_layer==NULL)printf("FUCK");
        matrix transposed = matrix_transposition(current->weights);
        current = current->previous_layer;
//        printf("%d\n", i);
        //matrix transposed = matrix_transposition(current->next_layer->weights);//invalid read
//        printf("frst invalid read\n");
        matrix multiplied = matrix_multiplication(transposed, dl);
        matrix_free(transposed);
        derived_results = matrix_copy(prediction[i]);//invalid read
//        printf("snd invalid read\n");
        if(current!=NULL)
        {
            current->activation_function_derivative(&derived_results);
        }
        matrix_free(dl);
        dl=matrix_multiplication_elements(multiplied, derived_results);
        matrix_free(derived_results);
//        matrix_free(multiplied);


    }
}

void learn_step2(network_start_layer network, double learning_rate, matrix start_layer,
                 matrix result_layer/*, double restriction*/) { //UNSURE
    int layer_number = count_hidden_layers(network);
    matrix *prediction = predict_all_layers(network, start_layer);
    neural_network *current = last_layer(network);
    matrix last_layer_prediction = prediction[layer_number];
    current->activation_function(&last_layer_prediction);

    matrix error = matrix_substact(result_layer, last_layer_prediction);

}

void learn_step3(network_start_layer network, double learning_rate, matrix start_layer,
                 matrix result_layer/*, double restriction*/) {
    int layer_number = count_hidden_layers(network);
    matrix *prediction = predict_all_layers(network, start_layer);
    neural_network *current = last_layer(network);
    matrix last_layer_prediction = matrix_copy(prediction[layer_number]);
    current->activation_function(&last_layer_prediction);

    matrix distributed_error = matrix_substact(result_layer, last_layer_prediction);

    matrix derivative = matrix_copy(prediction[layer_number]);
    current->activation_function_derivative(&derivative);
//    matrix delta = matrix_multiplication_elements(distributed_error, )
}

void print_network(network_start_layer network) {
    printf("startlayer\n");
    int i = 0;
    neural_network *current = network.next_layer;
    while (current != NULL) {
        i++;
        printf("layer %d exists\n", i);
        current = current->next_layer;
    }
}

matrix predict(network_start_layer network, matrix start_layer) {
    neural_network *current = network.next_layer;
    int i = 0;
    matrix current_results = start_layer;
//    printf("Step0\n");
    while (current != NULL) {
//        printf("Step\n");
        //matrix_print(current->weights);
        //printf("\n\n");
        matrix multiplication = matrix_multiplication(current->weights, current_results);
        if (i != 0) {
            matrix_free(current_results);
        } else {
            i++;
        }
        current_results = matrix_addition(multiplication, current->bias);
        current->activation_function(&current_results);
        current = current->next_layer;
    }
    return current_results;
}

double small_accuracy(network_start_layer network, matrix start_layer, matrix answers) {
    double accuracy;

    matrix prediction = predict(network, start_layer);
    for (int i = 0; i < answers.i; i++) {
        accuracy += fabs(answers.table[i][0] - prediction.table[i][0] + EPSILON) / (prediction.table[i][0] + EPSILON) /
                    answers.i;
    }
    matrix_free(prediction);
    return 1 - accuracy;
}

double accuracy(network_start_layer network, matrix *start_layers, matrix *answers, int len_of_accuracy) {
    double accuracy;
    for (int i = 0; i < len_of_accuracy; i++) {
        accuracy += small_accuracy(network, start_layers[i], answers[i]) / len_of_accuracy;
    }
    return accuracy;
}

void free_network(network_start_layer network) {
    neural_network *current = network.next_layer;
    while (current != NULL) {
        neural_network *current2 = current->next_layer;
        matrix_free(current->weights);
        free(current);
        current = current2;
    }
    network.next_layer = NULL;
}