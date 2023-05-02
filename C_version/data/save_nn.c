//
// Created by maximus on 26.04.23.
//

#include "save_nn.h"
#include "DAO.h"
#include "stdlib.h"
#include "bin_writers.h"
#include "../neural_network/neural_network.h"

void write_neural_network_structure(FILE *fp, neural_network *network) {
    save_matrix(fp, network->weights);
    save_matrix(fp, network->bias);
    fwrite(&network->activation_name, sizeof(activation_function_names), 1, fp);
}

neural_network *read_neural_network_structure(FILE *fp) {
    neural_network *network = calloc(sizeof(neural_network), 1);
    network->weights = read_matrix(fp);
    network->bias = read_matrix(fp);
    fread(&network->activation_name, sizeof(activation_function_names), 1, fp);
    add_function_with_derivative(network, network->activation_name);
    network->next_layer = network->previous_layer = NULL;
    return network;
}

network_start_layer read_network_start_layer(FILE *fp) {
    network_start_layer start_layer;
    fread(&start_layer.i, sizeof(int), 1, fp);
    start_layer.next_layer = NULL;
    return start_layer;
}

void write_network_start_layer(FILE *fp, network_start_layer network) {
    fwrite(&network.i, sizeof(int), 1, fp);
}

void save_neural_network(char *file_name, network_start_layer network) {
    FILE *fp = fopen(file_name, "wb");

    write_network_start_layer(fp, network);
    neural_network *network_layer_pointer = network.next_layer;
    while (network_layer_pointer != NULL) {
        write_neural_network_structure(fp, network_layer_pointer);
        network_layer_pointer = network_layer_pointer->next_layer;
    }

    fclose(fp);
}

network_start_layer read_neural_network(char *file_name) {
    FILE *fp = fopen(file_name, "rb");

    network_start_layer network = read_network_start_layer(fp);
    network.next_layer = read_neural_network_structure(fp);
    neural_network *network_layer_pointer = network.next_layer;
    while (!check_end(fp)) {
        neural_network *new_layer = read_neural_network_structure(fp);
        network_layer_pointer->next_layer = new_layer;
        new_layer->previous_layer = network_layer_pointer;
        network_layer_pointer = network_layer_pointer->next_layer;
    }

    fclose(fp);
    return network;
}