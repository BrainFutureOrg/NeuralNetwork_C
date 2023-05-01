//
// Created by maximus on 26.04.23.
//

#ifndef C_VERSION_SAVE_NN_H
#define C_VERSION_SAVE_NN_H

#include "../neural_network/neural_network.h"

void save_neural_network(char *file_name, network_start_layer network);

network_start_layer read_neural_network(char *file_name);

#endif //C_VERSION_SAVE_NN_H
