//
// Created by maximus on 17.04.23.
//

#ifndef C_VERSION_WEIGHT_INITIALIZERS_H
#define C_VERSION_WEIGHT_INITIALIZERS_H

#include "../neural_network.h"

enum weight_init {
    GAUSSIAN,
    XAVIER,
    XAVIER_NORMALIZED,
    HE_WEIGHT_INITIALIZATION
};

void set_weights(regularization_params *params, enum weight_init weight_name);

#endif //C_VERSION_WEIGHT_INITIALIZERS_H
