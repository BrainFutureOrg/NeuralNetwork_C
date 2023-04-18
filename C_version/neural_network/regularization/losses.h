//
// Created by maximus on 10.04.23.
//

#ifndef C_VERSION_LOSSES_H
#define C_VERSION_LOSSES_H

#include "../../math/matrix_operations.h"

double mse(matrix prediction, matrix expected);

matrix mse_derived(matrix prediction, matrix expected);

double crossentropy(matrix prediction, matrix expected);

matrix crossentropy_loss(matrix prediction, matrix expected);

#endif //C_VERSION_LOSSES_H
