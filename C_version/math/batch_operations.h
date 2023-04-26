//
// Created by maximus on 23.04.23.
//

#ifndef C_VERSION_BATCH_OPERATIONS_H
#define C_VERSION_BATCH_OPERATIONS_H

#include "matrix_operations.h"

typedef struct batch {
    int size;
    matrix *batch_elements;
} batch;

batch batch_create(int size);

void batch_free(batch element);

#endif //C_VERSION_BATCH_OPERATIONS_H
