//
// Created by maximus on 23.04.23.
//

#include <stdlib.h>
#include "batch_operations.h"

batch batch_create(int size) {
    batch result;
    result.batch_elements = calloc(size, sizeof(matrix));
    result.size = size;
    return result;
}

void batch_free(batch element) {
    for (int i = 0; i < element.size; ++i) {
        matrix_free(element.batch_elements[i]);
    }
    free(element.batch_elements);
}