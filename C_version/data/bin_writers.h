//
// Created by maximus on 27.04.23.
//

#ifndef C_VERSION_BIN_WRITERS_H
#define C_VERSION_BIN_WRITERS_H

#include "../math/matrix_operations.h"
#include <stdio.h>

void save_matrix(FILE *fp, matrix m);

matrix read_matrix(FILE *fp);

#endif //C_VERSION_BIN_WRITERS_H
