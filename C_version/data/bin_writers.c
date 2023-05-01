//
// Created by maximus on 27.04.23.
//

#include "bin_writers.h"
#include <stdlib.h>

void save_matrix(FILE *fp, matrix m) {
    fwrite(&m.i, sizeof(int), 1, fp);
    fwrite(&m.j, sizeof(int), 1, fp);
    for (int i = 0; i < m.i; ++i) {
        fwrite(m.table[i], sizeof(double), m.j, fp);
    }
}

matrix read_matrix(FILE *fp) {
    matrix m;
    fread(&m.i, sizeof(int), 1, fp);
    fread(&m.j, sizeof(int), 1, fp);
    m.table = calloc(m.i, sizeof(double *));
    for (int i = 0; i < m.i; ++i) {
        m.table[i] = calloc(m.j, sizeof(double));
        fread(m.table[i], sizeof(double), m.j, fp);
    }
    return m;
}