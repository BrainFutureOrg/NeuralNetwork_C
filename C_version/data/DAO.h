//
// Created by maximus on 25.01.23.
//

#ifndef C_VERSION_DAO_H
#define C_VERSION_DAO_H

#include <stdio.h>
#include "../math/matrix_operations.h"

typedef struct data_reader {
    FILE *fp;

    void (*data_prepare)(matrix);

    int sample_number;
    int batch_size;
    int this_elem;
} data_reader;

data_reader
create_data_reader(char *file_name, int from_line, int sample_number, int batch_size, void (*data_prepare)(matrix));

matrix **read_batch_from_data_nn(data_reader *reader);

void close_data_reader(data_reader reader);

void readline(char *file_name);

double *get_line_matrix(FILE *file);

FILE *open_file(char *file_name);

void pass_line(FILE *file);

#endif //C_VERSION_DAO_H
