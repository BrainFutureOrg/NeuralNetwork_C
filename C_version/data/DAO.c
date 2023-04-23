//
// Created by maximus on 25.01.23.
//

#include <stdlib.h>
#include <stdio.h>
#include "DAO.h"
#include "errno.h"

void readline(char *file_name) {
    FILE *file_csv = fopen(file_name, "r");
    int c;
    while ((c = getc(file_csv)) != '\n') {
        putchar(c);
    }
    fclose(file_csv);
}

FILE *open_file(char *file_name) {
    FILE *file_csv = fopen(file_name, "r");
    return file_csv;
}

void pass_line(FILE *file) {
    while (getc(file) != '\n');
}


double *get_line_matrix(FILE *file) {
    double *numbers = calloc(785, sizeof(double));
    for (int i = 0; i < 785; ++i) {
        fscanf(file, "%lf", &numbers[i]);
        getc(file);
    }
    return numbers;
}

data_reader
create_data_reader(char *file_name, int from_line, int sample_number, int batch_size, void (*data_prepare)(matrix)) {
    FILE *fp = open_file(file_name);
    pass_line(fp); // pass header
    for (int i = 0; i < from_line; ++i) {
        pass_line(fp);
    }
    return (data_reader) {fp, data_prepare, sample_number, batch_size, 0};
}

matrix **read_batch_from_data_nn(data_reader *reader) {
    int delta = reader->sample_number - reader->this_elem;
    if (delta < 0) {
        printf("ERROR, ");
        errno = EDOM;
    }
    int sample_number = reader->batch_size < delta ? reader->batch_size : delta;
    matrix **data = calloc(2, sizeof(matrix *));
    matrix *answers = calloc(sample_number, sizeof(matrix));
    matrix *for_predict = calloc(sample_number, sizeof(matrix));
    data[1] = answers;
    data[0] = for_predict;
    for (int w = 0; w < sample_number; w++) {
        double *numbers = get_line_matrix(reader->fp);
        for_predict[w] = make_matrix_from_array(&numbers[1], 28 * 28, 1);
        reader->data_prepare(for_predict[w]);
        answers[w] = create_vector(10, (int) numbers[0]);
        free(numbers);
    }
    reader->this_elem += sample_number;
    return data;
}

void close_data_reader(data_reader reader) {
    fclose(reader.fp);
}