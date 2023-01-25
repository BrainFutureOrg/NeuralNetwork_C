//
// Created by maximus on 25.01.23.
//

#include <stdlib.h>
#include <stdio.h>
#include "DAO.h"

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
