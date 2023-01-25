//
// Created by maximus on 25.01.23.
//

#ifndef C_VERSION_DAO_H
#define C_VERSION_DAO_H

#include <stdio.h>

void readline(char *file_name);

double *get_line_matrix(FILE *file);

FILE *open_file(char *file_name);

void pass_line(FILE *file);

#endif //C_VERSION_DAO_H
