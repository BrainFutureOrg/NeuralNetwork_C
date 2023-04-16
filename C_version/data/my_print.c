//
// Created by maximus on 10.04.23.
//

#include "my_print.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define num_length(a) a == 0 ? 2 : (int)(ceil(log10(a))+1)

void reverse(char s[]) {
    int i, j;
    char c;

    for (i = 0, j = strlen(s) - 1; i < j; i++, j--) {
        c = s[i];
        s[i] = s[j];
        s[j] = c;
    }
}

void itoa(int n, char s[]) {
    int i, sign;

    if ((sign = n) < 0)
        n = -n;
    i = 0;
    do {
        s[i++] = n % 10 + '0';
    } while ((n /= 10) > 0);
    if (sign < 0)
        s[i++] = '-';
    s[i] = '\0';
    reverse(s);
}

void print_double_number(double number, int integer_allocate, int fraction_allocate) {
    int integer_allocate_size = num_length(integer_allocate);
    int fraction_allocate_size = num_length(fraction_allocate);
    int fraction_len = 7;

    char *s_int = calloc(integer_allocate_size, sizeof(char));
    char *s_frac = calloc(fraction_allocate_size, sizeof(char));

    char *str = calloc(integer_allocate_size + fraction_allocate_size + fraction_len, sizeof(char));
    str[0] = '\0';
    itoa(integer_allocate, s_int);
    itoa(fraction_allocate, s_frac);

    strcat(str, "%");
    strcat(str, s_int);
    strcat(str, ".");
    strcat(str, s_frac);
    strcat(str, "f");
    printf(str, number);
//    char *string =
    free(s_int);
    free(s_frac);
    free(str);
}