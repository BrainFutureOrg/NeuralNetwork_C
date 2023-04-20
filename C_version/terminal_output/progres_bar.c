//
// Created by maximus on 19.04.23.
//

#include "progres_bar.h"
#include <stdio.h>

#define bar_len 50

progress_bar create_bar(int number_of_all_elements) {
    return (progress_bar) {bar_len, number_of_all_elements, 0};
}

void print_char_n_times(char c, int n) {
    for (int i = 0; i < n; ++i) {
        putchar(c);
    }
}

void bar_step(progress_bar *bar) {
    bar->this_number += bar->this_number < bar->number_of_all_elements ? 1 : 0;

    printf("%d / %d", bar->this_number, bar->number_of_all_elements);
    printf("  [");
    int equ_num = bar->this_number * bar->bar_length / bar->number_of_all_elements;

    print_char_n_times('=', equ_num);
    if (equ_num < bar->bar_length) {
        putchar('>');
        print_char_n_times(' ', bar->bar_length - 1 - equ_num);
    }
    putchar(']');
    printf("\n\033[1A");
//    putchar('\r');

}

void delete_bar(progress_bar *bar) {
    for (int i = 0; i < bar->bar_length * 2; ++i) {
        putchar(' ');
    }
    putchar('\n');
    printf("\033[1A");
}