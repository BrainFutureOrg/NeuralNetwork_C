cmake_minimum_required(VERSION 3.24)
project(C_version C)

set(CMAKE_C_STANDARD 11)

add_executable(C_version main.c matrix_operations.c matrix_operations.h neural_network/activation_functions.c neural_network/activation_functions.h neural_network/neural_network.c neural_network/neural_network.h)
