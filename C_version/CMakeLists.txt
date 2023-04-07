cmake_minimum_required(VERSION 3.24)
project(C_version C)

set(CMAKE_C_STANDARD 11)


add_executable(C_version main.c matrix_operations.c matrix_operations.h neural_network/activation_functions.c neural_network/activation_functions.h neural_network/neural_network.c neural_network/neural_network.h neural_network/network_activation_functions.c neural_network/network_activation_functions.h data/DAO.c data/DAO.h neural_network/optimizers.c neural_network/optimizers.h statistical_random.c statistical_random.h)

target_link_libraries(C_version m)