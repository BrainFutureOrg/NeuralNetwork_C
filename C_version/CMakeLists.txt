cmake_minimum_required(VERSION 3.24)
project(C_version C)

set(CMAKE_C_STANDARD 11)

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()


add_executable(C_version main.c matrix_operations.c matrix_operations.h neural_network/activation_functions.c neural_network/activation_functions.h neural_network/neural_network.c neural_network/neural_network.h neural_network/network_activation_functions.c neural_network/network_activation_functions.h data/DAO.c data/DAO.h neural_network/optimizers.c neural_network/optimizers.h statistical_random.c statistical_random.h data/my_print.h data/my_print.c neural_network/regularization_params.c neural_network/regularization_params.h test/main_tests.c test/main_tests.h neural_network/losses.c neural_network/losses.h)

target_link_libraries(C_version m)