cmake_minimum_required(VERSION 3.24)
project(C_version C)

set(CMAKE_C_STANDARD 11)

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()


add_executable(C_version main.c neural_network/neural_network.c neural_network/weight_initialize/weight_initializers.c neural_network/regularization/losses.c neural_network/regularization/regularization_params.c test/main_tests.c neural_network/Optimizers/momentum_optimizer.c neural_network/Optimizers/optimizers_functions.c neural_network/Optimizers/SGD.c neural_network/activation_functions/activation_functions.c neural_network/activation_functions/network_activation_functions.c math/matrix_operations.c math/statistical_random.c data/my_print.c data/DAO.c neural_network/neural_structs.h neural_network/Optimizers/Nesterov_accelerated_gd.c neural_network/Optimizers/Nesterov_accelerated_gd.h neural_network/Optimizers/Adam_optimizer.c neural_network/Optimizers/Adam_optimizer.h terminal_output/progres_bar.c terminal_output/progres_bar.h math/batch_operations.c math/batch_operations.h data/save_nn.c data/save_nn.h data/bin_writers.c data/bin_writers.h neural_network/Optimizers/Adam_future_view_optimizer.c neural_network/Optimizers/Adam_future_view_optimizer.h neural_network/learning_rates.c neural_network/learning_rates.h)

target_link_libraries(C_version m)