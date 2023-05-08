//
// Created by maximus on 08.05.23.
//
#include <math.h>
#include "learning_rates.h"

double exponential_learning_rate(double start_value, double decay_rate, int epoch) {
    for (int i = 0; i < epoch; i++) {
        start_value *= decay_rate;
    }
    return start_value;
}

double decay_learning_rate(double start_value, double decay_rate, int epoch) {
    return start_value / (1 + decay_rate * epoch);
}

double cosine_learning_rate(double start_value, double final_value, int max_steps, int epoch) {
    return epoch <= max_steps ? final_value + (start_value - final_value) * (1 + cos(M_PI * epoch / max_steps)) / 2
                              : final_value;
}

double cosine_restart_learning_rate(double start_value, double final_value, int steps_till_restart, int epoch) {
    return final_value +
           (start_value - final_value) * (1 + cos(M_PI * (epoch % steps_till_restart) / steps_till_restart)) / 2;
}