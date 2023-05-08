//
// Created by maximus on 08.05.23.
//

#ifndef C_VERSION_LEARNING_RATES_H
#define C_VERSION_LEARNING_RATES_H

double exponential_learning_rate(double start_value, double decay_rate, int epoch);

double decay_learning_rate(double start_value, double decay_rate, int epoch);

double cosine_learning_rate(double start_value, double final_value, int max_steps, int epoch);

double cosine_restart_learning_rate(double start_value, double final_value, int steps_till_restart, int epoch);

#endif //C_VERSION_LEARNING_RATES_H
