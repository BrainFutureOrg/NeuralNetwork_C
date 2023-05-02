//
// Created by kosenko on 07.04.23.
//

#ifndef C_VERSION_STATISTICAL_RANDOM_H
#define C_VERSION_STATISTICAL_RANDOM_H

#define randint(a, b) (random() % (b-a) + a)

double randn();

double randu_range(double start, double end);

#endif //C_VERSION_STATISTICAL_RANDOM_H
