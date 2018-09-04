/**
 * array-utils.c
 * 
 * Utilities for array computations
 **/
#include "array-utils.h"

void array_constant_1d(double array[], const int length, const double value) {
    for (int i = 0; i < length; i++) {
        array[i] = value;
    }
}

void array_constant_2d(double array[][2], const int length, const double value) {
    for (int i = 0; i < length; i++) {
        array[i][0] = value;
        array[i][1] = value;
    }
}

void zeros_1d(double array[], const int length) {
    array_constant_1d(array, length, 0.0);
}

void zeros_2d(double array[][2], const int length) {
    array_constant_2d(array, length, 0.0);
}
