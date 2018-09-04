/**
 * array-utils.h
 * 
 * Utilities for array computations
 **/
#ifndef ARRAY_UTILS_H_
#define ARRAY_UTILS_H_

void zeros_2d(double array[][2], const int length);
void zeros_1d(double array[], const int length);
void array_constant_1d(double array[], const int length, const double value);
void array_constant_2d(double array[][2], const int length, const double value);

#endif
