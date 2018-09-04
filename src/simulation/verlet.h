/**
 * verlet.h
 * 
 * Verlet integration procedure
 **/

#ifndef VERLET_H_
#define VERLET_H_

void verlet(double x[][2], double v[][2], const double M[], const int n_bodies, const double h, const int t_max, const double eps);
void verlet_omp(double x[][2], double v[][2], const double M[], const int n_bodies, const double h, const int t_max, const double eps, const int n_threads);
#endif
