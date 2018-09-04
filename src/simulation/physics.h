/**
 * Physics.h
 * 
 * Definition file for physical constants and equations
 **/

#ifndef PHYSICS_H_
#define PHYSICS_H_

void get_acceleration(double x[][2],
                      const double M[],
                      double a[][2],
                      const int n_bodies,
                      const double eps);

void get_ai(double x[][2],
            const double M[],
            double a[][2],
            const int i,
            const int n_bodies,
            const double eps);

#endif
