/**
 * physics.c
 * 
 * Implementation of physical equations
 **/

#include <math.h>

#include "omp.h"
#include "physics.h"

void get_acceleration(double x[][2], const double M[], double a[][2], const int n_bodies, const double eps) {
    for (int i = 0; i < n_bodies; i++) {
        for (int j = 0; j < n_bodies; j++) {
            if (i == j) {
                continue;
            }

            double dx = x[j][0] - x[i][0];
            double dy = x[j][1] - x[i][1];

            double distance = sqrt(dx * dx + dy * dy);
            double d3 = distance * distance * distance;
            double denom;

            if (d3 < 1e-15) {
                denom = d3 + eps;
            } else {
                denom = d3;
            }

            double a_i_val = (M[j] / denom);
            a[i][0] += a_i_val * dx;
            a[i][1] += a_i_val * dy;
        }
    }
}

/**
 * Get acceleration on body i-th
 * 
 * Parameters
 * x: body positions, shape (n_bodies, 2)
 * M: body masses, shape (n_bodies, 1)
 * a: body accelerations, shape (n_bodies, 2)
 * i: index of the body to calculate acceleration for
 * n_bodies: number of bodies
 * eps: denominator for numerical stability
 * 
 * Return value
 * void
 * 
 * at the end of the routine, a[i][0] and a[i][1] contain the ax and ay acceleration
 * of the i-th body
 **/
void get_ai(double x[][2], const double M[], double a[][2], const int i, const int n_bodies, const double eps) {
    // make the acceleration = 0
    a[i][0] = 0.0;
    a[i][1] = 0.0;

    // compute acceleration on body i-th
    for (int j = 0; j < n_bodies; j++) {
        if (i == j) {
            continue;
        }

        double dx = x[j][0] - x[i][0];
        double dy = x[j][1] - x[i][1];

        double distance = sqrt(dx * dx + dy * dy);
        double d3 = distance * distance * distance;
        double denom;

        if (d3 < 1e-15) {
            denom = d3 + eps;
        } else {
            denom = d3;
        }

        double a_i_val = (M[j] / denom);
        a[i][0] += a_i_val * dx;
        a[i][1] += a_i_val * dy;
    }
}
