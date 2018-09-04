/**
 * verlet.c
 * 
 * Implementation of Verlet integration
 **/

#include <stdio.h>
#include <cuda.h>

#include "omp.h"

#include "verlet.h"
#include "physics.h"
#include "../utils/array-utils.h"

void verlet(double x[][2], double v[][2], const double M[], const int n_bodies, const double h, const int t_max, const double eps) {
    // declare acceleration
    double a[n_bodies][2];
    zeros_2d(a, n_bodies);

    // start integration
    // get the acceleration at x_t0
    printf("\nGetting initial acceleration...\n");
    get_acceleration(x, M, a, n_bodies, eps);

    // time steps
    for (int t = 1; t < t_max + 1; t++) {
        if ((t % 100) == 0) {
            printf("\nSingle thread, time step %d", t);
        }

        // compute velocities and new positions
        // v_t+1/2 = v_t + (1/2) * h * a_t
        for (int i = 0; i < n_bodies; i++) {
            // step v_(t+1/2)
            v[i][0] += 0.5 * h * a[i][0];
            v[i][1] += 0.5 * h * a[i][1];

            // step x_(t+1)
            // x_t+1 = x_t + h * v_t+1/2
            x[i][0] += h * v[i][0];
            x[i][1] += h * v[i][1];
        }

        // compute v_t+1 = v_t+1/2 + (1/2) * h * a_t+1
        // 1. get a_t+1 first
        zeros_2d(a, n_bodies);
        get_acceleration(x, M, a, n_bodies, eps);

        // 2. compute velocity
        for (int i = 0; i < n_bodies; i++) {
            v[i][0] += 0.5 * h * a[i][0];
            v[i][1] += 0.5 * h * a[i][1];
        }
    }
}

void verlet_omp(double x[][2], double v[][2], const double M[], const int n_bodies, const double h, const int t_max, const double eps, const int n_threads) {
    // declare acceleration
    double a[n_bodies][2];

    // serial: zeros_2d(a, n_bodies)
    // serial: get_acceleration(x, M, a, n_bodies, eps)
    //  shared(a, x, M, n_bodies, eps)
    printf("\nGetting initial acceleration...\n");
    # pragma omp parallel default(shared) num_threads(n_threads)
    {
        # pragma omp for schedule(static)
        for (int i = 0; i < n_bodies; i++) {
            // get acceleration of the i-th body
            get_ai(x, M, a, i, n_bodies, eps);
        }
    }
    
    for (int t = 1; t <= t_max; t++) {
        if ((t % 100) == 0) {
            printf("\nMultithreading execution at timestep %d...", t);
        }

        // compute velocities and new positions
        // v_t+1/2 = v_t + (1/2) * h * a_t
        # pragma omp parallel default(shared) num_threads(n_threads)
        {
            # pragma omp for schedule(static)
            for (int i = 0; i < n_bodies; i++) {
                // step v_(t+1/2)
                v[i][0] += 0.5 * h * a[i][0];
                v[i][1] += 0.5 * h * a[i][1];

                // step x_(t+1)
                // x_t+1 = x_t + h * v_t+1/2
                x[i][0] += h * v[i][0];
                x[i][1] += h * v[i][1];
            }

            # pragma omp for schedule(static)
            for (int i = 0; i < n_bodies; i++) {
                // get a_t+1
                get_ai(x, M, a, i, n_bodies, eps);

                // get v_t+1 = v_t+1/2 + (1/2) * h * a_t+1
                v[i][0] += 0.5 * h * a[i][0];
                v[i][1] += 0.5 * h * a[i][1];
            }
        }

        if ((t % 100) == 0) {
            printf("done timestep %d", t);
        }
    }
}

void verlet_cuda(x, v, M, n_bodies, h, t_max, EPS) {
    printf("\nMoving data to GPU...");
    // TODO: move data to GPU
    
    
    // calcolo accelerazioni iniziali
    compute_a<<<>>>(dev_M, dev_x, dev_a, n_bodies, eps);
    
    // integra nel tempo
    for (int t = 1; t <= t_max; t++) {
        if ((t % 100) == 0) {
            printf("\nCUDA execution at timestep %d...", t);
        }
        
        // compute velocities and new positions
        // v_t+1/2 = v_t + (1/2) * h * a_t
        // lancio kernel per calcolo velocità v_t+1/2
        compute_v<<<>>>(dev_v, dev_a, h);  // dopo questo passaggio, dev_v contiene le velocità aggiornate al tempo t + 1/2
        
        // lancio kernel per calcolo posizione x_t+1
        integrate_position<<<>>>(dev_x, dev_v, h);  // ora x := x_t+1
        
        // ora per calcolare la v_t+1 mi servono le nuove accelerazioni
        // lancio kernel per calcolo accelerazioni in posizioni x_t+1
        compute_a<<<>>>(dev_M, dev_x, dev_a, n_bodies, eps);
        
        // lancio kernel per calcolo velocità v_t+1
        compute_v<<<>>>(dev_v, dev_a, h);

        if ((t % 100) == 0) {
            printf("done timestep %d", t);
        }
    }
}