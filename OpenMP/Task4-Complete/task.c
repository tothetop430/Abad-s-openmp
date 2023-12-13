#include <stdio.h>
#include <math.h>
#include <omp.h>

#define M 100  // Number of grid points in the y-direction
#define N 100  // Number of grid points in the x-direction
#define EPSILON 0.0001  // Tolerance for convergence
#define MAX_ITERATIONS 1000  // Maximum number of iterations

// Function to apply Gauss-Seidel method for heat diffusion
void applyGaussSeidel(double t_old[M][N], double t_new[M][N]) {
    double diff, difmax;
    int iter = 0;
    do {
        difmax = 0.0;
        for (int i = 1; i <= M-1; i++) {
            for (int j = 1; j <= N-1; j++) {
                t_new[i][j] = 0.25 * (t_new[i+1][j] + t_old[i-1][j] + t_new[i][j+1] + t_old[i][j-1]);
                diff = fabs(t_new[i][j] - t_old[i][j]);
                if (diff > difmax) {
                    difmax = diff;
                }
            }
        }
        iter++;
    } while (difmax > EPSILON && iter < MAX_ITERATIONS);
}

// Function to apply Jacobi method for heat diffusion
void applyJacobi(double t_old[M][N], double t_new[M][N]) {
    double diff, difmax;
    int iter = 0;
    do {
        difmax = 0.0;
        for (int i = 1; i <= M-1; i++) {
            for (int j = 1; j <= N-1; j++) {
                t_new[i][j] = 0.25 * (t_old[i+1][j] + t_old[i-1][j] + t_old[i][j+1] + t_old[i][j-1]);
                diff = fabs(t_new[i][j] - t_old[i][j]);
                if (diff > difmax) {
                    difmax = diff;
                }
            }
        }
        iter++;
    } while (difmax > EPSILON && iter < MAX_ITERATIONS);
}

int main() {
    double temperature_old[M][N];
    double temperature_new[M][N];

    // Initialize temperature grids with boundary conditions and initial values
    for (int i = 1; i <= M; i++) {
        temperature_old[i][0] = 40.0;     // Left boundary
        temperature_old[i][N + 1] = 50.0;  // Right boundary
    }
    for (int j = 1; j <= N; j++) {
        temperature_old[0][j] = 10.0;     // Top boundary
        temperature_old[M + 1][j] = 30.0;  // Bottom boundary
    }

    // Apply Gauss-Seidel method
    double start_time = omp_get_wtime();
    applyGaussSeidel(temperature_old, temperature_new);
    double end_time = omp_get_wtime();
    printf("Time taken for Gauss-Seidel method: %f seconds\n", end_time - start_time);

    // Apply Jacobi method with OpenMP parallelization
    start_time = omp_get_wtime();
    applyJacobi(temperature_old, temperature_new);
    end_time = omp_get_wtime();
    printf("Time taken for Jacobi method: %f seconds\n", end_time - start_time);

    return 0;
}
