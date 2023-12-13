#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    int m = 100; // Increase problem size to 100x100
    int n = 100;
    double tol = 0.001; // Set consistent tolerance

    double t[m + 2][n + 2], tnew[m + 1][n + 1], diff, difmax;

    // Initialise temperature array
#pragma omp parallel for
    for (int i = 0; i <= m + 1; i++) {
        for (int j = 0; j <= n + 1; j++) {
            t[i][j] = 30.0;
        }
    }

    // Fix boundary conditions
    for (int i = 1; i <= m; i++) {
        t[i][0] = 40.0;     // Left boundary
        t[i][n + 1] = 50.0;  // Right boundary
    }
    for (int j = 1; j <= n; j++) {
        t[0][j] = 10.0;     // Top boundary
        t[m + 1][j] = 30.0;  // Bottom boundary
    }

    // Main loop
    int iter = 0;
    difmax = 1000000.0;
    double start_time = omp_get_wtime(); // Start timer

    while (difmax > tol) {
        iter++;
        difmax = 0.0;

#pragma omp parallel for reduction(max : difmax)
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                tnew[i][j] = (t[i - 1][j] + t[i + 1][j] + t[i][j - 1] + t[i][j + 1]) / 4.0;

                // Work out maximum difference between old and new temperatures
                diff = fabs(tnew[i][j] - t[i][j]);
                if (diff > difmax) {
                    difmax = diff;
                }
                t[i][j] = tnew[i][j];
            }
        }
    }

    double end_time = omp_get_wtime(); // Stop timer
    double runtime = end_time - start_time;

    printf("Runtime: %f seconds\n", runtime);

    return 0;
}
