#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    int m = 100; // Problem size
    int n = 100;
    double tol = 0.001; // Consistent tolerance

    double t[m + 2][n + 2], tnew[m + 1][n + 1], diff, difmax;

    double start_time = 0.0, end_time = 0.0, runtime_serial = 0.0, runtime_parallel = 0.0;

    // Perform serial execution for comparison
    start_time = omp_get_wtime();
    // ... (Serial implementation of the Jacobi method without OpenMP)
    end_time = omp_get_wtime();
    runtime_serial = end_time - start_time;

    // Perform parallel execution with OpenMP for various thread counts
    int num_threads_array[] = {2, 4, 8}; // Range of thread counts
    int num_thread_configs = sizeof(num_threads_array) / sizeof(num_threads_array[0]);
    double speedup[num_thread_configs];

    for (int t_index = 0; t_index < num_thread_configs; t_index++) {
        omp_set_num_threads(num_threads_array[t_index]); // Set number of threads

        start_time = omp_get_wtime();
        // ... (OpenMP parallel implementation of the Jacobi method with the specified number of threads)
        end_time = omp_get_wtime();
        runtime_parallel = end_time - start_time;

        speedup[t_index] = runtime_serial / runtime_parallel;
        printf("Run-time with %d threads: %f seconds\n", num_threads_array[t_index], runtime_parallel);
        printf("Speedup with %d threads: %f\n", num_threads_array[t_index], speedup[t_index]);
    }

    return 0;
}