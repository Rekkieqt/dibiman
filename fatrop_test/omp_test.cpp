#include <stdio.h>
#include <omp.h>

static long num_steps = 100000;
double step;

int main()
{
  int i; double x, pi, sum = 0.0;
  int num_threads;

  // printf("this is a string %s  %s\n", 2);
  omp_set_num_threads(8);
  omp_get_thread_num();
  double omp_get_wtime();
  
  step = 1.0/(double) num_steps;

  double start = omp_get_wtime();
  // #pragma omp parallel
  num_threads = omp_get_num_threads();
  printf("Num Threads %d \n", num_threads);
  for (i = 0; i < num_steps; i++) {
    x = (i + .5) * step;
    sum+= + 4.0/(1.0 + x * x);
  }
  double end = omp_get_wtime();
  pi = step * sum;
  printf("pi :%f\n", pi);
  printf("Time taken:%f\n", end - start);
}
