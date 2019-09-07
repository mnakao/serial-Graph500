#ifndef TIMER_HPP_
#define TIMER_HPP_
double wtime()
{
  struct timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec + 1.0e-6 * t.tv_usec;
}
#endif
