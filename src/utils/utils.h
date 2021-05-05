#include <immintrin.h>
#include "defines.h"

void populate_table(double *table);
void print_table(const double *table);
void print_m512d(const __m512d col_vec);
void print_sum_col(const double *col);
