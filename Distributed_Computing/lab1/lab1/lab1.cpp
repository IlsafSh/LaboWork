//#include <iostream>
//#include "omp.h"
//
//int main() {
//	int A;
//	A = 5;
//#pragma omp parallel 
//	{
//	printf("Hello! A=%d\n", A);
//	}
//	return 0;

//#include"omp.h"
//#include <iostream>
//
//int main() {
//	int A[8] = {10, 11, 12, 13, 14, 15, 16, 17};
//	int rank = 498, size;
//	omp_set_dynamic(0);		// Bb130B KaK npaBV1no He 0693aTeneH
//	omp_set_num_threads(8);
//#pragma omp parallel private(rank,size)
//	{
//		rank = omp_get_thread_num();
//		size = omp_get_num_threads();
//		printf("Hello! rank=%d size=%d A[rankl=%d\n", rank, size, A[rank]);
//		printf("rank=%d\n", rank);
//	}
//	printf("rank=%d", rank);
//	return 0;
//}

#include <omp.h>
#include <stdio.h>

int main() {
    const int N = 13;
    const int ProcCount = 4;

    int B[10 * N][10 * N];
    int i, j, numer, size, sum = 0;

    omp_set_dynamic(0);
    omp_set_num_threads(ProcCount);

    // Первая параллельная область - заполнение массива B
#pragma omp parallel for private(j)
    for (i = 0; i < 10 * N; i++) {
        for (j = 0; j < 10 * N; j++) {
            B[i][j] = 1;
        }
    }

    // Вторая параллельная область - нахождение суммы эл-тов массива B
#pragma omp parallel private(numer, size, i, j) reduction(+:sum)
    {
        size = omp_get_num_threads();
        numer = omp_get_thread_num();

#pragma omp sections
        {
#pragma omp section
            for (i = 0; i < 3 * N; i++) {
                for (j = 0; j < 10 * N; j++) {
                    sum += B[i][j];
                }
            }
#pragma omp section
            for (i = 3 * N; i < 6 * N; i++) {
                for (j = 0; j < 10 * N; j++) {
                    sum += B[i][j];
                }
            }
#pragma omp section
            for (i = 6 * N; i < 10 * N; i++) {
                for (j = 0; j < 10 * N; j++) {
                    sum += B[i][j];
                }
            }
        }
    }

    printf("Sum of elements massive B[10*%d][10*%d]: %d\n", N, N, sum);

    return 0;
}
