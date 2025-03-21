﻿//#include <stdio.h> 
//#include <mpi.h> 
//#include <stdlib.h> 
//
//int main(int argc, char* argv[])
//{
//	int myid, numprocs, namelen;
//	char processor_name[MPI_MAX_PROCESSOR_NAME];
//
//	MPI_Init(&argc, &argv);        // starts MPI
//	MPI_Comm_rank(MPI_COMM_WORLD, &myid);  // get current process id
//	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);      // get number of processeser
//	MPI_Get_processor_name(processor_name, &namelen);
//
//	if (myid == 0) printf("number of processes: %d\n...", numprocs);
//	printf("%s: Hello world from process %d \n", processor_name, myid);
//
//	MPI_Finalize();
//
//	return 0;
//}

//#include<mpi.h>
//#include<iostream>
//using namespace std;
//
//int main(int argc, char** argv)
//{
//    int rank, size;
//    MPI_Init(&argc, &argv);
//    MPI_Comm_size(MPI_COMM_WORLD, &size);
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//    cout << "The number of processes: " << size << " my number is " << rank << endl;
//    MPI_Finalize();
//    return 0;
//}

	//Пример последовательной программы
//#include <stdio.h>
//#define N1 32
//#define N2 64
//
//int main(int argc, char** argv)
//{
//	float A[N1][N2], B[N1], C[N1];
//	int i, j, k;
//
//	// Инициализация исходных данных
//	for (i = 0; i < N1; i++) {
//		for (j = 0; j < N2; j++) A[i][j] = i + j;
//		B[i] = i;
//	}
//
//	// Вычисления
//	for (i = 0; i < N1; i++) {
//		C[i] = 0;
//		for (j = 0; j < N2; j++) C[i] += B[i] * A[i][j];
//	}
//
//	//Вывод результатов
//	for (i = 0; i < N1; i++) printf("%f ", C[i]);
//
//	return 0;
//}

//	//Пример параллельной программы
//#include <stdio.h>
//#include <mpi.h>
//#define N1 32
//#define N2 64
//
//int main(int argc, char** argv)
//{
//	float A[N1][N2], B[N1], C[N1];
//	int rank, size, cicle, i, j, k;
//	double time, time1, time2;
//	MPI_Status status;		// Переменная-структура, в кот будут сохр параметры принимаемых сооб с данными
//	MPI_Init(&argc, &argv);
//	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//	MPI_Comm_size(MPI_COMM_WORLD, &size);
//
//	printf("Hello (%d)-%d\n", rank, size);
//	if (rank == 0) {	// Если процесс нулевой
//		for (i = 0; i < N1; i++) {
//			for (j = 0; j < N2; j++)
//				A[i][j] = i + j;
//			B[i] = i;
//		}
//
//		time1 = MPI_Wtime();	// Определение времени начала обработки
//		for (cicle = 0; cicle < 10000; cicle++) {	// Цикл кратности
//			for (i = 1; i < size; i++) {	// Распределение элем массива B между процессами
//				MPI_Send(&B[(int)i * N1 / size], (int)N1 / size, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
//			}
//
//
//			// Вычисления
//			for (i = 0; i < N1 / size; i++) {
//				C[i] = 0;
//				for (j = 0; j < N2; j++)
//					C[i] += B[i] * A[i][j];
//			}
//			for (i = 1; i < size; i++)		// Сбор результатов в массив C
//				MPI_Recv(&C[(int)i * N1 / size], (int)N1 / size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);
//		}
//		time2 = MPI_Wtime();	// Определение времени конца обработки
//		time = time2 - time1;
//		for (i = 0; i < N1; i++)
//			printf("%f ", C[i]);	// Вывод результатов и времени выполнения
//		if (rank == 0)
//			printf("\nTIME=%f\n", time);
//	}
//
//	else {
//		for (i = 0; i < N1; i++)
//			for (j = 0; j < N2; j++)
//				A[i][j] = i + j;
//		for (cicle = 0; cicle < 10000; cicle++) {	// Цикл кратности
//			MPI_Recv(&B[(int)rank * N1 / size], (int)N1 / size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
//			for (i = rank * N1 / size; i < (rank + 1) * N1 / size; i++) {
//				C[i] = 0;
//				for (j = 0; j < N2; j++)
//					C[i] += B[i] * A[i][j];
//			}
//			// Передача результатов нулевому процессу
//			MPI_Send(&C[(int)rank * N1 / size], (int)N1 / size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
//		}
//	}
//
//	MPI_Finalize();
//
//	return 0;
//}

#include <iostream>
#include <stdio.h>
#include <mpi.h>
#include <cstdlib>

#define N 131072
#define max 10
#define S1 10
#define S2 15

void sequent_pr(int rnd);
int parallel_pr(int argc, char** argv, int rnd);

int main(int argc, char** argv) {
    int rnd = 42;		// Фиксированный seed генерации
    //printf("----------Sequential program----------\n");
    //sequent_pr(rnd);
    printf("----------Parallel program----------\n");
    parallel_pr(argc, argv, rnd);

    return 0;
}

void sequent_pr(int rnd) {
    float* A = (float*)malloc(N * sizeof(float));
    float* B = (float*)malloc(N * sizeof(float));
    float* C = (float*)malloc(N * sizeof(float));
    float* Y = (float*)malloc(N * sizeof(float));

    srand(rnd);
    for (int i = 0; i < N; i++) {
        A[i] = rand() % max;
        B[i] = rand() % max;
        C[i] = rand() % max;
    }

    for (int i = 0; i < N; i++) {
        Y[i] = (A[i] * B[i] + S1) / S2 + C[i];
    }

    // Вывод результатов
    // for (int i = 0; i < N; i++)
    //     printf("Y[%d] = %f\n", i, Y[i]);

    free(A);
    free(B);
    free(C);
    free(Y);

    return;
}

int parallel_pr(int argc, char** argv, int rnd) {
    int rank, size;
    double time1, time2;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    printf("Rank %d, Size %d\n", rank, size);

    int chunk = N / size; // Размер блока данных

    // Выделение памяти в куче
    float* A = (float*)malloc(N * sizeof(float));
    float* B = (float*)malloc(N * sizeof(float));
    float* C = (float*)malloc(N * sizeof(float));
    float* Y = (float*)malloc(N * sizeof(float));

    if (rank == 0) {
        srand(rnd);
        for (int i = 0; i < N; i++) {
            A[i] = rand() % max;
            B[i] = rand() % max;
            C[i] = rand() % max;
        }

        time1 = MPI_Wtime(); // Начало измерения времени

        for (int cicl = 0; cicl < 10000; cicl++) { // Цикл кратности
            for (int i = 1; i < size; i++) {
                MPI_Send(&A[i * chunk], chunk, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
                MPI_Send(&B[i * chunk], chunk, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
                MPI_Send(&C[i * chunk], chunk, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            }
            // Цикл вычислений
            for (int i = 0; i < chunk; i++) {
                Y[i] = (A[i] * B[i] + S1) / S2 + C[i];
            }
            // Сбор результатов в массив Y
            for (int i = 1; i < size; i++) {
                MPI_Recv(&Y[i * chunk], chunk, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);
            }
        }

        time2 = MPI_Wtime();    // Определение времени конца обработки
        double time = time2 - time1;

        // Вывод результатов
        // for (int i = 0; i < N; i++)
        //     printf("Y[%d] = %f\n", i, Y[i]);

        printf("TIME=%f\n", time);
    }
    else {  // Для всех процессов, кроме нулевого
        for (int cicl = 0; cicl < 10000; cicl++) {
            // Прием массивов
            MPI_Recv(&A[rank * chunk], chunk, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(&B[rank * chunk], chunk, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(&C[rank * chunk], chunk, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
            // Вычисления
            for (int i = 0; i < chunk; i++) {
                Y[rank * chunk + i] = (A[rank * chunk + i] * B[rank * chunk + i] + S1) / S2 + C[rank * chunk + i];
            }
            // Передача результатов нулевому процессу
            MPI_Send(&Y[rank * chunk], chunk, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        }
    }

    // Освобождение памяти
    free(A);
    free(B);
    free(C);
    free(Y);

    MPI_Finalize();
    return 0;
}
