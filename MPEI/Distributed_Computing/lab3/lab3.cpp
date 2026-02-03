#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

// === Параметры системы ===
#define NUM_CLIENTS 3         // Количество клиентов (по варианту)
#define NUM_EXECUTORS 6       // Количество исполнителей (по варианту)
#define NUM_JOB_TYPES 3       // Количество типов заданий (по варианту)
#define NUM_CONTROLLERS 4     // Менеджер, планировщик, 2 диспетчера
#define TOTAL_THREADS (NUM_CLIENTS + NUM_EXECUTORS + NUM_CONTROLLERS) // Общее число потоков в системе

// === Структура задания ===
typedef struct {
    int type;        // Тип вычислений (0, 1, 2)
    int size;        // Объём вычислений
    int client_id;   // Идентификатор клиента-заказчика
} Job;

// === Структура исполнителя ===
typedef struct {
    int id;             // Уникальный идентификатор исполнителя
    int job_type;       // Тип вычислений, который может выполнять исполнитель
    int performance;    // Производительность (1, 2 или 4)
    int is_busy;        // Занят ли исполнитель (1 — да, 0 — нет)
    Job current_job;    // Задание, которое в данный момент выполняется
} Executor;

// === Глобальные переменные ===
Executor executors[NUM_EXECUTORS];         // Массив исполнителей
Job client_jobs[NUM_CLIENTS];              // Задания от клиентов
Job dispatcher_results[NUM_CLIENTS];       // Результаты, полученные диспетчерами

// === Инициализация исполнителей ===
void init_executors() {
    for (int i = 0; i < NUM_EXECUTORS; i++) {
        executors[i].id = i;
        executors[i].job_type = i % NUM_JOB_TYPES;                      // Распределение по типам заданий
        executors[i].performance = (i % 3 == 0) ? 1 : ((i % 3 == 1) ? 2 : 4);  // Произвольное распределение производительности
        executors[i].is_busy = 0;                                       // Все свободны по умолчанию
    }
}

// === Генерация случайных заданий для клиентов ===
void generate_jobs() {
    for (int i = 0; i < NUM_CLIENTS; i++) {
        client_jobs[i].type = rand() % NUM_JOB_TYPES;                   // Случайный тип задания
        client_jobs[i].size = (rand() % 10 + 1) * 10;                   // Случайный объём (10, 20, ..., 100)
        client_jobs[i].client_id = i;                                   // ID клиента
    }
}

// === Планировщик распределяет задания исполнителям ===
void planner(Job* jobs, Executor* executors, int assignments[NUM_CLIENTS]) {
    for (int i = 0; i < NUM_CLIENTS; i++) {
        dispatcher_results[i].type = -1;         // По умолчанию — результата нет
        dispatcher_results[i].size = -1;
        dispatcher_results[i].client_id = i;
        assignments[i] = -1;                     // Исполнитель ещё не назначен

        for (int j = 0; j < NUM_EXECUTORS; j++) {
            if (!executors[j].is_busy && executors[j].job_type == jobs[i].type) {
                assignments[i] = j;              // Назначение исполнителя
                executors[j].is_busy = 1;        // Пометить как занятого
                executors[j].current_job = jobs[i];
                break;                           // Прекращаем поиск
            }
        }
    }
}

// === Выполнение задания исполнителем ===
void perform_job(Executor* executor) {
    int work_time = executor->current_job.size / executor->performance;   // Время работы зависит от объёма и производительности

    // Информационное сообщение
#pragma omp critical
    {
        printf("Executor %d starts job from Client %d of type %d, size %d, perf %d\n",
            executor->id,
            executor->current_job.client_id,
            executor->current_job.type,
            executor->current_job.size,
            executor->performance);
    }

    // Симуляция работы — просто задержка
    for (volatile int i = 0; i < work_time * 1000000; i++);

    // Сохраняем результат для диспетчера
    dispatcher_results[executor->current_job.client_id] = executor->current_job;

    // Освобождаем исполнителя
    executor->is_busy = 0;
}

int main() {
    srand(time(NULL));                        // Инициализация генератора случайных чисел
    omp_set_num_threads(TOTAL_THREADS);      // Установка числа потоков

    init_executors();                         // Инициализация исполнителей
    generate_jobs();                          // Генерация клиентских заданий

    int assignments[NUM_CLIENTS] = { -1, -1, -1 };      // Назначения: клиент -> исполнитель

    printf("Max OpenMP threads available: %d\n", omp_get_max_threads());

    // Модель запускается параллельно
#pragma omp parallel
    {
        int tid = omp_get_thread_num();       // ID текущего потока

        // === Этап 1 === Регистрация узлов и генерация заданий
#pragma omp barrier
        if (tid >= NUM_CLIENTS + NUM_EXECUTORS) { // Контроллеры
            if (tid == NUM_CLIENTS + NUM_EXECUTORS) {
                printf("Resource Manager: received executor statuses\n");
            }
            else if (tid == NUM_CLIENTS + NUM_EXECUTORS + 1) {
                printf("Planner: received system state\n");
            }
        }
        else if (tid < NUM_CLIENTS) {       // Клиенты отправляют задания
            printf("Client %d sends job: type %d, size %d\n",
                tid, client_jobs[tid].type, client_jobs[tid].size);
        }

        // === Этап 2 === Планирование заданий
#pragma omp barrier
        if (tid == NUM_CLIENTS + NUM_EXECUTORS + 1) { // Планировщик
            planner(client_jobs, executors, assignments);
            printf("Planner: assigned jobs\n");
        }
        else if (tid < NUM_CLIENTS) {       // Клиенты посылают задания диспетчерам
            printf("Client %d sends job to dispatcher\n", tid);
        }

        // === Этап 3 === Отправка заданий исполнителям и выполнение
#pragma omp barrier
        if (tid == NUM_CLIENTS + NUM_EXECUTORS + 2 || tid == NUM_CLIENTS + NUM_EXECUTORS + 3) { // Диспетчеры
            for (int i = 0; i < NUM_CLIENTS; i++) {
                // Чётные клиенты → диспетчер 1, нечётные → диспетчер 2
                if ((tid == NUM_CLIENTS + NUM_EXECUTORS + 2 && i % 2 == 0) ||
                    (tid == NUM_CLIENTS + NUM_EXECUTORS + 3 && i % 2 == 1)) {

                    int exec_id = assignments[i];
                    if (exec_id != -1) {
                        printf("Dispatcher sends job of Client %d to Executor %d\n", i, exec_id);
                    }
                }
            }
        }

        if (tid >= NUM_CLIENTS && tid < NUM_CLIENTS + NUM_EXECUTORS) {   // Исполнители работают
            int executor_id = tid - NUM_CLIENTS;
            if (executors[executor_id].is_busy) {
                perform_job(&executors[executor_id]);
            }
        }

        // === Этап 4 === Возврат результата клиентам
#pragma omp barrier
        if (tid == NUM_CLIENTS + NUM_EXECUTORS + 2 || tid == NUM_CLIENTS + NUM_EXECUTORS + 3) {
            for (int i = 0; i < NUM_CLIENTS; i++) {
                if ((tid == NUM_CLIENTS + NUM_EXECUTORS + 2 && i % 2 == 0) ||
                    (tid == NUM_CLIENTS + NUM_EXECUTORS + 3 && i % 2 == 1)) {
                    if (dispatcher_results[i].type != -1) {
                        printf("Dispatcher returns result to Client %d: type %d, size %d\n",
                            i, dispatcher_results[i].type, dispatcher_results[i].size);
                    }
                    else {
                        printf("Dispatcher has no result for Client %d\n", i);
                    }
                }
            }
        }
    }

    return 0;
}
