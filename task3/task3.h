#ifndef TASK3_H
#define TASK3_H

#include <pthread.h>
#include <stdbool.h>

typedef struct rwlock_t
{
    pthread_mutex_t mutex;
    pthread_cond_t readers_cond;
    pthread_cond_t writers_cond;

    int readers;

    int waiting_readers;
    int waiting_writers;

    bool writer_active;
} rwlock_t;

void rwlock_init(rwlock_t *lock);
void rwlock_destroy(rwlock_t *lock);

void rwlock_rdlock(rwlock_t *lock);
void rwlock_wrlock(rwlock_t *lock);
void rwlock_unlock(rwlock_t *lock);

#endif /* TASK3_H */