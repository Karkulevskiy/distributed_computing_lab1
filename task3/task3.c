#include "task3.h"
#include <pthread.h>

void rwlock_init(rwlock_t *lock)
{
    pthread_mutex_init(&lock->mutex, NULL);
    pthread_cond_init(&lock->readers_cond, NULL);
    pthread_cond_init(&lock->writers_cond, NULL);
    lock->readers = 0;
    lock->waiting_readers = 0;
    lock->waiting_writers = 0;
    lock->writer_active = false;
}

void rwlock_destroy(rwlock_t *lock)
{
    pthread_mutex_destroy(&lock->mutex);
    pthread_cond_destroy(&lock->readers_cond);
    pthread_cond_destroy(&lock->writers_cond);
}

void rwlock_rdlock(rwlock_t *lock)
{
    pthread_mutex_lock(&lock->mutex);
    if (lock->writer_active || lock->waiting_writers > 0)
    {
        lock->waiting_readers++;
        while (lock->writer_active || lock->waiting_writers > 0)
        {
            pthread_cond_wait(&lock->readers_cond, &lock->mutex);
        }
        lock->waiting_readers--;
    }
    lock->readers++;
    pthread_mutex_unlock(&lock->mutex);
}

void rwlock_wrlock(rwlock_t *lock)
{
    pthread_mutex_lock(&lock->mutex);
    lock->waiting_writers++;
    while (lock->writer_active || lock->readers > 0)
    {
        pthread_cond_wait(&lock->writers_cond, &lock->mutex);
    }
    lock->waiting_writers--;
    lock->writer_active = true;
    pthread_mutex_unlock(&lock->mutex);
}

void rwlock_unlock(rwlock_t *lock)
{
    pthread_mutex_lock(&lock->mutex);
    if (lock->writer_active)
    {
        lock->writer_active = false;
        if (lock->waiting_writers > 0)
        {
            pthread_cond_signal(&lock->writers_cond);
        }
        else if (lock->waiting_readers > 0)
        {
            pthread_cond_broadcast(&lock->readers_cond);
        }
    }
    else
    {
        lock->readers--;
        if (lock->readers == 0 && lock->waiting_writers > 0)
        {
            pthread_cond_signal(&lock->writers_cond);
        }
    }
    pthread_mutex_unlock(&lock->mutex);
}