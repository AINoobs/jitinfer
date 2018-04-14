/*******************************************************************************
 * Copyright 2018 Tensor Tang. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*******************************************************************************/
/**
 * This file defines some utilities that do not depends on jitinfer itself
 */
#pragma once

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "omp_thread.h"
#ifdef WIN32
#include <malloc.h>
#include <windows.h>
#endif

namespace jitinfer {

namespace util {

template <typename T>
inline size_t array_product(const T *p, size_t num) {
  size_t out = 1;
  for (size_t i = 0; i < num; ++i) {
    out *= size_t(p[i]);
  }
  return out;
}

template <typename T, typename P>
inline bool one_of(T val, P item) {
  return val == item;
}
template <typename T, typename P, typename... Args>
inline bool one_of(T val, P item, Args... item_others) {
  return val == item || one_of(val, item_others...);
}

template <typename T>
inline bool all_true(T expr) {
  return expr;
}

template <typename T, typename... Args>
inline bool all_true(T expr, Args... others_expr) {
  return expr && all_true(others_expr...);
}

inline int dividable_of(int val, int divisor) {
  if (val % divisor == 0) {
    return divisor;
  } else {
    return 1;
  }
}

template <typename... Args>
inline int dividable_of(int val, int divisor, Args... others_divisor) {
  if (val % divisor == 0) {
    return divisor;
  } else {
    return dividable_of(val, others_divisor...);
  }
}

inline int find_dividable(int val, int divisor) {
  if (divisor <= 1) {
    return 1;
  }
  if (divisor > val) {
    return val;
  }
  if (val % divisor == 0) {
    return divisor;
  } else {
    return find_dividable(val, divisor - 1);
  }
}

template <typename T>
inline void copy_array(T *dst, const T *src, size_t sz) {
// do not use memcpy, in case of memory aligment
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < sz; ++i) {
    dst[i] = src[i];
  }
}

template <typename T, typename U>
inline void set_array(T *arr, const U &val, size_t size) {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < size; ++i) {
    arr[i] = static_cast<T>(val);
  }
}

template <typename T>
struct remove_reference {
  typedef T type;
};
template <typename T>
struct remove_reference<T &> {
  typedef T type;
};
template <typename T>
struct remove_reference<T &&> {
  typedef T type;
};

template <typename T, typename U>
inline typename remove_reference<T>::type div_up(const T a, const U b) {
  assert(b);
  return (a + b - 1) / b;
}
template <typename T>
inline T &&forward(typename remove_reference<T>::type &t) {
  return static_cast<T &&>(t);
}
template <typename T>
inline T &&forward(typename remove_reference<T>::type &&t) {
  return static_cast<T &&>(t);
}

template <typename T>
inline typename remove_reference<T>::type zero() {
  auto zero = typename remove_reference<T>::type();
  return zero;
}

// divide jobs on workers
// for example 4 jobs to 3 worker get 2,1,1
template <typename T, typename U>
inline void balance211(T n, U team, U tid, T &n_start, T &n_end) {
  T n_min = 1;
  T &n_my = n_end;
  if (team <= 1 || n == 0) {
    n_start = 0;
    n_my = n;
  } else if (n_min == 1) {
    // team = T1 + T2
    // n = T1*n1 + T2*n2  (n1 - n2 = 1)
    T n1 = div_up(n, (T)team);
    T n2 = n1 - 1;
    T T1 = n - n2 * (T)team;
    n_my = (T)tid < T1 ? n1 : n2;
    n_start = (T)tid <= T1 ? tid * n1 : T1 * n1 + ((T)tid - T1) * n2;
  }

  n_end += n_start;
}

template <typename T>
inline T nd_iterator_init(T start) {
  return start;
}
template <typename T, typename U, typename W, typename... Args>
inline T nd_iterator_init(T start, U &x, const W &X, Args &&... tuple) {
  start = nd_iterator_init(start, forward<Args>(tuple)...);
  x = start % X;
  return start / X;
}

inline bool nd_iterator_step() { return true; }
template <typename U, typename W, typename... Args>
inline bool nd_iterator_step(U &x, const W &X, Args &&... tuple) {
  if (nd_iterator_step(forward<Args>(tuple)...)) {
    x = (x + 1) % X;
    return x == 0;
  }
  return false;
}

template <typename U, typename W, typename Y>
inline bool nd_iterator_jump(U &cur, const U end, W &x, const Y &X) {
  U max_jump = end - cur;
  U dim_jump = X - x;
  if (dim_jump <= max_jump) {
    x = 0;
    cur += dim_jump;
    return true;
  } else {
    cur += max_jump;
    x += max_jump;
    return false;
  }
}
template <typename U, typename W, typename Y, typename... Args>
inline bool nd_iterator_jump(
    U &cur, const U end, W &x, const Y &X, Args &&... tuple) {
  if (nd_iterator_jump(cur, end, forward<Args>(tuple)...)) {
    x = (x + 1) % X;
    return x == 0;
  }
  return false;
}

namespace timer {
inline double get_current_ms() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+3 * time.tv_sec + 1e-3 * time.tv_usec;
};
}

namespace env {
int _getenv(char *value, const char *name, int length);
bool profiling_time();
bool jit_dump_code();
}
}

void *aligned_malloc(size_t size, int alignment);
void free(void *p);
}
