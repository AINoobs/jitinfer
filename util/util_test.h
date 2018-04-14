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
 * This file defines some utilities that used in gtest
 */

#pragma once
#include <cmath>
#include "gtest/gtest.h"
#include "jitinfer.h"
#include "omp_thread.h"
#include "util.h"
#include "util_jitinfer.h"

namespace jitinfer {
namespace util {

template <typename data_t>
static inline data_t set_value(size_t index) {
  using data_type = jitinfer::memory::dtype;
  if (type2dtype<data_t>::dtype == data_type::f32) {
    double mean = 1., deviation = 1e-2;
    return static_cast<data_t>(mean + deviation * sinf(float(index % 37)));
  } else if (one_of(type2dtype<data_t>::dtype, data_type::s8, data_type::s32)) {
    return data_t(rand() % 21 - 10);
  } else if (type2dtype<data_t>::dtype == data_type::u8) {
    return data_t(rand() % 17);
  } else {
    return data_t(0);
  }
}

template <typename T>
void fill_data(T* p, size_t sz) {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < sz; i++) {
    p[i] = set_value<T>(i);
  }
}

template <typename data_t>
static inline data_t set_value(size_t index, data_t mmin, data_t mmax) {
  using data_type = jitinfer::memory::dtype;
  if (type2dtype<data_t>::dtype == data_type::f32) {
    return static_cast<data_t>(mmin +
                               (mmax - mmin) * (float)(rand() % 100) / 100.f);
  } else if (one_of(type2dtype<data_t>::dtype,
                    data_type::s8,
                    data_type::u8,
                    data_type::s32)) {
    return data_t(mmin + rand() % (s32)(mmax - mmin));
  } else {
    return data_t(0);
  }
}

template <typename T>
void fill_data(T* p, size_t sz, T a, T b) {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < sz; i++) {
    p[i] = set_value<T>(i, std::min(a, b), std::max(a, b));
  }
}

template <typename T>
void compare_array(T* dst, T* ref, size_t sz) {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < sz; ++i) {
    if (std::is_same<T, f32>::value) {
      f32 diff = dst[i] - ref[i];
      f32 e = (std::fabs(ref[i]) > (f32)1e-4) ? diff / ref[i] : diff;
      EXPECT_NEAR(e, 0.f, (f32)1e-4) << "Index: " << i << " Total: " << sz;
    } else {
      EXPECT_EQ(dst[i], ref[i]) << "Index: " << i << " Total: " << sz;
    }
  }
}
}
}
