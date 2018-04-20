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
#include "util_benchmark.h"
#include <sstream>
#include "log.h"
#include "omp_thread.h"

namespace jitinfer {
namespace util {

#ifdef WITH_COLD_CACHE
dummy_memory::dummy_memory(size_t num_bytes) {
  int max_nthr = omp_get_max_threads();
  debug("Max OMP threads: %d", max_nthr);
  size_ = num_bytes * max_nthr;
  p_ = (unsigned char*)aligned_malloc(size_, 64);
}
dummy_memory::~dummy_memory() { free(p_); }
void dummy_memory::clear_cache() {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < size_; ++i) {
    p_[i] = p_[i] * 2;
  }
}

// skx, L3: 1.375MB * n
//      L2: 1MB
//      L1: 32KB
constexpr size_t PAGE_2MB = 2 * 1024 * 1024;
static dummy_memory dummy_mem(PAGE_2MB);
void clear_cache() { dummy_mem.clear_cache(); }
#else
// hot cache, do nothing
void clear_cache() { ; }
#endif

const char* dtype2str(memory::dtype dt) {
  using dtype = memory::dtype;
  switch (dt) {
    case dtype::f32:
      return "f32";
    case dtype::s32:
      return "s32";
    case dtype::s8:
      return "s8";
    case dtype::u8:
      return "u8";
    default:
      error_and_exit("Unknow data type");
      return NULL;
  }
}

memory::dtype str2dtype(const std::string& str) {
  using dtype = memory::dtype;
  if (str == "f32") {
    return dtype::f32;
  } else if (str == "s32") {
    return dtype::s32;
  } else if (str == "s8") {
    return dtype::s8;
  } else if (str == "u8") {
    return dtype::u8;
  } else {
    error_and_exit("Unknow data type %s", str.c_str());
    return dtype::undef;
  }
}

memory::dtype str2dtype(const char* str) { return str2dtype(std::string(str)); }

std::vector<std::string> split(const std::string& s, char delimiter) {
  std::stringstream ss(s);
  std::string item;
  std::vector<std::string> tokens;
  while (std::getline(ss, item, delimiter)) {
    tokens.push_back(item);
  }
  return tokens;
}
}
}
