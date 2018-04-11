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
#include "jitinfer.h"
#include "op_concat.h"
#include "op_conv.h"
#include "util_jitinfer.h"

namespace jitinfer {

memory::dims nchw2format(const memory::nchw_dims &dm,
                         const memory::format fmt) {
  using format = memory::format;
  memory::dims out;
  switch (fmt) {
    case format::nhwc:
      out.resize(4);
      out[0] = dm[0];
      out[1] = dm[2];
      out[2] = dm[3];
      out[3] = dm[1];
      break;
    case format::nchw:
      out.resize(4);
      out[0] = dm[0];
      out[1] = dm[1];
      out[2] = dm[2];
      out[3] = dm[3];
      break;
    case format::OIhw4i16o4i:
      // o/16, i/16, h, w, 4i, 16o, 4i
      out.resize(7);
      out[0] = dm[0] / 16;
      out[1] = dm[1] / 16;
      out[2] = dm[2];
      out[3] = dm[3];
      out[4] = 4;
      out[5] = 16;
      out[6] = 4;
      break;
    default:
      error_and_exit("bad type");
  }
  check_eq(util::array_product<int>(dm.data(), dm.size()),
           util::array_product<int>(out.data(), out.size()));
  return out;
}

memory::memory(const nchw_dims &dm,
               const format fmt,
               const dtype dt,
               int alignment)
    : std_dims_(dm), fmt_(fmt), dt_(dt) {
  dims_ = nchw2format(dm, fmt);
  allocate_buffer(alignment);
}

memory::~memory() { free(data_); }

void memory::allocate_buffer(int alignment) {
  assert(buffer_size() > 0);
  data_ = aligned_malloc(buffer_size(), alignment);
  assert(data_ != NULL);
}

size_t memory::size() {
  return util::array_product<int>(dims_.data(), dims_.size());
}

size_t memory::buffer_size() { return size() * util::dtype_size(dt_); }

void op::submit() {
#ifdef WITH_VERBOSE
  double t_start = 0;
  if (util::env::profiling_time()) {
    t_start = util::timer::get_current_ms();
  }
#endif
  infer();
#ifdef WITH_VERBOSE
  if (util::env::profiling_time()) {
    info("%s infer %f", this->name(), util::timer::get_current_ms() - t_start);
  }
#endif
}

std::unique_ptr<op> concat(const std::vector<std::unique_ptr<memory>> &srcs,
                           std::unique_ptr<memory> &dst,
                           bool post_relu) {
  switch (dst->data_type()) {
#define CASE(tp)          \
  case memory::dtype::tp: \
    return std::unique_ptr<op>(new op_concat<tp>(srcs, dst, post_relu))
    CASE(f32);
    CASE(s32);
    CASE(s8);
    CASE(u8);
#undef CASE
    default:
      assert(!"bad data_type");
  }
  return nullptr;
}

std::unique_ptr<op> conv(const std::unique_ptr<memory> &src,
                         const std::unique_ptr<memory> &wei,
                         const std::unique_ptr<memory> &bia,
                         std::array<int, 2> sz_stride,
                         std::array<int, 2> sz_padding,
                         const std::unique_ptr<memory> &wei1x1,
                         const std::unique_ptr<memory> &bia1x1,
                         std::unique_ptr<memory> &dst,
                         bool conv0_relu,
                         std::vector<float> conv0_scales,
                         round_mode conv0_round_mode,
                         bool conv1_relu,
                         std::vector<float> conv1_scales,
                         round_mode conv1_round_mode) {
  switch (dst->data_type()) {
#define CASE(tp)                                                 \
  case memory::dtype::tp:                                        \
    return std::unique_ptr<op>(new op_conv<tp>(src,              \
                                               wei,              \
                                               bia,              \
                                               sz_stride,        \
                                               sz_padding,       \
                                               dst,              \
                                               conv0_scales,     \
                                               conv1_scales,     \
                                               wei1x1,           \
                                               bia1x1,           \
                                               conv0_relu,       \
                                               conv1_relu,       \
                                               conv0_round_mode, \
                                               conv1_round_mode))
    CASE(f32);
    CASE(s32);
    CASE(s8);
    CASE(u8);
#undef CASE
    default:
      assert(!"bad data_type");
  }
  return nullptr;
}

std::unique_ptr<op> conv(const std::unique_ptr<memory> &src,
                         const std::unique_ptr<memory> &wei,
                         const std::unique_ptr<memory> &bia,
                         std::array<int, 2> sz_stride,
                         std::array<int, 2> sz_padding,
                         std::unique_ptr<memory> &dst,
                         bool conv0_relu,
                         std::vector<float> conv0_scales,
                         round_mode conv0_round_mode) {
  return conv(src,
              wei,
              bia,
              sz_stride,
              sz_padding,
              nullptr,
              nullptr,
              dst,
              conv0_relu,
              conv0_scales,
              conv0_round_mode);
}
}
