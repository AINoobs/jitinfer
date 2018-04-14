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
#pragma once

#include <jitinfer.h>
#include "jit_conv_kernel.h"

namespace jitinfer {

template <typename dst_data_t>
class op_conv : public op {
  typedef u8 src_data_t;
  typedef s8 wei_data_t;
  // typedef s32 bia_data_t;
  typedef s32 acc_data_t;

public:
  explicit op_conv(const std::unique_ptr<memory> &src,
                   const std::unique_ptr<memory> &wei,
                   const std::unique_ptr<memory> &bia,
                   std::array<int, 2> sz_stride,
                   std::array<int, 2> sz_padding,
                   std::unique_ptr<memory> &dst,
                   const std::vector<float> &conv0_scales,
                   const std::vector<float> &conv1_scales,
                   const std::unique_ptr<memory> &wei1x1 = nullptr,
                   const std::unique_ptr<memory> &bia1x1 = nullptr,
                   bool conv0_relu = false,
                   bool conv1_relu = false,
                   round_mode conv0_round_mode = round_mode::nearest,
                   round_mode conv1_round_mode = round_mode::nearest);

  ~op_conv();

protected:
  bool init_conf(jit::jit_conv_conf_t &conf,
                 const std::unique_ptr<memory> &src,
                 const std::unique_ptr<memory> &wei,
                 const std::unique_ptr<memory> &bia,
                 int ngroups,  // only enabled on conv0
                 std::array<int, 2> sz_stride,
                 std::array<int, 2> sz_padding,
                 std::unique_ptr<memory> &dst,
                 const std::vector<float> &conv0_scales,
                 const std::vector<float> &conv1_scales,
                 const std::unique_ptr<memory> &wei1x1,
                 const std::unique_ptr<memory> &bia1x1,
                 bool conv0_relu,
                 bool conv1_relu,
                 round_mode conv0_round_mode,
                 round_mode conv1_round_mode);
  void infer() override;
  inline void infer_conv0();
  inline void infer_conv0conv1();
  const char *name() { return "conv"; }

private:
  bool fuse_conv1x1_;
  const src_data_t *src_data_;
  const wei_data_t *wei_data_, *wei1x1_data_;
  const void *bia_data_, *bia1x1_data_;
  float *conv0_scales_data_, *conv1_scales_data_;
  dst_data_t *dst_data_;
  jit::jit_conv_kernel *kernel_;
  size_t ws_per_thread_;
  size_t ws1x1_per_thread_;
  acc_data_t *ws_;
  acc_data_t *ws1x1_;
};
}
