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
#include "log.h"
#include "omp_thread.h"

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
                   std::vector<float> conv0_scales = {1.f},
                   std::vector<float> conv1_scales = {1.f},
                   const std::unique_ptr<memory> &wei1x1 = nullptr,
                   const std::unique_ptr<memory> &bia1x1 = nullptr,
                   bool conv0_relu = false,
                   bool conv1_relu = false,
                   round_mode conv0_round_mode = round_mode::nearest,
                   round_mode conv1_round_mode = round_mode::nearest)
      : op(), fuse_conv1x1_(wei1x1 != nullptr) {
    jit::jit_conv_conf_t conf;
    if (!init_conf(conf,
                   src,
                   wei,
                   bia,
                   1,
                   sz_stride,
                   sz_padding,
                   dst,
                   conv0_scales,
                   conv1_scales,
                   wei1x1,
                   bia1x1,
                   conv0_relu,
                   conv1_relu,
                   conv0_round_mode,
                   conv1_round_mode)) {
      error_and_exit("Init Conv op failed!");
    }
    kernel_ = new jit::jit_conv_kernel(conf);
    const auto &jcp = kernel_->jcp;
    const int nthreads = omp_get_max_threads();
    ws_per_thread_ = jcp.oh * jcp.ow * jcp.oc_block * jcp.nb_oc_blocking;
    ws_ = (acc_data_t *)aligned_malloc(
        nthreads * ws_per_thread_ * sizeof(acc_data_t), 4096);  // 64??
    // acc format (h, oc/16, ow, 16o)
    ws1x1_per_thread_ = jcp.oh * jcp.ow * jcp.oc1x1;
    ws1x1_ = (acc_data_t *)aligned_malloc(
        nthreads * ws1x1_per_thread_ * sizeof(acc_data_t), 4096);  // 64??

    // save data point
    // TODO: enable update data handle from outside
    src_data_ = reinterpret_cast<const src_data_t *>(src->data());
    wei_data_ = reinterpret_cast<const wei_data_t *>(wei->data());
    dst_data_ = reinterpret_cast<dst_data_t *>(dst->data());
    bia_data_ =
        bia != nullptr ? reinterpret_cast<const void *>(bia->data()) : NULL;
    wei1x1_data_ = wei1x1 != nullptr
                       ? reinterpret_cast<const wei_data_t *>(wei1x1->data())
                       : NULL;
    bia1x1_data_ = bia1x1 != nullptr
                       ? reinterpret_cast<const void *>(bia1x1->data())
                       : NULL;
    // TODO:refine scale!! malloc scale buffer, align 64?
    conv0_scales_data_ = reinterpret_cast<const float *>(conv0_scales.data());
    conv1_scales_data_ = reinterpret_cast<const float *>(conv1_scales.data());
  }

  ~op_conv() {
    free(ws_);
    free(ws1x1_);
    delete kernel_;
  }

protected:
  bool init_conf(jit::jit_conv_conf_t &conf,
                 const std::unique_ptr<memory> &src,
                 const std::unique_ptr<memory> &wei,
                 const std::unique_ptr<memory> &bia,
                 int ngroups,  // only enabled on conv0
                 std::array<int, 2> sz_stride,
                 std::array<int, 2> sz_padding,
                 std::unique_ptr<memory> &dst,
                 std::vector<float> conv0_scales,
                 std::vector<float> conv1_scales,
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
  const float *conv0_scales_data_, *conv1_scales_data_;
  dst_data_t *dst_data_;
  jit::jit_conv_kernel *kernel_;
  size_t ws_per_thread_;
  size_t ws1x1_per_thread_;
  acc_data_t *ws_;
  acc_data_t *ws1x1_;
};
}
