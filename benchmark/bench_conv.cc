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
#include <gflags/gflags.h>
#include <mkldnn.hpp>
#include <sstream>
#include "jitinfer.h"
#include "log.h"
#include "util_benchmark.h"
#include "util_mkldnn.h"
#include "util_params.h"

DEFINE_int32(burning_iter, 50, "Burning iterations");
DEFINE_int32(iter, 100, "Iterations for average");
DEFINE_int32(bs, 0, "Batch size, number of images");
DEFINE_int32(ih, 0, "Input image height");
DEFINE_int32(iw, 0, "Input image width");
DEFINE_int32(kh, 0, "Kernel height");
DEFINE_int32(kw, 0, "Kernel width");
DEFINE_int32(sh, 0, "Stride height");
DEFINE_int32(sw, 0, "Stride width");
DEFINE_int32(ph, 0, "Padding height");
DEFINE_int32(pw, 0, "Padding width");
DEFINE_int32(ic, 0, "Input channels of first conv");
DEFINE_int32(oc, 0, "Output channels of first conv");
DEFINE_int32(oc1x1, 0, "Output channels of 1x1 conv");
DEFINE_string(dtype, "u8", "Dst data type");

static mkldnn::engine eng = mkldnn::engine(mkldnn::engine::cpu, 0);
static const jitinfer::memory::dtype src_dt = jitinfer::memory::dtype::u8;
static const jitinfer::memory::dtype wei_dt = jitinfer::memory::dtype::s8;
static const jitinfer::memory::dtype bia_dt = jitinfer::memory::dtype::s8;
static std::vector<float> scales = {0.3f};
static jitinfer::round_mode rmode = jitinfer::round_mode::nearest;

double bench_mkldnn(const jitinfer::util::conv_params& pm) {
  std::unique_ptr<mkldnn::convolution_forward::desc> desc0, desc1;
  std::unique_ptr<mkldnn::convolution_forward::primitive_desc> pd0, pd1;
  std::unique_ptr<mkldnn::primitive> fwd0, fwd1;
  std::vector<mkldnn::primitive> pp0, pp1;
  // mkldnn memory
  std::unique_ptr<mkldnn::memory> mkldnn_src, mkldnn_wei, mkldnn_bia;
  std::unique_ptr<mkldnn::memory> mkldnn_dst, mkldnn_dst1x1;
  std::unique_ptr<mkldnn::memory> mkldnn_wei1x1, mkldnn_bia1x1;

  // conv0 desc and pd
  desc0 = jitinfer::util::get_conv_desc(
      pm,
      jitinfer::util::exchange::dtype(src_dt),
      jitinfer::util::exchange::dtype(wei_dt),
      jitinfer::util::exchange::dtype(bia_dt),
      jitinfer::util::exchange::dtype(jitinfer::memory::dtype::u8));
  pd0 = jitinfer::util::get_conv_pd(
      desc0, eng, scales, jitinfer::util::exchange::round_mode(rmode), true);
  // memory
  mkldnn_src.reset(new mkldnn::memory(pd0->src_primitive_desc()));
  mkldnn_wei.reset(new mkldnn::memory(pd0->weights_primitive_desc()));
  mkldnn_dst.reset(new mkldnn::memory(pd0->dst_primitive_desc()));
  mkldnn_bia.reset(new mkldnn::memory(pd0->bias_primitive_desc()));
  fwd0.reset(new mkldnn::convolution_forward(
      *pd0, *mkldnn_src, *mkldnn_wei, *mkldnn_bia, *mkldnn_dst));
  pp0.push_back(*fwd0);

  // change the size used for init conv1x1 desc
  jitinfer::util::conv_params pm_conv1 = pm;
  pm_conv1.ic = pm_conv1.oc;
  pm_conv1.ih = pm_conv1.oh;
  pm_conv1.iw = pm_conv1.ow;
  pm_conv1.oc = pm_conv1.oc1x1;
  pm_conv1.kh = 1;
  pm_conv1.kw = 1;
  pm_conv1.ph = 0;
  pm_conv1.pw = 0;
  pm_conv1.sh = 1;
  pm_conv1.sw = 1;
  desc1 = jitinfer::util::get_conv_desc(
      pm_conv1,
      jitinfer::util::exchange::dtype(src_dt),
      jitinfer::util::exchange::dtype(wei_dt),
      jitinfer::util::exchange::dtype(bia_dt),
      jitinfer::util::exchange::dtype(jitinfer::util::str2dtype(FLAGS_dtype)));
  pd1 = jitinfer::util::get_conv_pd(
      desc1, eng, scales, jitinfer::util::exchange::round_mode(rmode), true);
  mkldnn_wei1x1.reset(new mkldnn::memory(pd1->weights_primitive_desc()));
  mkldnn_bia1x1.reset(new mkldnn::memory(pd1->bias_primitive_desc()));
  mkldnn_dst1x1.reset(new mkldnn::memory(pd1->dst_primitive_desc()));
  fwd1.reset(new mkldnn::convolution_forward(
      *pd1, *mkldnn_dst, *mkldnn_wei1x1, *mkldnn_bia1x1, *mkldnn_dst1x1));
  pp1.push_back(*fwd1);

  for (auto i = 0; i < FLAGS_burning_iter; ++i) {
    jitinfer::util::clear_cache();
    mkldnn::stream(mkldnn::stream::kind::eager).submit(pp0).wait();
    mkldnn::stream(mkldnn::stream::kind::eager).submit(pp1).wait();
    jitinfer::util::clear_cache();
  }

  // cal time
  double sum_conv3x3 = 0;
  double sum_conv1x1 = 0;
  for (auto i = 0; i < FLAGS_iter; ++i) {
    jitinfer::util::clear_cache();
    auto s1 = jitinfer::util::timer::get_current_ms();
    mkldnn::stream(mkldnn::stream::kind::eager).submit(pp0).wait();
    auto s2 = jitinfer::util::timer::get_current_ms();
    mkldnn::stream(mkldnn::stream::kind::eager).submit(pp1).wait();
    auto s3 = jitinfer::util::timer::get_current_ms();
    sum_conv3x3 += (s2 - s1);
    sum_conv1x1 += (s3 - s2);
    jitinfer::util::clear_cache();
  }

  auto avg_conv3x3 = sum_conv3x3 / (double)FLAGS_iter;
  auto avg_conv1x1 = sum_conv1x1 / (double)FLAGS_iter;
  std::ostringstream oss;
  oss << "MKL-DNN Conv3x3 fused ReLU + Conv1x1 fused ReLU, avg time: ("
      << avg_conv3x3 << " + " << avg_conv1x1 << ") "
      << avg_conv3x3 + avg_conv1x1 << " ms";
  info("%s", oss.str().c_str());

  return avg_conv3x3 + avg_conv1x1;
}

double bench_jitinfer(const jitinfer::util::conv_params& p) {
  using namespace jitinfer;
  using format = memory::format;
  constexpr format fmt = memory::format::nhwc;

  std::unique_ptr<memory> src, wei, bia, dst, wei1x1, bia1x1, dst1x1;
  auto dst_dt = jitinfer::util::str2dtype(FLAGS_dtype);
  std::array<int, 2> sz_stride = {p.sh, p.sw};
  std::array<int, 2> sz_padding = {p.ph, p.pw};
  src.reset(new memory({p.bs, p.ic, p.ih, p.iw}, fmt, src_dt));
  wei.reset(new memory({p.oc, p.ic, p.kh, p.kw}, format::OIhw4i16o4i, wei_dt));
  bia.reset(new memory({p.oc}, bia_dt));
  wei1x1.reset(new memory({p.oc1x1, p.oc, 1, 1}, format::OIhw4i16o4i, wei_dt));
  bia1x1.reset(new memory({p.oc1x1}, bia_dt));
  dst1x1.reset(new memory({p.bs, p.oc1x1, p.oh, p.ow}, fmt, dst_dt));
  auto c = conv(src,
                wei,
                bia,
                sz_stride,
                sz_padding,
                wei1x1,
                bia1x1,
                dst1x1,
                true,
                scales,
                rmode,
                true,
                scales,
                rmode);

  for (auto i = 0; i < FLAGS_burning_iter; ++i) {
    jitinfer::util::clear_cache();
    c->submit();
    jitinfer::util::clear_cache();
  }

  // cal time
  double sum = 0;
  for (auto i = 0; i < FLAGS_iter; ++i) {
    jitinfer::util::clear_cache();
    auto s1 = jitinfer::util::timer::get_current_ms();
    c->submit();
    auto s2 = jitinfer::util::timer::get_current_ms();
    sum += (s2 - s1);
    jitinfer::util::clear_cache();
  }

  auto avg = sum / (double)FLAGS_iter;
  std::ostringstream oss;
  oss << "Jitinfer Conv3x3 fused ReLU fused Conv1x1 fused ReLU, avg time: "
      << avg << " ms";
  info("%s", oss.str().c_str());

  return avg;
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  using namespace jitinfer::util;
  conv_params defualt_cases[] = {
      /*bs, gp, ic, ih, iw, oc, oh, ow, kh, kw, ph, pw, sh, sw, oc1x1*/
      {2, 1, 32, 4, 4, 32, 4, 4, 3, 3, 1, 1, 1, 1, 32},
      {2, 1, 32, 13, 13, 32, 11, 11, 3, 3, 0, 0, 1, 1, 64},
      {2, 1, 32, 13, 13, 32, 13, 13, 3, 3, 1, 1, 1, 1, 32},
      {2, 1, 32, 120, 360, 64, 120, 360, 3, 3, 1, 1, 1, 1, 32},
      {2, 1, 64, 60, 180, 128, 60, 180, 3, 3, 1, 1, 1, 1, 64},
      {2, 1, 128, 30, 90, 256, 30, 90, 3, 3, 1, 1, 1, 1, 128},
      {2, 1, 256, 15, 45, 512, 15, 45, 3, 3, 1, 1, 1, 1, 256},
      {2, 1, 1024, 15, 45, 512, 15, 45, 3, 3, 1, 1, 1, 1, 80}};
  for (size_t i = 0; i < sizeof(defualt_cases) / sizeof(conv_params); ++i) {
    conv_params& pm = defualt_cases[i];
    std::ostringstream oss;
    info("==========================================");
    oss << "Benchmark with data type: u8s8s8" << FLAGS_dtype;
    oss << "\nData sizes: In(" << pm.bs << ", " << pm.ic << ", " << pm.ih
        << ", " << pm.iw << ")@NCHW ==> Kernel(" << pm.kh << ", " << pm.kw
        << "), ==> Out(" << pm.bs << ", " << pm.oc << ", " << pm.oh << ", "
        << pm.ow << ")@NCHW ==> Kernel(1, 1) ==> Out(" << pm.bs << ", "
        << pm.oc1x1 << ", " << pm.oh << "," << pm.ow << ")@NCHW";
    info("%s", oss.str().c_str());

    auto m = bench_mkldnn(pm);
    auto j = bench_jitinfer(pm);
    info("Jitinfer promote: %.2f %%", (m - j) / j * 100);
  }
  return 0;
}
