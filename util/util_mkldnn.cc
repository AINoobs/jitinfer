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
#include "util_mkldnn.h"
#include "log.h"

namespace jitinfer {
namespace util {
// TODO: get concat pd
std::unique_ptr<mkldnn::eltwise_forward::primitive_desc> get_mkldnn_relu_pd(
    const mkldnn::memory::desc md, const mkldnn::engine& eng) {
  using namespace mkldnn;
  auto relu_desc = eltwise_forward::desc(
      prop_kind::forward_inference, algorithm::eltwise_relu, md, 0.f, 0.f);
  return std::unique_ptr<eltwise_forward::primitive_desc>(
      new eltwise_forward::primitive_desc(relu_desc, eng));
}

std::unique_ptr<mkldnn::convolution_forward::desc> get_conv_desc(
    const conv_params& p,
    mkldnn::memory::data_type src_dt,
    mkldnn::memory::data_type wei_dt,
    mkldnn::memory::data_type bia_dt,
    mkldnn::memory::data_type dst_dt) {
  mkldnn::memory::format src_fmt = mkldnn::memory::format::any;
  mkldnn::memory::format wei_fmt = mkldnn::memory::format::any;
  mkldnn::memory::format bia_fmt = mkldnn::memory::format::x;
  mkldnn::memory::format dst_fmt = mkldnn::memory::format::any;
  auto aprop_kind = mkldnn::prop_kind::forward_inference;
  auto aalgorithm = mkldnn::algorithm::convolution_direct;
  auto pad_kind = mkldnn::padding_kind::zero;

  // memory desc
  auto c_src_desc =
      mkldnn::memory::desc({p.bs, p.ic, p.ih, p.iw}, src_dt, src_fmt);
  auto c_weights_desc =
      p.gp > 1
          ? mkldnn::memory::desc(
                {p.gp, p.oc / p.gp, p.ic / p.gp, p.kh, p.kw}, wei_dt, wei_fmt)
          : mkldnn::memory::desc({p.oc, p.ic, p.kh, p.kw}, wei_dt, wei_fmt);
  auto c_bias_desc = bia_dt != mkldnn::memory::data_type::data_undef
                         ? mkldnn::memory::desc({p.oc}, bia_dt, bia_fmt)
                         : mkldnn::memory::desc({}, bia_dt, bia_fmt);
  auto c_dst_desc =
      mkldnn::memory::desc({p.bs, p.oc, p.oh, p.ow}, dst_dt, dst_fmt);

  std::vector<int> padR = {p.ph, p.pw};
  for (int i = 0; i < 2; ++i) {
    if ((p.ih - ((p.kh - 1) * (p.dh + 1) + 1) + p.ph + padR[0]) / p.sh + 1 !=
        p.oh) {
      ++padR[0];
    }
    if ((p.iw - ((p.kw - 1) * (p.dw + 1) + 1) + p.pw + padR[1]) / p.sw + 1 !=
        p.ow) {
      ++padR[1];
    }
  }

  if (bia_dt != mkldnn::memory::data_type::data_undef) {
    return std::unique_ptr<mkldnn::convolution_forward::desc>(
        new mkldnn::convolution_forward::desc(aprop_kind,
                                              aalgorithm,
                                              c_src_desc,
                                              c_weights_desc,
                                              c_bias_desc,
                                              c_dst_desc,
                                              {p.sh, p.sw},
                                              {p.dh, p.dw},
                                              {p.ph, p.pw},
                                              padR,
                                              pad_kind));
  } else {
    return std::unique_ptr<mkldnn::convolution_forward::desc>(
        new mkldnn::convolution_forward::desc(aprop_kind,
                                              aalgorithm,
                                              c_src_desc,
                                              c_weights_desc,
                                              c_dst_desc,
                                              {p.sh, p.sw},
                                              {p.dh, p.dw},
                                              {p.ph, p.pw},
                                              padR,
                                              pad_kind));
  }
}

std::unique_ptr<mkldnn::convolution_forward::primitive_desc> get_conv_pd(
    const std::unique_ptr<mkldnn::convolution_forward::desc>& conv_desc,
    const mkldnn::engine& eng,
    std::vector<float> scales,
    mkldnn::round_mode rmode,
    bool with_relu) {
  // attribute
  mkldnn::primitive_attr attr = mkldnn::primitive_attr();
  attr.set_int_output_round_mode(rmode);
  const int count = scales.size();
  const int mask = count > 1 ? 1 << 1 : 0;
  check_ge(count, 1);
  attr.set_output_scales(mask, scales);

  if (with_relu) {
    mkldnn::post_ops ops;
    const float negative_slope = 0.f;
    ops.append_eltwise(
        1.0f, mkldnn::algorithm::eltwise_relu, negative_slope, 0.f);
    attr.set_post_ops(ops);
  }

  return std::unique_ptr<mkldnn::convolution_forward::primitive_desc>(
      new mkldnn::convolution_forward::primitive_desc(*conv_desc, attr, eng));
}

namespace exchange {
mkldnn::round_mode round_mode(jitinfer::round_mode rmode) {
  switch (rmode) {
#define CASE(tp)                 \
  case jitinfer::round_mode::tp: \
    return mkldnn::round_mode::round_##tp
    CASE(nearest);
    CASE(down);
#undef CASE
    default:
      error_and_exit("Unkown round mode %d", rmode);
  }
}

memory::dtype dtype(mkldnn::memory::data_type dt) {
  switch (dt) {
#define CASE(tp)                      \
  case mkldnn::memory::data_type::tp: \
    return memory::dtype::tp
    CASE(f32);
    CASE(s32);
    CASE(s8);
    CASE(u8);
#undef CASE
    case mkldnn::memory::data_type::data_undef:
      return jitinfer::memory::dtype::undef;
    default:
      error_and_exit("Unkown type %d", dt);
  }
}

mkldnn::memory::data_type dtype(memory::dtype dt) {
  switch (dt) {
#define CASE(tp)                    \
  case jitinfer::memory::dtype::tp: \
    return mkldnn::memory::data_type::tp
    CASE(f32);
    CASE(s32);
    CASE(s8);
    CASE(u8);
#undef CASE
    case jitinfer::memory::dtype::undef:
      return mkldnn::memory::data_type::data_undef;
    default:
      error_and_exit("Unkown type %d", dt);
  }
}

mkldnn::memory::dims dims(const memory::nchw_dims& nchwdims) {
  mkldnn::memory::dims out(nchwdims.size());
  for (size_t i = 0; i < out.size(); ++i) {
    out[i] = nchwdims[i];
  }
  return out;
}
}
}
}
