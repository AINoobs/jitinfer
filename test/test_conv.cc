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
#include "util_jitinfer.h"
#include "util_mkldnn.h"
#include "util_test.h"

#define CONV0_PARAMS(bias)                                \
  src, wei, bias, sz_stride, sz_padding, dst, conv0_relu, \
      (conv0_multi_scales ? conv0_scales_c : conv0_scales_1), conv0_round_mode
#define CONV0(bias)                   \
  auto c0 = conv(CONV0_PARAMS(bias)); \
  c0->submit();                       \
  check_result(CONV0_PARAMS(bias))

#define CONV1_PARAMS(bias, bias1x1)                                           \
  src, wei, bias, sz_stride, sz_padding, wei1x1, bias1x1, dst1x1, conv0_relu, \
      (conv0_multi_scales ? conv0_scales_c : conv0_scales_1),                 \
      conv0_round_mode, conv1_relu,                                           \
      (conv1_multi_scales ? conv1_scales_c : conv1_scales_1), conv1_round_mode
#define CONV1(bias, bias1x1)                   \
  auto c1 = conv(CONV1_PARAMS(bias, bias1x1)); \
  c1->submit();                                \
  check_result(CONV1_PARAMS(bias, bias1x1))

namespace jitinfer {

struct test_conv_params {
  test_conv_params(int bs,
                   int gp,
                   int ic,
                   int ih,
                   int iw,
                   int oc,
                   int oh,
                   int ow,
                   int kh,
                   int kw,
                   int padh,
                   int padw,
                   int strh,
                   int strw,
                   int oc1x1)
      : bs(bs),
        gp(gp),
        ic(ic),
        ih(ih),
        iw(iw),
        oc(oc),
        oh(oh),
        ow(ow),
        kh(kh),
        kw(kw),
        ph(padh),
        pw(padw),
        sh(strh),
        sw(strw),
        oc1x1(oc1x1) {}
  int bs;
  int gp;
  int ic, ih, iw;
  int oc, oh, ow;
  int kh, kw;
  int ph, pw;
  int sh, sw;
  int oc1x1;
};

template <typename src_t, typename wei_t, typename bia_t, typename dst_t>
class test_conv : public ::testing::TestWithParam<test_conv_params> {
  void check_result(const std::unique_ptr<memory> &src,
                    const std::unique_ptr<memory> &wei,
                    const std::unique_ptr<memory> &bia,
                    std::array<int, 2> sz_stride,
                    std::array<int, 2> sz_padding,
                    std::unique_ptr<memory> &dst,
                    bool conv0_relu,
                    std::vector<float> conv0_scales,
                    round_mode conv0_round_mode) {
    check_result(src,
                 wei,
                 bia,
                 sz_stride,
                 sz_padding,
                 nullptr,
                 nullptr,
                 dst,
                 conv0_relu,
                 conv0_scales,
                 conv0_round_mode,
                 false,
                 {1.f},
                 nearest);
  }

  // conv and fuse conv1x1_relu
  void check_result(const std::unique_ptr<memory> &src,
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
    mkldnn::engine eng = mkldnn::engine(mkldnn::engine::cpu, 0);
  }

protected:
  virtual void SetUp() {
    using format = memory::format;
    test_conv_params p = ::testing::TestWithParam<test_conv_params>::GetParam();
    std::unique_ptr<memory> src, wei, bia, dst, wei1x1, bia1x1, dst1x1;
    EXPECT_EQ(p.gp, 1);
    constexpr format fmt = format::nhwc;
    auto src_dt = util::type2dtype<src_t>::dtype;
    auto wei_dt = util::type2dtype<wei_t>::dtype;
    auto bia_dt = util::type2dtype<bia_t>::dtype;
    auto dst_dt = util::type2dtype<dst_t>::dtype;
    std::array<int, 2> sz_stride = {p.sh, p.sw};
    std::array<int, 2> sz_padding = {p.ph, p.pw};

    src.reset(new memory({p.bs, p.ic, p.ih, p.iw}, fmt, src_dt));
    wei.reset(
        new memory({p.oc, p.ic, p.kh, p.kw}, format::OIhw4i16o4i, wei_dt));
    bia.reset(new memory({p.oc}, bia_dt));
    wei1x1.reset(
        new memory({p.oc1x1, p.oc, 1, 1}, format::OIhw4i16o4i, wei_dt));
    bia1x1.reset(new memory({p.oc1x1}, bia_dt));
    // here bia1x1 dtype always == bia dtype
    util::fill_data<src_t>(static_cast<src_t *>(src->data()), src->size());
    util::fill_data<wei_t>(static_cast<wei_t *>(wei->data()), wei->size());
    util::fill_data<bia_t>(static_cast<bia_t *>(bia->data()), bia->size());
    util::fill_data<wei_t>(static_cast<wei_t *>(wei1x1->data()),
                           wei1x1->size());
    util::fill_data<bia_t>(static_cast<bia_t *>(bia1x1->data()),
                           bia1x1->size());

    // dst and dst1x1 using two different buffers here is just for gtest
    dst.reset(new memory({p.bs, p.oc, p.oh, p.ow}, fmt, dst_dt));
    dst1x1.reset(new memory({p.bs, p.oc1x1, p.oh, p.ow}, fmt, dst_dt));

    // scales
    std::vector<float> conv0_scales_1(1);
    std::vector<float> conv0_scales_c(p.oc);
    std::vector<float> conv1_scales_1(1);
    std::vector<float> conv1_scales_c(p.oc1x1);
    const int a = 0.01f, b = 1.f;
    util::fill_data<float>(conv0_scales_1.data(), conv0_scales_1.size(), a, b);
    util::fill_data<float>(conv0_scales_c.data(), conv0_scales_c.size(), a, b);
    util::fill_data<float>(conv1_scales_1.data(), conv1_scales_1.size(), a, b);
    util::fill_data<float>(conv1_scales_c.data(), conv1_scales_c.size(), a, b);

    for (bool conv0_bias : {true, false}) {
      for (bool conv0_relu : {true, false}) {
        for (bool conv0_multi_scales : {true, false}) {
          for (round_mode conv0_round_mode : {nearest, down}) {
            // test non-fuse
            if (conv0_bias) {
              CONV0(bia);
            } else {
              CONV0(nullptr);
            }

            // test fuse conv1x1
            for (bool conv1_bias : {true, false}) {
              for (bool conv1_relu : {true, false}) {
                for (bool conv1_multi_scales : {true, false}) {
                  for (round_mode conv1_round_mode : {nearest, down}) {
                    if (conv0_bias) {
                      if (conv1_bias) {
                        CONV1(bia, bia1x1);
                      } else {
                        CONV1(bia, nullptr);
                      }
                    } else {
                      if (conv1_bias) {
                        CONV1(nullptr, bia1x1);
                      } else {
                        CONV1(nullptr, nullptr);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
};

// @note: the srcs, wei and dst are always given as nchw
// TODO: add more test cases
#define test_conv_case(src, wei, bia, dst)                              \
  using test_conv_##src##wei##bia##dst = test_conv<src, wei, bia, dst>; \
  TEST_P(test_conv_##src##wei##bia##dst, TestsConv) {}                  \
  INSTANTIATE_TEST_CASE_P(                                              \
      TestConv,                                                         \
      test_conv_##src##wei##bia##dst,                                   \
      ::testing::Values(                                                \
          test_conv_params{                                             \
              2, 1, 32, 13, 13, 32, 12, 12, 3, 3, 0, 0, 1, 1, 64},      \
          test_conv_params{                                             \
              2, 1, 32, 13, 13, 32, 11, 11, 3, 3, 1, 1, 1, 1, 32},      \
          test_conv_params{                                             \
              2, 1, 32, 120, 360, 64, 120, 360, 3, 3, 1, 1, 1, 1, 32}))

// data type src, weight, bias, dst
test_conv_case(u8, s8, s8, u8);
test_conv_case(u8, s8, s8, s8);
test_conv_case(u8, s8, s8, s32);
test_conv_case(u8, s8, s8, f32);
test_conv_case(u8, s8, s32, u8);
test_conv_case(u8, s8, s32, s8);
test_conv_case(u8, s8, s32, s32);
test_conv_case(u8, s8, s32, f32);
}
