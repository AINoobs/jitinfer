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
  void check_result(const test_conv_params& pm,
                    const std::unique_ptr<memory>& src,
                    const std::unique_ptr<memory>& dst,
                    bool post_relu) {
    mkldnn::engine eng = mkldnn::engine(mkldnn::engine::cpu, 0);
  }

protected:
  virtual void SetUp() {
    using format = memory::format;
    test_conv_params p = ::testing::TestWithParam<test_conv_params>::GetParam();
    std::unique_ptr<memory> src, wei, bia, wei1x1, bias1x1, dst;
    EXPECT_EQ(p.gp, 1);
    constexpr format fmt = format::nhwc;
    auto src_dt = util::type2dtype<src_t>::dtype;
    auto wei_dt = util::type2dtype<wei_t>::dtype;
    auto bia_dt = util::type2dtype<bia_t>::dtype;
    auto dst_dt = util::type2dtype<dst_t>::dtype;

    src.reset(new memory({p.bs, p.ic, p.ih, p.iw}, fmt, src_dt));
    wei.reset(
        new memory({p.oc, p.ic, p.kh, p.kw}, format::OIhw4i16o4i, wei_dt));
    util::fill_data<src_t>(static_cast<src_t*>(src->data()), src->size());
    util::fill_data<wei_t>(static_cast<wei_t*>(wei->data()), wei->size());

    for (auto fuse_conv : {false, true}) {
      if (fuse_conv) {
        dst.reset(new memory({p.bs, p.oc1x1, p.oh, p.ow}, fmt, dst_dt));
      } else {
        dst.reset(new memory({p.bs, p.oc, p.oh, p.ow}, fmt, dst_dt));
      }
      for (memory::dtype bia1x1_dt : {memory::dtype::undef,
                                      memory::dtype::s8,
                                      memory::dtype::s32,
                                      memory::dtype::f32}) {
        for (auto with_relu : {false, true}) {
          for (auto with_bias : {false, true}) {
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
