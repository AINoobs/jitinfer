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

struct test_concat_params {
  std::vector<memory::nchw_dims> srcs_dims;
  memory::nchw_dims dst_dims;
};

template <typename dtype>
class test_concat : public ::testing::TestWithParam<test_concat_params> {
  void check_result(const test_concat_params& pm,
                    const std::vector<std::unique_ptr<memory>>& srcs,
                    const std::unique_ptr<memory>& dst,
                    bool post_relu) {
    mkldnn::engine eng = mkldnn::engine(mkldnn::engine::cpu, 0);

    std::unique_ptr<mkldnn::primitive> fwd_concat, fwd_relu;
    std::unique_ptr<mkldnn::concat::primitive_desc> concat_pd;
    std::unique_ptr<mkldnn::eltwise_forward::primitive_desc> relu_pd;
    std::vector<mkldnn::primitive> pp_concat;

    // below is input
    int concat_dimension = 1;
    mkldnn::memory::format fmt = mkldnn::memory::format::nhwc;
    auto mkldnn_dt = util::exchange::dtype(dst->data_type());
    // allocate srcs memory
    std::vector<mkldnn::memory::primitive_desc> srcs_pd;
    std::vector<mkldnn::memory> mkldnn_srcs;
    for (size_t i = 0; i < srcs.size(); ++i) {
      auto mkldnn_dims = util::exchange::dims(pm.srcs_dims[i]);
      auto desc = mkldnn::memory::desc(mkldnn_dims, mkldnn_dt, fmt);
      auto mpd = mkldnn::memory::primitive_desc(desc, eng);
      auto src_memory = mkldnn::memory(mpd);
      assert(srcs[i]->size() ==
             src_memory.get_primitive_desc().get_size() / sizeof(dtype));
      util::copy_array<dtype>((dtype*)(src_memory.get_data_handle()),
                              (dtype*)(srcs[i]->data()),
                              srcs[i]->size());
      srcs_pd.push_back(mpd);
      mkldnn_srcs.push_back(src_memory);
    }
    // dst memory
    auto dst_desc =
        mkldnn::memory::desc(util::exchange::dims(pm.dst_dims), mkldnn_dt, fmt);
    concat_pd.reset(new mkldnn::concat::primitive_desc(
        dst_desc, concat_dimension, srcs_pd));
    auto mkldnn_dst = mkldnn::memory(concat_pd->dst_primitive_desc());

    // concat
    std::vector<mkldnn::primitive::at> inputs;
    for (size_t i = 0; i < mkldnn_srcs.size(); i++) {
      inputs.push_back(mkldnn_srcs[i]);
    }
    fwd_concat.reset(new mkldnn::concat(*concat_pd, inputs, mkldnn_dst));
    pp_concat.clear();
    pp_concat.push_back(*fwd_concat);

    if (post_relu) {
      // add relu
      relu_pd = jitinfer::util::get_mkldnn_relu_pd(dst_desc, eng);
      fwd_relu.reset(
          new mkldnn::eltwise_forward(*relu_pd, mkldnn_dst, mkldnn_dst));
      pp_concat.push_back(*fwd_relu);
    }

    mkldnn::stream(mkldnn::stream::kind::eager).submit(pp_concat).wait();
    dtype* ref_data = (dtype*)(mkldnn_dst.get_data_handle());
    dtype* jit_data = (dtype*)(dst->data());
    util::compare_array<dtype>(jit_data, ref_data, dst->size());
  }

protected:
  virtual void SetUp() {
    test_concat_params p =
        ::testing::TestWithParam<test_concat_params>::GetParam();
    auto dt = util::type2dtype<dtype>::dtype;
    std::vector<std::unique_ptr<memory>> srcs(p.srcs_dims.size());
    std::unique_ptr<memory> dst;
    memory::format fmt = memory::format::nhwc;
    for (size_t i = 0; i < p.srcs_dims.size(); ++i) {
      srcs[i].reset(new memory(p.srcs_dims[i], fmt, dt));
      util::fill_data<dtype>(static_cast<dtype*>(srcs[i]->data()),
                             srcs[i]->size());
    }
    dst.reset(new memory(p.dst_dims, fmt, dt));

    for (bool post_relu : {true, false}) {
      auto c = concat(srcs, dst, post_relu);
      c->submit();
      check_result(p, srcs, dst, post_relu);
    }
  }
};

using test_concat_f32 = test_concat<f32>;
using test_concat_s32 = test_concat<s32>;
using test_concat_s8 = test_concat<s8>;
using test_concat_u8 = test_concat<u8>;

TEST_P(test_concat_f32, TestsConcat) {}
TEST_P(test_concat_s32, TestsConcat) {}
TEST_P(test_concat_s8, TestsConcat) {}
TEST_P(test_concat_u8, TestsConcat) {}

// @note: the srcs and dst are always given as nchw
#define BASIC_TEST_CASES                                                  \
  test_concat_params{{{2, 64, 1, 1}, {2, 96, 1, 1}}, {2, 160, 1, 1}},     \
      test_concat_params{{{2, 64, 4, 4}, {2, 32, 4, 4}}, {2, 96, 4, 4}},  \
      test_concat_params{{{2, 16, 8, 8}, {2, 32, 8, 8}}, {2, 48, 8, 8}},  \
      test_concat_params{{{2, 32, 9, 9}, {2, 96, 9, 9}}, {2, 128, 9, 9}}, \
      test_concat_params{{{2, 16, 3, 3}, {2, 32, 3, 3}, {2, 64, 3, 3}},   \
                         {2, 112, 3, 3}},                                 \
      test_concat_params{{{2, 256, 16, 16}, {2, 256, 16, 16}},            \
                         {2, 512, 16, 16}},                               \
      test_concat_params {                                                \
    {{4, 128, 14, 14}, {4, 256, 14, 14}}, { 4, 384, 14, 14 }              \
  }

// f32 and s32 should support 4x, 8x or 16x of ic
INSTANTIATE_TEST_CASE_P(
    TestConcat,
    test_concat_f32,
    ::testing::Values(
        BASIC_TEST_CASES,
        test_concat_params{{{2, 4, 4, 4}, {2, 8, 4, 4}}, {2, 12, 4, 4}},
        test_concat_params{{{2, 16, 4, 4}, {2, 8, 4, 4}}, {2, 24, 4, 4}}));

INSTANTIATE_TEST_CASE_P(
    TestConcat,
    test_concat_s32,
    ::testing::Values(
        BASIC_TEST_CASES,
        test_concat_params{{{2, 4, 4, 4}, {2, 8, 4, 4}}, {2, 12, 4, 4}},
        test_concat_params{{{2, 16, 4, 4}, {2, 8, 4, 4}}, {2, 24, 4, 4}}));

INSTANTIATE_TEST_CASE_P(TestConcat,
                        test_concat_s8,
                        ::testing::Values(BASIC_TEST_CASES));

INSTANTIATE_TEST_CASE_P(TestConcat,
                        test_concat_u8,
                        ::testing::Values(BASIC_TEST_CASES));
}
