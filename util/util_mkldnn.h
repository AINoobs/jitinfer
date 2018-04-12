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
 * This file defines some utilities of MKL-DNN, or exchange with jitinfer
 */
#pragma once

#include "jitinfer.h"
#include "mkldnn.hpp"
#include "util_params.h"

namespace jitinfer {
namespace util {

std::unique_ptr<mkldnn::eltwise_forward::primitive_desc> get_mkldnn_relu_pd(
    const mkldnn::memory::desc md, const mkldnn::engine& eng);

std::unique_ptr<mkldnn::convolution_forward::desc> get_conv_desc(
    const conv_params& p,
    bool with_bias,
    mkldnn::memory::data_type src_dt,
    mkldnn::memory::data_type wei_dt,
    mkldnn::memory::data_type bia_dt,
    mkldnn::memory::data_type dst_dt);

std::unique_ptr<mkldnn::convolution_forward::primitive_desc> get_conv_pd(
    const std::unique_ptr<mkldnn::convolution_forward::desc>& conv_desc,
    const mkldnn::engine& eng,
    std::vector<float> scales,
    mkldnn::round_mode rmode,
    bool with_relu = false);

// exchange btw mkldnn and jitinfer
namespace exchange {
mkldnn::round_mode round_mode(jitinfer::round_mode rmode);
mkldnn::memory::dims dims(const memory::nchw_dims& nchwdims);
memory::dtype dtype(mkldnn::memory::data_type dt);
mkldnn::memory::data_type dtype(memory::dtype dt);
}
}
}
