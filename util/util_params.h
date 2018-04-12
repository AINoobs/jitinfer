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
 * This file defines the params for benchmark and unit test
 */
#pragma once
#include <stdint.h>

namespace jitinfer {
namespace util {

struct conv_params {
  conv_params(int bs,
              int gp,
              int ic,
              int ih,
              int iw,
              int oc,
              int oh,
              int ow,
              int kh,
              int kw,
              int ph,
              int pw,
              int sh,
              int sw,
              int oc1x1,
              int dh = 0,
              int dw = 0)
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
        ph(ph),
        pw(pw),
        sh(sh),
        sw(sw),
        oc1x1(oc1x1),
        dh(dh),
        dw(dw) {}
  int bs;
  int gp;
  int ic, ih, iw;
  int oc, oh, ow;
  int kh, kw;
  int ph, pw;
  int sh, sw;
  int oc1x1;
  int dh, dw;  // dilation, do not use yet
};
}
}
