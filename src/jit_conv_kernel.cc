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
#include "jit_conv_kernel.h"
#include "util_jitinfer.h"

#define GET_OFF(field) offsetof(jit_conv_call_s, field)

namespace jitinfer {
namespace jit {

using namespace Xbyak;

void jit_conv_kernel::prepare_1x1output(int ur_w) {
  Label l_first_load, l_ret;
  mov(reg_ocb3x3, ptr[param1 + GET_OFF(ocb3x3)]);
  cmp(reg_ocb3x3, 0);  // FISRT load
  je(l_first_load, T_NEAR);

  for (int j = 0; j < ur_w; j++) {
    Zmm zmm = zmm_1x1out(j);
    // acc1x1 format is (oc1x1/16, ow, 16o)
    int offset = jcp.typesize_acc * j * jcp.oc1x1_block;
    vmovups(zmm, EVEX_compress_addr(aux_reg_ptr_acc1x1, offset));
  }
  jmp(l_ret, T_NEAR);

  L(l_first_load);
  for (int j = 0; j < ur_w; j++) {
    Zmm zmm = zmm_1x1out(j);
    vpxord(zmm, zmm, zmm);
  }

  L(l_ret);
}

void jit_conv_kernel::store_1x1output(int ur_w, int ocb1x1) {
  using data_type = memory::dtype;
  Label l_update_acc, l_ret;
  mov(reg_ocb3x3, ptr[param1 + GET_OFF(ocb3x3)]);
  int adjusment =
      jcp.nb_oc - 1 - ((jcp.nb_oc_blocking <= 1) ? 0 : jcp.nb_oc_blocking);
  cmp(reg_ocb3x3, adjusment);  // LAST channel
  jl(l_update_acc, T_NEAR);

  // prepare bias
  mov(reg_ptr_bia1x1, ptr[param1 + GET_OFF(bia1x1)]);
  mov(reg_ptr_scales1x1, ptr[param1 + GET_OFF(scales1x1)]);
  int scale_offset =
      jcp.conv1_multi_oc_scale ? sizeof(float) * ocb1x1 * jcp.oc1x1_block : 0;

  auto zmm_bias = zmm_tmp;
  if (jcp.conv1_with_bias) {
    int bias_offset = jcp.typesize_conv1_bia * ocb1x1 * jcp.oc1x1_block;
    auto bias_addr = EVEX_compress_addr(reg_ptr_bia1x1, bias_offset);
    switch (jcp.conv1_bias_dt) {
      case data_type::f32:
      case data_type::s32:
        vmovups(zmm_bias, bias_addr);
        break;
      case data_type::s8:
        vpmovsxbd(zmm_bias, bias_addr);
        break;
      case data_type::u8:
        vpmovzxbd(zmm_bias, bias_addr);
        break;
      default:
        assert(!"unsupported dst data type");
    }
    if (jcp.conv1_bias_dt != data_type::f32) {
      vcvtdq2ps(zmm_bias, zmm_bias);
    }
  }

  vpxord(zmm_zero, zmm_zero, zmm_zero);
  for (int jw = 0; jw < ur_w; jw++) {
    Zmm zmm = zmm_1x1out(jw);
    Xmm xmm = xmm_1x1out(jw);
    // out format is nhw,c/16,16o
    int offset = jcp.typesize_out * (jw * jcp.oc1x1 + ocb1x1 * jcp.oc1x1_block);
    auto addr = EVEX_compress_addr(reg_ptr_out1x1, offset);
    // cvt to f32
    vcvtdq2ps(zmm, zmm);
    if (jcp.conv1_with_bias) {
      vaddps(zmm, zmm, zmm_bias);
    }
    vmulps(zmm, zmm, EVEX_compress_addr(reg_ptr_scales1x1, scale_offset));
    // relu
    if (jcp.conv1_with_relu) {
      vmaxps(zmm, zmm_zero, zmm);
    }
    if (jcp.dst_dt != data_type::f32) {
      if (jcp.conv1_round_mode == round_mode::nearest)
        vcvtps2dq(zmm | T_rn_sae, zmm);  // cvt back
      else if (jcp.conv1_round_mode == round_mode::down)
        vcvtps2dq(zmm | T_rd_sae, zmm);
      else
        assert(!"unimplemented");
    }
    // 1x1 dst
    switch (jcp.dst_dt) {
      case data_type::f32:
      case data_type::s32:
        vmovups(addr, zmm);
        break;
      case data_type::s8:
        vpmovsdb(xmm, zmm);
        vmovups(addr, xmm);
        break;
      case data_type::u8:
        vpmovusdb(xmm, zmm);
        vmovups(addr, xmm);
        break;
      default:
        assert(!"unknown dst_dt");
    }
  }
  jmp(l_ret, T_NEAR);

  L(l_update_acc);
  for (int j = 0; j < ur_w; j++) {
    Zmm zmm = zmm_1x1out(j);
    // acc1x1 format is (oc1x1/16, ow, 16o)
    int offset = jcp.typesize_acc * j * jcp.oc1x1_block;
    vmovups(EVEX_compress_addr(aux_reg_ptr_acc1x1, offset), zmm);
  }
  L(l_ret);
}

void jit_conv_kernel::compute1x1_loop(int ur_w) {
  // this lamda function should be the same with compute_loop
  auto compute = [=](Zmm vreg_acc, Zmm vreg_wei, Zmm vreg_src) {
    if (jcp.use_vnni) {
      vpdpbusd(vreg_acc, vreg_src, vreg_wei);
    } else {
      vpmaddubsw(zmm_tmp, vreg_src, vreg_wei);
      vpmaddwd(zmm_tmp, zmm_tmp, zmm_one);
      vpaddd(vreg_acc, vreg_acc, zmm_tmp);
    }
  };
  // reg sum_scale, scales, bias, channel are avaible now
  mov(reg_ptr_wei1x1, ptr[param1 + GET_OFF(wei1x1)]);  // ic1x1 offsetted
  mov(aux_reg_ptr_acc1x1, reg_ptr_acc1x1);             // oh, ow offsetted.
  // acc1x1 format is (oc1x1/16, ow, 16o)
  int acc1x1_nboc_shift = jcp.typesize_acc * jcp.ow * jcp.oc1x1_block;
  int wei1x1_shift = jcp.typesize_in * 4 * jcp.oc1x1_block;  // == 64*s8
  // compute all oc3x3 for all ur_w
  for (int oc1x1_idx = 0; oc1x1_idx < jcp.nb_oc1x1; ++oc1x1_idx) {
    prepare_1x1output(ur_w);
    // 1x1 weight format is OIhw4i16o4i
    // [oc1x1/16,ic1x1/16, 4i,16o,4i]
    const int wei_oc_offset =
        jcp.typesize_in * (oc1x1_idx * jcp.oc * jcp.oc1x1_block);
    mov(aux_reg_ptr_wei1x1, reg_ptr_wei1x1);
    add(aux_reg_ptr_wei1x1, wei_oc_offset);
    // compute 16o of 1x1conv for all ur_w
    for (int k = 0; k < jcp.nb_oc_blocking; ++k) {
      for (int i4 = 0; i4 < 4; ++i4) {  // jcp.oc_block / 4
        // load 1x1 wei, load 16o4i *s8 one, the format is OIhw4i16o4i
        // [oc1x1/16,ic1x1/16, 4i,16o,4i]
        vmovups(zmm_1x1_wei, EVEX_compress_addr(aux_reg_ptr_wei1x1, 0));
        add(aux_reg_ptr_wei1x1, wei1x1_shift);
        for (int jw = 0; jw < ur_w; ++jw) {
          if (i4 == 0) {
            vmovd(reg_1x1_src_4u8, xmm_out(jw, k));  // get lower 4*u8
          } else {
            vpextrd(
                reg_1x1_src_4u8, xmm_out(jw, k), i4);  // get 4u8 from index i4
          }
          vpbroadcastd(zmm_1x1_src_bcast_u8, reg_1x1_src_4u8);
          compute(zmm_1x1out(jw), zmm_1x1_wei, zmm_1x1_src_bcast_u8);
        }
      }
    }
    store_1x1output(ur_w, oc1x1_idx);  // update acc, or last then relu to dst
    add(aux_reg_ptr_acc1x1, acc1x1_nboc_shift);
  }
}

void jit_conv_kernel::prepare_output(int ur_w) {
  Label l_first_load, l_ret;
  mov(reg_channel, ptr[param1 + GET_OFF(channel)]);
  cmp(reg_channel, 0);  // FISRT load
  je(l_first_load, T_NEAR);

  for (int k = 0; k < jcp.nb_oc_blocking; k++) {
    for (int j = 0; j < ur_w; j++) {
      Zmm zmm = zmm_out(j, k);
      int offset = jcp.typesize_acc * (k * ur_w + j) * jcp.oc_block;
      vmovups(zmm, EVEX_compress_addr(reg_acc_s32, offset));
    }
  }
  jmp(l_ret, T_NEAR);

  L(l_first_load);
  for (int k = 0; k < jcp.nb_oc_blocking; k++) {
    for (int j = 0; j < ur_w; j++) {
      Zmm zmm = zmm_out(j, k);
      vpxord(zmm, zmm, zmm);
    }
  }
  L(l_ret);
}

void jit_conv_kernel::store_output(int ur_w) {
  using data_type = memory::dtype;
  Label l_update_acc, l_ret;

  mov(reg_channel, ptr[param1 + GET_OFF(channel)]);
  int adjusment =
      jcp.nb_ic - ((jcp.nb_ic_blocking <= 1) ? 0 : jcp.nb_ic_blocking) - 1;
  cmp(reg_channel, adjusment);  // LAST channel
  jl(l_update_acc, T_NEAR);

  mov(reg_bias, ptr[param1 + GET_OFF(bia)]);
  mov(reg_ptr_scales, ptr[param1 + GET_OFF(scales)]);
  vpxord(zmm_zero, zmm_zero, zmm_zero);
  for (int k = 0; k < jcp.nb_oc_blocking; k++) {
    int scale_offset =
        jcp.conv0_multi_oc_scale ? sizeof(float) * k * jcp.oc_block : 0;
    auto zmm_bias = zmm_tmp;
    if (jcp.conv0_with_bias) {
      int bias_offset = jcp.typesize_conv0_bia * k * jcp.oc_block;
      auto bias_addr = EVEX_compress_addr(reg_bias, bias_offset);
      switch (jcp.conv0_bias_dt) {
        case data_type::f32:
        case data_type::s32:
          vmovups(zmm_bias, bias_addr);
          break;
        case data_type::s8:
          vpmovsxbd(zmm_bias, bias_addr);
          break;
        case data_type::u8:
          vpmovzxbd(zmm_bias, bias_addr);
          break;
        default:
          assert(!"unsupported dst data type");
      }
      if (jcp.conv0_bias_dt != data_type::f32) {
        vcvtdq2ps(zmm_bias, zmm_bias);
      }
    }
    for (int j = 0; j < ur_w; j++) {
      Xmm xmm = xmm_out(j, k);
      Zmm zmm = zmm_out(j, k);
      vcvtdq2ps(zmm, zmm);
      if (jcp.conv0_with_bias) {
        vaddps(zmm, zmm, zmm_bias);
      }
      vmulps(zmm, zmm, EVEX_compress_addr(reg_ptr_scales, scale_offset));
      if (jcp.conv0_with_relu) {
        vmaxps(zmm, zmm_zero, zmm);
      }
      if (jcp.dst_dt != data_type::f32) {
        if (jcp.conv0_round_mode == round_mode::nearest)
          vcvtps2dq(zmm | T_rn_sae, zmm);
        else if (jcp.conv0_round_mode == round_mode::down)
          vcvtps2dq(zmm | T_rd_sae, zmm);
        else
          assert(!"unimplemented");
      }
      if (jcp.fuse_conv1x1) {
        // always convert to u8, as src of 1x1 conv
        vpmovusdb(xmm, zmm);
      } else {
        int aux_output_offset =
            jcp.typesize_out * (k * jcp.oc_block + j * jcp.oc * jcp.gp);
        auto addr = EVEX_compress_addr(reg_out, aux_output_offset);
        switch (jcp.dst_dt) {
          case data_type::f32:
          case data_type::s32:
            vmovups(addr, zmm);
            break;
          case data_type::s8:
            vpmovsdb(xmm, zmm);
            vmovups(addr, xmm);
            break;
          case data_type::u8:
            vpmovusdb(xmm, zmm);
            vmovups(addr, xmm);
            break;
          default:
            assert(!"unknown dst_dt");
        }
      }
    }
  }

  if (jcp.fuse_conv1x1) {
    compute1x1_loop(ur_w);
  }
  jmp(l_ret, T_NEAR);

  L(l_update_acc);
  for (int k = 0; k < jcp.nb_oc_blocking; k++)
    for (int j = 0; j < ur_w; j++) {
      Zmm zmm = zmm_out(j, k);
      int offset = jcp.typesize_acc * (k * ur_w + j) * jcp.oc_block;
      vmovups(EVEX_compress_addr(reg_acc_s32, offset), zmm);
    }
  L(l_ret);
}

void jit_conv_kernel::compute_loop(int ur_w, int pad_l, int pad_r) {
  int kw = jcp.kw;
  int stride_w = jcp.sw;
  int ic_block = jcp.ic_block;
  int oc_block = jcp.oc_block;
  int nb_oc_block = jcp.nb_oc_blocking;
  int nb_ic_block = jcp.nb_ic_blocking;

  Label kh_label, skip_kh_loop;
  int shift_kernel_ptr = jcp.typesize_in * jcp.kw * jcp.oc_block * jcp.ic_block;
  int shift_input_ptr = jcp.typesize_in * jcp.iw * jcp.ic * jcp.gp;

  auto input_offset = [=](int oi, int nb_ic, int ic, int ki) {
    return jcp.typesize_in * ((ki + oi * stride_w - pad_l) * jcp.ic * jcp.gp +
                              4 * ic + nb_ic * jcp.ic_block);
  };
  auto kernel_offset = [=](int ii, int nb_ic, int ic, int ki) {
    return jcp.typesize_in *
           (ii * jcp.nb_ic * jcp.kh * jcp.kw * ic_block * oc_block +
            ki * ic_block * oc_block + 4 * ic * oc_block +
            jcp.kh * jcp.kw * nb_ic * jcp.ic_block * oc_block);
  };
  auto compute = [=](Zmm vreg_acc, Zmm vreg_wei, Zmm vreg_src) {
    if (jcp.use_vnni) {
      vpdpbusd(vreg_acc, vreg_src, vreg_wei);
    } else {
      vpmaddubsw(zmm_tmp, vreg_src, vreg_wei);
      vpmaddwd(zmm_tmp, zmm_tmp, zmm_one);
      vpaddd(vreg_acc, vreg_acc, zmm_tmp);
    }
  };

  prepare_output(ur_w);

  mov(aux_reg_inp, reg_inp);
  mov(aux_reg_ker, reg_ker);
  mov(reg_kj, reg_kh);
  if (jcp.kh <= jcp.t_pad) {
    cmp(reg_kj, 0);
    je(skip_kh_loop, T_NEAR);
  }
  L(kh_label);
  {
    for (int ki = 0; ki < kw; ki++) {
      int jj_start = get_ow_start(ki, pad_l);
      int jj_end = get_ow_end(ur_w, ki, pad_r);

      for (int cc = 0; cc < nb_ic_block; cc++) {
        for (int ic = 0; ic < ic_block / 4; ic++) {
          for (int jj = jj_start; jj < jj_end; jj++) {
            int aux_input_offset = input_offset(jj, cc, ic, ki);
            vpbroadcastd(zmm_inp(jj, nb_oc_block),
                         ptr[aux_reg_inp + aux_input_offset]);
          }

          for (int ii = 0; ii < nb_oc_block; ii++) {
            int aux_kernel_offset = kernel_offset(ii, cc, ic, ki);
            if (jj_end - jj_start > 0)
              vmovups(zmm_wei,
                      EVEX_compress_addr(aux_reg_ker, aux_kernel_offset));
            for (int jj = jj_start; jj < jj_end; jj++) {
              compute(zmm_out(jj, ii), zmm_wei, zmm_inp(jj, nb_oc_block));
            }
          }
        }
      }
    }
    add(aux_reg_ker, shift_kernel_ptr);
    add(aux_reg_inp, shift_input_ptr);
    dec(reg_kj);
    cmp(reg_kj, 0);
    jg(kh_label, T_NEAR);
  }
  L(skip_kh_loop);

  store_output(ur_w);
}

void jit_conv_kernel::generate() {
  int inp_shift_pad =
      jcp.typesize_in * (jcp.ur_w * jcp.sw - jcp.l_pad) * jcp.ic * jcp.gp;
  int inp_shift = jcp.typesize_in * (jcp.ur_w * jcp.sw * jcp.ic * jcp.gp);
  int acc_shift =
      jcp.typesize_acc * (jcp.ur_w * jcp.oc_block * jcp.nb_oc_blocking);
  int out_shift = 0, out1x1_shift = 0, acc1x1_shift = 0;
  if (jcp.fuse_conv1x1) {
    // here is for shifting ur_w
    out1x1_shift = jcp.typesize_out * (jcp.ur_w * jcp.oc1x1);
    // acc1x1 format is oc/16, ow, 16
    acc1x1_shift = jcp.typesize_acc * (jcp.ur_w * jcp.oc1x1_block);
  } else {
    out_shift = jcp.typesize_out * (jcp.ur_w * jcp.oc * jcp.gp);
  }

  preamble();

  if (jcp.fuse_conv1x1) {
    xor_(reg_scratch_1x1, reg_scratch_1x1);
    Reg16 _t = reg_scratch_1x1.cvt16();
    mov(_t, 0x1);
    vpbroadcastw(zmm_one, _t);
  } else {
    xor_(reg_scratch_3x3, reg_scratch_3x3);
    Reg16 _t = reg_scratch_3x3.cvt16();
    mov(_t, 0x1);
    vpbroadcastw(zmm_one, _t);
  }

  mov(reg_inp, ptr[param1 + GET_OFF(src)]);
  if (jcp.fuse_conv1x1) {
    mov(reg_ptr_out1x1, ptr[param1 + GET_OFF(dst)]);
    mov(reg_ptr_acc1x1, ptr[param1 + GET_OFF(acc1x1)]);
  } else {
    mov(reg_out, ptr[param1 + GET_OFF(dst)]);
  }
  mov(reg_ker, ptr[param1 + GET_OFF(wei)]);
  mov(reg_kh, ptr[param1 + GET_OFF(kh_padding)]);
  mov(reg_acc_s32, ptr[param1 + GET_OFF(acc_s32)]);

  int r_pad = std::max(
      0, (jcp.ow - 1) * jcp.sw + (jcp.kw - 1) - (jcp.iw + jcp.l_pad - 1));
  int n_oi = jcp.ow / jcp.ur_w;
  int r_pad1 =
      (jcp.ur_w * n_oi - 1) * jcp.sw + jcp.kw - 1 - (jcp.iw + jcp.l_pad - 1);
  if (r_pad1 > 0) n_oi--;

  xor_(reg_oi, reg_oi);
  if (jcp.ow == jcp.ur_w) {
    compute_loop(jcp.ur_w, jcp.l_pad, r_pad);
  } else {
    if (n_oi == 0) {
      compute_loop(jcp.ur_w, jcp.l_pad, r_pad1);
      add(reg_inp, inp_shift_pad);
      if (jcp.fuse_conv1x1) {
        add(reg_ptr_out1x1, out1x1_shift);
        add(reg_ptr_acc1x1, acc1x1_shift);
      } else {
        add(reg_out, out_shift);
      }
      add(reg_acc_s32, acc_shift);
      if (jcp.ur_w_tail != 0) {
        compute_loop(jcp.ur_w_tail, 0, r_pad);
      }
    } else {
      if (jcp.l_pad > 0) {
        compute_loop(jcp.ur_w, jcp.l_pad, 0);
        add(reg_inp, inp_shift_pad);
        if (jcp.fuse_conv1x1) {
          add(reg_ptr_out1x1, out1x1_shift);
          add(reg_ptr_acc1x1, acc1x1_shift);
        } else {
          add(reg_out, out_shift);
        }
        add(reg_acc_s32, acc_shift);
        inc(reg_oi);
      }
      if ((jcp.l_pad <= 0 && n_oi > 0) || (jcp.l_pad > 0 && n_oi > 1)) {
        if (jcp.l_pad <= 0 && r_pad1 > 0) n_oi--;
        Label ow_loop_label;
        L(ow_loop_label);
        {
          compute_loop(jcp.ur_w, 0, 0);
          add(reg_inp, inp_shift);
          if (jcp.fuse_conv1x1) {
            add(reg_ptr_out1x1, out1x1_shift);
            add(reg_ptr_acc1x1, acc1x1_shift);
          } else {
            add(reg_out, out_shift);
          }
          add(reg_acc_s32, acc_shift);
          inc(reg_oi);
          cmp(reg_oi, n_oi);
          jl(ow_loop_label, T_NEAR);
        }
      }
      if (r_pad1 > 0) {
        compute_loop(jcp.ur_w, 0, r_pad1);
        add(reg_inp, inp_shift);
        if (jcp.fuse_conv1x1) {
          add(reg_ptr_out1x1, out1x1_shift);
          add(reg_ptr_acc1x1, acc1x1_shift);
        } else {
          add(reg_out, out_shift);
        }
        add(reg_acc_s32, acc_shift);
      }
      if (jcp.ur_w_tail != 0) {
        compute_loop(jcp.ur_w_tail, 0, r_pad);
      }
    }
  }

  postamble();
}

bool jit_conv_kernel::init_conf(jit_conv_conf_t &jcp,
                                const std::unique_ptr<memory> &src,
                                const std::unique_ptr<memory> &wei,
                                const std::unique_ptr<memory> &bia,
                                int ngroups,
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
                                round_mode conv1_round_mode) {
  using namespace util;
  jcp = zero<decltype(jcp)>();
  // Check data type
  if (!all_true(src->data_type() == memory::dtype::u8,
                wei->data_type() == memory::dtype::s8,
                wei1x1 == nullptr || wei1x1->data_type() == memory::dtype::s8,
                one_of(dst->data_type(),
                       memory::dtype::f32,
                       memory::dtype::s32,
                       memory::dtype::s8,
                       memory::dtype::u8),
                bia == nullptr || one_of(bia->data_type(),
                                         memory::dtype::f32,
                                         memory::dtype::s32,
                                         memory::dtype::s8,
                                         memory::dtype::u8),
                bia1x1 == nullptr || one_of(bia1x1->data_type(),
                                            memory::dtype::f32,
                                            memory::dtype::s32,
                                            memory::dtype::s8,
                                            memory::dtype::u8))) {
    return false;
  }
  // Check format
  if (!all_true(one_of(src->dim_format(), memory::format::nhwc),
                one_of(dst->dim_format(), memory::format::nhwc),
                one_of(wei->dim_format(),
                       memory::format::OIhw4i16o4i,
                       memory::format::gOIhw4i16o4i),
                bia == nullptr || one_of(bia->dim_format(), memory::format::x),
                wei1x1 == nullptr || one_of(wei1x1->dim_format(),
                                            memory::format::OIhw4i16o4i,
                                            memory::format::gOIhw4i16o4i),
                bia1x1 == nullptr ||
                    one_of(bia1x1->dim_format(), memory::format::x))) {
    return false;
  }

  jcp.gp = ngroups;
  assert(ngroups == 1);
  auto src_dims = src->std_dims();  // nchw
  auto wei_dims = wei->std_dims();  // oihw
  auto dst_dims = dst->std_dims();  // nchw
  jcp.bs = src_dims[0];
  jcp.ic = wei_dims[1];
  jcp.ih = src_dims[2];
  jcp.iw = src_dims[3];
  jcp.oc = wei_dims[0];
  jcp.oh = dst_dims[2];
  jcp.ow = dst_dims[3];
  jcp.kh = wei_dims[2];
  jcp.kw = wei_dims[3];
  jcp.sh = sz_stride[0];
  jcp.sw = sz_stride[1];
  jcp.t_pad = sz_padding[0];
  jcp.l_pad = sz_padding[1];
  jcp.ic_block = 16;
  jcp.oc_block = 16;
  jcp.nb_ic = jcp.ic / jcp.ic_block;
  jcp.nb_oc = jcp.oc / jcp.oc_block;
  if (!all_true(jcp.ic % jcp.ic_block == 0, jcp.oc % jcp.oc_block == 0)) {
    return false;
  }
  jcp.use_vnni = mayiuse(avx512_core_vnni);
  // pick loop order
  jcp.loop_order = loop_cgn;
  if (jcp.gp > 1) {
    jcp.loop_order = loop_ngc;
  }

  // check wei dim ic
  if (wei_dims[1] != src_dims[1]) {
    return false;
  }

  jcp.fuse_conv1x1 = wei1x1 != nullptr;
  if (jcp.fuse_conv1x1) {
    if (jcp.oc_block % 4 != 0) {
      // for 4 bcast
      return false;
    }
    assert(wei1x1 != nullptr);
    auto wei1x1_dims = wei1x1->std_dims();  // oihw
    jcp.oc1x1 = wei1x1_dims[0];
    if (!all_true(jcp.oc == wei1x1_dims[1],
                  1 == wei1x1_dims[2],
                  1 == wei1x1_dims[3])) {
      return false;
    }
    jcp.oc1x1_block = 16;
    jcp.nb_oc1x1 = jcp.oc1x1 / jcp.oc1x1_block;
    if (jcp.oc1x1 % jcp.oc1x1_block != 0) {
      return false;
    }

  } else {
    // check wei dim oc
    if (wei_dims[0] != dst_dims[1]) {
      return false;
    }
  }

  auto undef_dt = memory::dtype::undef;
  jcp.conv0_with_bias = bia != nullptr;
  jcp.conv1_with_bias = bia1x1 != nullptr;
  jcp.conv0_bias_dt = jcp.conv0_with_bias ? bia->data_type() : undef_dt;
  jcp.conv1_bias_dt = jcp.conv1_with_bias ? bia1x1->data_type() : undef_dt;
  jcp.dst_dt = dst->data_type();
  jcp.typesize_in = dtype_size(src->data_type());
  jcp.typesize_out = dtype_size(dst->data_type());
  jcp.typesize_acc = sizeof(s32);
  jcp.typesize_conv0_bia =
      jcp.conv0_with_bias ? dtype_size(bia->data_type()) : 0;
  jcp.typesize_conv1_bia =
      jcp.conv1_with_bias ? dtype_size(bia1x1->data_type()) : 0;
  jcp.conv0_with_relu = conv0_relu;
  jcp.conv1_with_relu = conv1_relu;

  jcp.conv0_round_mode = conv0_round_mode;
  jcp.conv1_round_mode = conv1_round_mode;
  assert(one_of(jcp.conv0_round_mode, round_mode::nearest, round_mode::down));
  assert(one_of(jcp.conv1_round_mode, round_mode::nearest, round_mode::down));

  // conv 3x3 blocking settings
  jcp.nb_ic_blocking = dividable_of(jcp.nb_ic, 8, 4, 2, 1);
  if (jcp.kh >= 7 || jcp.kw >= 7) {  // Note: maybe have large code issue on SKX
    jcp.nb_ic_blocking = dividable_of(jcp.nb_ic, 4, 2, 1);
  }
  jcp.nb_oc_blocking = jcp.nb_oc > 4 ? 4 : jcp.nb_oc;
  if (jcp.nb_oc % jcp.nb_oc_blocking != 0) {
    jcp.nb_oc_blocking = find_dividable(jcp.nb_oc, jcp.nb_oc_blocking);
  }

  // the rest 1 size of ur_w is for src input zmm
  jcp.ur_w = ker_reg_base_idx / (jcp.nb_oc_blocking + 1);
  if (jcp.ow < jcp.ur_w) jcp.ur_w = jcp.ow;
  jcp.ur_w_tail = jcp.ow % jcp.ur_w;

  int r_pad_no_tail = std::max(
      0, (jcp.ow - jcp.ur_w_tail - 1) * jcp.sw + jcp.kw - jcp.iw - jcp.l_pad);
  if (jcp.l_pad > jcp.ur_w || r_pad_no_tail > jcp.ur_w) {
    return false;
  }

  jcp.conv0_multi_oc_scale = conv0_scales.size() > 1;
  jcp.conv1_multi_oc_scale = conv1_scales.size() > 1;
  if (!one_of(conv0_scales.size(), 1, jcp.oc)) {
    return false;
  }
  if (jcp.fuse_conv1x1 && !one_of(conv1_scales.size(), 1, jcp.oc1x1)) {
    return false;
  }

  return true;
}
}
}
