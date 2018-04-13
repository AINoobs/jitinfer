/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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

// This file is changed from Intel MKL-DNN
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

#include <type_traits>

#define XBYAK64
#define XBYAK_NO_OP_NAMES
/* in order to make selinux happy memory that would be marked with X-bit should
 * be obtained with mmap */
#define XBYAK_USE_MMAP_ALLOCATOR
#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
/* turn off `size_t to other-type implicit casting` warning
 * currently we have a lot of jit-generated instructions that
 * take uint32_t, but we pass size_t (e.g. due to using sizeof).
 * FIXME: replace size_t parameters with the appropriate ones */
#pragma warning(disable : 4267)
#endif
#include "util_jitinfer.h"
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

#define DECLARE_JIT_KERNEL(jit_name)                      \
  const char *name() const override { return #jit_name; } \
  const char *source_file() const override { return __FILE__; }

namespace jitinfer {
namespace jit {
static Xbyak::util::Cpu cpu;
typedef enum {
  isa_any,
  sse42,
  avx2,
  avx512_common,
  avx512_core,
  avx512_core_vnni,
  avx512_mic,
  avx512_mic_4ops,
} cpu_isa_t;  // Instruction set architecture

template <cpu_isa_t>
struct cpu_isa_traits {}; /* ::vlen -> 32 (for avx2) */

template <>
struct cpu_isa_traits<sse42> {
  static constexpr int vlen_shift = 4;
  static constexpr int vlen = 16;
  static constexpr int n_vregs = 16;
};
template <>
struct cpu_isa_traits<avx2> {
  static constexpr int vlen_shift = 5;
  static constexpr int vlen = 32;
  static constexpr int n_vregs = 16;
};
template <>
struct cpu_isa_traits<avx512_common> {
  static constexpr int vlen_shift = 6;
  static constexpr int vlen = 64;
  static constexpr int n_vregs = 32;
};
template <>
struct cpu_isa_traits<avx512_core> : public cpu_isa_traits<avx512_common> {};

template <>
struct cpu_isa_traits<avx512_mic> : public cpu_isa_traits<avx512_common> {};

template <>
struct cpu_isa_traits<avx512_mic_4ops> : public cpu_isa_traits<avx512_common> {
};

static inline bool mayiuse(const cpu_isa_t cpu_isa) {
  using namespace Xbyak::util;

  switch (cpu_isa) {
    case sse42:
      return cpu.has(Cpu::tSSE42);
    case avx2:
      return cpu.has(Cpu::tAVX2);
    case avx512_common:
      return cpu.has(Cpu::tAVX512F);
    case avx512_core:
      return true && cpu.has(Cpu::tAVX512F) && cpu.has(Cpu::tAVX512BW) &&
             cpu.has(Cpu::tAVX512VL) && cpu.has(Cpu::tAVX512DQ);
    case avx512_core_vnni:
      return true && cpu.has(Cpu::tAVX512F) && cpu.has(Cpu::tAVX512BW) &&
             cpu.has(Cpu::tAVX512VL) && cpu.has(Cpu::tAVX512DQ) &&
             cpu.has(Cpu::tAVX512_VNNI);
    case avx512_mic:
      return true && cpu.has(Cpu::tAVX512F) && cpu.has(Cpu::tAVX512CD) &&
             cpu.has(Cpu::tAVX512ER) && cpu.has(Cpu::tAVX512PF);
    case avx512_mic_4ops:
      return true && mayiuse(avx512_mic) && cpu.has(Cpu::tAVX512_4FMAPS) &&
             cpu.has(Cpu::tAVX512_4VNNIW);
    case isa_any:
      return true;
  }
  return false;
}

inline unsigned int get_cache_size(int level, bool per_core = true) {
  unsigned int l = level - 1;
  // Currently, if XByak is not able to fetch the cache topology
  // we default to 32KB of L1, 512KB of L2 and 1MB of L3 per core.
  if (cpu.data_cache_levels == 0) {
    const int L1_cache_per_core = 32000;
    const int L2_cache_per_core = 512000;
    const int L3_cache_per_core = 1024000;
    int num_cores = per_core ? 1 : omp_get_max_threads();
    switch (l) {
      case (0):
        return L1_cache_per_core * num_cores;
      case (1):
        return L2_cache_per_core * num_cores;
      case (2):
        return L3_cache_per_core * num_cores;
      default:
        return 0;
    }
  }
  if (l < cpu.data_cache_levels) {
    return cpu.data_cache_size[l] /
           (per_core ? cpu.cores_sharing_data_cache[l] : 1);
  } else
    return 0;
}

#ifdef XBYAK64
constexpr Xbyak::Operand::Code abi_save_gpr_regs[] = {
    Xbyak::Operand::RBX,
    Xbyak::Operand::RBP,
    Xbyak::Operand::R12,
    Xbyak::Operand::R13,
    Xbyak::Operand::R14,
    Xbyak::Operand::R15,
#ifdef _WIN
    Xbyak::Operand::RDI,
    Xbyak::Operand::RSI,
#endif
};

#ifdef _WIN
static const Xbyak::Reg64 abi_param1(Xbyak::Operand::RCX),
    abi_param2(Xbyak::Operand::RDX), abi_param3(Xbyak::Operand::R8),
    abi_param4(Xbyak::Operand::R9), abi_not_param1(Xbyak::Operand::RDI);
#else
static const Xbyak::Reg64 abi_param1(Xbyak::Operand::RDI),
    abi_param2(Xbyak::Operand::RSI), abi_param3(Xbyak::Operand::RDX),
    abi_param4(Xbyak::Operand::RCX), abi_not_param1(Xbyak::Operand::RCX);
#endif
#endif

class jit_generator : public Xbyak::CodeGenerator {
private:
  const size_t xmm_len = 16;
#ifdef _WIN
  const size_t xmm_to_preserve_start = 6;
  const size_t xmm_to_preserve = 10;
#else
  const size_t xmm_to_preserve_start = 0;
  const size_t xmm_to_preserve = 0;
#endif

  const size_t num_abi_save_gpr_regs =
      sizeof(abi_save_gpr_regs) / sizeof(abi_save_gpr_regs[0]);

  const size_t size_of_abi_save_regs =
      num_abi_save_gpr_regs * rax.getBit() / 8 + xmm_to_preserve * xmm_len;

public:
  enum {
    _cmp_eq_oq = 0u,
    _cmp_lt_os = 1u,
    _cmp_le_os = 2u,
    _cmp_neq_uq = 4u,
    _cmp_nlt_us = 5u,
    _cmp_nle_us = 6u,
  };

  Xbyak::Reg64 param1 = abi_param1;
  const int EVEX_max_8b_offt = 0x200;
  const Xbyak::Reg64 reg_EVEX_max_8b_offt = rbp;

  inline size_t get_size_of_abi_save_regs() { return size_of_abi_save_regs; }

  void preamble() {
    if (xmm_to_preserve) {
      sub(rsp, xmm_to_preserve * xmm_len);
      for (size_t i = 0; i < xmm_to_preserve; ++i)
        movdqu(ptr[rsp + i * xmm_len], Xbyak::Xmm(xmm_to_preserve_start + i));
    }
    for (size_t i = 0; i < num_abi_save_gpr_regs; ++i)
      push(Xbyak::Reg64(abi_save_gpr_regs[i]));
    if (mayiuse(avx512_common)) {
      mov(reg_EVEX_max_8b_offt, 2 * EVEX_max_8b_offt);
    }
  }

  void postamble() {
    for (size_t i = 0; i < num_abi_save_gpr_regs; ++i)
      pop(Xbyak::Reg64(abi_save_gpr_regs[num_abi_save_gpr_regs - 1 - i]));
    if (xmm_to_preserve) {
      for (size_t i = 0; i < xmm_to_preserve; ++i)
        movdqu(Xbyak::Xmm(xmm_to_preserve_start + i), ptr[rsp + i * xmm_len]);
      add(rsp, xmm_to_preserve * xmm_len);
    }
    ret();
  }

  template <typename T>
  Xbyak::Address EVEX_compress_addr(Xbyak::Reg64 base,
                                    T raw_offt,
                                    bool bcast = false) {
    using Xbyak::Zmm;
    using Xbyak::Reg64;
    using Xbyak::Address;
    using Xbyak::RegExp;

    auto offt = static_cast<int>(raw_offt);

    int scale = 0;

    if (EVEX_max_8b_offt <= offt && offt < 3 * EVEX_max_8b_offt) {
      offt = offt - 2 * EVEX_max_8b_offt;
      scale = 1;
    } else if (3 * EVEX_max_8b_offt <= offt && offt < 5 * EVEX_max_8b_offt) {
      offt = offt - 4 * EVEX_max_8b_offt;
      scale = 2;
    }

    auto re = RegExp() + base + offt;
    if (scale) re = re + reg_EVEX_max_8b_offt * scale;

    if (bcast)
      return zword_b[re];
    else
      return zword[re];
  }

  void L(const char *label) { Xbyak::CodeGenerator::L(label); }
  void L(const Xbyak::Label &label) { Xbyak::CodeGenerator::L(label); }

  void dump_code(const Xbyak::uint8 *code) const {
    if (code) {
      static int counter = 0;
#define MAX_FNAME_LEN 256
      char fname[MAX_FNAME_LEN + 1];
      snprintf(
          fname, MAX_FNAME_LEN, "jitinfer_dump_%s.%d.bin", name(), counter);
      counter++;

      FILE *fp = fopen(fname, "w+");
      // Failure to dump code is not fatal
      if (fp) {
        fwrite(code, getSize(), 1, fp);
        fclose(fp);
      }
    }
#undef MAX_FNAME_LEN
  }

public:
  jit_generator(void *code_ptr = nullptr, size_t code_size = 256 * 1024)
      : Xbyak::CodeGenerator(code_size, code_ptr) {}

  virtual const char *name() const = 0;
  virtual const char *source_file() const = 0;

  // XXX: use normal_case name and update all callees (?)
  const Xbyak::uint8 *getCode() {
    const Xbyak::uint8 *code = CodeGenerator::getCode();

#ifdef WITH_DUMP_CODE
    // only can dump code when cmake option is enabled
    if (util::env::jit_dump_code()) dump_code(code);
#endif

    return code;
  }

  template <typename F>
  const F getCode() {
    // XXX (Roma): Xbyak code probably has a bug here
    return (const F)getCode();
  }
};
}
}
