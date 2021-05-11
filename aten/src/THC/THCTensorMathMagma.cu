#include <ATen/native/cuda/MiscUtils.h>
#include <THC/THCGeneral.h>
#include <THC/THCStorage.hpp>
#include <THC/THCTensor.hpp>
#include <THC/THCTensorCopy.h>
#include <THC/THCTensorMath.h>
#include <THC/THCTensorMathMagma.cuh>
#include <algorithm>

#ifdef USE_MAGMA
#include <magma_v2.h>
#endif

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y)-1) / (y))
#endif

#define NoMagma(name)                 \
  "No CUDA implementation of '" #name \
  "'. Install MAGMA and rebuild cutorch (http://icl.cs.utk.edu/magma/)"

void THCMagma_init(THCState* state) {
#ifdef USE_MAGMA
  magma_init();
#endif
}

// clang-format off
#include <THC/generic/THCTensorMathMagma.cu>
#include <THC/THCGenerateAllTypes.h>
// clang-format on
