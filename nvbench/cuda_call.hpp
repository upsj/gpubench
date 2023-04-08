/*
 *  Copyright 2021 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 with the LLVM exception
 *  (the "License"); you may not use this file except in compliance with
 *  the License.
 *
 *  You may obtain a copy of the License at
 *
 *      http://llvm.org/foundation/relicensing/LICENSE.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <nvbench/types.hpp>

#include <string>

/// Throws a std::runtime_error if `call` doesn't return `gpuSuccess`.
#define NVBENCH_CUDA_CALL(call)                                                                    \
  do                                                                                               \
  {                                                                                                \
    const gpuError_t nvbench_cuda_call_error = call;                                               \
    if (nvbench_cuda_call_error != gpuSuccess)                                                     \
    {                                                                                              \
      nvbench::cuda_call::throw_error(__FILE__, __LINE__, #call, nvbench_cuda_call_error);         \
    }                                                                                              \
  } while (false)

/// Throws a std::runtime_error if `call` doesn't return `CUDA_SUCCESS`.
#define NVBENCH_DRIVER_API_CALL(call)                                                              \
  do                                                                                               \
  {                                                                                                \
    const CUresult nvbench_cuda_call_error = call;                                                 \
    if (nvbench_cuda_call_error != CUDA_SUCCESS)                                                   \
    {                                                                                              \
      nvbench::cuda_call::throw_error(__FILE__, __LINE__, #call, nvbench_cuda_call_error);         \
    }                                                                                              \
  } while (false)

/// Terminates process with failure status if `call` doesn't return
/// `gpuSuccess`.
#define NVBENCH_CUDA_CALL_NOEXCEPT(call)                                                           \
  do                                                                                               \
  {                                                                                                \
    const gpuError_t nvbench_cuda_call_error = call;                                               \
    if (nvbench_cuda_call_error != gpuSuccess)                                                     \
    {                                                                                              \
      nvbench::cuda_call::exit_error(__FILE__, __LINE__, #call, nvbench_cuda_call_error);          \
    }                                                                                              \
  } while (false)

namespace nvbench::cuda_call
{

void throw_error(const std::string &filename,
                 std::size_t lineno,
                 const std::string &call,
                 gpuError_t error);

#ifndef USE_HIP
void throw_error(const std::string &filename,
                 std::size_t lineno,
                 const std::string &call,
                 CUresult error);
#endif

void exit_error(const std::string &filename,
                std::size_t lineno,
                const std::string &command,
                gpuError_t error);

} // namespace nvbench::cuda_call
