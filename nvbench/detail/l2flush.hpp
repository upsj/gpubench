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

#include <nvbench/cuda_call.hpp>
#include <nvbench/types.hpp>

namespace nvbench::detail
{

struct l2flush
{
  __forceinline__ l2flush()
  {
    int dev_id{};
    NVBENCH_CUDA_CALL(gpuGetDevice(&dev_id));
#ifdef USE_HIP
    gpuDeviceProp prop;
    NVBENCH_CUDA_CALL(gpuGetDeviceProperties(&prop, dev_id));
    m_l2_size = prop.l2CacheSize;
#else
    NVBENCH_CUDA_CALL(gpuDeviceGetAttribute(&m_l2_size, gpuDevAttrL2CacheSize, dev_id));
#endif
    if (m_l2_size > 0)
    {
      void *buffer = m_l2_buffer;
      NVBENCH_CUDA_CALL(gpuMalloc(&buffer, m_l2_size));
      m_l2_buffer = reinterpret_cast<int *>(buffer);
    }
  }

  __forceinline__ ~l2flush()
  {
    if (m_l2_buffer)
    {
      NVBENCH_CUDA_CALL_NOEXCEPT(gpuFree(m_l2_buffer));
    }
  }

  __forceinline__ void flush(gpuStream_t stream)
  {
    if (m_l2_size > 0)
    {
      NVBENCH_CUDA_CALL(gpuMemsetAsync(m_l2_buffer, 0, m_l2_size, stream));
    }
  }

private:
  int m_l2_size{};
  int *m_l2_buffer{};
};

} // namespace nvbench::detail
