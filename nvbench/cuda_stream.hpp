/*
 *  Copyright 2021-2022 NVIDIA Corporation
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

#include <memory>

namespace nvbench
{

/**
 * Manages and provides access to a CUDA stream.
 *
 * May be owning or non-owning. If the stream is owned, it will be freed with
 * `gpuStreamDestroy` when the `cuda_stream`'s lifetime ends. Non-owning
 * `cuda_stream`s are sometimes referred to as views.
 *
 * @sa nvbench::make_cuda_stream_view
 */
struct cuda_stream
{
  /**
   * Constructs a cuda_stream that owns a new stream, created with
   * `gpuStreamCreate`.
   */
  cuda_stream()
      : m_stream{[]() {
                   gpuStream_t s;
                   NVBENCH_CUDA_CALL(gpuStreamCreate(&s));
                   return s;
                 }(),
                 stream_deleter{true}}
  {}

  /**
   * Constructs a `cuda_stream` from an explicit gpuStream_t.
   *
   * @param owning If true, `gpuStreamCreate(stream)` will be called from this
   * `cuda_stream`'s destructor.
   *
   * @sa nvbench::make_cuda_stream_view
   */
  cuda_stream(gpuStream_t stream, bool owning)
      : m_stream{stream, stream_deleter{owning}}
  {}

  ~cuda_stream() = default;

  // move-only
  cuda_stream(const cuda_stream &)            = delete;
  cuda_stream &operator=(const cuda_stream &) = delete;
  cuda_stream(cuda_stream &&)                 = default;
  cuda_stream &operator=(cuda_stream &&)      = default;

  /**
   * @return The `gpuStream_t` managed by this `cuda_stream`.
   * @{
   */
  operator gpuStream_t() const { return m_stream.get(); }

  gpuStream_t get_stream() const { return m_stream.get(); }
  /**@}*/

private:
  struct stream_deleter
  {
    using pointer = gpuStream_t;
    bool owning;

    constexpr void operator()(pointer s) const noexcept
    {
      if (owning)
      {
        NVBENCH_CUDA_CALL_NOEXCEPT(gpuStreamDestroy(s));
      }
    }
  };

  std::unique_ptr<gpuStream_t, stream_deleter> m_stream;
};

/**
 * Creates a non-owning view of the specified `stream`.
 */
inline nvbench::cuda_stream make_cuda_stream_view(gpuStream_t stream)
{
  return cuda_stream{stream, false};
}

} // namespace nvbench
