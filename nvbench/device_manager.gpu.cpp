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

#include <nvbench/cuda_call.hpp>
#include <nvbench/detail/device_scope.hpp>
#include <nvbench/device_manager.hpp>
#include <nvbench/types.hpp>

namespace nvbench
{

device_manager &device_manager::get()
{
  static device_manager the_manager;
  return the_manager;
}

device_manager::device_manager()
{
  int num_devs{};
  NVBENCH_CUDA_CALL(gpuGetDeviceCount(&num_devs));
  m_devices.reserve(static_cast<std::size_t>(num_devs));

  for (int i = 0; i < num_devs; ++i)
  {
    m_devices.emplace_back(i);
  }
}

} // namespace nvbench
