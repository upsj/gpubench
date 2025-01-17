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

#include <nvbench/config.hpp>
#include <nvbench/internal/nvml.hpp>

#include <stdexcept>

#include <fmt/format.h>
#include <nvml.h>

namespace
{

// RAII struct that initializes and shuts down NVML
struct NVMLLifetimeManager
{
  NVMLLifetimeManager()
  {
    try
    {
      NVBENCH_NVML_CALL_NO_API(nvmlInit());
      m_inited = true;
    }
    catch (std::exception &e)
    {
      fmt::print("NVML initialization failed:\n {}", e.what());
    }
  }

  ~NVMLLifetimeManager()
  {
    if (m_inited)
    {
      try
      {
        NVBENCH_NVML_CALL_NO_API(nvmlShutdown());
      }
      catch (std::exception &e)
      {
        fmt::print("NVML shutdown failed:\n {}", e.what());
      }
    }
  }

private:
  bool m_inited{false};
};

// NVML's lifetime should extend for the entirety of the process, so store in a
// global.
auto nvml_lifetime = NVMLLifetimeManager{};

} // namespace
