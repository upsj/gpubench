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

#include <nvbench/benchmark.hpp>
#include <nvbench/benchmark_base.hpp>
#include <nvbench/benchmark_manager.hpp>
#include <nvbench/callable.hpp>
#include <nvbench/config.hpp>
#include <nvbench/cpu_timer.hpp>
#include <nvbench/create.hpp>
#include <nvbench/cuda_call.hpp>
#include <nvbench/cuda_stream.hpp>
#include <nvbench/cuda_timer.hpp>
#include <nvbench/enum_type_list.hpp>
#include <nvbench/exec_tag.hpp>
#include <nvbench/launch.hpp>
#include <nvbench/main.hpp>
#include <nvbench/range.hpp>
#include <nvbench/state.hpp>
#include <nvbench/type_list.hpp>
#include <nvbench/types.hpp>
