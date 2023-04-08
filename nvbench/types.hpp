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

#include <nvbench/config.hpp>

#include <cstdint>

#ifdef USE_HIP

#include <hip/hip_runtime.h>

#define gpuStreamSynchronize hipStreamSynchronize
#define gpuError_t hipError_t
#define gpuStream_t hipStream_t
#define gpuEvent_t hipEvent_t
#define gpuGetErrorName hipGetErrorName
#define gpuGetErrorString hipGetErrorString
#define gpuGetDevice hipGetDevice
#define gpuGetDeviceCount hipGetDeviceCount
#define gpuGetDeviceProperties hipGetDeviceProperties
#define gpuDeviceReset hipDeviceReset
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuSetDevice hipSetDevice
#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuEventRecord hipEventRecord
#define gpuEventCreate hipEventCreate
#define gpuEventDestroy hipEventDestroy
#define gpuEventQuery hipEventQuery
#define gpuEventSynchronize hipEventSynchronize
#define gpuEventElapsedTime hipEventElapsedTime
#define gpuSuccess hipSuccess
#define gpuErrorNotReady hipErrorNotReady
#define gpuStreamCreate hipStreamCreate
#define gpuStreamDestroy hipStreamDestroy
#define gpuDeviceProp hipDeviceProp_t
#define gpuMemsetAsync hipMemsetAsync
#define gpuFuncAttributes hipFuncAttributes
#define gpuFuncGetAttributes hipFuncGetAttributes
#define gpuMemGetInfo hipMemGetInfo
#define gpuDeviceGetAttribute hipDeviceGetAttribute
#define gpuDevAttrL2CacheSize hipDevAttrL2CacheSize
#else
#include <cuda.h>
#include <cuda_runtime_api.h>
#define gpuStreamSynchronize cudaStreamSynchronize
#define gpuError_t cudaError_t
#define gpuStream_t cudaStream_t
#define gpuEvent_t cudaEvent_t
#define gpuGetErrorName cudaGetErrorName
#define gpuGetErrorString cudaGetErrorString
#define gpuGetDevice cudaGetDevice
#define gpuGetDeviceCount cudaGetDeviceCount
#define gpuGetDeviceProperties cudaGetDeviceProperties
#define gpuDeviceReset cudaDeviceReset
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuSetDevice cudaSetDevice
#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuEventRecord cudaEventRecord
#define gpuEventCreate cudaEventCreate
#define gpuEventDestroy cudaEventDestroy
#define gpuEventQuery cudaEventQuery
#define gpuEventSynchronize cudaEventSynchronize
#define gpuEventElapsedTime cudaEventElapsedTime
#define gpuSuccess cudaSuccess
#define gpuErrorNotReady cudaErrorNotReady
#define gpuStreamCreate cudaStreamCreate
#define gpuStreamDestroy cudaStreamDestroy
#define gpuDeviceProp cudaDeviceProp
#define gpuMemsetAsync cudaMemsetAsync
#define gpuFuncAttributes cudaFuncAttributes
#define gpuFuncGetAttributes cudaFuncGetAttributes
#define gpuMemGetInfo cudaMemGetInfo
#define gpuDeviceGetAttribute cudaDeviceGetAttribute
#define gpuDevAttrL2CacheSize cudaDevAttrL2CacheSize
#endif

namespace nvbench
{

using int8_t    = std::int8_t;
using int16_t   = std::int16_t;
using int32_t   = std::int32_t;
using int64_t   = std::int64_t;
using uint8_t   = std::uint8_t;
using uint16_t  = std::uint16_t;
using uint32_t  = std::uint32_t;
using uint64_t  = std::uint64_t;
using float32_t = float;
using float64_t = double;

} // namespace nvbench
