// Copyright 2026 Meta Platforms, Inc. and affiliates.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "frame_kernel.hpp"

namespace executorch_policy_demo
{
namespace
{

__global__ void float_frame(float * frame, size_t count, size_t plane, uint32_t index)
{
  const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < count) {
    const size_t pixel = i % plane;
    // pixel / 256 makes a row differ from the rows near it, so a frame with moved rows
    // gives another action.
    // A multiplication, as PyTorch divides by a scalar on the GPU.
    frame[i] = static_cast<float>((pixel + pixel / 256 + 7 * index + i / plane) % 256) *
      (1.0f / 255.0f);
  }
}

__global__ void byte_frame(uint8_t * frame, size_t count, uint32_t index)
{
  const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < count) {
    // The contrast changes with the frame, so the mean does too. A frame of 130 or more
    // bytes is never all zeros, which an output that was never written could match.
    frame[i] = static_cast<uint8_t>((i + i / 256 + 7 * index) % 256 * (index + 2) / 257);
  }
}

}  // namespace

cudaError_t render_frame(
  void * frame, FrameFormat format, size_t count, size_t plane, uint32_t index,
  cudaStream_t stream)
{
  constexpr unsigned int kThreads = 256;
  index %= kFramePeriod;
  const auto blocks = static_cast<unsigned int>((count + kThreads - 1) / kThreads);
  if (format == FrameFormat::kFloatChw) {
    float_frame<<<blocks, kThreads, 0, stream>>>(static_cast<float *>(frame), count, plane, index);
  } else {
    byte_frame<<<blocks, kThreads, 0, stream>>>(static_cast<uint8_t *>(frame), count, index);
  }
  return cudaGetLastError();
}

}  // namespace executorch_policy_demo
