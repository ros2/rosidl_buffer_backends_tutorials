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

#ifndef FRAME_KERNEL_HPP_
#define FRAME_KERNEL_HPP_

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>

namespace executorch_policy_demo
{

enum class FrameFormat
{
  kFloatChw,  // [1, C, H, W] float in [0, 1], as a policy consumes it
  kByteHwc,  // [H, W, C] uint8, as a camera produces it
};

/// Frames repeat after this many indices.
constexpr uint32_t kFramePeriod = 256;

/// The robot state sent with frame `index`. Element k is the cosine (even k) or
/// sine (odd k) of 2 pi (k / 2 + 1) index / kFramePeriod. With two or more
/// elements, each frame of a period has its own state, and an action computed for
/// another frame differs.
inline float robot_state(uint32_t index, size_t element)
{
  constexpr double kPi = 3.14159265358979323846;
  const double angle = 2.0 * kPi * static_cast<double>(element / 2 + 1) * index / kFramePeriod;
  return static_cast<float>(element % 2 == 0 ? std::cos(angle) : std::sin(angle));
}

/// Writes a frame that changes with `index` into device memory, on `stream`.
/// `count` is the number of elements, and `plane` the elements per channel of a
/// `kFloatChw` frame.
cudaError_t render_frame(
  void * frame, FrameFormat format, size_t count, size_t plane, uint32_t index,
  cudaStream_t stream);

}  // namespace executorch_policy_demo

#endif  // FRAME_KERNEL_HPP_
