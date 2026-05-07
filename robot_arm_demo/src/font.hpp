// Copyright 2026 Open Source Robotics Foundation, Inc.
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

#ifndef FONT_HPP_
#define FONT_HPP_

#include <torch/torch.h>
#include <string>

// Render a text string into an anti-aliased [H, W] float bitmap (0..1).
// scale: screen pixels per glyph pixel.
torch::Tensor make_text_bitmap(const std::string & text, int scale, torch::Device dev);

#endif  // FONT_HPP_
