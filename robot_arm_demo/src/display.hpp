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

#ifndef DISPLAY_HPP_
#define DISPLAY_HPP_

#include <torch/torch.h>
#include <string>

struct SDL_Window;
typedef void * SDL_GLContext;

enum class DisplayMode { CudaGL, IntelGL, SoftwareSDL, Headless };

class FrameDisplay
{
public:
  FrameDisplay();
  ~FrameDisplay();

  bool init(
    int width, int height, bool headless, bool use_cuda, bool fullscreen = false,
    int max_win_w = 0, int max_win_h = 0,
    int win_x = -1, int win_y = -1, bool borderless = false);
  void present(const torch::Tensor & frame);
  bool poll_events();

  static void save_ppm(const torch::Tensor & frame_bgra, const std::string & path);

  DisplayMode mode() const;
  SDL_Window * window() const;
  int win_width() const;
  int win_height() const;

private:
  void init_gl_resources();
  void present_gl(const torch::Tensor & frame);
  void present_sdl_software(const torch::Tensor & frame);
  void draw_fullscreen_quad();

  DisplayMode mode_ = DisplayMode::Headless;
  int W_ = 0, H_ = 0;
  int winW_ = 0, winH_ = 0;

  SDL_Window * window_ = nullptr;
  SDL_GLContext gl_ctx_ = nullptr;

  unsigned int tex_ = 0;
  unsigned int vao_ = 0;
  unsigned int shader_program_ = 0;

#ifdef USE_CUDA_GL_INTEROP
  unsigned int pbo_ = 0;
  void * cuda_pbo_ = nullptr;
#endif

  void * sdl_renderer_ = nullptr;
  void * sdl_texture_ = nullptr;
};

#endif  // DISPLAY_HPP_
