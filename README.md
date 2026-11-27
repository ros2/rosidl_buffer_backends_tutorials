# rosidl_buffer_backends_tutorials

End-to-end tutorials for the [rosidl_buffer_backends](https://github.com/ros2/rosidl_buffer_backends).

## robot_arm_demo

A ROS 2 demo that renders an SDF-based pencil-sketch robot arm animation with
LibTorch tensor ops, publishes BGRA frames as `tensor_msgs/msg/ExperimentalTensor`,
and displays them in an SDL2/OpenGL window.

The demo uses **`torch_conversions`** to publish frames as a first-class tensor
message. Under the hood, `cuda_buffer_backend` carries the frame bytes
zero-copy via CUDA IPC when the CUDA path is enabled.

Animation is frame-count driven (fixed dt = 1/60 s per frame), so low FPS results in slower but smooth playback rather than frame skipping.

The demo exercises the full bridge pub/sub pipeline:

1. **renderer_node** -- renders BGRA frames with LibTorch SDF operations, allocates a `tensor_msgs::msg::ExperimentalTensor` via `torch_conversions::allocate_tensor_msg`, copies the rendered frame into it with `to_tensor_msg`, and publishes.
2. **display_node** -- subscribes, wraps the received `ExperimentalTensor` as an `at::Tensor` via `torch_conversions::from_input_tensor_msg`, renders into an SDL2/OpenGL window, and reports FPS.

### Dependencies

- CUDA Toolkit (>= 11.8)
- LibTorch (provided automatically by `libtorch_vendor` at build time)
- SDL2, GLEW, OpenGL, X11

### Build

Requires a ROS 2 Rolling source workspace; see
[Building ROS 2 on Ubuntu](https://docs.ros.org/en/rolling/Installation/Alternatives/Ubuntu-Development-Setup.html)
for the canonical setup. After cloning both
`rosidl_buffer_backends` and `rosidl_buffer_backends_tutorials` into your
workspace's `src/` directory:

```bash
# Install system dependencies (CUDA toolkit, SDL2, GLEW, OpenGL, X11).
rosdep install --from-paths src --ignore-src -y \
  --skip-keys "fastcdr rti-connext-dds-7.7.0 urdfdom_headers qt6-svg-dev"

colcon build --symlink-install --packages-up-to cuda_buffer_backend && \
  source install/setup.sh && \
  colcon build --symlink-install --packages-up-to robot_arm_demo && \
  source install/setup.sh
```

The intermediate `source install/setup.sh` is required so that
`torch_conversions` can discover `cuda_buffer` at CMake configure time and
compile the CUDA fast path.

### Run

```bash
ros2 launch robot_arm_demo robot_arm_demo.launch.py
```

Arguments:
- `width` (default `1920`), `height` (default `1080`) --- render resolution.
- `headless` (default `false`) --- run without a display window.

### Compare CUDA and CPU paths

```bash
ros2 launch robot_arm_demo robot_arm_compare.launch.py
```

This starts two renderer/display pairs side by side: the CUDA path renders and
transports tensors through `torch_conversions` with CUDA acceleration, while
the CPU path uses the same `torch_conversions` APIs with CPU tensors. Each
display window labels the mode and FPS. By default, compare mode uses the
same render resolution as the main launch (`width:=1920`, `height:=1080`)
and scales each display window to `960x540`; pass `width:=W height:=H` to
compare at another render resolution.

### Benchmark methodology

Measured inter-process, headless mode (`headless:=true`), FastRTPS DDS:

```bash
ros2 launch robot_arm_demo robot_arm_demo.launch.py headless:=true width:=W height:=H
```

### Reference numbers

| Resolution | Image Size | `torch_conversions` + CUDA IPC | CPU fallback |
|---|---:|---:|---:|
| 1920x1080 | 7.9 MB | 116.6 FPS | 35.5 FPS |
| 2560x1440 | 14.1 MB | 90.6 FPS | 21.3 FPS |
| 3840x2160 | 31.6 MB | 59.5 FPS | 10.3 FPS |

The `torch_conversions` path keeps the frame bytes on the registered
`rosidl::Buffer` backend, so the CUDA path goes through the same
`cuda_buffer_backend` IPC transport used by lower-level buffer examples.

## executorch_policy_demo

A GPU camera feeding an [ExecuTorch](https://github.com/pytorch/executorch)
policy, in 2 processes. The camera renders a frame on the GPU and publishes it
with a robot state. The policy node runs an exported PyTorch program (`.pte`) on
the received tensors and publishes the action back. The camera then sends the
next frame, as a control loop does, and reports the round trip time.

The demo uses **`executorch_conversions`**, so neither node loads LibTorch. With
`transport:=cuda`, the program reads the frame from the publisher's GPU memory in
place and writes the action into a message the camera reads from GPU memory. With
`transport:=cpu`, the same loop uses host messages. Given the outputs the export
computed for each frame, the camera checks every action against them.

See [executorch_policy_demo/README.md](executorch_policy_demo/README.md) for
model export and launch commands.

### Reference numbers

Median round trip, measured inter-process with Fast DDS on an NVIDIA A100, with a
`[H, W, 4]` uint8 frame and a program that reduces it to 4 values. Each round
trip sends one frame to the policy and gets one action back. The ranges cover 4
runs on a GPU shared with other jobs.

| Resolution | Image Size | `executorch_conversions` + CUDA IPC | CPU messages |
|---|---:|---:|---:|
| 1920x1080 | 7.9 MB | 0.48 to 0.60 ms | 41 to 51 ms |
| 3840x2160 | 31.6 MB | 0.51 to 0.61 ms | 126 to 166 ms |

## License

Apache-2.0
