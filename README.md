# rosidl_buffer_backends_tutorials

End-to-end tutorials for the [rosidl_buffer_backends](https://github.com/ros2/rosidl_buffer_backends).

## robot_arm_demo

A ROS 2 demo that renders an SDF-based pencil-sketch robot arm animation entirely on the GPU via LibTorch tensor ops, publishes BGRA frames as `tensor_msgs/msg/ExperimentalTensor`, and displays them in an SDL2/OpenGL window with CUDA-GL interop.

The demo uses the **`torch_conversions`** to transport frames as a first-class, DLPack-aligned tensor message. Under the hood, `cuda_buffer_backend` still carries the bytes zero-copy via CUDA IPC; the bridge provides the tensor metadata layer (shape, strides, dtype, device) as a standard ROS 2 message instead of piggybacking on `sensor_msgs/Image`.

Animation is frame-count driven (fixed dt = 1/60 s per frame), so low FPS results in slower but smooth playback rather than frame skipping.

The demo exercises the full bridge pub/sub pipeline:

1. **renderer_node** -- renders BGRA frames on the GPU using LibTorch SDF operations, allocates a `tensor_msgs::msg::ExperimentalTensor` via `torch_conversions::allocate_tensor_msg`, copies the rendered frame into it with `to_tensor_msg`, and publishes.
2. **display_node** -- subscribes, wraps the received `Tensor` as an `at::Tensor` via `torch_conversions::from_input_tensor_msg` (routed through `at::fromDLPack`), renders into an SDL2/OpenGL window (with CUDA-GL interop), and reports FPS.

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

# Build the bridge + its CUDA transport dependency, source, then the demo.
colcon build --symlink-install --packages-up-to torch_conversions && \
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

### Benchmark methodology

Measured inter-process, headless mode (`headless:=true`), FastRTPS DDS:

```bash
ros2 launch robot_arm_demo robot_arm_demo.launch.py headless:=true width:=W height:=H
```

### Reference numbers

These numbers were collected with the original `torch_buffer_backend`
variant of this tutorial (pre-bridge) and are retained here as a baseline
for comparison. Re-run on your hardware with the current bridge-based
tutorial and fill in the third column.

| Resolution | Image Size | Backend (`torch_buffer_backend`) | Bridge (`torch_conversions`) | CPU fallback |
|---|---|---:|---:|---:|
| 1920x1080 | 7.9 MB | 116.6 FPS | TBD | 35.5 FPS |
| 2560x1440 | 14.1 MB | 90.6 FPS | TBD | 21.3 FPS |
| 3840x2160 | 31.6 MB | 59.5 FPS | TBD | 10.3 FPS |

Expected behavior: the bridge-based path should measure **within a few percent** of the original `torch_buffer_backend` column. Both paths end up going through the same `cuda_buffer_backend` IPC transport for the frame bytes; the only per-frame overhead the bridge adds over the backend approach is two small heap allocations (a `DLManagedTensor` + its context struct, ~120 bytes total) per publish and per subscribe, plus the metadata passes through `at::toDLPack` / `at::fromDLPack` rather than bespoke descriptor serialization. At MB-scale frame payloads, neither difference is measurable.

If you see a large divergence, likely suspects are:
- **CPU fallback**: something stopped `cuda_buffer_backend` from offering IPC (different user, different GPU, different host, or VMM unavailable). Check `ros2 topic info -v /image` and the backend discovery logs.
- **Extra copy**: verify the renderer's `to_tensor_msg` path is the only copy; no intermediate CPU tensor should sit in the pipeline.

## License

Apache-2.0
