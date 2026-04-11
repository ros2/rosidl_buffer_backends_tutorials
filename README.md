# rosidl_buffer_backends_tutorials

End-to-end tutorials for the [rosidl_buffer_backends](https://github.com/ros2/rosidl_buffer_backends).

## robot_arm_demo

A ROS 2 demo that renders an SDF-based pencil-sketch robot arm animation entirely on the GPU via LibTorch tensor ops, publishes BGRA frames as `sensor_msgs/msg/Image`, and displays them in an SDL2/OpenGL window with CUDA-GL interop.

Two transport modes are compared:

- **cuda** -- uses the `torch_buffer_backend` with CUDA IPC for zero-copy GPU-to-GPU frame transport between processes.
- **cpu** -- publishes raw `sensor_msgs/msg/Image` data (GPU render, then `cudaMemcpy` to host, serialised via standard ROS 2 middleware). No buffer backend is involved.

Both modes render on the GPU. The only difference is the transport path, making this a clean comparison of zero-copy CUDA IPC vs traditional CPU-serialised image transport.

Animation is frame-count driven (fixed dt = 1/60 s per frame), so low FPS results in slower but smooth playback rather than frame skipping.

The demo exercises the full buffer-aware pub/sub pipeline:

1. **renderer_node** -- renders BGRA frames on the GPU using LibTorch SDF operations and publishes them as `sensor_msgs/msg/Image` via either CUDA IPC or raw CPU transport.
2. **display_node** -- subscribes, displays frames in an SDL2/OpenGL window (with CUDA-GL interop for the CUDA path, or CPU texture upload for the raw path), and reports FPS.

### Dependencies

- [rcl_buffer_ws](https://github.com/yuanknv/ros2) workspace with `pixi`
- LibTorch (see [torch_buffer_backend prerequisites](https://github.com/yuanknv/torch_buffer_backend#prerequisites))
- CUDA toolkit, SDL2, GLEW, OpenGL

### Build

Build order matters -- device backends must be installed before
`torch_buffer` so CUDA support is compiled in:

```bash
# 1. Build CUDA device backend
pixi run build cuda_buffer_backend

# 2. Build the demo
pixi run build robot_arm_demo
```

The demo discovers all installed buffer backend plugins automatically
via `source install/setup.sh`. If `cuda_buffer_backend` is installed,
the CUDA IPC path is used; otherwise it falls back to CPU.

### Run

```bash
# CUDA zero-copy (default)
ros2 launch robot_arm_demo robot_arm_demo.launch.py

# CPU transport
ros2 launch robot_arm_demo robot_arm_demo.launch.py use_cuda:=false

# Side-by-side CUDA vs CPU
ros2 launch robot_arm_demo robot_arm_compare.launch.py
```

All launch files accept `headless:=true` to run without a display window.

### Benchmark results

Measured on a single machine (inter-process, headless mode, RTX 3090, rmw_fastrtps_cpp).
To reproduce, run headless and note the stabilized FPS:

```bash
ros2 launch robot_arm_demo robot_arm_demo.launch.py headless:=true width:=W height:=H use_cuda:=true|false
```

| Resolution | Image Size | Transport | FPS | Speedup |
|---|---|---|---:|---:|
| 1920x1080 | 7.9 MB | CUDA | 116.6 | 3.3x |
| 1920x1080 | 7.9 MB | CPU | 35.5 | -- |
| 2560x1440 | 14.1 MB | CUDA | 90.6 | 4.3x |
| 2560x1440 | 14.1 MB | CPU | 21.3 | -- |
| 3840x2160 | 31.6 MB | CUDA | 59.5 | 5.8x |
| 3840x2160 | 31.6 MB | CPU | 10.3 | -- |

The CUDA path maintains high throughput across resolutions because zero-copy IPC transfers only a handle, not the pixel data. The CPU path must copy frames from GPU to host and serialise them through the middleware, so throughput drops as image size grows. At 4K (31.6 MB/frame) the CUDA backend is ~6x faster than the raw CPU path.

## License

Apache-2.0
