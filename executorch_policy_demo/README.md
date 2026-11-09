# executorch_policy_demo

A GPU camera feeding an ExecuTorch policy, in 2 processes. It shows that a
frame can go from the camera, through a model, to an action, without the CPU
touching the frame, and compares that with the same pipeline on plain CPU
messages.

```
camera_node                                policy_node
  render frame on GPU ----- image ------->   view message storage (no copy)
  robot state         ----- state ------->   Module::execute (ExecuTorch)
  read action, check, <---- action -------   outputs written to reply storage
  time, next frame
```

- `camera_node` renders a frame with a CUDA kernel into a `tensor_msgs/ExperimentalTensor`,
  publishes it with a robot state that changes every frame, waits for the action,
  and times the round trip.
  It sends the next frame only after the action arrives, as a control loop does.
  Given the outputs the program should produce, it also checks every action.
- `policy_node` runs an ExecuTorch program on tensor messages. Each input topic
  feeds one program input, and each program output goes to one output topic.
  Messages whose storage is already where the program expects it are passed in
  place. A host message for a GPU program, or a GPU message for a CPU program,
  is copied once into a staging buffer, so the same node also runs the CPU
  baseline. A message with the wrong dtype or shape is logged and dropped on
  arrival, and the messages held for the other inputs stay. A message that
  cannot be viewed, for example one on another GPU, drops the whole set when it
  runs.

With `transport:=cuda`, messages use `cuda_buffer_backend`. With `transport:=cpu`,
the camera copies each frame to host memory and the policy copies it back to the
GPU, which is what happens without `cuda_buffer_backend`.

## Build

The demo needs `executorch_conversions` from `rosidl_buffer_backends`, CMake 3.28
and a CUDA 13 toolkit. With both repositories in the workspace's `src/`:

```bash
colcon build --packages-up-to executorch_policy_demo
source install/setup.bash
```

With a CUDA toolkit of another major version, for example 12.8,
`executorch_cuda_vendor` installs no ExecuTorch, and this package skips itself
with a CMake warning: the build succeeds, but no nodes or launch files are
installed. Without any CUDA toolkit, `cuda_buffer` does not build.

The export script needs `torch` and `executorch` in Python.

## Programs

The policy node passes message storage to the program, so the program must take
its inputs and outputs unplanned (left out of memory planning), in the
contiguous layout. The node refuses other programs when it loads them. It links
ExecuTorch's optimized CPU kernels and its CUDA backend, so a program that needs
another backend, such as XNNPACK, or other kernels fails to load. A program
must not change its inputs in place: they are the received messages, which other
subscribers may share. Each
output must have storage of its own: a program that returns an input, or the
same tensor twice, fails every run with "output N is not written to its reply".
`scripts/export_demo_models.py` exports 2 kinds
of program for the CUDA backend, which needs a CUDA 13 toolkit:

- `tiny_policy`: image `[1, 3, 480, 640]` and state `[1, 14]` to action `[1, 100, 14]`,
  the shapes of a LeRobot ACT policy.
- `frame_stats_HxW`: a `[H, W, 4]` uint8 frame to its per-channel mean, to time
  transport at 1080p and 4K (`--stats` takes other sizes). A mean does not depend
  on pixel order, so its check shows that each frame arrived, not that its layout
  is right.

Each program directory also gets `expected_outputs.bin`, the outputs for every
frame the camera renders. Export on the GPU that runs the program, since the
CUDA backend compiles kernels for it with `nvcc`, found through `PATH`,
`CUDA_HOME` or `CUDACXX`. The versions are pinned because the programs must
match the ExecuTorch 1.6.0 runtime that `executorch_cuda_vendor` installs, and
that release needs exactly torch 2.15.0, here the CUDA 13.2 build. From this
package's directory:

```bash
python3 -m venv ~/export-venv && source ~/export-venv/bin/activate
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132
pip install executorch==1.6.0
python scripts/export_demo_models.py ~/models
deactivate
```

Leave the venv before building, or the vendor packages reuse its ExecuTorch.

## Run

```bash
ros2 launch executorch_policy_demo policy_demo.launch.py \
  model_file_path:=$HOME/models/tiny_policy/model.pte \
  data_file_paths:="['$HOME/models/tiny_policy/aoti_cuda_blob.ptd']" \
  expected_file_path:=$HOME/models/tiny_policy/expected_outputs.bin \
  transport:=cuda
```

The camera prints one `SUMMARY` line, and the demo then stops. It has the round
trip percentiles, the loop rate, the buffer backend the action arrived on, and
`wrong_actions`, the number of actions that did not match `expected_outputs.bin`
or could not be read. The image and the state change every frame, and the check is
tight enough that an action computed from another frame's image or state counts as
wrong. The frames repeat every 256, so frames 256 apart give the same action.

`ros2 launch` exits with 0 either way, so the camera's own status is the result:

- 0: every measured frame got an action. With `expected_file_path`, every action
  also matched. Without it, the values are not checked, and `SUMMARY` says
  `checked=no`.
- 1: an action was wrong or could not be read, no policy connected within 60 s,
  no action came for 10 s, the policy failed, or the run was stopped.
- A signal, in 2 cases. On CUDA transport, `cuda_buffer` makes a process wait at
  exit until every reader has released its buffers, so while the policy holds one
  of the camera's messages, the camera cannot exit and the launch file ends it.
  And when the launch stops the camera while ROS is still creating its tensor
  publishers, for example because the policy failed to load, the camera can
  crash. The same crash happens in a program with no code from this demo.

Neither node sets a CUDA stream. So on each thread the camera's kernel, the
message views and the program all use the per-thread default stream, which
orders them as the Streams section of the `executorch_conversions` README
describes.

Every `report_every` runs, the policy prints a `POLICY` line:

- `callback_p50_ms`: time spent in the subscription callback that runs the
  program. On CUDA, the GPU may still be working when it returns; the camera's
  round trip includes that.
- `staged_copies`: copies this node made since the last `POLICY` line. Copies
  inside the program, for example by a delegate, are not counted.
- `gpu_total_used_mb`: memory in use on the whole GPU, by every process. 0 for a
  program that neither uses the CUDA backend nor has CUDA inputs or outputs.

A frame-size test at 1080p:

```bash
ros2 launch executorch_policy_demo policy_demo.launch.py \
  model_file_path:=$HOME/models/frame_stats_1080x1920/model.pte \
  data_file_paths:="['$HOME/models/frame_stats_1080x1920/aoti_cuda_blob.ptd']" \
  expected_file_path:=$HOME/models/frame_stats_1080x1920/expected_outputs.bin \
  format:=byte_hwc channels:=4 width:=1920 height:=1080 state_shape:='[0]' \
  transport:=cuda warmup:=500
```

Launch arguments:

- `model_file_path`, `data_file_paths`, `backend_libraries`: the program, its
  data files, and extra delegate libraries to load at run time. A delegate
  library registers with the shared ExecuTorch runtime it links, so it must be
  built for the release `executorch_cuda_vendor` installs, and a GPU delegate
  must queue its work on the calling thread's per-thread default stream, as the
  CUDA backend does here. Programs and libraries run as native code in the
  node's process, so load only files you trust.
- `transport`: `cuda` or `cpu`.
- `format` (`float_chw` or `byte_hwc`), `width`, `height`, `channels`: the frame.
- `state_shape`: the robot state, `[0]` for a program that takes only the image.
- `frames`, `warmup`: measured frames, and frames sent before them.
- `expected_file_path`: the outputs to check every action against.
- `report_every`: runs between `POLICY` lines.

## TensorRT

With Torch-TensorRT 2.15, `--backend tensorrt` exports `tiny_policy_tensorrt`:
TensorRT runs the network and the CUDA backend joins the image features and the
robot state, so the program uses both. `torch_tensorrt[executorch]` installs
PyTorch, TensorRT, the CUDA build of ExecuTorch and the TensorRT delegate
library, and the policy node loads that library at run time from
`backend_libraries`. The delegate in Torch-TensorRT 2.15 is built for ExecuTorch
1.6, the release `executorch_cuda_vendor` installs, and queues its work on the
per-thread default stream, as that parameter requires. The install is pinned to
2.15 for that reason. A later release brings a delegate for a later ExecuTorch,
so move this pin together with the vendor's ExecuTorch pin. The split puts both
backends in one program and is not tuned for speed. `--stats` with no sizes
skips the frame_stats programs. From this package's directory:

```bash
python3 -m venv ~/trt-venv && source ~/trt-venv/bin/activate
pip install "torch_tensorrt[executorch]==2.15.*"
python scripts/export_demo_models.py ~/models --backend tensorrt --stats
TRT_LIB=$(python -c "import importlib.util as u, os; print(os.path.join(u.find_spec( \
'torch_tensorrt_executorch_runtime').submodule_search_locations[0], 'lib', \
'libexecutorch_backend_tensorrt.so'))")
deactivate
ros2 launch executorch_policy_demo policy_demo.launch.py \
  model_file_path:=$HOME/models/tiny_policy_tensorrt/model.pte \
  data_file_paths:="['$HOME/models/tiny_policy_tensorrt/aoti_cuda_blob.ptd']" \
  backend_libraries:="['${TRT_LIB:?}']" \
  expected_file_path:=$HOME/models/tiny_policy_tensorrt/expected_outputs.bin \
  transport:=cuda
```

Without `backend_libraries`, the node cannot load this program. The node loads
the delegate library from `~/trt-venv` at run time, so keep that venv. The
TensorRT engine in the program only loads with the TensorRT version that built
it, so export again after changing the venv. A TensorRT engine runs on the GPU
number it was exported on, 0 with this script, so export with the same
`CUDA_VISIBLE_DEVICES` as the node.

## Limits

- The node runs once every input topic has a new message. Messages carry no
  frame number, so the inputs of a run are paired by arrival.
- A program with dynamic shapes takes its inputs at their largest shape. Its
  outputs are published at the shape the program returns.
- CUDA programs run on `cuda:0`. The node refuses a program on another GPU, and
  drops the set of messages when one is on another GPU. It selects `cuda:0` for
  its own work, but in a container where another component left the thread on
  another GPU, received `cuda_buffer` messages had no data, so run such
  components in another process.
- With `transport:=cuda`, a message the policy holds while it waits for the
  other inputs keeps one of the camera's buffers, and `cuda_buffer` waits for
  its buffers when the camera exits. So if an input never arrives, for example
  on a wrong topic, stop the launch with Ctrl-C.

## License

Apache-2.0
