# Copyright 2026 Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Export the demo programs for the ExecuTorch CUDA backend.

tiny_policy: image [1, 3, H, W] float and state [1, 14] -> action [1, 100, 14],
    the input and output shapes of a LeRobot ACT policy, with a small network.
frame_stats: frame [H, W, 4] uint8 -> per-channel mean [4], to time large frames
    at camera resolutions.

Every program keeps inputs and outputs in device memory and leaves them
unplanned, so the policy node passes message storage to it in place. Export on
the GPU that runs the program: the CUDA backend compiles kernels for it.

Each program directory also gets expected_outputs.bin: the outputs for every
frame camera_node renders, which the camera checks each action against.

Usage: python export_demo_models.py OUT_DIR [--stats HxW ...]
"""

import argparse
import math
import os

from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.exir import ExecutorchBackendConfig, to_edge_transform_and_lower
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.passes.propagate_device_config import PropagateDeviceConfig
import torch


class TinyPolicy(torch.nn.Module):

    def __init__(self, state_dim=14, chunk=100):
        super().__init__()
        self.chunk = chunk
        self.state_dim = state_dim
        self.vision = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 5, stride=4, padding=2), torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, stride=2, padding=1), torch.nn.ReLU(),
            # A grid rather than one average, so an image from another frame changes the action.
            torch.nn.AdaptiveAvgPool2d(4), torch.nn.Flatten())
        self.head = torch.nn.Linear(32 * 4 * 4 + state_dim, chunk * state_dim)

    def forward(self, image, state):
        features = torch.cat([self.vision(image), state], dim=1)
        return self.head(features).reshape(1, self.chunk, self.state_dim)


class FrameStats(torch.nn.Module):

    def forward(self, frame):
        return frame.to(torch.float32).mean(dim=(0, 1))


# kFramePeriod in frame_kernel.hpp.
FRAME_PERIOD = 256


def camera_frame(index, shape, dtype):
    """Return the frame camera_node renders for index, as frame_kernel.cu does."""
    index %= FRAME_PERIOD
    flat = torch.arange(math.prod(shape), device='cuda')
    if dtype == torch.uint8:
        value = (flat + flat // 256 + 7 * index) % 256 * (index + 2) // 257
        return value.to(torch.uint8).reshape(shape)
    plane = shape[-2] * shape[-1]
    pixel = flat % plane
    value = (pixel + pixel // 256 + 7 * index + flat // plane) % 256
    return value.float().div(255).reshape(shape)


def robot_state(index, shape):
    """Return the state camera_node sends with frame index, as frame_kernel.hpp does."""
    element = torch.arange(math.prod(shape), device='cuda', dtype=torch.float64)
    angle = 2 * math.pi * (element // 2 + 1) * index / FRAME_PERIOD
    return torch.where(element % 2 == 0, angle.cos(), angle.sin()).float().reshape(shape)


def export(module, camera_inputs, out, name):
    directory = os.path.join(out, name)
    os.makedirs(directory, exist_ok=True)
    module = module.cuda().eval()
    with torch.no_grad():
        expected = [module(*camera_inputs(index)).flatten() for index in range(FRAME_PERIOD)]
        program = torch.export.export(module, camera_inputs(0))
        partitioner = CudaPartitioner([CudaBackend.generate_method_name_compile_spec('forward')])
        edge = to_edge_transform_and_lower(program, partitioner=[partitioner])
        executorch_program = edge.to_executorch(ExecutorchBackendConfig(
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=False, alloc_graph_output=False),
            propagate_device_config=PropagateDeviceConfig(
                skip_h2d_for_method_inputs=True, skip_d2h_for_method_outputs=True),
            enable_non_cpu_memory_planning=True,
        ))
        with open(os.path.join(directory, 'model.pte'), 'wb') as f:
            executorch_program.write_to_file(f)
        executorch_program.write_tensor_data_to_file(directory)
        # Last, so a failed export does not leave new outputs next to an old program.
        torch.cat(expected).float().cpu().numpy().tofile(
            os.path.join(directory, 'expected_outputs.bin'))
    print('wrote', directory, sorted(os.listdir(directory)))


def resolution(value):
    try:
        height, width = (int(part) for part in value.split('x'))
    except ValueError:
        raise argparse.ArgumentTypeError(f'{value} is not HxW')
    return height, width


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out')
    parser.add_argument('--stats', nargs='*', type=resolution,
                        default=[(1080, 1920), (2160, 3840)],
                        help='frame_stats resolutions as HxW')
    args = parser.parse_args()

    torch.manual_seed(0)
    # The camera checks actions with an FP32 tolerance, and cuDNN may otherwise pick TF32.
    torch.backends.cudnn.allow_tf32 = False
    export(TinyPolicy(),
           lambda index: (camera_frame(index, (1, 3, 480, 640), torch.float32),
                          robot_state(index, (1, 14))),
           args.out, 'tiny_policy')
    for height, width in args.stats:
        export(FrameStats(),
               lambda index: (camera_frame(index, (height, width, 4), torch.uint8),),
               args.out, f'frame_stats_{height}x{width}')


if __name__ == '__main__':
    main()
