# Copyright 2026 Open Source Robotics Foundation, Inc.
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

import array
import atexit
import glob
import os
import re
import tempfile
import unittest

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import launch_testing
import launch_testing.actions
import launch_testing.asserts
import pytest

# The cases are skipped when this is unset, since exporting needs PyTorch and a GPU. Set
# it to an exported tiny_policy directory (see scripts/export_demo_models.py).
MODEL_DIR = os.environ.get('EXECUTORCH_POLICY_DEMO_TEST_MODEL_DIR', '')
EXPECTED = os.path.join(MODEL_DIR, 'expected_outputs.bin')
# kFramePeriod in frame_kernel.hpp: the expected outputs hold one block per frame of it.
FRAME_PERIOD = 256
# With the warmup frames the run passes one full period, so the frame index also wraps.
FRAMES = 240
WARMUP = 20
REPORT_EVERY = 20
CAMERA = 'camera_node'
POLICY = 'policy_node'
CHECKED = ('checked=yes wrong_actions=0 ', CAMERA)
REFUSED = ('policy: skipping a message on input', POLICY)
IN_PLACE = [CHECKED, ('action_backend=cuda ', CAMERA)]
# The staged copies every policy report must show: none in place, and on host messages
# the 2 inputs and the action for each run.
COPIES = {'checked': 0, 'cpu': 3 * REPORT_EVERY}

# Launch arguments, the camera's exit status, and lines each process must log.
CASES = {
    'checked': ({'expected_file_path': EXPECTED}, 0, IN_PLACE),
    # Host messages: the camera copies each frame off the GPU, and the policy copies it
    # onto the GPU and the action off it.
    'cpu': ({'expected_file_path': EXPECTED, 'transport': 'cpu'}, 0,
            [CHECKED, ('action_backend=cpu ', CAMERA)]),
    # References for the previous frame, so every action, warmup included, must count as wrong.
    'wrong_actions': ({}, 1, [(f'checked=yes wrong_actions={FRAMES + WARMUP} ', CAMERA)]),
    # A message the policy refuses must not keep the other input's buffer, or the camera
    # cannot exit. These refuse the state or the image, which can arrive in either order.
    'wrong_state': ({'state_shape': '[1, 13]'}, 1, [REFUSED]),
    'wrong_image': ({'width': '320'}, 1, [REFUSED]),
    # One input topic for the 2-input program, which the policy must refuse when it loads
    # it. The launch then stops the camera, which can exit either way (see the README), so
    # only the policy's exit status is checked.
    'wrong_program': ({'state_shape': '[0]'}, None,
                      [('1 input and 1 output topics, but the program takes 2 inputs', POLICY)]),
}


def previous_frame_expected_file():
    """Write each frame's expected outputs for the frame before it, and return the path."""
    values = array.array('f')
    with open(EXPECTED, 'rb') as source:
        values.frombytes(source.read())
    block = len(values) // FRAME_PERIOD
    descriptor, path = tempfile.mkstemp(suffix='.bin')
    atexit.register(os.remove, path)
    with os.fdopen(descriptor, 'wb') as target:
        (values[-block:] + values[:-block]).tofile(target)
    return path


@pytest.mark.launch_test
@launch_testing.parametrize('case', list(CASES))
def generate_test_description(case):
    """Camera and policy, from the package's launch file."""
    if not MODEL_DIR:
        raise unittest.SkipTest('EXECUTORCH_POLICY_DEMO_TEST_MODEL_DIR is not set')
    data = sorted(glob.glob(os.path.join(MODEL_DIR, '*.ptd'))) or ['']
    arguments = {
        'model_file_path': os.path.join(MODEL_DIR, 'model.pte'),
        'data_file_paths': str(data),
        'transport': 'cuda',
        'frames': str(FRAMES),
        'warmup': str(WARMUP),
        'report_every': str(REPORT_EVERY),
        **CASES[case][0],
    }
    if case == 'wrong_actions':
        arguments['expected_file_path'] = previous_frame_expected_file()
    launch_file = os.path.join(
        os.path.dirname(__file__), '..', 'launch', 'policy_demo.launch.py')
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(launch_file), launch_arguments=arguments.items()),
        launch_testing.actions.ReadyToTest(),
    ])


class TestPolicyDemo(unittest.TestCase):

    def test_camera_exits(self, proc_info, proc_output, case):
        proc_info.assertWaitForShutdown(process=CAMERA, timeout=60)
        for line, process in CASES[case][2]:
            proc_output.assertWaitFor(line, process=process, timeout=1)


@launch_testing.post_shutdown_test()
class TestPolicyDemoExit(unittest.TestCase):

    def test_exit_codes(self, proc_info, case):
        if CASES[case][1] is None:
            # The exception from the node's constructor ends the process with abort().
            launch_testing.asserts.assertExitCodes(
                proc_info, allowable_exit_codes=[-6], process=POLICY)
            return
        launch_testing.asserts.assertExitCodes(
            proc_info, allowable_exit_codes=[CASES[case][1]], process=CAMERA)
        # The launch stops the policy when the camera exits; any other status is a crash.
        launch_testing.asserts.assertExitCodes(
            proc_info, allowable_exit_codes=[0, -2, -15], process=POLICY)

    def test_staged_copies(self, proc_output, case):
        if case not in COPIES:
            self.skipTest('this case checks no staged copies')
        text = ''.join(
            output.text.decode() for output in proc_output
            if output.process_name.startswith(POLICY))
        reports = [int(copies) for copies in re.findall(r' staged_copies=(\d+) ', text)]
        # The launch can stop the policy before it logs the report of its last run.
        self.assertGreaterEqual(len(reports), (FRAMES + WARMUP) // REPORT_EVERY - 1)
        self.assertEqual(set(reports), {COPIES[case]})
