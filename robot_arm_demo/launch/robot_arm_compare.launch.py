# Copyright 2026 Open Source Robotics Foundation, Inc.
# Licensed under the Apache License, Version 2.0

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('width', default_value='1920'),
        DeclareLaunchArgument('height', default_value='1080'),
        DeclareLaunchArgument('headless', default_value='false'),

        Node(
            package='robot_arm_demo',
            executable='renderer_node',
            namespace='cuda',
            parameters=[{
                'image_width': LaunchConfiguration('width'),
                'image_height': LaunchConfiguration('height'),
                'use_cuda': True,
            }],
            arguments=['--log-level', 'WARN'],
        ),

        Node(
            package='robot_arm_demo',
            executable='renderer_node',
            namespace='cpu',
            parameters=[{
                'image_width': LaunchConfiguration('width'),
                'image_height': LaunchConfiguration('height'),
                'use_cuda': False,
            }],
            arguments=['--log-level', 'WARN'],
        ),

        Node(
            package='robot_arm_demo',
            executable='display_node',
            namespace='cuda',
            parameters=[{
                'use_cuda': True,
                'headless': LaunchConfiguration('headless'),
            }],
        ),

        Node(
            package='robot_arm_demo',
            executable='display_node',
            namespace='cpu',
            parameters=[{
                'use_cuda': False,
                'headless': LaunchConfiguration('headless'),
            }],
        ),
    ])
