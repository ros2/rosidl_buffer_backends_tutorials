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
        DeclareLaunchArgument('use_cuda', default_value='true'),
        DeclareLaunchArgument('headless', default_value='false'),

        Node(
            package='robot_arm_demo',
            executable='renderer_node',
            parameters=[{
                'image_width': LaunchConfiguration('width'),
                'image_height': LaunchConfiguration('height'),
                'use_cuda': LaunchConfiguration('use_cuda'),
            }],
        ),

        Node(
            package='robot_arm_demo',
            executable='display_node',
            parameters=[{
                'use_cuda': LaunchConfiguration('use_cuda'),
                'headless': LaunchConfiguration('headless'),
            }],
        ),
    ])
