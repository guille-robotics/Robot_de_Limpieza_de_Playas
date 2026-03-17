"""
Simple launch file to spawn a robot in Gazebo.
"""
__author__ = "C. Mauricio Arteaga-Escamilla from 'Robotica Posgrado' (Youtube channel)"
__contact__ = "cmauricioae8@gmail.com"

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro

package_name = 'robot_tesis_cuerpo'
robot_model = 'robot_cuerpo' #caster_diffbot, sensors_diffbot, macro_light_diffbot, omnibot (static wheels)
robot_base_color = 'Red' #Gazebo color of the robot's main body
pose = ['0.0', '0.0', '0.0', '0.0'] #Initial robot pose in gz: x,y,z,th
gz_robot_name = robot_model #robot name used in Gazebo


#Launch argument
#world_file = 'empty.world' #empty.world
#world_file = 'varias_cajas_test_robot_grande.world' #empty.world
world_file = 'varias_cajas_test_robot_grande.world'#mundo_irregular_rocas.world' #empty.world

def generate_launch_description():

    pkg_robot_simulation = get_package_share_directory(package_name)

    world_arg = DeclareLaunchArgument(
        'world',
        default_value=[os.path.join(pkg_robot_simulation, 'worlds', world_file), ''],
        description='Custom SDF world file')    

    simu_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='True',
        description='Use simulation (Gazebo) clock if true')

    robot_description_path =  os.path.join( pkg_robot_simulation, "urdf", robot_model + '.xacro', )
    
    #robot_description = {"robot_description": xacro.process_file(robot_description_path).toxml()} #simple model
    robot_description = {"robot_description": xacro.process_file(robot_description_path, 
        mappings={'base_color' : robot_base_color}).toxml()}


    #joint_state_publisher =  Node(package='joint_state_publisher',executable='joint_state_publisher',name='joint_state_publisher')

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description],
    )

    gazebo_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join( get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py'),
        )
    )

    robot_spawner = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='urdf_spawner',
        output='screen',
        arguments=["-topic", "/robot_description", "-entity", gz_robot_name, 
            "-x", pose[0], "-y", pose[1], "-z", pose[2], "-Y", pose[3] ]
    )


    print("STARTING ALL NODES ...")

    return LaunchDescription([
        world_arg,
        simu_time,
        #joint_state_publisher, #node is omitted, 'libgazebo_ros_joint_state_publisher.so' plugin is used
        gazebo_node,
        robot_state_publisher_node,
        robot_spawner
    ])
