import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro

package_name    = 'robot_tesis_cuerpo'
robot_model     = 'robot_cuerpo'
robot_color     = 'Red'
pose            = ['0.0', '0.0', '0.0', '0.0']
world_file      = 'lab_rapido.world'

def generate_launch_description():
    pkg_share = get_package_share_directory(package_name)
    world_path = os.path.join(pkg_share, 'worlds', world_file)

    # 1) Argumento para usar tiempo simulado
    use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='True',
        description='Use simulation (Gazebo) clock if true'
    )

    # 2) Lanzar sólo gzserver (no GUI)
    gazebo_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('gazebo_ros'),
                'launch', 'gazebo.launch.py'
            )
        ),
        launch_arguments={
            'world': world_path,
            'gui'  : 'false'       # <— deshabilita gzclient
        }.items(),
    )

    # 3) Publicador de estado del robot (TF desde URDF)
    robot_description_path = os.path.join(
        pkg_share, 'urdf', f'{robot_model}.xacro'
    )
    robot_description = {
        'robot_description': xacro.process_file(
            robot_description_path,
            mappings={'base_color': robot_color}
        ).toxml()
    }
    state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[robot_description, {'use_sim_time': True}],
        output='screen'
    )

    # 4) Spawnear la entidad en el servidor
    spawner = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', '/robot_description',
            '-entity', robot_model,
            '-x', pose[0], '-y', pose[1],
            '-z', pose[2], '-Y', pose[3]
        ],
        output='screen'
    )

    # 5) Lanzar RViz2
    rviz_config = os.path.join(pkg_share, 'rviz', 'robot_view.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': True}]
    )

    return LaunchDescription([
        use_sim_time,
        gazebo_server,
        state_publisher,
        spawner,
        rviz_node
    ])

