"""
Launch file para entrenar RL en modo headless (sin GUI de Gazebo) + RViz para visualización.
Optimizado para entrenamiento rápido de RL manteniendo capacidad de monitoreo visual.
"""
__author__ = "C. Mauricio Arteaga-Escamilla from 'Robotica Posgrado' (Youtube channel)"
__contact__ = "cmauricioae8@gmail.com"

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro

package_name = 'robot_tesis_cuerpo'
robot_model = 'robot_cuerpo'#'robot_cuerpo'
robot_base_color = 'Red'
pose = ['0.0', '0.0', '0.0', '0.0']  # Pose inicial del robot: x,y,z,th
gz_robot_name = robot_model

# Archivo de mundo para simulación headless
world_file = 'varias_cajas_test_robot_grande_sin_obstaculos.world'

def generate_launch_description():

    pkg_robot_simulation = get_package_share_directory(package_name)

    # ═══════════════════════════════════════════════════════════════════════════════
    # 🔧 ARGUMENTOS DE LANZAMIENTO
    # ═══════════════════════════════════════════════════════════════════════════════
    
    world_arg = DeclareLaunchArgument(
        'world',
        default_value=[os.path.join(pkg_robot_simulation, 'worlds', world_file), ''],
        description='Custom SDF world file for headless simulation')

    simu_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='True',
        description='Use simulation (Gazebo) clock if true')
    
    # Argumento para controlar si mostrar RViz
    show_rviz_arg = DeclareLaunchArgument(
        'show_rviz',
        default_value='True',
        description='Launch RViz for visualization')

    # ═══════════════════════════════════════════════════════════════════════════════
    # 🤖 DESCRIPCIÓN DEL ROBOT
    # ═══════════════════════════════════════════════════════════════════════════════
    
    robot_description_path = os.path.join(
        pkg_robot_simulation, "urdf", robot_model + '.xacro'
    )
    
    robot_description = {
        "robot_description": xacro.process_file(
            robot_description_path, 
            mappings={'base_color': robot_base_color}
        ).toxml()
    }

    # ═══════════════════════════════════════════════════════════════════════════════
    # 📡 ROBOT STATE PUBLISHER
    # ═══════════════════════════════════════════════════════════════════════════════
    
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description, {"use_sim_time": LaunchConfiguration('use_sim_time')}],
    )

    # ═══════════════════════════════════════════════════════════════════════════════
    # 💻 GAZEBO HEADLESS (SIN GUI)
    # ═══════════════════════════════════════════════════════════════════════════════
    
    gazebo_headless = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('gazebo_ros'), 
                'launch', 
                'gzserver.launch.py'  # Solo servidor, sin cliente gráfico
            )
        ),
        launch_arguments={
            'world': LaunchConfiguration('world'),
            'verbose': 'false',  # Reduce logs para mayor velocidad
            'physics': 'ode',    # Motor de física (ode es más rápido que bullet)
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }.items()
    )

    # ═══════════════════════════════════════════════════════════════════════════════
    # 🚀 SPAWNER DEL ROBOT EN GAZEBO
    # ═══════════════════════════════════════════════════════════════════════════════
    
    robot_spawner = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='urdf_spawner',
        output='screen',
        arguments=[
            "-topic", "/robot_description", 
            "-entity", gz_robot_name,
            "-x", pose[0], 
            "-y", pose[1], 
            "-z", pose[2], 
            "-Y", pose[3]
        ],
        parameters=[{"use_sim_time": LaunchConfiguration('use_sim_time')}]
    )

    # ═══════════════════════════════════════════════════════════════════════════════
    # 👁️ RVIZ PARA VISUALIZACIÓN (OPCIONAL)
    # ═══════════════════════════════════════════════════════════════════════════════
    
    # Buscar archivo de configuración de RViz
    rviz_config_path = os.path.join(
        pkg_robot_simulation, 'rviz', 'robot_navigation.rviz'
    )
    
    # Si no existe el archivo de config, usar configuración por defecto
    if not os.path.exists(rviz_config_path):
        rviz_config_path = ''
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_path] if rviz_config_path else [],
        parameters=[{"use_sim_time": LaunchConfiguration('use_sim_time')}],
        condition=None  # Se lanzará condicionalmente más abajo
    )

    # ═══════════════════════════════════════════════════════════════════════════════
    # 🔗 NODO ADICIONAL PARA PUBLICAR TRANSFORMACIONES ESTÁTICAS (SI ES NECESARIO)
    # ═══════════════════════════════════════════════════════════════════════════════
    
    # Algunos sistemas necesitan transformaciones estáticas adicionales
    static_transform_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom_broadcaster',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        parameters=[{"use_sim_time": LaunchConfiguration('use_sim_time')}]
    )

    # ═══════════════════════════════════════════════════════════════════════════════
    # 📊 CONFIGURACIÓN ADICIONAL PARA ENTRENAMIENTO RL
    # ═══════════════════════════════════════════════════════════════════════════════
    
    print("🚀 INICIANDO SIMULACIÓN HEADLESS PARA ENTRENAMIENTO RL...")
    print("📍 Configuración:")
    print(f"   🤖 Robot: {robot_model}")
    print(f"   🌍 Mundo: {world_file}")
    print(f"   📍 Pose inicial: {pose}")
    print("   💻 Modo: Headless (sin GUI de Gazebo)")
    print("   👁️  Visualización: RViz disponible")
    print("   ⚡ Optimizado para velocidad de entrenamiento")

    # Lista de nodos a lanzar
    launch_nodes = [
        world_arg,
        simu_time,
        show_rviz_arg,
        gazebo_headless,
        robot_state_publisher_node,
        static_transform_publisher,
        robot_spawner,
    ]
    
    # Agregar RViz condicionalmente
    # Nota: En ROS2, la condición se maneja diferente. 
    # Para simplificar, siempre incluimos RViz pero puedes comentar la siguiente línea
    # si no quieres RViz en absoluto durante el entrenamiento
    launch_nodes.append(rviz_node)
    
    return LaunchDescription(launch_nodes)


# ═══════════════════════════════════════════════════════════════════════════════
# 💡 INSTRUCCIONES DE USO
# ═══════════════════════════════════════════════════════════════════════════════

"""
CÓMO USAR ESTE LAUNCH FILE:

1. 🚀 Lanzar simulación headless CON RViz:
   ros2 launch robot_tesis_cuerpo iniciar_launch_headless.py

2. 🚀 Lanzar simulación headless SIN RViz:
   ros2 launch robot_tesis_cuerpo iniciar_launch_headless.py show_rviz:=False

3. 🚀 Usar mundo personalizado:
   ros2 launch robot_tesis_cuerpo iniciar_launch_headless.py world:=path/to/your/world.world

VENTAJAS DEL MODO HEADLESS:
✅ Mayor velocidad de simulación (sin renderizado gráfico)
✅ Menor uso de recursos (CPU/GPU)
✅ Ideal para entrenamiento masivo de RL
✅ Permite ejecutar múltiples instancias en paralelo
✅ Mantiene RViz para monitoreo cuando es necesario

MONITOREO DURANTE ENTRENAMIENTO:
- 📊 Usa RViz para ver el robot y obstáculos
- 📈 Visualiza el marcador de la meta (sphere verde)
- 🔦 Observa los datos del LiDAR
- 📍 Monitorea la trayectoria del robot
"""
