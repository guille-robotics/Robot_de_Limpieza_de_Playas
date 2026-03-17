#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────
#  ENTORNO GYM + ROS 2 + GAZEBO (adaptado a Humble: sin StepWorld,
#  con visualización de meta en RViz2 mediante Marker)
#  -----------------------------------------------
#  • Generación de obstáculos reproducible mediante seed
#  • Lectura de sensores tras teleport para evitar datos "fantasma"
#  • Logging reducido para acelerar entrenamiento
#  • Uso de pause/unpause “manual” para simular paso síncrono
#  • Publicación de Marker para visualizar meta en RViz2
#  • Función de recompensa inspirada en Tao & Kim 2024
#  • Cálculo de orientación, velocidad y penalización por paso
#  • Comentarios detallados por toda la lógica
# ──────────────────────────────────────────────────────────

import threading
import time
import math
import random

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist, PoseStamped, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from gazebo_msgs.srv import SetEntityState, SpawnEntity, DeleteEntity
from gazebo_msgs.msg import EntityState
from visualization_msgs.msg import Marker  # Para el marcador de la meta

import gymnasium as gym


device_random = random  # alias para claridad


def seed_all(seed: int):
    """
    Fija la semilla para random y numpy para reproducibilidad.
    """
    device_random.seed(seed)
    np.random.seed(seed)


class RobotSimulacion(gym.Env, Node):
    """
    Entorno RL para robot diferencial en Gazebo+ROS2 con obstáculos aleatorios.
    (Adaptado a Humble eliminando StepWorld y usando pause/unpause “manual”,
    más Marker para visualizar meta en RViz2)
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        step_duration: float = 0.15,#0.015,
        n_obs: int = 4,  # Espacio 2de 20x20: 14
        obs_area: tuple = (-8.5, 8.5, -8.5, 8.5),  # Espacio de 20x20: (-9.5, 9.5, -9.5, 9.5)
        obs_min_dist: float = 3.0
    ):
        # ─── Iniciar nodo y executor ROS2 (multithreaded) ───
        Node.__init__(self, "rl_robot_env_improved")
        self.step_dt = step_duration  # tiempo (s) para “simular un paso” en cada acción

        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self)
        threading.Thread(target=self._executor.spin, daemon=True).start()

        # ─── Parámetros de entorno y obstáculos ───
        self.n_obstacles = n_obs
        self.obs_area = obs_area
        self.obs_min_dist = obs_min_dist
        self.box_names = [f"box_{i:02d}" for i in range(self.n_obstacles)]

        # ─── Parámetros de recompensa de evento ───
        self.rg = 200    # recompensa al llegar al objetivo
        self.rc = -150   # penalización por colisión
        self.rep = -100   # penalización por no llegar a tiempo
        self.goal_thresh = 1.2       # umbral para considerar "llegado"
        self.collision_thresh = 0.8 # umbral LiDAR para choque
        self.safety_margin=1.20

        # ─── Variables de episodios ───
        self.paso_en_episodio = 0
        self.max_steps = 800  # Espacio de 20x20: 800

        # ─── Truncamiento adaptativo ───
        self.no_prog_limit = 200      # pasos consecutivos sin avance suficiente
        self.min_prog_thresh = 0.05   # metros de avance considerados “progreso”
        self.no_prog_counter = 0

        self.verbose = False  # activa si quieres ver logs detallados


        # ─── Acción ───
        # ─── Acción (DESPUÉS) ───
        self.v_max = 0.8              # m/s avance/retroceso
        self.w_max = 0.6              # rad/s giro en sitio
        self.action_space = gym.spaces.Discrete(3)   # 0-3

        # Tabla para interpretar cada acción
        # 0: FWD, 1: REV, 2: LEFT, 3: RIGHT
        self._action_table = np.array([
            [ 1.0,  0.0],   # FWD  ->  v=+v_max , w=0
            [ 0.0, +0.8],   # LEFT ->  v=0      , w=+w_max
            [ 0.0, -0.8],   # RIGHT->  v=0      , w=-w_max
        ], dtype=np.float32)


        # ─── Observación: LiDAR 360° normalizado + [d_n, dir_x, dir_y, v_n, w_n] ───
        self.n_sect = 36              # Número de sectores en que dividimos el LiDAR
        self.r_min, self.r_max = 0.15, 8.0  # Límites de rango físico del LiDAR

        low = np.concatenate([
            np.zeros(self.n_sect),            # mínimos LiDAR normalizados
            [0.0, -1.0, -1.0, -1.0, -1.0]      # límites inferiores para [d_n, dir_x, dir_y, v_n, w_n]
        ]).astype(np.float32)
        high = np.concatenate([
            np.ones(self.n_sect),             # máximos LiDAR normalizados
            [1.0, 1.0, 1.0, 1.0, 1.0]          # límites superiores para [d_n, dir_x, dir_y, v_n, w_n]
        ]).astype(np.float32)

        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

        # ─── Publicadores y subscriptores ROS ───
        self.cmd_pub    = self.create_publisher(Twist, "/cmd_vel", 10)
        self.goal_pub   = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self.marker_pub = self.create_publisher(Marker, "/goal_marker", 10)  # Publisher de Marker
        self.create_subscription(Odometry, "/odom", self._odom_cb, 10)
        self.create_subscription(LaserScan, "/scan", self._scan_cb, 10)

        # ─── Clientes de servicios Gazebo ───
        self.pause_srv = self.create_client(Empty, "/pause_physics")
        self.unpause_srv = self.create_client(Empty, "/unpause_physics")
        self.reset_w_srv = self.create_client(Empty, "/reset_world")
        self.set_ent_srv = self.create_client(SetEntityState, "/set_entity_state")
        self.spawn_srv   = self.create_client(SpawnEntity, "/spawn_entity")
        self.delete_srv  = self.create_client(DeleteEntity, "/delete_entity")

        # Esperar a que servicios estén listos
        for cli, name in [
            (self.pause_srv,   "pause_physics"),
            (self.unpause_srv, "unpause_physics"),
            (self.reset_w_srv, "reset_world"),
            (self.set_ent_srv, "set_entity_state"),
            (self.spawn_srv,   "spawn_entity"),
            (self.delete_srv,  "delete_entity"),
        ]:
            while not cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f"⏳ Esperando /{name}...")

        # ─── Variables internas y primer reset ───
        self._reset_internal_vars()
        # Radio aproximado del robot (m), usado en TTC
        self.get_logger().info("🚀 Entorno mejorado inicializado. (Humble sin StepWorld)")

    def _reset_internal_vars(self):
        """Variables de estado al inicio de cada episodio."""
        self.pos_x = self.pos_y = self.yaw = 0.0
        self.v_act = self.w_act = 0.0
        # LiDAR full 360°, inicializamos a rango máximo
        self.lidar = np.ones(self.n_sect, dtype=np.float32) * self.r_max
        self.target_x = self.target_y  = 0.0
        self.paso_en_episodio = 0
        self.no_prog_counter = 0

    # ─────────────────────────────────────────────
    # 3) CALLBACKS ROS
    # ─────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        """Callback de odometría: actualiza posición, orientación y velocidades."""
        self.pos_x = msg.pose.pose.position.x
        self.pos_y = msg.pose.pose.position.y
        _, _, self.yaw = self._quat_to_rpy(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )
        self.v_act = msg.twist.twist.linear.x
        self.w_act = msg.twist.twist.angular.z

    def _scan_cb(self, msg: LaserScan):
        """
        Callback de LiDAR: segmenta los datos crudos en 'n_sect' sectores,
        obtiene la distancia mínima válida de cada sector y la almacena en
        un vector de longitud 'n_sect'.
        """
        rays_sec = max(1, len(msg.ranges) // self.n_sect)
        full_vals = []
        for i in range(self.n_sect):
            sect = msg.ranges[i * rays_sec : (i + 1) * rays_sec]
            m = min(
                (v if not math.isinf(v) else self.r_max)
                for v in sect
            )
            full_vals.append(min(m, self.r_max))
        self.lidar = np.array(full_vals, dtype=np.float32)

    # ─────────────────────────────────────────────
    # 4) UTILIDADES GEOMÉTRICAS
    # ─────────────────────────────────────────────

    @staticmethod
    def _quat_to_rpy(x, y, z, w):
        """
        Convierte un cuaternión (x, y, z, w) a ángulos roll–pitch–yaw,
        pero en este entorno solo usamos el yaw (rotación alrededor de Z).
        """
        t3 = 2 * (w * z + x * y)
        t4 = 1 - 2 * (y * y + z * z)
        return 0.0, 0.0, math.atan2(t3, t4)

    @staticmethod
    def _yaw_to_quat(yaw):
        """
        Convierte un ángulo yaw (rotación Z) a un cuaternión unitario.
        """
        q = Quaternion()
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        q.x = 0.0
        q.y = 0.0
        return q

    def _dist_to_goal(self):
        """
        Calcula la distancia Euclidiana entre la posición actual del robot
        y la meta.
        """
        dx = self.target_x - self.pos_x
        dy = self.target_y - self.pos_y
        return math.hypot(dx, dy)

    def _yaw_error(self):
        """
        Calcula el error angular entre la orientación actual del robot (self.yaw)
        y el ángulo hacia el objetivo (self.target_x, self.target_y).
        """
        # 1. Ángulo hacia el objetivo
        angle_to_goal = math.atan2(self.target_y - self.pos_y,
                                self.target_x - self.pos_x)

        # 2. Error angular (ángulo que debe girar el robot)
        err = angle_to_goal - self.yaw

        # 3. Normaliza entre [-pi, pi]
        return (err + math.pi) % (2 * math.pi) - math.pi


    # ─────────────────────────────────────────────
    # Función para publicar Marker de meta
    # ─────────────────────────────────────────────

    def _publish_goal_marker(self):
        """
        Publica un Marker de RViz en la posición de la meta para verla como esfera/arrow.
        """
        m = Marker()
        m.header.frame_id = "odom"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "goal"
        m.id = 0
        m.type = Marker.SPHERE       # puede ser ARROW, SPHERE, CUBE, etc.
        m.action = Marker.ADD
        m.pose.position.x = float(self.target_x)
        m.pose.position.y = float(self.target_y)
        m.pose.position.z = 0.0       # ajusta z si quieres elevar la esfera
        m.pose.orientation.w = 1.0    # sin rotación
        # Escala del marcador (diámetro de la esfera)
        m.scale.x = 0.4
        m.scale.y = 0.4
        m.scale.z = 0.4
        # Color (RGBA)
        m.color.r = 0.0
        m.color.g = 1.0
        m.color.b = 0.0
        m.color.a = 0.8
        self.marker_pub.publish(m)

    # ─────────────────────────────────────────────
    # 5) RESET (episodio)
    # ─────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        """
        Reinicia el entorno para un nuevo episodio:
        1) Pausa y reinicia la física de Gazebo para partir de un estado limpio.
        2) Teletransporta el robot y los obstáculos a posiciones aleatorias válidas.
        3) Publica meta y marcador en RViz.
        4) Estabiliza los sensores antes de arrancar el episodio.
        """
        super().reset(seed=seed)
        if seed is not None:
            seed_all(seed)

        # 1) Pausar física y resetear todo el mundo
        self._publish_cmd(0.0, 0.0)
        self._call_srv(self.pause_srv)
        self._call_srv(self.reset_w_srv)

        # 2) Reiniciar variables internas y marcar tiempo inicial
        self._reset_internal_vars()
        self.episode_start_time = time.time()

        # 3) Teletransportar robot
        spawn_range = 8.0 
        x0 = random.uniform(-spawn_range, spawn_range)
        y0 = random.uniform(-spawn_range, spawn_range)
        yaw0 = random.uniform(-math.pi, math.pi)
        self._teleport_entity('robot_cuerpo', x0, y0, yaw0)

        # 4) Definir y publicar meta aleatoria (Pose + Marker)
        # Meta con distancia mínima garantizada
        min_goal_dist = 3.0  # Mínimo 3m de distancia inicial
        for _ in range(100):  # Intentos para encontrar posición válida
            self.target_x = random.uniform(-spawn_range, spawn_range)
            self.target_y = random.uniform(-spawn_range, spawn_range)
            if math.hypot(self.target_x - x0, self.target_y - y0) > min_goal_dist:
                break
        #self.target_yaw = random.uniform(-math.pi, math.pi)
        self._publish_goal_pose()
        self._publish_goal_marker()   # <-- publicamos el Marker en RViz
        self.prev_dist = self._dist_to_goal()

        # 5) Teletransportar obstáculos sin chocar con robot/meta
        # Obstáculos con margen extra para robot grande
        coords = []
        for name in self.box_names:
            for _ in range(100):  # Más intentos para robot grande
                x = random.uniform(*self.obs_area[:2])
                y = random.uniform(*self.obs_area[2:])
                # Verificar distancia a robot Y meta
                dist_robot = math.hypot(x - x0, y - y0)
                dist_goal = math.hypot(x - self.target_x, y - self.target_y)
                if dist_robot > self.obs_min_dist and dist_goal > self.obs_min_dist:
                    coords.append((x, y))
                    break
        for name, (x, y) in zip(self.box_names, coords):
            self._teleport_entity(name, x, y, 0.0)

        # 6) Reactivar física brevemente para estabilizar sensores
        self._publish_cmd(0.0, 0.0)
        self._call_srv(self.unpause_srv)
        time.sleep(self.step_dt)
        self._call_srv(self.pause_srv)

        return self._get_obs(), {}

    # ─────────────────────────────────────────────
    # 6) STEP (acción)
    # ─────────────────────────────────────────────

    def _manual_step(self):
        """
        “Simula” un número de pasos de física en Gazebo haciendo:
        unpause → dormir un breve intervalo → pause.
        """
        self._call_srv(self.unpause_srv)
        time.sleep(self.step_dt)
        self._call_srv(self.pause_srv)

    def step(self, action):
        """
        Ejecuta un paso en el ambiente de RL para navegación robótica.
        
        Args:
            action: Acción discreta del espacio de acciones
            
        Returns:
            tuple: (observación, recompensa, terminado, truncado, info)
        """
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 1️⃣ CONVERSIÓN DE ACCIÓN DISCRETA A VELOCIDADES REALES
        # ═══════════════════════════════════════════════════════════════════════════════
        
        # Verifica que la acción esté dentro del rango válido
        assert self.action_space.contains(action), "Acción fuera de rango"
        
        # Convierte la acción discreta en velocidades normalizadas usando la tabla de acciones
        v_norm, w_norm = self._action_table[action]
        
        # Escala las velocidades normalizadas a valores reales
        v = v_norm * self.v_max  # Velocidad lineal real (m/s)
        w = w_norm * self.w_max  # Velocidad angular real (rad/s)
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 2️⃣ EJECUCIÓN DE LA ACCIÓN EN EL ROBOT
        # ═══════════════════════════════════════════════════════════════════════════════
        
        # Envía el comando de velocidad al robot
        self._publish_cmd(v, w)
        
        # Espera un paso de simulación física para que se ejecute la acción
        self._manual_step()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 3️⃣ LECTURA Y PROCESAMIENTO DE SENSORES
        # ═══════════════════════════════════════════════════════════════════════════════
        
        # 🎯 Errores de posición y orientación respecto al objetivo
        error_pos = self._dist_to_goal()      # Distancia euclidiana al objetivo (m)
        error_angular = self._yaw_error()     # Error angular respecto al objetivo (rad)
        
        # 🔦 Información del sensor LiDAR
        min_lidar = float(min(self.lidar))    # Distancia al obstáculo más cercano (m)
        max_lidar = float(max(self.lidar))    # Distancia al obstáculo más lejano (m)
        
        # 🚗 Velocidades actuales del robot (feedback de los encoders)
        v_cur = self.v_act                    # Velocidad lineal actual (m/s)
        w_cur = self.w_act                    # Velocidad angular actual (rad/s)
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 4️⃣ CÁLCULO DE LA RECOMPENSA (FUNCIÓN DE RECOMPENSA COMPLEJA)
        # ═══════════════════════════════════════════════════════════════════════════════
        
        # 📊 Progreso hacia el objetivo
        raw_prog = self.prev_dist - error_pos  # Progreso bruto: cuánto se acercó al objetivo
        # Normaliza el progreso entre 0 y 1 para evitar valores explosivos
        prog = max(0.0, min(1.0, raw_prog / (self.prev_dist + 1e-6)))
        
        # 🎯 Recompensa por movimiento suave hacia el objetivo
        # Premia avanzar rápido cuando está bien alineado con el objetivo
        smooth_drive = max(0.0, v_cur) * max(0.0, math.cos(error_angular))
        
        # 🔄 Penalización por giros excesivos (queremos movimiento eficiente)
        spin_pen = -1.5 * abs(w_cur)
        
        # 🛑 Penalización por quedarse detenido (evita comportamiento pasivo)
        stall_pen = -0.8 if abs(v) < 0.03 else 0.0
        
        # 🚧 Penalización progresiva por acercarse demasiado a obstáculos
        obs_pen = -5.0 * max(0.0, (self.collision_thresh - min_lidar) / self.collision_thresh)
        
        # 🛡️ Bonificación por mantener distancia segura de obstáculos
        safe_bonus = 2.0 if min_lidar > self.safety_margin else 0.0
        
        # 🧭 Bonificación extra por estar bien alineado con el objetivo
        alignment_bonus = 3.0 if abs(error_angular) < 0.1 else 0.0  # < 0.1 rad ≈ 5.7°
        
        # SUMA TOTAL DE LA RECOMPENSA
        reward = (
            8.0 * prog +           # 🎯 Progreso hacia meta (componente principal)
            3.0 * smooth_drive +   # 🚗 Movimiento suave y dirigido
            spin_pen +             # 🔄 Penalización por giro excesivo
            stall_pen +            # 🛑 Penalización por quedarse quieto
            obs_pen +              # 🚧 Penalización por cercanía a obstáculos
            safe_bonus +           # 🛡️ Bonus por mantener distancia segura
            alignment_bonus        # 🧭 Bonus por buena alineación
        )
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 5️⃣ VERIFICACIÓN DE CONDICIONES DE TÉRMINO
        # ═══════════════════════════════════════════════════════════════════════════════
        
        # Inicialización de variables de estado
        done = False
        estado = "🚗 En ruta"
        
        # 🏆 Condición de éxito: llegó al objetivo
        if error_pos < self.goal_thresh:
            reward = self.rg  # Recompensa grande por completar la tarea
            done = True
            estado = "🏆 Objetivo alcanzado"
        
        # 💥 Condición de fallo: colisionó con un obstáculo
        elif min_lidar < self.collision_thresh:
            reward = self.rc  # Penalización grande por colisionar
            done = True
            estado = "💥 Colisión"
        
        # Actualiza la distancia previa para el siguiente paso
        self.prev_dist = error_pos
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 6️⃣ VERIFICACIÓN DE CONDICIONES DE TRUNCAMIENTO
        # ═══════════════════════════════════════════════════════════════════════════════
        
        # 📈 Control de progreso: cuenta pasos sin avance significativo
        if prog < self.min_prog_thresh:
            self.no_prog_counter += 1
        else:
            self.no_prog_counter = 0  # Resetea contador si hay progreso
        
        # ⏱️ Incrementa contador de pasos en el episodio
        self.paso_en_episodio += 1
        
        # Verifica condiciones de truncamiento
        time_out = self.paso_en_episodio >= self.max_steps           # Límite de pasos alcanzado
        stuck_too_long = self.no_prog_counter >= self.no_prog_limit # Sin progreso por mucho tiempo
        truncated = time_out or stuck_too_long
        
        # Aplica penalización y actualiza estado si hay truncamiento
        if truncated:
            reward = self.rep  # Penalización por no completar la tarea
            estado = "⌛ Timeout" if time_out else "🛑 Sin progreso"
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 7️⃣ OBTENCIÓN DE NUEVA OBSERVACIÓN
        # ═══════════════════════════════════════════════════════════════════════════════
        
        obs = self._get_obs()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 8️⃣ LOGGING INTELIGENTE (solo cuando es relevante)
        # ═══════════════════════════════════════════════════════════════════════════════
        
        log = self.get_logger().info
        
        # Determina cuándo hacer logging para evitar spam en la consola
        should_log = (
            self.paso_en_episodio % 10 == 0 or          # Cada 10 pasos
            done or truncated or                         # Al terminar/truncar
            min_lidar < self.collision_thresh * 1.5      # Cuando está cerca de obstáculos
        )
        
        if should_log:
            log(f"Step {self.paso_en_episodio:3d} | {estado}")
            log(f"  🎯 Dist: {error_pos:.2f}m  🧭 Yaw: {math.degrees(error_angular):+.0f}°")
            log(f"  🚗 Vel: v={v:.2f} w={w:.2f}  🔦 LiDAR: {min_lidar:.2f}m")
            log(f"  💰 Reward: {reward:+.2f}  {'✅' if done else '🔄' if not truncated else '⏹️'}")
            
            if done or truncated:
                log("─" * 50)  # Separador visual para episodios terminados
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 9️⃣ CONSTRUCCIÓN DEL DICCIONARIO DE INFORMACIÓN
        # ═══════════════════════════════════════════════════════════════════════════════
        
        info = {
            "distance": error_pos,                      # Distancia actual al objetivo
            "error_yaw": error_angular,                 # Error angular actual
            "min_lidar": min_lidar,                     # Distancia mínima a obstáculos
            "max_lidar": max_lidar,                     # Distancia máxima a obstáculos
            "v_cmd": v,                                 # Velocidad lineal comandada
            "w_cmd": w,                                 # Velocidad angular comandada
            "progress": prog,                           # Progreso normalizado
            "reward_components": {                      # Desglose de componentes de recompensa
                "progress": 8.0 * prog,
                "smooth_drive": 3.0 * smooth_drive,
                "spin_penalty": spin_pen,
                "stall_penalty": stall_pen,
                "obstacle_penalty": obs_pen,
                "safety_bonus": safe_bonus,
                "alignment_bonus": alignment_bonus
            },
            "truncated_reason": (
                "timeout" if time_out else
                "no_progress" if stuck_too_long else
                None
            )
        }
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 🔟 RETORNO DE RESULTADOS
        # ═══════════════════════════════════════════════════════════════════════════════
        
        return obs, reward, done, truncated, info

    # ─────────────────────────────────────────────
    # 7) OBSERVACIÓN NORMALIZADA
    # ─────────────────────────────────────────────

    def _get_obs(self):
        """
        Construye el vector de observación completo para el agente:
        1) Lecturas de LiDAR segmentadas y normalizadas (n_sect valores).
        2) Distancia al objetivo, normalizada en [0,1].
        3) Orientación al objetivo como coseno y seno del error de yaw.
        4) Velocidades lineal y angular normalizadas.
        Resultado final: array de float32 con forma (n_sect + 5,).
        """
        lidar_n = np.clip(
            (self.lidar - self.r_min) / (self.r_max - self.r_min),
            0.0, 1.0
        )
        d_n = np.clip(self._dist_to_goal() / 10.0, 0.0, 1.0)
        dir_x = math.cos(self._yaw_error())
        dir_y = math.sin(self._yaw_error())
        v_n = max(0.0, self.v_act / self.v_max)
        w_n = self.w_act / self.w_max

        obs = np.concatenate([
            lidar_n,
            [d_n, dir_x, dir_y, v_n, w_n]
        ]).astype(np.float32)

        return obs

    # ─────────────────────────────────────────────
    # 8) RENDER (opcional)
    # ─────────────────────────────────────────────

    def render(self, mode="human"):
        if mode == "rgb_array":
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot(self.pos_x, self.pos_y, 'bo', label="Robot")
            ax.plot(self.target_x, self.target_y, 'rx', label="Goal")
            ax.set(xlim=(-8, 8), ylim=(-8, 8), title="Robot & Goal")
            ax.grid(True)
            ax.legend()
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            h, w = fig.canvas.get_width_height()
            plt.close(fig)
            return img.reshape((h, w, 3))

    # ─────────────────────────────────────────────
    # 9) CIERRE
    # ─────────────────────────────────────────────

    def close(self):
        self._publish_cmd(0.0, 0.0)
        # Aseguramos que la física esté despausada antes de apagar
        self._call_srv(self.unpause_srv)
        self._executor.shutdown()
        rclpy.shutdown()

    # ─────────────────────────────────────────────
    # 10) UTILIDADES ROS (privadas)
    # ─────────────────────────────────────────────

    def _publish_cmd(self, v, w):
        """Publica Twist en /cmd_vel."""
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.cmd_pub.publish(msg)

    def _publish_goal_pose(self):
        """Publica la posición del objetivo como PoseStamped en /goal_pose."""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        msg.pose.position.x = float(self.target_x)
        msg.pose.position.y = float(self.target_y)

        # Orientación neutra (sin giro)
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0

        self.goal_pub.publish(msg)


    def _teleport_entity(self, name, x, y, yaw):
        """Teletransporta entidad en Gazebo a (x,y,yaw)."""
        req = SetEntityState.Request()
        req.state = EntityState()
        req.state.name = name
        req.state.pose.position.x = float(x)
        req.state.pose.position.y = float(y)
        req.state.pose.position.z = 0.0
        req.state.pose.orientation = self._yaw_to_quat(yaw)
        req.state.reference_frame = "world"
        self._call_srv(self.set_ent_srv, req)

    def _call_srv(self, client, request=None, timeout=2.0):
        """Llamada síncrona a un servicio ROS2."""
        if request is None:
            request = client.srv_type.Request()
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        return future.result()


# ──────────────────────────────────────────────────────────
# TEST RÁPIDO
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    rclpy.init()
    env = RobotSimulacion()
    obs, _ = env.reset()
    print("Obs0 shape:", obs.shape)
    for ep in range(100):
        obs, _ = env.reset()
        done = truncated = False
        while not (done or truncated):
            action = env.action_space.sample()
            obs, rew, done, truncated, _ = env.step(action)
        print(f"✅ Episodio {ep+1} terminado. Rew final: {rew:.2f}")
    env.close()
