#!/usr/bin/env python3
# ────────────────────────────────────────────────────────────────
# train_robot_rl_with_PPO_mask.py
# ────────────────────────────────────────────────────────────────
# Entrenamiento con PPO + Action Masking para navegación autónoma
# Mantiene todos los callbacks originales (CSV, TensorBoard, checkpoints)
# ────────────────────────────────────────────────────────────────

import os
import csv
import time
import random
from datetime import datetime
from collections import deque

import numpy as np
import gymnasium as gym
import torch  

from sb3_contrib import MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import (
    BaseCallback,
    StopTrainingOnMaxEpisodes,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure

import rclpy                    # ROS 2 Python API
import robot_env                # registra RobotSimulacion-v0

# ─────────────────── 0. Reproducibilidad ───────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
set_random_seed(SEED)

# ─────────────────── 1. Parámetros generales ───────────────────
TOTAL_TIMESTEPS    = 1_000_000
SAVE_EVERY_STEPS   = 200_000

RUN_NAME           = datetime.now().strftime("rl_mask_run_%Y%m%d_%H%M")
LOG_DIR            = os.path.join("runs", RUN_NAME)
os.makedirs(LOG_DIR, exist_ok=True)

CSV_PATH           = os.path.join(LOG_DIR, "episode_metrics.csv")
CHECKPOINT_DIR     = os.path.join(LOG_DIR, "checkpoints")
BEST_DIR           = os.path.join(LOG_DIR, "best_model")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(BEST_DIR, exist_ok=True)

# detectar GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"CUDA available: {torch.cuda.is_available()}  → usando device: {DEVICE}")

# ─────────────────── 2. Función de Action Masking ───────────────────
def mask_fn(env):
    """
    Función que determina qué acciones están permitidas basándose en el estado actual.
    
    Acciones en tu entorno:
    0: FWD   (avanzar)
    1: LEFT  (girar izquierda) 
    2: RIGHT (girar derecha)
    
    Lógica de masking inteligente:
    - Si hay obstáculo muy cerca adelante → prohibir FWD
    - Si hay obstáculo muy cerca a la izquierda → prohibir LEFT
    - Si hay obstáculo muy cerca a la derecha → prohibir RIGHT
    - Siempre permitir al menos una acción para evitar deadlock
    """
    # Obtener el entorno base RobotSimulacion desde los wrappers
    base_env = env
    while hasattr(base_env, 'env'):
        base_env = base_env.env
    
    # Obtener distancias LiDAR del entorno base
    lidar_readings = base_env.lidar  # Array con lecturas de cada sector
    
    # Definir umbrales de seguridad
    DANGER_THRESH = 1.2   # Distancia considerada peligrosa (m)
    CRITICAL_THRESH = 0.9 # Distancia crítica que fuerza masking (m)
    
    # Calcular sectores relevantes para cada dirección
    n_sectors = len(lidar_readings)
    
    # Sector frontal (±30° adelante)
    front_sectors = []
    sector_angle = 360 / n_sectors
    for i in range(n_sectors):
        angle = (i * sector_angle - 180) % 360 - 180  # Convertir a [-180, 180]
        if -30 <= angle <= 30:
            front_sectors.append(i)
    
    # Sectores izquierda (60° a 120°)
    left_sectors = []
    for i in range(n_sectors):
        angle = (i * sector_angle - 180) % 360 - 180
        if 60 <= angle <= 120:
            left_sectors.append(i)
    
    # Sectores derecha (-120° a -60°)
    right_sectors = []
    for i in range(n_sectors):
        angle = (i * sector_angle - 180) % 360 - 180
        if -120 <= angle <= -60:
            right_sectors.append(i)
    
    # Obtener distancia mínima en cada dirección
    front_dist = min([lidar_readings[i] for i in front_sectors]) if front_sectors else float('inf')
    left_dist = min([lidar_readings[i] for i in left_sectors]) if left_sectors else float('inf')
    right_dist = min([lidar_readings[i] for i in right_sectors]) if right_sectors else float('inf')
    
    # Crear máscara (True = acción permitida, False = prohibida)
    mask = np.array([True, True, True], dtype=bool)  # [FWD, LEFT, RIGHT]
    
    # Aplicar restricciones basadas en proximidad a obstáculos
    if front_dist < CRITICAL_THRESH:
        mask[0] = False  # Prohibir avanzar
    
    if left_dist < CRITICAL_THRESH:
        mask[1] = False  # Prohibir girar izquierda
        
    if right_dist < CRITICAL_THRESH:
        mask[2] = False  # Prohibir girar derecha
    
    # Medida de seguridad: asegurar que al menos una acción esté disponible
    if not np.any(mask):
        # Si todas están prohibidas, permitir giros (más seguro que avanzar)
        mask[1] = True  # Permitir LEFT
        mask[2] = True  # Permitir RIGHT
    
    return mask

# ─────────────────── 3. Callback para CSV & Checkpoints (IDÉNTICO) ───────────────────
class CsvAndCheckpointCallback(BaseCallback):
    def __init__(self, csv_path, save_every, checkpoint_dir, verbose=0):
        super().__init__(verbose)
        self.csv_path        = csv_path
        self.save_every      = save_every
        self.checkpoint_dir  = checkpoint_dir
        self.start_time      = time.time()
        self.successes       = 0
        self.episode_count   = 0
        self.last_rewards    = deque(maxlen=20)
        self.last_success    = deque(maxlen=20)

        # inicializar CSV
        if not os.path.isfile(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["episode","reward","length","time_sec","success"]
                )

    def _on_step(self) -> bool:
        # fin de episodio
        if self.locals.get("dones") and self.locals["dones"][0]:
            info    = self.locals["infos"][0]["episode"]
            ep_rew  = info["r"]
            ep_len  = info["l"]
            elapsed = time.time() - self.start_time

            self.episode_count += 1
            success = 1 if ep_rew > 0 else 0
            self.successes    += success
            self.last_rewards.append(ep_rew)
            self.last_success.append(success)

            # escribir CSV
            with open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    self.episode_count,
                    ep_rew,
                    ep_len,
                    elapsed,
                    success
                ])

            # imprimir cada 20 episodios
            if len(self.last_rewards) == self.last_rewards.maxlen:
                avg_rew   = sum(self.last_rewards) / len(self.last_rewards)
                succ_rate = sum(self.last_success) / len(self.last_success) * 100
                print("──── Estadísticas ────")
                print(f"Recompensa media (últ.20): {avg_rew:.2f}")
                print(f"Tasa éxito  (últ.20): {succ_rate:.1f}%")

        # guardar checkpoint con timesteps en el nombre
        if self.num_timesteps % self.save_every == 0:
            fname = os.path.join(
                self.checkpoint_dir,
                f"ppo_mask_robot_{self.num_timesteps}"
            )
            self.model.save(fname)
            if self.verbose:
                print(f"💾 Checkpoint guardado: {fname}.zip")

        return True

    def _on_training_end(self) -> None:
        if self.episode_count > 0:
            rate = self.successes / self.episode_count * 100
            print(f"\n✅ Éxito total: {self.successes}/{self.episode_count} ({rate:.1f}%)")
        super()._on_training_end()

# ─────────────────── 4. Entorno con Action Masking ───────────────────
def make_env():
    """Crea el entorno base y le aplica el wrapper de Action Masking"""
    try:
        rclpy.init(args=None)
    except RuntimeError:
        pass
    
    # Crear entorno base
    base_env = gym.make("RobotSimulacion-v0", step_duration=0.015)
    
    # Aplicar wrapper de Action Masking
    masked_env = ActionMasker(base_env, mask_fn)
    
    # Aplicar Monitor para logging
    return Monitor(masked_env)

def make_eval_env():
    """Crea entorno de evaluación (puede ser sin masking si prefieres evaluar sin restricciones)"""
    try:
        rclpy.init(args=None)
    except RuntimeError:
        pass
    
    base_env = gym.make("RobotSimulacion-v0", step_duration=0.015)
    # Para evaluación, puedes decidir si usar masking o no
    masked_env = ActionMasker(base_env, mask_fn)  # Con masking
    # masked_env = base_env  # Sin masking para evaluación "pura"
    
    return Monitor(masked_env)

env = make_env()

# ─────────────────── 5. Modelo MaskablePPO ───────────────────
# ──────────────────────────────────────────────────────────
# Configuración del agente MaskablePPO 
# (Proximal Policy Optimization con Action Masking)
# ──────────────────────────────────────────────────────────
model = MaskablePPO(
    # ── Política y entorno ──
    policy="MlpPolicy",        # Red MLP: capas densas que aprenden patrones de estado→acción
    env=env,                   # Entorno de entrenamiento con ActionMasker

    # ── Rollouts y actualización ──
    n_steps=4096,              # Pasos por rollout: mantén el mismo valor que funcionó
    batch_size=512,            # Tamaño de batch: mantén consistencia con PPO original

    # ── Ventajas y descuento ──
    gae_lambda=0.85,           # λ en GAE: mismo valor optimizado
    gamma=0.99,                # γ de descuento: mismo valor que funcionó

    # ── Tasa de aprendizaje ──
    learning_rate=1e-4,        # Mismo learning rate que demostró funcionar

    # ── Entropía para exploración ──
    ent_coef=0.05,             # Coeficiente de entropía: mismo valor
                               # En MaskablePPO esto es aún más importante porque 
                               # el masking puede reducir la exploración

    # ── Clipping específico de PPO ──
    clip_range=0.2,            # Rango de clipping para PPO (valor estándar)
    
    # ── Épocas de optimización ──
    n_epochs=10,               # Número de épocas por actualización (estándar)

    # ── Verbosidad y logging ──
    verbose=1,                 # Mostrar información clave en consola
    tensorboard_log=LOG_DIR,   # Directorio para logs de TensorBoard

    # ── Dispositivo de cómputo ──
    device=DEVICE,             # 'cpu' o 'cuda'

    # ── Reproducibilidad ──
    seed=SEED                  # Semilla para reproducibilidad
)

print("SB3 MaskablePPO model.device:", model.device)
tb_logger = configure(LOG_DIR, ["stdout", "tensorboard"])
model.set_logger(tb_logger)

# ─────────────────── 6. Callbacks (IDÉNTICOS) ───────────────────
csv_chkpt_cb = CsvAndCheckpointCallback(
    CSV_PATH,
    SAVE_EVERY_STEPS,
    CHECKPOINT_DIR,
    verbose=1
)

# EvalCallback para guardar el mejor modelo según recompensa media de evaluación
eval_env = make_eval_env()
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=BEST_DIR,
    log_path=LOG_DIR,
    eval_freq=SAVE_EVERY_STEPS,
    n_eval_episodes=5,
    deterministic=True,
    render=False
)

# ─────────────────── 7. Entrenamiento ───────────────────
try:
    print("🚀 Comenzando entrenamiento con MaskablePPO...")
    print("🎭 Action Masking habilitado para seguridad mejorada")
    start_time = time.time()

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[csv_chkpt_cb, eval_callback],
        tb_log_name=RUN_NAME
    )

    elapsed = time.time() - start_time
    print(f"✅ Entrenamiento finalizado en {elapsed/60:.1f} min")

    # guardar tiempo
    with open(os.path.join(LOG_DIR, "tiempo_entrenamiento.txt"), "w") as f:
        f.write(
            f"Tiempo total: {elapsed:.2f}s "
            f"({elapsed/60:.2f}min)\n"
            f"Algoritmo: MaskablePPO\n"
            f"Action Masking: Habilitado\n"
        )

except KeyboardInterrupt:
    print("\n⏸️ Interrumpido por usuario. Guardando modelo…")
    model.save(os.path.join(CHECKPOINT_DIR, f"ppo_mask_robot_{model.num_timesteps}"))

# ─────────────────── 8. Guardado final ───────────────────
model.save(os.path.join(CHECKPOINT_DIR, f"ppo_mask_robot_{model.num_timesteps}"))
print(f"🏁 Artefactos en: {LOG_DIR}")
print("📊 Logs de TensorBoard disponibles para comparar con PPO tradicional")

# intentar cerrar ROS2
try:
    rclpy.shutdown()
except RuntimeError:
    pass