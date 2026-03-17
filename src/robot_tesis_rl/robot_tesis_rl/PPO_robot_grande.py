#!/usr/bin/env python3
# ────────────────────────────────────────────────────────────────
# train_robot_rl_with_checkpoints.py
# ────────────────────────────────────────────────────────────────
#Este codigo lo tire a correr el 9 de Mayo 
import os
import csv
import time
import random
from datetime import datetime
from collections import deque

import numpy as np
import gymnasium as gym
import torch  

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    StopTrainingOnMaxEpisodes,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure

import rclpy                    # ROS 2 Python API
import robot_env                # registra RobotSimulacion-v0

# ─────────────────── 0. Reproducibilidad ───────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
set_random_seed(SEED)

# ─────────────────── 1. Parámetros generales ───────────────────
TOTAL_TIMESTEPS    = 500_000
SAVE_EVERY_STEPS   = 200_000
#MAX_EPISODES       = 50_000

RUN_NAME           = datetime.now().strftime("rl_run_%Y%m%d_%H%M")
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

# ─────────────────── 2. Callback para CSV & Checkpoints ───────────────────
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
                f"ppo_robot_{self.num_timesteps}"
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


# ─────────────────── 3. Entorno ───────────────────
def make_env():
    try:
        rclpy.init(args=None)
    except RuntimeError:
        pass
    env = gym.make("RobotSimulacion-v0", step_duration=0.15)
    return Monitor(env)

env = make_env()

# ─────────────────── 4. Modelo PPO ───────────────────
# ──────────────────────────────────────────────────────────
# Configuración del agente PPO (Proximal Policy Optimization)
# ──────────────────────────────────────────────────────────
model = PPO(
    # ── Política y entorno ──
    policy="MlpPolicy",        # Red MLP: capas densas que aprenden patrones de estado→acción
    env=env,                   # Entorno de entrenamiento: RobotSimulacion

    # ── Rollouts y actualización ──
    n_steps=4096,               # Agrega 800 interacciones antes de cada actualización:
                               # equivale a un episodio completo, para aprender del ciclo entero.
    batch_size=512,            # Divide esas 800 muestras en batches de 200 cada uno:
                               # 4 mini-batches por paso de optimización, balance entre estabilidad y rapidez.

    # ── Ventajas y descuento ──
    gae_lambda=0.85,           # λ en GAE: 0.90 suaviza el cálculo de ventajas,
                               # parecido a “mirar un poco hacia adelante” sin perder estabilidad.
    gamma=0.99,                # γ de descuento: 0.95 hace que el agente valore
                               # recompensas a corto plazo más que las muy lejanas.

    # ── Tasa de aprendizaje ──
    learning_rate=1e-4,        # Paso de actualización de pesos:
                               # 1e-4 es pequeño para ajustes finos (evita saltos bruscos),
                               # pero suficiente para que el robot “aprenda a buen ritmo” sin desestabilizarse.

    # ── Entropía para exploración ──
    ent_coef=0.05,             # Coeficiente de entropía:
                               # 0.01 equivale a “curiosidad moderada”,
                               # anima al agente a probar acciones nuevas sin caer en aleatoriedad total.

    # ── Verbosidad y logging ──
    verbose=1,                 # 1: muestra información clave en consola (comienzo/fin de actualizaciones).
    tensorboard_log=LOG_DIR,   # Directorio para logs de TensorBoard:
                               # puedes ver curvas de pérdida, recompensa y métricas de entrenamiento.

    # ── Dispositivo de cómputo ──
    device=DEVICE,             # 'cpu' o 'cuda': elije GPU si quieres entrenar más rápido,
                               # de lo contrario CPU es suficiente para pruebas pequeñas.

    # ── Reproducibilidad ──
    seed=SEED                  # Semilla global para PyTorch, NumPy y el entorno,
                               # garantiza que repitas experimentos con resultados idénticos.
)

print("SB3 model.device:", model.device)
tb_logger = configure(LOG_DIR, ["stdout", "tensorboard"])
model.set_logger(tb_logger)

# ─────────────────── 5. Callbacks ───────────────────
csv_chkpt_cb = CsvAndCheckpointCallback(
    CSV_PATH,
    SAVE_EVERY_STEPS,
    CHECKPOINT_DIR,
    verbose=1
)
#stop_cb = StopTrainingOnMaxEpisodes(max_episodes=MAX_EPISODES, verbose=1)

# EvalCallback para guardar el mejor modelo según recompensa media de evaluación
eval_env = make_env()
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=BEST_DIR,
    log_path=LOG_DIR,
    eval_freq=SAVE_EVERY_STEPS,
    n_eval_episodes=5,
    deterministic=True,
    render=False
)

# ─────────────────── 6. Entrenamiento ───────────────────
try:
    print("🚀 Comenzando entrenamiento...")
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
        )

except KeyboardInterrupt:
    print("\n⏸️ Interrumpido por usuario. Guardando modelo…")
    model.save(os.path.join(CHECKPOINT_DIR, f"ppo_robot_{model.num_timesteps}"))

# ─────────────────── 7. Guardado final ───────────────────
model.save(os.path.join(CHECKPOINT_DIR, f"ppo_robot_{model.num_timesteps}"))
print(f"🏁 Artefactos en: {LOG_DIR}")

# intentar cerrar ROS2
try:
    rclpy.shutdown()
except RuntimeError:
    pass
