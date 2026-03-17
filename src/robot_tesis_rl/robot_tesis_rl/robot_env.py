"""
robot_env.py
Registra el entorno RobotSimulacion‑v0 para usarlo con gym.make().
Se ejecuta una sola vez, cuando el módulo se importa.
En este codigo es donde se agrega el ambiente con el cual se realizara el entrenamiento
"""

import gymnasium as gym
from gymnasium.envs.registration import register

# ── REGISTRO SEGURO ────────────────────────────────────────────
# Gym lanza un Error si se intenta registrar el mismo id dos veces
try:
    register(
        id="RobotSimulacion-v0",
        entry_point="ambiente_yaw_arreglado:RobotSimulacion",  # <nombre_del_codigo>:<clase> aca es donde se coloca el nombre del codigo que se trabajara
        max_episode_steps=None      # dejamos que el propio env trunque
    )
except gym.error.Error:
    # Ya estaba registrado en esta sesión; lo ignoramos
    pass

