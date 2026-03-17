# 🌊 Robot Autónomo de Limpieza de Playas

<p align="center">
  <img src="assets/robot.gif" width="600"/>
</p>

Sistema robótico autónomo diseñado para la limpieza eficiente de playas mediante navegación inteligente, percepción con sensores y algoritmos de aprendizaje por refuerzo.

---

## 🧠 Descripción

Este proyecto presenta el desarrollo de un robot móvil autónomo capaz de desplazarse en entornos tipo playa, detectar obstáculos y navegar hacia objetivos definidos de manera autónoma.

El sistema combina:
- 🧭 Navegación autónoma
- 🤖 Aprendizaje por Refuerzo (Reinforcement Learning)
- 📡 Sensado con LiDAR y odometría
- 🗺️ Simulación en entornos complejos

El objetivo es contribuir a la reducción de contaminación en playas mediante soluciones tecnológicas de bajo costo y alta eficiencia.

---

## ⚙️ Tecnologías utilizadas

- ROS 2 (Robot Operating System)
- Gazebo (simulación)
- Python
- PyTorch / Stable-Baselines3
- LiDAR
- Encoders (odometría)
- Control diferencial

---

## 🧩 Arquitectura del sistema

El sistema está dividido en dos grandes módulos:

### 🔹 1. Modelo del robot (ROS2)
Ubicado en:

```bash
src/robot_tesis_cuerpo/
```

Incluye:
- Modelado URDF/XACRO del robot
- Launch files para simulación
- Mundos de prueba en Gazebo
- Configuración de sensores

---

### 🔹 2. Inteligencia y entrenamiento (RL)
Ubicado en:

```bash
src/robot_tesis_rl/
```

Incluye:
- Entorno personalizado tipo Gym (`robot_env.py`)
- Implementación de PPO
- Scripts de entrenamiento
- Procesamiento de datos de sensores

---

## 🧠 Aprendizaje por Refuerzo

El robot utiliza algoritmos de Reinforcement Learning para aprender a:

- Evitar obstáculos
- Minimizar la distancia al objetivo
- Mantener orientación adecuada
- Navegar de forma eficiente

### Observaciones del agente:
- Datos de LiDAR (segmentados)
- Distancia al objetivo
- Error angular
- Velocidades del robot

### Acciones:
- Velocidad lineal
- Velocidad angular

---

## 🌍 Simulación

Se utilizan múltiples entornos en Gazebo para entrenar y evaluar el robot:

- Escenarios con obstáculos
- Terrenos irregulares
- Ambientes dinámicos
- Pruebas con y sin paredes

---

## ▶️ Cómo ejecutar

### 1. Clonar repositorio
```bash
git clone https://github.com/guille-robotics/Robot_de_Limpieza_de_Playas.git
cd Robot_de_Limpieza_de_Playas
```

### 2. Compilar workspace
```bash
colcon build
source install/setup.bash
```

### 3. Ejecutar simulación
```bash
ros2 launch robot_tesis_cuerpo one_robot_gz_launch.py
```

### 4. Entrenar modelo
```bash
python3 src/robot_tesis_rl/robot_tesis_rl/train_con_PPO_MASK.py
```

---

## 📊 Resultados esperados

- Navegación autónoma sin colisiones
- Adaptación a distintos entornos
- Optimización de trayectorias
- Transferencia sim2real (simulación → robot real)

---

## 🔬 Aplicaciones

- Limpieza automatizada de playas
- Robots de servicio autónomos
- Investigación en navegación autónoma
- Sistemas basados en inteligencia artificial aplicada

---

## 📌 Estado del proyecto

🚧 En desarrollo activo

---

## 👨‍💻 Autor

Guillermo Cid Ampuero  
Ingeniería Electrónica  
Robótica & Inteligencia Artificial

---


## 📄 Licencia

Este proyecto se distribuye bajo la licencia incluida en el repositorio.