# Clash Royale Engine

Motor de simulación headless de Clash Royale (Arena 1) optimizado para entrenamiento masivo de agentes de Reinforcement Learning.

Implementa 8 cartas específicas con física continua realista, sistema de combate con proyectiles, targeting oficial, interfaz Gymnasium completa, sistema de grabación y extracción de episodios para Imitation Learning, fog-of-war, pocket placement (colocación en lado enemigo tras destruir torres), y visualización GUI con Pygame.

---

## Tabla de Contenidos

- [Requisitos del Sistema](#requisitos-del-sistema)
- [Instalación](#instalación)
- [Guía de Inicio Rápido](#guía-de-inicio-rápido)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Cartas Implementadas](#cartas-implementadas)
- [Arquitectura del Motor](#arquitectura-del-motor)
- [Entorno Gymnasium](#entorno-gymnasium)
- [Fog-of-War](#fog-of-war)
- [Pocket Placement](#pocket-placement)
- [Grabación y Extracción de Episodios (IL)](#grabación-y-extracción-de-episodios-il)
- [Visualización GUI](#visualización-gui)
- [Ejemplos](#ejemplos)
- [Agentes PPO: Baseline vs CNN+LSTM](#agentes-ppo-baseline-vs-cnnlstm)
  - [PPO Baseline (MLP)](#1-ppo-baseline-mlp)
  - [PPO CNN+LSTM (Recurrente)](#2-ppo-cnnlstm-recurrente)
  - [Comparación directa](#comparación-directa)
- [Testing](#testing)
- [Compatibilidad con Google Colab](#compatibilidad-con-google-colab)
- [Licencia](#licencia)

---

## Requisitos del Sistema

### Sistema Operativo

| OS | Soporte | Notas |
|---|---|---|
| **Windows 10/11** | Completo | Desarrollo y entrenamiento local |
| **Ubuntu 20.04+** | Completo | Servidores, Colab |
| **macOS 12+** | Solo CPU | Sin aceleración CUDA |

### Python

- **Python 3.10 – 3.12** (recomendado: **3.12**)
- Compatible con Python 3.13, pero las dependencias de RL (`torch`, `stable-baselines3`) aún no tienen soporte oficial completo en 3.13.

### GPU / NVIDIA (para entrenamiento con RL)

> **Nota:** El motor de simulación en sí es **100 % CPU** y no requiere GPU. La GPU solo es necesaria para entrenar redes neuronales (PPO, A2C, etc.) con PyTorch.

#### Compatibilidad con Google Colab

Google Colab provee GPUs compartidas. Estas son las configuraciones típicas y la compatibilidad con el proyecto:

| Tipo de GPU (Colab) | VRAM | CUDA Compute | Driver NVIDIA | CUDA Toolkit | Compatible |
|---|---|---|---|---|---|
| **Tesla T4** (gratuito) | 16 GB | 7.5 | ≥ 525.x | 12.1 – 12.4 | **Sí** |
| **A100** (Colab Pro) | 40 GB | 8.0 | ≥ 525.x | 12.1 – 12.4 | **Sí** |
| **L4** (Colab Pro) | 24 GB | 8.9 | ≥ 525.x | 12.1 – 12.4 | **Sí** |
| **V100** (ocasional) | 16 GB | 7.0 | ≥ 525.x | 12.1 – 12.4 | **Sí** |

#### Versiones NVIDIA recomendadas para entorno local

| Componente | Versión Mínima | Versión Recomendada |
|---|---|---|
| **Driver NVIDIA** | 525.60+ | 550.x+ |
| **CUDA Toolkit** | 12.1 | 12.4 |
| **cuDNN** | 8.9 | 9.x |
| **PyTorch** | 2.1+ | 2.5+ (`pytorch-cuda=12.4`) |

#### Usuarios con GPU AMD (ROCm)

Si tienes GPU AMD, el motor de simulación funciona sin problemas ya que no usa GPU.
Para entrenamiento de RL con PyTorch en AMD, necesitarías [ROCm](https://rocm.docs.amd.com/) (solo Linux).
La alternativa recomendada es entrenar en **Google Colab** con GPU NVIDIA gratuita.

---

## Instalación

### 1. Instalar Miniforge (Conda)

Descargar e instalar [Miniforge](https://github.com/conda-forge/miniforge) para tu sistema operativo

### 2. Crear el entorno virtual

**Opción A — Desde `environment.yaml` (recomendado para entrenamiento con GPU NVIDIA):**

```bash
mamba env create -f environment.yaml
mamba activate cr-engine
```

Esto instala Python 3.12, PyTorch con CUDA 12.1 y el paquete en modo editable con dependencias de RL.

**Opción B — Entorno manual (CPU-only o GPU AMD):**

```bash
mamba create -n cr-engine python=3.12 -y
mamba activate cr-engine
pip install -e ".[dev]"
```

**Opción C — Con soporte de RL (sin CUDA, solo CPU):**

```bash
mamba create -n cr-engine python=3.12 -y
mamba activate cr-engine
pip install -e ".[dev,rl]"
```

### 3. Verificar la instalación

```bash
python -c "from clash_royale_engine import ClashRoyaleEnv; print('OK')"
python -m pytest tests/ -v
```

---

## Guía de Inicio Rápido

### Ejecutar una partida simulada

```python
from clash_royale_engine import ClashRoyaleEnv

env = ClashRoyaleEnv()
obs, info = env.reset()

done = False
while not done:
    action = env.action_space.sample()  # Acción aleatoria
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

print(f"Resultado: {'Victoria' if reward > 0 else 'Derrota' if reward < 0 else 'Empate'}")
```

### Usar el motor directamente (sin Gymnasium)

```python
from clash_royale_engine import ClashRoyaleEngine
from clash_royale_engine.players.player_interface import HeuristicBot

engine = ClashRoyaleEngine(
    player1=HeuristicBot(),
    player2=HeuristicBot(),
    fps=30,
    time_limit=180.0,
)

state_p0, state_p1, done = engine.step(frames=1)

print(f"Elixir P0: {state_p0.numbers.elixir:.1f}")
print(f"Torres P0: {state_p0.numbers.king_hp}")
print(f"Tiempo restante: {state_p0.numbers.time_remaining:.0f}s")
```

### Ambiente con reward shaping denso

```python
env = ClashRoyaleEnv(
    reward_shaping="dense",    # Rewards intermedios (daño a torres, etc.)
    time_limit=180.0,          # 3 minutos
    speed_multiplier=5.0,      # 5x más rápido
)
```

### Benchmark de rendimiento

```python
from clash_royale_engine import ClashRoyaleEnv
import time

env = ClashRoyaleEnv(time_limit=10.0, speed_multiplier=10.0)
start = time.time()

for _ in range(50):
    env.reset()
    done = False
    while not done:
        obs, r, te, tr, _ = env.step(env.action_space.sample())
        done = te or tr

elapsed = time.time() - start
print(f"{(50 / elapsed) * 3600:.0f} episodios/hora")
# Target: >1000 ep/h — resultado típico: ~2800 ep/h
```

---

## Estructura del Proyecto

```
cr-engine/
├── clash_royale_engine/
│   ├── __init__.py              # Exports públicos del paquete
│   ├── core/
│   │   ├── engine.py            # ClashRoyaleEngine — orquestador principal
│   │   ├── arena.py             # Contenedor de entidades, spawn de tropas/hechizos
│   │   ├── recorder.py          # Grabación de partidas y extracción de episodios IL
│   │   ├── scheduler.py         # Reloj de simulación, detección de overtime
│   │   └── state.py             # Dataclasses del estado (compatible BuildABot)
│   ├── entities/
│   │   ├── base_entity.py       # Clase base Entity + Projectile
│   │   ├── troops/
│   │   │   ├── giant.py         # Gigante (tanque, target: buildings)
│   │   │   ├── musketeer.py     # Mosquetera (ranged, target: all)
│   │   │   ├── archers.py       # Arqueras ×2 (ranged, target: all)
│   │   │   ├── mini_pekka.py    # Mini P.E.K.K.A (melee rápido, target: ground)
│   │   │   ├── knight.py        # Caballero (tanque melee, target: ground)
│   │   │   └── skeletons.py     # Esqueletos ×3 (swarm, target: ground)
│   │   ├── buildings/
│   │   │   ├── princess_tower.py  # Torres princesa (×2 por jugador)
│   │   │   └── king_tower.py     # Torre del rey (se activa con daño)
│   │   └── spells/
│   │       ├── arrows.py        # Flechas (daño de área grande)
│   │       └── fireball.py      # Bola de fuego (daño + knockback)
│   ├── systems/
│   │   ├── physics.py           # Movimiento continuo, colisiones circulares
│   │   ├── combat.py            # Ataques, proyectiles, aplicación de daño
│   │   ├── targeting.py         # Selección de objetivos (lógica oficial CR)
│   │   ├── pathfinding.py       # Navegación por puentes, restricción de río
│   │   └── elixir.py            # Generación y gasto de elixir
│   ├── players/
│   │   ├── player.py            # Deck, mano, ciclado de cartas
│   │   └── player_interface.py  # Interfaz abstracta + HeuristicBot, RLAgentPlayer
│   ├── env/
│   │   ├── gymnasium_env.py     # ClashRoyaleEnv (wrapper Gymnasium, fog-of-war, recording)
│   │   └── multi_agent_env.py   # MultiAgentEnv + VectorizedEnv
│   ├── utils/
│   │   ├── constants.py         # Constantes del juego (stats, coordenadas, velocidades)
│   │   ├── converters.py        # Conversiones tile ↔ pixel (BuildABot-compatible)
│   │   └── validators.py        # Validación de acciones, pocket placement
│   └── visualization/
│       └── renderer.py          # Visualización GUI con Pygame
│   models/
│   ├── __init__.py
│   ├── cnn_lstm_policy.py       # CNN + LSTM actor-critic (CnnLstmPolicy)
│   ├── recurrent_rollout_buffer.py  # Buffer con (h,c) y secuencias
│   └── recurrent_ppo.py        # Trainer PPO recurrente (mismo algoritmo)
├── examples/
│   ├── 01_headless_quickstart.py   # Partida headless bot vs bot
│   ├── 02_gymnasium_random_agent.py # Gymnasium con agente aleatorio
│   ├── 03_recording_and_il.py      # Grabación + 4 episodios IL por simetría
│   ├── 04_manual_placement.py      # Colocación manual de cartas
│   ├── 05_pocket_placement.py      # Demo pocket placement (carriles/torres)
│   ├── 06_state_inspection.py      # Inspección de estado y fog-of-war
│   ├── demo_gui.py                 # Demo de visualización GUI con Pygame
│   ├── train_ppo_baseline.py       # Entrenamiento PPO MLP (SB3)
│   └── train_ppo_cnn_lstm.py       # Entrenamiento PPO CNN+LSTM (custom)
├── tests/
│   ├── test_engine.py           # 69 tests (motor, física, combate, gym, IL, fog, pocket)
│   └── test_cnn_lstm.py         # 20 tests (shapes, hidden reset, buffer, GAE)
├── environment.yaml             # Entorno Mamba con PyTorch + CUDA
├── pyproject.toml               # Configuración del paquete y herramientas
└── AGENTS.md                    # Especificación completa del motor
```

---

## Cartas Implementadas

### Tropas

| Carta | Elixir | HP | Daño | Velocidad | Rango | Target | Cantidad |
|---|---|---|---|---|---|---|---|
| **Giant** | 5 | 2000 | 120 | Lento (45 px/s) | 1.0 | Buildings | 1 |
| **Musketeer** | 4 | 340 | 100 | Media (60 px/s) | 6.0 | All | 1 |
| **Archers** | 3 | 125 | 40 | Media (60 px/s) | 5.0 | All | 2 |
| **Mini P.E.K.K.A** | 4 | 600 | 325 | Rápida (90 px/s) | 1.2 | Ground | 1 |
| **Knight** | 3 | 600 | 75 | Media (60 px/s) | 1.2 | Ground | 1 |
| **Skeletons** | 1 | 32 | 32 | Rápida (90 px/s) | 1.0 | Ground | 3 |

### Hechizos

| Carta | Elixir | Daño | Radio | Daño a Torres | Efecto |
|---|---|---|---|---|---|
| **Arrows** | 3 | 115 | 4.0 tiles | 46 | Daño de área |
| **Fireball** | 4 | 325 | 2.5 tiles | 130 | Daño de área + knockback |

### Torres

| Torre | HP | Daño | Hit Speed | Rango |
|---|---|---|---|---|
| **Princess Tower** | 1400 | 50 | 0.8s | 7.5 tiles |
| **King Tower** | 2400 | 50 | 1.0s | 7.0 tiles |

> Stats de nivel 1, Arena 1. La King Tower comienza inactiva y se activa al recibir daño o cuando una tropa enemiga cruza el puente.

---

## Arquitectura del Motor

```
┌──────────────────────────────────────────────────────┐
│                  ClashRoyaleEngine                    │
│                                                      │
│   ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│   │ Scheduler│  │  Arena   │  │  ElixirSystem    │  │
│   │ (reloj)  │  │(entidades│  │  (generación,    │  │
│   │          │  │ towers)  │  │   gasto, cap)    │  │
│   └──────────┘  └──────────┘  └──────────────────┘  │
│                                                      │
│   ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│   │ Physics  │  │ Combat   │  │  Targeting       │  │
│   │(movimiento│ │(ataques, │  │  (selección de   │  │
│   │colisiones)│ │proyectil)│  │   objetivos)     │  │
│   └──────────┘  └──────────┘  └──────────────────┘  │
│                                                      │
│   ┌──────────────────────────────────────────────┐   │
│   │         Player 0          Player 1           │   │
│   │    (deck, hand, elixir) (deck, hand, elixir) │   │
│   └──────────────────────────────────────────────┘   │
└──────────────┬───────────────────────┬───────────────┘
               │                       │
      ┌────────▼────────┐    ┌────────▼────────┐
      │ PlayerInterface │    │ PlayerInterface │
      │  (HeuristicBot, │    │  (RLAgentPlayer,│
      │   RLAgent, etc.) │    │   HumanPlayer)  │
      └─────────────────┘    └─────────────────┘
```

### Ciclo de un frame

1. Obtener acciones de ambos jugadores
2. Validar y aplicar acciones (spawn tropas, lanzar hechizos)
3. Actualizar targeting (buscar objetivos válidos)
4. Actualizar física (movimiento, colisiones, pathfinding)
5. Procesar combate (ataques melee/ranged, proyectiles)
6. Generar elixir
7. Verificar activación de King Tower
8. Eliminar entidades muertas
9. Verificar condiciones de victoria
10. Avanzar reloj

---

## Entorno Gymnasium

### Espacio de Acciones

`Discrete(2305)` — 2304 combinaciones de colocación + 1 no-op:

- Índices `0–2303`: `(tile_x, tile_y, card_idx)` donde `tile_x ∈ [0,17]`, `tile_y ∈ [0,31]`, `card_idx ∈ [0,3]`
- Índice `2304`: no hacer nada (pasar turno)

### Espacio de Observaciones

`Box(0, 1, shape=(1196,), float32)`:

| Segmento | Dimensiones | Descripción |
|---|---|---|
| Elixir | 2 | Propio + enemigo (normalizado /10) |
| Torres HP | 6 | 3 propias + 3 enemigas (normalizado) |
| Cartas en mano | 36 | 4 × (one-hot 8 cartas + costo normalizado) |
| Grid aliados | 576 | 32 × 18 occupancy map |
| Grid enemigos | 576 | 32 × 18 occupancy map |

> Con `fog_of_war=True` (por defecto), el elixir enemigo (índice 1) se fija a 0.

### Reward Shaping

| Modo | Descripción |
|---|---|
| `"sparse"` | +1 victoria, −1 derrota, 0 empate (solo al final) |
| `"dense"` | Rewards intermedios: daño a torres (+), torres perdidas (−), acción inválida (−0.1), victoria (+10), derrota (−10) |

---

## Fog-of-War

El entorno soporta **fog-of-war** (`fog_of_war=True` por defecto) que oculta información privilegiada del oponente:

- El **elixir enemigo** se reporta como 0 en la observación.
- Todas las unidades enemigas visibles en el campo siguen siendo observables (no hay fog espacial).

```python
# Observación con información perfecta
env_god = ClashRoyaleEnv(fog_of_war=False)

# Observación parcial (por defecto, recomendado para entrenamiento)
env_fog = ClashRoyaleEnv(fog_of_war=True)
```

---

## Pocket Placement

Implementa la regla oficial de Clash Royale para colocación en el lado enemigo:

| Regla | Descripción |
|---|---|
| **Tropas en lado propio** | Siempre permitido |
| **Tropas en lado enemigo** | Bloqueado por defecto |
| **Hechizos** | Se pueden lanzar en cualquier parte del mapa |
| **Pocket izquierdo** | Se desbloquea al destruir la torre princesa izquierda enemiga |
| **Pocket derecho** | Se desbloquea al destruir la torre princesa derecha enemiga |

El **pocket** es un área de `POCKET_DEPTH=3` tiles de profundidad pasando el río, restringida al carril cuya torre princesa fue destruida:

- **Carril izquierdo:** `tile_x < 9`
- **Carril derecho:** `tile_x >= 9`
- **Profundidad P0:** `y = 17..19` (atacando hacia arriba)
- **Profundidad P1:** `y = 12..14` (atacando hacia abajo)

---

## Grabación y Extracción de Episodios (IL)

El sistema de **recording** permite grabar partidas completas y extraer episodios para **Imitation Learning**:

### Grabar una partida

```python
env = ClashRoyaleEnv(record=True, fog_of_war=True)
obs, info = env.reset()

# ... jugar episodio completo ...

env.reset()  # finaliza la grabación
record = env.get_game_record()
```

### Extraer 4 episodios por simetría

De cada partida grabada se generan **4 trayectorias** de entrenamiento:

| Episodio | Transformación | Propósito |
|---|---|---|
| P0 original | Ninguna | Trayectoria directa del jugador 0 |
| P1 y-flip | Volteo vertical | Normaliza perspectiva de P1 al fondo del mapa |
| P0 x-flip | Espejo horizontal | Data augmentation |
| P1 y+x-flip | Ambas | Combinación de normalización + augmentation |

```python
episodes = env.extract_il_episodes(record)  # Lista de 4 episodios

# Convertir a numpy batch
from clash_royale_engine.core.recorder import EpisodeExtractor
batch = EpisodeExtractor.episodes_to_numpy(episodes)
# batch["states"].shape     → (N, 1196)
# batch["actions"].shape    → (N,)
# batch["rewards"].shape    → (N,)
# batch["next_states"].shape → (N, 1196)
# batch["dones"].shape      → (N,)
```

---

## Visualización GUI

El motor incluye una GUI de visualización en tiempo real con **Pygame**:

```bash
python examples/demo_gui.py
```

Características:
- Renderizado de la arena 18×32 con tiles de color
- Torres (princesa y rey) con barras de HP
- Tropas con colores por tipo y bandos diferenciados
- Proyectiles en vuelo
- Barra de elixir en tiempo real
- Río y puentes
- Panel de información (frame, tiempo, elixir)

> Requiere `pygame` instalado (`pip install pygame`).

---

## Ejemplos

La carpeta `examples/` contiene scripts listos para ejecutar:

| Archivo | Descripción |
|---|---|
| `01_headless_quickstart.py` | Partida bot vs bot sin GUI, muestra estado cada 5s |
| `02_gymnasium_random_agent.py` | Entorno Gymnasium con agente aleatorio, sparse vs dense |
| `03_recording_and_il.py` | Grabación de partida + extracción de 4 episodios IL |
| `04_manual_placement.py` | Colocación manual de cartas con `step_with_action` |
| `05_pocket_placement.py` | Demo completo de pocket placement (carriles, torres, reglas) |
| `06_state_inspection.py` | Inspección del feature vector, fog-of-war, estructura interna |
| `demo_gui.py` | Visualización GUI con Pygame |

```bash
# Ejecutar cualquier ejemplo
python examples/01_headless_quickstart.py
python examples/02_gymnasium_random_agent.py
python examples/03_recording_and_il.py
python examples/04_manual_placement.py
python examples/05_pocket_placement.py
python examples/06_state_inspection.py
python examples/demo_gui.py  # requiere pygame
```

---

## Agentes PPO: Baseline vs CNN+LSTM

El proyecto incluye **dos variantes de PPO** que comparten exactamente el mismo entorno, la misma observación, las mismas seeds y los mismos hiperparámetros del algoritmo. La única diferencia es la **arquitectura de la red neuronal** y el manejo de la memoria temporal.

### Observación compartida

Ambos agentes reciben el mismo vector de observación de **1 196 dimensiones** (normalizado a `[0, 1]`):

| Rango | Dim | Contenido |
|---|---|---|
| `[0, 2)` | 2 | Elixir propio y enemigo (÷ 10) |
| `[2, 8)` | 6 | HP de las 6 torres (÷ HP máximo) |
| `[8, 44)` | 36 | Mano de 4 cartas: cada una = 8-dim one-hot + coste (÷ 10) |
| `[44, 620)` | 576 | Grid **aliado** 32×18 (presencia binaria por celda) |
| `[620, 1196)` | 576 | Grid **enemigo** 32×18 (presencia binaria por celda) |

> Con fog-of-war activado (por defecto), el elixir enemigo se reporta como 0.

### Espacio de acción

`Discrete(2305)` — 18 × 32 × 4 = 2 304 combinaciones de colocación `(tile_x, tile_y, card_idx)` + 1 acción "no hacer nada" (no-op).

---

### 1. PPO Baseline (MLP)

**Archivo:** `examples/train_ppo_baseline.py`

Usa [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) con la política `MlpPolicy` estándar.

#### Arquitectura

```
obs (1196) ─→ Linear(1196, 64) ─→ ReLU
             ─→ Linear(64, 64)   ─→ ReLU  ─→ π(a|s)  [Linear(64, 2305)]
                                           ─→ V(s)    [Linear(64, 1)]
```

- **2 capas ocultas** de 64 unidades (MLP por defecto de SB3).
- Redes de policy y value **comparten** las capas ocultas.
- Sin memoria: cada step es una decisión independiente (Markoviano).

#### Algoritmo PPO

| Parámetro | Valor por defecto |
|---|---|
| `learning_rate` | 3 × 10⁻⁴ |
| `n_steps` | 2 048 (por env) |
| `batch_size` | 64 |
| `n_epochs` | 10 |
| `gamma` | 0.99 |
| `gae_lambda` | 0.95 |
| `clip_range` | 0.2 |
| `ent_coef` | 0.01 |
| `vf_coef` | 0.5 |
| `max_grad_norm` | 0.5 |
| `n_envs` | 4 (DummyVecEnv) |

#### Ejecución

```bash
# Entrenamiento rápido (50k steps, ~10 min)
python examples/train_ppo_baseline.py

# Entrenamiento largo
python examples/train_ppo_baseline.py --timesteps 500000

# Sin fog-of-war (observación perfecta)
python examples/train_ppo_baseline.py --no-fog
```

Logs en `runs/ppo_baseline_<timestamp>/`.

---

### 2. PPO CNN+LSTM (Recurrente)

**Archivos:**
- `clash_royale_engine/models/cnn_lstm_policy.py` — Arquitectura
- `clash_royale_engine/models/recurrent_rollout_buffer.py` — Buffer recurrente
- `clash_royale_engine/models/recurrent_ppo.py` — Trainer PPO recurrente
- `examples/train_ppo_cnn_lstm.py` — Script de entrenamiento

#### Motivación

Clash Royale es un juego con **observabilidad parcial** (fog-of-war, elixir enemigo oculto) y **dependencias temporales** (el agente debe recordar dónde colocó tropas, rastrear el ciclo de cartas del oponente, etc.). Una LSTM permite al agente mantener un **estado interno** que resume la historia de observaciones pasadas.

#### Arquitectura

```
obs (1196) ──┬── scalars (44) ─→ MLP(44→64→64) ──────────────────→ 64-d
             │
             └── grid (1152)  ─→ reshape (2, 32, 18)
                               ─→ Conv2d(2→32, 3×3, s=1, pad=1) ─→ ReLU
                               ─→ Conv2d(32→64, 3×3, s=2, pad=1) ─→ ReLU
                               ─→ Conv2d(64→64, 3×3, s=2, pad=1) ─→ ReLU
                               ─→ Flatten ─→ Linear(→128) ───────→ 128-d

                          concat(128 + 64) = 192-d
                                     │
                                     ▼
                         LSTM(192→128, 1 capa)
                                     │
                              ┌──────┴──────┐
                              ▼              ▼
                       π(a|s,h)           V(s,h)
                     Linear(128→2305)   Linear(128→1)
```

**Componentes:**

| Módulo | Entrada | Salida | Descripción |
|---|---|---|---|
| `ScalarEncoder` | `(B, 44)` | `(B, 64)` | MLP de 2 capas para elixir, HP torres, mano |
| `CnnEncoder` | `(B, 2, 32, 18)` | `(B, 128)` | 3 capas Conv2d con stride progresivo |
| `LSTM` | `(T, B, 192)` | `(T, B, 128)` | 1 capa, hidden=128 |
| `policy_head` | `(T, B, 128)` | `(T, B, 2305)` | Logits de la distribución categórica |
| `value_head` | `(T, B, 128)` | `(T, B, 1)` | Estimación de V(s) |

#### Manejo del estado oculto (h, c)

1. **Durante rollout:** se mantiene un `(h, c)` por cada env paralelo. Cuando un episodio termina (`done=True`), el hidden se **resetea a cero** para ese env.
2. **Almacenamiento:** el buffer guarda `(h, c)` en cada step para poder reconstruir la secuencia durante entrenamiento.
3. **Entrenamiento:** los minibatches son **secuencias contiguas** de longitud `seq_len` (16 por defecto). El forward pass usa `done_mask` para resetear el hidden en los límites de episodio dentro de cada secuencia.

#### Rollout buffer recurrente

A diferencia del `RolloutBuffer` estándar de SB3 (que muestrea transiciones individuales), el `RecurrentRolloutBuffer`:

- Almacena datos en layout `(n_steps, n_envs, ...)`.
- Guarda `hidden_h` y `hidden_c` en cada step.
- GAE se calcula de forma **idéntica** al baseline (mismo `gamma`, `gae_lambda`).
- El generador de minibatches parte el rollout en chunks de `seq_len` y los baraja entre envs.

```
Rollout (n_steps=2048, n_envs=4):
├── Chunk 0: steps [0, 16)   env 0
├── Chunk 1: steps [0, 16)   env 1
├── Chunk 2: steps [0, 16)   env 2
├── ...
├── Chunk K: steps [2032, 2048) env 3
└── → Shuffle → agrupar en minibatches
```

#### Algoritmo PPO (idéntico)

Los hiperparámetros de PPO son **exactamente los mismos** que el baseline, sin ningún cambio en:

- ✅ GAE (Generalized Advantage Estimation)
- ✅ Clipped surrogate objective
- ✅ Value function MSE loss (sin clipping de V)
- ✅ Entropy bonus
- ✅ Adam optimiser con gradient clipping
- ✅ Normalización de ventajas por minibatch

| Parámetro extra (CNN+LSTM) | Valor por defecto |
|---|---|
| `seq_len` | 16 |
| `lstm_hidden` | 128 |
| `lstm_layers` | 1 |

#### Ejecución

```bash
# Entrenamiento rápido (50k steps)
python examples/train_ppo_cnn_lstm.py

# Entrenamiento largo con secuencia de 32
python examples/train_ppo_cnn_lstm.py --timesteps 500000 --seq-len 32

# LSTM más grande
python examples/train_ppo_cnn_lstm.py --lstm-hidden 256
```

Logs en `runs/ppo_cnn_lstm_<timestamp>/`.

---

### Comparación directa

Ambos scripts aceptan los **mismos flags CLI** para hiperparámetros PPO, por lo que se pueden comparar de forma justa:

```bash
# Entrenar ambos con misma configuración
python examples/train_ppo_baseline.py  --timesteps 200000 --seed 42
python examples/train_ppo_cnn_lstm.py  --timesteps 200000 --seed 42

# Visualizar en TensorBoard
tensorboard --logdir runs/
```

Métricas disponibles en TensorBoard:

| Métrica | Baseline | CNN+LSTM |
|---|---|---|
| `rollout/ep_rew_mean` | ✅ | ✅ |
| `rollout/ep_len_mean` | ✅ | ✅ |
| `train/pg_loss` | ✅ | ✅ |
| `train/vf_loss` | ✅ | ✅ |
| `train/entropy` | ✅ | ✅ |
| `game/win_rate` | ✅ | — (usar eval) |
| `game/valid_action_pct` | ✅ | — (usar eval) |
| `time/fps` | ✅ | ✅ |

Ambos scripts escriben un `eval_summary.txt` al final con win rate, reward medio y longitud de episodio.

#### Resumen de diferencias

| Aspecto | PPO Baseline (MLP) | PPO CNN+LSTM |
|---|---|---|
| Framework | Stable-Baselines3 | Custom (PyTorch puro) |
| Arquitectura | MLP 64→64 | CNN + MLP → LSTM(128) |
| Memoria temporal | ❌ | ✅ (LSTM carry h,c) |
| Observación | Flat vector (1196) | Misma, pero split en grid + scalars |
| Info nueva | Ninguna | Ninguna (mismos features) |
| Minibatch | Transiciones individuales | Secuencias de T=16 |
| Buffer | SB3 RolloutBuffer | RecurrentRolloutBuffer |
| Parámetros (aprox.) | ~160K | ~350K |
| Ventaja esperada | Simple, rápido | Mejor en observabilidad parcial |

---

### Entrenamiento con Stable-Baselines3 (snippet rápido)

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from clash_royale_engine import ClashRoyaleEnv

def make_env():
    def _init():
        return ClashRoyaleEnv(
            reward_shaping="dense",
            speed_multiplier=5.0,
            time_limit=180.0,
        )
    return _init

# 16 ambientes en paralelo
vec_env = SubprocVecEnv([make_env() for _ in range(16)])

model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=3e-4)
model.learn(total_timesteps=1_000_000)
model.save("clash_royale_ppo")
```

---

## Testing

```bash
# Ejecutar todos los tests
python -m pytest tests/ -v

# Solo tests rápidos (sin benchmark)
python -m pytest tests/ -v -k "not benchmark"

# Con coverage
python -m pytest tests/ --cov=clash_royale_engine --cov-report=term-missing
```

Los **89 tests** cubren:
- Inicialización y reset del motor
- Generación, cap y gasto de elixir
- Spawn de cartas (×1, ×2, ×3 según la carta)
- Sistema de combate y muerte de entidades
- Destrucción de torres y detección de victoria
- Validación de acciones (bounds, lado enemigo, elixir)
- Física (velocidades, separación por colisión)
- Conversión de coordenadas tile ↔ pixel
- Restricción de río (tropas no cruzan, hechizos sí)
- Interfaz Gymnasium completa (obs/action spaces, no-op, terminación, rewards)
- Ambiente vectorizado
- **Fog-of-war** (ocultación de elixir enemigo)
- **Action encoding/decoding** (codificación bidireccional de acciones)
- **State transforms** (y-flip, x-flip para normalización de perspectiva)
- **Recording** (grabación de frames, transiciones, GameRecord)
- **Extracción de IL** (4 episodios por simetría, conversión a numpy)
- **Pocket placement** (13 tests: restricción de lado, desbloqueo por torre, carriles, profundidad, hechizos libres, ambos jugadores)
- Benchmark de rendimiento (~2800 ep/hora)
- **CNN+LSTM policy** (20 tests: shapes de obs split, encoders, forward/act/evaluate, hidden reset on done, rollout buffer add/GAE/generator/reset)

---

## Compatibilidad con Google Colab

El motor está diseñado para ejecutarse directamente en Google Colab sin modificaciones.

### Setup en Colab (celda de instalación)

```python
!git clone https://github.com/<tu-usuario>/cr-engine.git
%cd cr-engine
!pip install -e ".[rl]"
```

### Entornos de Colab

| Entorno Colab | Python | CUDA | PyTorch | Funciona |
|---|---|---|---|---|
| **CPU runtime** | 3.10/3.11 | — | CPU | Sí (solo simulación) |
| **GPU T4** (free) | 3.10/3.11 | 12.1 | 2.x+cu121 | **Sí (recomendado)** |
| **GPU A100** (Pro) | 3.10/3.11 | 12.1 | 2.x+cu121 | Sí |

### Verificar GPU en Colab

```python
import torch
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

> **Para usuarios con GPU AMD local:** el motor de simulación funciona en CPU sin problemas. Para entrenar redes neuronales con GPU, se recomienda usar Google Colab con GPU T4 gratuita o configurar [ROCm](https://rocm.docs.amd.com/) en Linux.

---

## Licencia

MIT