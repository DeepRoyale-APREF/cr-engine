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
- [Entrenamiento con Stable-Baselines3](#entrenamiento-con-stable-baselines3)
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
├── examples/
│   ├── 01_headless_quickstart.py   # Partida headless bot vs bot
│   ├── 02_gymnasium_random_agent.py # Gymnasium con agente aleatorio
│   ├── 03_recording_and_il.py      # Grabación + 4 episodios IL por simetría
│   ├── 04_manual_placement.py      # Colocación manual de cartas
│   ├── 05_pocket_placement.py      # Demo pocket placement (carriles/torres)
│   ├── 06_state_inspection.py      # Inspección de estado y fog-of-war
│   └── demo_gui.py                 # Demo de visualización GUI con Pygame
├── tests/
│   └── test_engine.py           # 69 tests (motor, física, combate, gym, IL, fog, pocket)
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

## Entrenamiento con Stable-Baselines3

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

Los **69 tests** cubren:
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