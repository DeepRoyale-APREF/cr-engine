# Motor de Simulación Completo de Clash Royale (Arena 1)

## OBJETIVO PRINCIPAL
Construir un motor de simulación completo de Clash Royale (Arena 1) enfocado en 8 cartas específicas, con física realista, arquitectura modular y optimizado para entrenamiento de agentes de RL a escala masiva.

---

## CONTEXTO Y ARQUITECTURA BASE

### Repositorios de Referencia
- **MSU-AI/clash-royale-gym**: Framework base incompleto con estructura Gymnasium
- **Pbatch/ClashRoyaleBuildABot**: Sistema de detección y formato de estado

### Stack Tecnológico
- Python 3.12.12 (compatible con Google Colab)
- Gymnasium (compatibilidad OpenAI Gym)
- NumPy para cálculos numéricos
- Dataclasses para estructuras
- Type hints obligatorios
- Sin GUI en primera fase (headless)

---

## PARTE 1: DISEÑO DE ARQUITECTURA MODULAR

### 1.1 Estructura de Directorios

```
clash_royale_engine/
├── core/
│   ├── engine.py           # GameEngine principal
│   ├── arena.py            # Arena con grid y física
│   ├── scheduler.py        # Manejo de tiempo y frames
│   └── state.py            # Definición de State compatible con BuildABot
├── entities/
│   ├── base_entity.py      # Clase base Entity
│   ├── troops/
│   │   ├── giant.py
│   │   ├── musketeer.py
│   │   ├── archers.py
│   │   ├── mini_pekka.py
│   │   ├── knight.py
│   │   └── skeletons.py
│   ├── buildings/
│   │   ├── princess_tower.py
│   │   └── king_tower.py
│   └── spells/
│       ├── arrows.py
│       └── fireball.py
├── systems/
│   ├── physics.py          # Motor de física continua
│   ├── combat.py           # Sistema de combate
│   ├── targeting.py        # Sistema de selección de objetivos
│   ├── pathfinding.py      # Navegación y colisiones
│   └── elixir.py           # Gestión de elixir
├── players/
│   ├── player.py           # Gestión de mano, deck, elixir
│   └── player_interface.py # Interfaz agnóstica (Agent/Human)
├── env/
│   ├── gymnasium_env.py    # Wrapper Gymnasium
│   └── multi_agent_env.py  # Soporte multi-agente
├── utils/
│   ├── constants.py        # Constantes del juego
│   ├── converters.py       # Conversiones tile<->pixel
│   └── validators.py       # Validación de acciones
└── visualization/ (Fase 2)
    └── renderer.py         # GUI opcional
```

---

## PARTE 2: ESPECIFICACIÓN DEL SISTEMA DE ESTADO

### 2.1 Formato de Estado Compatible con BuildABot

```python
from dataclasses import dataclass
from typing import List, Tuple
from enum import Enum

# EXACTAMENTE como en BuildABot
@dataclass
class Position:
    bbox: Tuple[int, int, int, int]  # (left, top, right, bottom) en pixels
    conf: float                       # Confidence (1.0 para simulador)
    tile_x: int                       # Posición en grid [0-17]
    tile_y: int                       # Posición en grid [0-31]

@dataclass
class Unit:
    name: str                         # "giant", "musketeer", etc.
    category: str                     # "troop" | "building"
    target: str                        # "all" | "ground" | "buildings"
    transport: str                      # "ground" | "air"

@dataclass
class UnitDetection:
    unit: Unit
    position: Position

@dataclass
class Numbers:
    elixir: float                              # [0-10]
    enemy_elixir: float                        # Estimado o perfecto según config
    left_princess_hp: float                    # HP aliada
    right_princess_hp: float                   # HP aliada
    king_hp: float                             # HP aliada
    left_enemy_princess_hp: float              # HP enemiga
    right_enemy_princess_hp: float             # HP enemiga
    enemy_king_hp: float                       # HP enemiga
    time_remaining: float                      # Segundos restantes

@dataclass
class Card:
    name: str
    is_spell: bool
    cost: int
    units: List[Unit]
    
@dataclass
class State:
    """Estado compatible con BuildABot"""
    allies: List[UnitDetection]                # Unidades del jugador activo
    enemies: List[UnitDetection]               # Unidades del oponente
    numbers: Numbers
    cards: Tuple[Card, Card, Card, Card]      # Mano actual
    ready: List[int]                           # Índices de cartas jugables
    screen: str = "battle"                    # Siempre "battle" en simulación
```

### 2.2 Sistema de Coordenadas

```python
# Configuración EXACTA de BuildABot
DISPLAY_WIDTH = 720
DISPLAY_HEIGHT = 1280
TILE_WIDTH = 34.0
TILE_HEIGHT = 27.6
N_WIDE_TILES = 18
N_HEIGHT_TILES = 32  # Incluye zona no jugable
PLAYABLE_HEIGHT = 15  # Zona donde se pueden colocar cartas por jugador
TILE_INIT_X = 52
TILE_INIT_Y = 188

# Conversiones requeridas
def pixel_to_tile(x_pixel: float, y_pixel: float) -> Tuple[int, int]:
    """Convierte coordenadas pixel a tile"""
    tile_x = round(((x_pixel - TILE_INIT_X) / TILE_WIDTH) - 0.5)
    tile_y = round(((DISPLAY_HEIGHT - TILE_INIT_Y - y_pixel) / TILE_HEIGHT) - 0.5)
    return tile_x, tile_y

def tile_to_pixel(tile_x: int, tile_y: int) -> Tuple[float, float]:
    """Convierte tile a coordenadas pixel (centro del tile)"""
    x_pixel = TILE_INIT_X + (tile_x + 0.5) * TILE_WIDTH
    y_pixel = DISPLAY_HEIGHT - TILE_INIT_Y - (tile_y + 0.5) * TILE_HEIGHT
    return x_pixel, y_pixel
```

---

## PARTE 3: ESPECIFICACIÓN DE LAS 8 CARTAS (Arena 1)

### 3.1 Stats Oficiales de Clash Royale (Nivel 1)

```python
GIANT = {
    "name": "giant",
    "elixir": 5,
    "hp": 2000,
    "damage": 120,
    "hit_speed": 1.5,        # segundos entre ataques
    "speed": 1.0,            # "slow" = 45 pixels/segundo
    "range": 1.0,            # melee (1 tile)
    "sight_range": 5.5,      # tiles
    "target": "buildings",
    "deploy_time": 1.0,      # segundos
    "count": 1,
    "hitbox_radius": 0.9     # tiles
}

MUSKETEER = {
    "name": "musketeer",
    "elixir": 4,
    "hp": 340,
    "damage": 100,
    "hit_speed": 1.0,
    "speed": 1.4,            # "medium" = 60 pixels/segundo
    "range": 6.0,            # tiles
    "sight_range": 6.0,
    "target": "all",         # air + ground
    "deploy_time": 1.0,
    "count": 1,
    "hitbox_radius": 0.6,
    "projectile_speed": 1000  # pixels/segundo
}

ARCHERS = {
    "name": "archers",
    "elixir": 3,
    "hp": 125,               # cada arquera
    "damage": 40,            # cada arquera
    "hit_speed": 1.2,
    "speed": 1.4,            # medium
    "range": 5.0,
    "sight_range": 5.5,
    "target": "all",
    "deploy_time": 1.0,
    "count": 2,              # IMPORTANTE: spawnean 2 unidades
    "spawn_offset": 1.0,     # separación entre arqueras (tiles)
    "hitbox_radius": 0.5,
    "projectile_speed": 800
}

MINI_PEKKA = {
    "name": "mini_pekka",
    "elixir": 4,
    "hp": 600,
    "damage": 325,           # alto daño
    "hit_speed": 1.8,
    "speed": 2.1,            # "fast" = 90 pixels/segundo
    "range": 1.2,            # melee extendido
    "sight_range": 5.5,
    "target": "ground",
    "deploy_time": 1.0,
    "count": 1,
    "hitbox_radius": 0.7
}

KNIGHT = {
    "name": "knight",
    "elixir": 3,
    "hp": 600,
    "damage": 75,
    "hit_speed": 1.2,
    "speed": 1.4,            # medium
    "range": 1.2,
    "sight_range": 5.5,
    "target": "ground",
    "deploy_time": 1.0,
    "count": 1,
    "hitbox_radius": 0.7
}

SKELETONS = {
    "name": "skeletons",
    "elixir": 1,
    "hp": 32,                # cada esqueleto
    "damage": 32,
    "hit_speed": 1.0,
    "speed": 2.1,            # fast
    "range": 1.0,
    "sight_range": 5.5,
    "target": "ground",
    "deploy_time": 1.0,
    "count": 3,              # IMPORTANTE: spawnean 3 unidades
    "spawn_pattern": "triangle",  # formación inicial
    "hitbox_radius": 0.4,
    "is_swarm": True         # comportamiento especial
}
```

### HECHIZOS

```python
ARROWS = {
    "name": "arrows",
    "elixir": 3,
    "damage": 115,
    "radius": 4.0,           # tiles de radio
    "is_spell": True,
    "target": "all",
    "deploy_time": 0.0,      # instantáneo
    "damage_type": "area",
    "crown_tower_damage": 46  # 40% del daño a torres
}

FIREBALL = {
    "name": "fireball",
    "elixir": 4,
    "damage": 325,
    "radius": 2.5,
    "knockback": 2.0,        # tiles de empuje
    "is_spell": True,
    "target": "all",
    "deploy_time": 0.0,
    "damage_type": "area",
    "crown_tower_damage": 130
}
```

### EDIFICIOS (Torres)

```python
PRINCESS_TOWER = {
    "name": "princess_tower",
    "hp": 1400,              # Arena 1
    "damage": 50,
    "hit_speed": 0.8,
    "range": 7.5,            # tiles
    "sight_range": 7.5,
    "target": "all",
    "is_building": True,
    "position_left": (3, 1),   # tile_x, tile_y
    "position_right": (14, 1),
    "hitbox_radius": 1.5
}

KING_TOWER = {
    "name": "king_tower",
    "hp": 2400,
    "damage": 50,
    "hit_speed": 1.0,
    "range": 7.0,
    "sight_range": 7.0,
    "target": "all",
    "is_building": True,
    "position": (8.5, 0),
    "hitbox_radius": 2.0,
    "starts_active": False    # se activa si recibe daño o unidad cruza puente
}
```

---

## PARTE 4: MOTOR DE FÍSICA REALISTA

### 4.1 Sistema de Movimiento

```python
class PhysicsEngine:
    """
    Motor de física con:
    - Movimiento continuo (no discreto)
    - Colisiones circulares entre unidades
    - Pathfinding con vector fields
    - Push (empuje) de hechizos
    """
    
    def __init__(self, fps: int = 30):
        self.fps = fps
        self.dt = 1.0 / fps  # delta time
        
    def update_positions(self, entities: List[Entity], delta_frames: int = 1):
        """
        Actualiza posiciones usando física realista:
        1. Calcular velocidades deseadas (hacia target)
        2. Resolver colisiones
        3. Aplicar movimiento
        """
        dt = self.dt * delta_frames
        
        for entity in entities:
            if entity.is_static or entity.target is None:
                continue
                
            # Calcular dirección hacia objetivo
            desired_velocity = self._calculate_desired_velocity(entity)
            
            # Resolver colisiones con otras entidades
            final_velocity = self._resolve_collisions(entity, entities, desired_velocity)
            
            # Actualizar posición
            entity.x += final_velocity[0] * dt
            entity.y += final_velocity[1] * dt
            
            # Clamp a bounds de arena
            entity.x = np.clip(entity.x, 0, DISPLAY_WIDTH)
            entity.y = np.clip(entity.y, 0, DISPLAY_HEIGHT)
    
    def _resolve_collisions(self, entity, all_entities, desired_velocity):
        """
        Separación de círculos para evitar overlap
        Implementar algoritmo de separation steering
        """
        separation_force = np.array([0.0, 0.0])
        
        for other in all_entities:
            if other is entity:
                continue
                
            distance = self._distance(entity, other)
            min_distance = entity.hitbox_radius + other.hitbox_radius
            
            if distance < min_distance and distance > 0:
                # Calcular fuerza de separación
                direction = self._direction_vector(other, entity)
                overlap = min_distance - distance
                separation_force += direction * overlap * 2.0
        
        # Combinar velocidad deseada + separación
        return desired_velocity + separation_force
```

### 4.2 Sistema de Targeting

```python
class TargetingSystem:
    """
    Implementa la lógica oficial de Clash Royale:
    1. Buscar enemigos en sight_range
    2. Priorizar según tipo de unidad (buildings first, etc.)
    3. Si hay múltiples, elegir el más cercano
    4. Mantener target hasta que muera o salga de rango
    """
    
    def find_target(self, entity: Entity, potential_targets: List[Entity]) -> Optional[Entity]:
        # Filtrar por rango de visión
        in_range = [t for t in potential_targets 
                    if self._distance(entity, t) <= entity.sight_range]
        
        if not in_range:
            return None
        
        # Filtrar por tipo de target permitido
        valid_targets = self._filter_by_target_type(entity, in_range)
        
        if not valid_targets:
            return None
        
        # Priorizar edificios si la unidad solo ataca buildings
        if entity.target_type == "buildings":
            buildings = [t for t in valid_targets if t.is_building]
            if buildings:
                return min(buildings, key=lambda t: self._distance(entity, t))
        
        # Retornar el más cercano
        return min(valid_targets, key=lambda t: self._distance(entity, t))
```

### 4.3 Sistema de Combate

```python
class CombatSystem:
    """
    Maneja ataques, proyectiles y daño
    """
    
    def __init__(self):
        self.active_projectiles: List[Projectile] = []
    
    def process_attacks(self, entities: List[Entity], current_frame: int):
        for entity in entities:
            if entity.target is None:
                continue
            
            # Check si puede atacar (cooldown terminado)
            if current_frame >= entity.next_attack_frame:
                
                # Check si target está en rango de ataque
                distance = self._distance(entity, entity.target)
                if distance <= entity.attack_range:
                    self._execute_attack(entity, current_frame)
    
    def _execute_attack(self, attacker: Entity, current_frame: int):
        if attacker.has_projectile:
            # Crear proyectil
            projectile = Projectile(
                source=attacker,
                target=attacker.target,
                damage=attacker.damage,
                speed=attacker.projectile_speed
            )
            self.active_projectiles.append(projectile)
        else:
            # Daño instantáneo (melee)
            self._apply_damage(attacker.target, attacker.damage)
        
        # Set next attack time
        delay_frames = int(attacker.hit_speed * self.fps)
        attacker.next_attack_frame = current_frame + delay_frames
    
    def update_projectiles(self, dt: float):
        """Actualizar trayectorias de proyectiles"""
        for proj in self.active_projectiles[:]:
            # Mover hacia target
            proj.update_position(dt)
            
            # Check si impactó
            if proj.has_reached_target():
                self._apply_damage(proj.target, proj.damage)
                self.active_projectiles.remove(proj)
            
            # Check si target murió
            if proj.target.is_dead:
                self.active_projectiles.remove(proj)
```

---

## PARTE 5: INTERFAZ AGNÓSTICA PARA JUGADORES

### 5.1 Clase PlayerInterface

```python
from abc import ABC, abstractmethod

class PlayerInterface(ABC):
    """
    Interfaz abstracta que permite:
    - Agentes de RL
    - Humanos (input manual)
    - Bots heurísticos
    - Cualquier futura implementación
    """
    
    @abstractmethod
    def get_action(self, state: State) -> Optional[Tuple[int, int, int]]:
        """
        Recibe el estado actual y retorna acción.
        
        Returns:
            None: no hacer nada (pasar turno)
            (tile_x, tile_y, card_index): jugar carta
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Llamado al inicio de cada episodio"""
        pass

class RLAgentPlayer(PlayerInterface):
    """Wrapper para agentes de RL (Stable-Baselines3, RLlib, etc.)"""
    
    def __init__(self, model, policy_type: str = "stochastic"):
        self.model = model
        self.policy_type = policy_type
    
    def get_action(self, state: State) -> Optional[Tuple[int, int, int]]:
        # Convertir State a observación del agente
        obs = self._state_to_obs(state)
        
        # Obtener acción del modelo
        action, _ = self.model.predict(obs, deterministic=(self.policy_type == "deterministic"))
        
        # Convertir acción plana a (x, y, card_idx)
        return self._decode_action(action)
    
    def reset(self):
        pass  # Los modelos de RL no necesitan reset interno

class HumanPlayer(PlayerInterface):
    """Para juego manual o debugging"""
    
    def __init__(self, input_queue):
        self.input_queue = input_queue
    
    def get_action(self, state: State) -> Optional[Tuple[int, int, int]]:
        if not self.input_queue.empty():
            return self.input_queue.get()
        return None
    
    def reset(self):
        while not self.input_queue.empty():
            self.input_queue.get()

class HeuristicBot(PlayerInterface):
    """Bot simple basado en reglas (para testing)"""
    
    def get_action(self, state: State) -> Optional[Tuple[int, int, int]]:
        # Implementar lógica simple tipo BuildABot
        # Ej: si elixir == 10, jugar Giant en tile estratégico
        pass
```

### 5.2 GameEngine con Soporte Multi-Player

```python
class ClashRoyaleEngine:
    """
    Motor de juego agnóstico al tipo de jugadores
    """
    
    def __init__(
        self,
        player1: PlayerInterface,
        player2: PlayerInterface,
        deck1: List[str],
        deck2: List[str],
        fps: int = 30,
        time_limit: float = 180.0,  # 3 minutos
        speed_multiplier: float = 1.0  # Para acelerar simulación
    ):
        self.player1 = player1
        self.player2 = player2
        self.speed_multiplier = speed_multiplier
        # ... resto de inicialización
    
    def step(self, frames: int = 1) -> Tuple[State, State, bool]:
        """
        Avanza la simulación N frames
        
        Returns:
            state_p1: Estado desde perspectiva jugador 1
            state_p2: Estado desde perspectiva jugador 2
            done: Si el episodio terminó
        """
        # Aplicar speed multiplier
        effective_frames = int(frames * self.speed_multiplier)
        
        for _ in range(effective_frames):
            # Obtener acciones de ambos jugadores
            action1 = self.player1.get_action(self._get_state(player_id=0))
            action2 = self.player2.get_action(self._get_state(player_id=1))
            
            # Aplicar acciones si son válidas
            if action1:
                self._apply_action(player_id=0, action=action1)
            if action2:
                self._apply_action(player_id=1, action=action2)
            
            # Actualizar física
            self.physics.update_positions(self.all_entities)
            
            # Actualizar combate
            self.combat.process_attacks(self.all_entities, self.current_frame)
            
            # Actualizar elixir
            self._update_elixir()
            
            # Remover unidades muertas
            self._cleanup_dead_entities()
            
            # Check condición de victoria
            done = self._check_game_over()
            if done:
                break
            
            self.current_frame += 1
        
        return (
            self._get_state(player_id=0),
            self._get_state(player_id=1),
            done
        )
```

---

## PARTE 6: OPTIMIZACIÓN PARA ENTRENAMIENTO MASIVO

### 6.1 Vectorización de Ambientes

```python
class VectorizedClashRoyaleEnv:
    """
    Ejecuta múltiples instancias del juego en paralelo
    Fundamental para PPO, A2C, etc.
    """
    
    def __init__(
        self,
        num_envs: int = 16,
        **env_kwargs
    ):
        self.num_envs = num_envs
        self.envs = [ClashRoyaleEnv(**env_kwargs) for _ in range(num_envs)]
    
    def step(self, actions: np.ndarray):
        """
        actions: (num_envs, action_dim)
        
        Returns:
            obs: (num_envs, obs_dim)
            rewards: (num_envs,)
            dones: (num_envs,)
            infos: List[dict]
        """
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        
        obs = np.array([r[0] for r in results])
        rewards = np.array([r[1] for r in results])
        dones = np.array([r[2] for r in results])
        infos = [r[3] for r in results]
        
        # Auto-reset envs que terminaron
        for i, done in enumerate(dones):
            if done:
                obs[i] = self.envs[i].reset()
        
        return obs, rewards, dones, infos
    
    def reset(self):
        return np.array([env.reset() for env in self.envs])
```

### 6.2 Fast-Forward para Simulación Acelerada

```python
class FastSimulationMode:
    """
    Acelera simulación eliminando cálculos innecesarios
    """
    
    def __init__(self, engine: ClashRoyaleEngine):
        self.engine = engine
        self.render_enabled = False
        self.detailed_logging = False
    
    def run_episode_fast(
        self,
        max_steps: int = 5400,  # 3 min a 30 FPS
        step_multiplier: int = 1  # Cuántos frames por step
    ) -> dict:
        """
        Ejecuta episodio completo lo más rápido posible
        """
        obs = self.engine.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < max_steps:
            # Batch de frames (ej: simular 5 frames a la vez)
            for _ in range(step_multiplier):
                state1, state2, done = self.engine.step(frames=1)
                if done:
                    break
            
            steps += 1
            # Calcular reward solo cada N steps si es necesario
        
        return {
            "total_reward": total_reward,
            "steps": steps,
            "winner": self.engine.get_winner()
        }
```

### 6.3 Sistema de Checkpointing y Replay

```python
class ReplayBuffer:
    """
    Almacena episodios para entrenamiento offline
    """
    
    def __init__(self, max_episodes: int = 10000):
        self.max_episodes = max_episodes
        self.episodes = []
    
    def add_episode(self, episode: dict):
        """
        episode = {
            'states': List[State],
            'actions': List[Tuple[int, int, int]],
            'rewards': List[float],
            'metadata': dict
        }
        """
        if len(self.episodes) >= self.max_episodes:
            self.episodes.pop(0)  # FIFO
        
        self.episodes.append(episode)
    
    def save_to_disk(self, path: str):
        """Guardar buffer comprimido"""
        import pickle
        import gzip
        
        with gzip.open(path, 'wb') as f:
            pickle.dump(self.episodes, f)
    
    def sample_batch(self, batch_size: int):
        """Para entrenamiento offline"""
        import random
        return random.sample(self.episodes, min(batch_size, len(self.episodes)))
```

---

## PARTE 7: SISTEMA DE OBSERVACIONES FLEXIBLE

### 7.1 Múltiples Tipos de Observación

```python
from enum import Enum

class ObservationType(Enum):
    STATE_DICT = "state_dict"          # Estado completo (BuildABot compatible)
    FEATURE_VECTOR = "feature_vector"  # Vector numérico para RL
    IMAGE = "image"                    # Imagen 128x128x3 (futuro)
    HYBRID = "hybrid"                  # Combinación

class ObservationEncoder:
    """
    Convierte State interno a diferentes formatos de observación
    """
    
    def __init__(self, obs_type: ObservationType):
        self.obs_type = obs_type
    
    def encode(self, state: State) -> Any:
        if self.obs_type == ObservationType.STATE_DICT:
            return self._to_state_dict(state)
        elif self.obs_type == ObservationType.FEATURE_VECTOR:
            return self._to_feature_vector(state)
        # ... otros tipos
    
    def _to_feature_vector(self, state: State) -> np.ndarray:
        """
        Convierte State a vector numérico flat
        
        Features:
        - Elixir propio/enemigo (2)
        - HP torres propias/enemigas (6)
        - Cartas en mano + costs (8)
        - Grid 18x32 con occupancy de aliados/enemigos (1152)
        - Grid 18x32 con tipo de unidad (1152)
        
        Total: ~2320 features
        """
        features = []
        
        # Elixir
        features.append(state.numbers.elixir / 10.0)  # Normalizado
        features.append(state.numbers.enemy_elixir / 10.0)
        
        # Torres (HP normalizado)
        max_princess_hp = 1400
        max_king_hp = 2400
        features.extend([
            state.numbers.left_princess_hp / max_princess_hp,
            state.numbers.right_princess_hp / max_princess_hp,
            state.numbers.king_hp / max_king_hp,
            state.numbers.left_enemy_princess_hp / max_princess_hp,
            state.numbers.right_enemy_princess_hp / max_princess_hp,
            state.numbers.enemy_king_hp / max_king_hp,
        ])
        
        # Cartas en mano (one-hot encoding + cost)
        card_vocab = ["giant", "musketeer", "archers", "mini_pekka", 
                      "knight", "skeletons", "arrows", "fireball"]
        for card in state.cards:
            card_encoding = [1 if card.name == c else 0 for c in card_vocab]
            features.extend(card_encoding)
            features.append(card.cost / 10.0)  # Normalizado
        
        # Grid de unidades (simplificado)
        ally_grid = np.zeros((N_HEIGHT_TILES, N_WIDE_TILES))
        enemy_grid = np.zeros((N_HEIGHT_TILES, N_WIDE_TILES))
        
        for ally in state.allies:
            x, y = ally.position.tile_x, ally.position.tile_y
            if 0 <= x < N_WIDE_TILES and 0 <= y < N_HEIGHT_TILES:
                ally_grid[y, x] = 1.0  # Presencia
        
        for enemy in state.enemies:
            x, y = enemy.position.tile_x, enemy.position.tile_y
            if 0 <= x < N_WIDE_TILES and 0 <= y < N_HEIGHT_TILES:
                enemy_grid[y, x] = 1.0
        
        features.extend(ally_grid.flatten())
        features.extend(enemy_grid.flatten())
        
        return np.array(features, dtype=np.float32)
```

---

## PARTE 8: GYMNASIUM ENVIRONMENT WRAPPER

### 8.1 Implementación Completa

```python
import gymnasium as gym
from gymnasium import spaces

class ClashRoyaleEnv(gym.Env):
    """
    Ambiente Gymnasium completamente funcional
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array", None],
        "render_fps": 30
    }
    
    def __init__(
        self,
        opponent: PlayerInterface = None,
        deck: List[str] = None,
        opponent_deck: List[str] = None,
        obs_type: ObservationType = ObservationType.FEATURE_VECTOR,
        reward_shaping: str = "sparse",  # "sparse" | "dense"
        time_limit: float = 180.0,
        fps: int = 30,
        speed_multiplier: float = 1.0,
        render_mode: str = None
    ):
        super().__init__()
        
        # Configuración
        self.obs_type = obs_type
        self.reward_shaping = reward_shaping
        self.render_mode = render_mode
        
        # Deck por defecto (las 8 cartas)
        if deck is None:
            deck = ["giant", "musketeer", "archers", "mini_pekka",
                    "knight", "skeletons", "arrows", "fireball"]
        if opponent_deck is None:
            opponent_deck = deck.copy()
        
        # Oponente por defecto (bot heurístico)
        if opponent is None:
            opponent = HeuristicBot()
        
        # Motor de juego
        self.engine = ClashRoyaleEngine(
            player1=RLAgentPlayer(model=None),  # Placeholder
            player2=opponent,
            deck1=deck,
            deck2=opponent_deck,
            fps=fps,
            time_limit=time_limit,
            speed_multiplier=speed_multiplier
        )
        
        # Espacios de observación y acción
        self._setup_spaces()
        
        # Encoder de observaciones
        self.obs_encoder = ObservationEncoder(obs_type)
    
    def _setup_spaces(self):
        """Define observation_space y action_space"""
        
        # Action Space: (tile_x, tile_y, card_index)
        # Discretizado: 18 * 32 * 4 = 2304 acciones posibles
        self.action_space = spaces.Discrete(18 * 32 * 4)
        
        # Observation Space
        if self.obs_type == ObservationType.FEATURE_VECTOR:
            # Vector de ~2320 features
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(2320,),
                dtype=np.float32
            )
        elif self.obs_type == ObservationType.IMAGE:
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(128, 128, 3),
                dtype=np.uint8
            )
        # ... otros tipos
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Resetear motor
        state = self.engine.reset()
        
        # Codificar observación
        obs = self.obs_encoder.encode(state)
        
        info = {
            "raw_state": state,
            "episode_start_time": time.time()
        }
        
        return obs, info
    
    def step(self, action: int):
        """
        Ejecuta un step del ambiente
        
        Args:
            action: int [0, 2303] representing (tile_x, tile_y, card_idx)
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Decodificar acción
        tile_x, tile_y, card_idx = self._decode_action(action)
        
        # Aplicar acción al motor
        try:
            state, opponent_state, done = self.engine.step_with_action(
                player_id=0,
                action=(tile_x, tile_y, card_idx)
            )
            action_valid = True
        except InvalidActionError as e:
            # Acción inválida (no hay elixir, carta no existe, etc.)
            state = self.engine.get_state(player_id=0)
            done = False
            action_valid = False
        
        # Calcular reward
        reward = self._calculate_reward(state, action_valid)
        
        # Codificar observación
        obs = self.obs_encoder.encode(state)
        
        # Info
        info = {
            "raw_state": state,
            "action_valid": action_valid,
            "elixir": state.numbers.elixir,
            "towers_destroyed": self._count_towers_destroyed(state)
        }
        
        # Terminated vs Truncated
        terminated = done and self.engine.has_winner()
        truncated = done and not self.engine.has_winner()  # Timeout
        
        return obs, reward, terminated, truncated, info
    
    def _decode_action(self, action: int) -> Tuple[int, int, int]:
        """Convierte acción discreta a (tile_x, tile_y, card_idx)"""
        card_idx = action % 4
        remaining = action // 4
        tile_y = remaining % 32
        tile_x = remaining // 32
        return tile_x, tile_y, card_idx
    
    def _calculate_reward(self, state: State, action_valid: bool) -> float:
        """
        Sistema de rewards configurables
        """
        if self.reward_shaping == "sparse":
            # Solo reward al final del episodio
            if self.engine.is_done():
                winner = self.engine.get_winner()
                if winner == 0:
                    return 1.0  # Victoria
                elif winner == 1:
                    return -1.0  # Derrota
                else:
                    return 0.0  # Empate
            return 0.0
        
        elif self.reward_shaping == "dense":
            # Reward shaping detallado
            reward = 0.0
            
            # Penalización por acción inválida
            if not action_valid:
                reward -= 0.1
            
            # Reward por daño a torres
            reward += self._calculate_tower_damage_reward(state)
            
            # Penalización por perder torres
            reward -= self._calculate_tower_loss_penalty(state)
            
            # Small reward por usar elixir eficientemente
            if action_valid:
                reward += 0.01
            
            # Reward final por victoria
            if self.engine.is_done():
                winner = self.engine.get_winner()
                if winner == 0:
                    reward += 10.0
                elif winner == 1:
                    reward -= 10.0
            
            return reward
    
    def render(self):
        """Renderizado opcional (Fase 2)"""
        if self.render_mode is None:
            return None
        
        # Futuro: implementar con pygame o matplotlib
        # Por ahora retornar None
        return None
    
    def close(self):
        """Cleanup"""
        pass
```

---

## CRITERIOS DE ACEPTACIÓN Y TESTING

```python
import pytest

class TestGameEngine:
    
    def test_initialization(self):
        """Motor se inicializa correctamente"""
        engine = ClashRoyaleEngine(...)
        assert engine.current_frame == 0
        assert len(engine.all_entities) == 6  # 3 torres por jugador
    
    def test_elixir_generation(self):
        """Elixir se genera a velocidad correcta"""
        engine = ClashRoyaleEngine(fps=30)
        engine.reset()
        
        initial_elixir = engine.player1.elixir
        
        # Simular 2.8 segundos (84 frames a 30 FPS)
        for _ in range(84):
            engine.step(frames=1)
        
        # Debe haber generado 1 elixir
        assert abs(engine.player1.elixir - (initial_elixir + 1.0)) < 0.01
    
    def test_card_spawning(self):
        """Cartas se spawnean correctamente"""
        # Test Giant (1 unidad)
        # Test Archers (2 unidades)
        # Test Skeletons (3 unidades)
        pass
    
    def test_combat_damage(self):
        """Sistema de combate aplica daño correctamente"""
        pass
    
    def test_tower_destruction(self):
        """Detecta victoria por destrucción de torre"""
        pass
    
    def test_action_validation(self):
        """Rechaza acciones inválidas"""
        # No hay elixir
        # Carta no en mano
        # Posición fuera de bounds
        pass

class TestPhysics:
    
    def test_movement_speed(self):
        """Unidades se mueven a velocidades correctas"""
        # Giant: 45 px/s
        # Musketeer: 60 px/s
        # Mini PEKKA: 90 px/s
        pass
    
    def test_collision_detection(self):
        """Unidades no se superponen"""
        pass
    
    def test_projectile_mechanics(self):
        """Proyectiles viajan correctamente"""
        pass

class TestEnvironment:
    
    def test_gymnasium_interface(self):
        """Cumple con interfaz Gymnasium"""
        env = ClashRoyaleEnv()
        
        # Check API
        assert hasattr(env, 'reset')
        assert hasattr(env, 'step')
        assert hasattr(env, 'action_space')
        assert hasattr(env, 'observation_space')
        
        # Test basic episode
        obs, info = env.reset()
        assert obs in env.observation_space
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs in env.observation_space
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
    
    def test_vectorization(self):
        """Vectorización funciona correctamente"""
        vec_env = VectorizedClashRoyaleEnv(num_envs=4)
        obs = vec_env.reset()
        assert obs.shape[0] == 4
```

```python
def benchmark_simulation_speed():
    """
    Target: 1000+ episodios/hora en CPU moderna
    """
    env = ClashRoyaleEnv(speed_multiplier=10.0)
    
    import time
    start = time.time()
    
    episodes = 0
    for _ in range(100):
        env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        episodes += 1
    
    elapsed = time.time() - start
    eps_per_hour = (episodes / elapsed) * 3600
    
    print(f"📊 Benchmark: {eps_per_hour:.0f} episodes/hour")
    assert eps_per_hour > 1000, "Performance too slow!"
```

---

## PARTE 12: CONFIGURACIÓN Y CONSTANTES

### 12.1 Archivo constants.py

```python
"""
Todas las constantes del juego en un solo lugar
"""

# === SISTEMA DE COORDENADAS ===
DISPLAY_WIDTH = 720
DISPLAY_HEIGHT = 1280
TILE_WIDTH = 34.0
TILE_HEIGHT = 27.6
N_WIDE_TILES = 18
N_HEIGHT_TILES = 32
PLAYABLE_HEIGHT_TILES = 15  # Por jugador
TILE_INIT_X = 52
TILE_INIT_Y = 188

# === CONFIGURACIÓN DE SIMULACIÓN ===
DEFAULT_FPS = 30
GAME_DURATION = 180.0  # segundos
OVERTIME_DURATION = 60.0  # segundos

# === ELIXIR ===
MAX_ELIXIR = 10.0
STARTING_ELIXIR = 5.0
ELIXIR_PER_SECOND = 1.0 / 2.8  # ~0.357
DOUBLE_ELIXIR_TIME = 120.0  # último minuto
DOUBLE_ELIXIR_RATE = 2.0

# === VELOCIDADES DE MOVIMIENTO (pixels/segundo) ===
SPEED_SLOW = 45.0      # Giant
SPEED_MEDIUM = 60.0    # Musketeer, Knight, Archers
SPEED_FAST = 90.0      # Mini PEKKA, Skeletons

# === CONFIGURACIÓN DE TORRES ===
TOWER_CONFIGS = {
    "princess": {
        "hp": 1400,
        "damage": 50,
        "hit_speed": 0.8,
        "range": 7.5,
        "positions": {
            "left": (3, 1),
            "right": (14, 1)
        }
    },
    "king": {
        "hp": 2400,
        "damage": 50,
        "hit_speed": 1.0,
        "range": 7.0,
        "position": (8.5, 0)
    }
}

# === STATS DE CARTAS (Ver PARTE 3 para detalles completos) ===
# Importar desde archivo separado cards_stats.py
```

---

## PARTE 13: EJEMPLO DE USO COMPLETO

### 13.1 Entrenamiento de Agente con Stable-Baselines3

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from clash_royale_engine import ClashRoyaleEnv

# Crear ambiente vectorizado
def make_env():
    def _init():
        env = ClashRoyaleEnv(
            obs_type=ObservationType.FEATURE_VECTOR,
            reward_shaping="dense",
            speed_multiplier=5.0,  # 5x más rápido
            opponent=HeuristicBot()
        )
        return env
    return _init

vec_env = SubprocVecEnv([make_env() for _ in range(16)])

# Entrenar modelo PPO
model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    tensorboard_log="./logs/",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95
)

# Entrenar por 1M steps (~277 horas de juego)
model.learn(total_timesteps=1_000_000)

# Guardar modelo
model.save("clash_royale_agent")

# Evaluar
eval_env = ClashRoyaleEnv(opponent=HeuristicBot())
obs, _ = eval_env.reset()

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    
    if terminated or truncated:
        obs, _ = eval_env.reset()
```

### 13.2 Self-Play Training

```python
class SelfPlayWrapper:
    """
    Entrenamiento contra copias de sí mismo
    """
    
    def __init__(self, base_env_config: dict):
        self.config = base_env_config
        self.historical_opponents = []
    
    def create_env_vs_past_self(self, past_version_idx: int = -1):
        """Crea ambiente donde el oponente es una versión pasada del agente"""
        opponent_model = self.historical_opponents[past_version_idx]
        opponent = RLAgentPlayer(opponent_model)
        
        return ClashRoyaleEnv(
            opponent=opponent,
            **self.config
        )
    
    def save_checkpoint(self, model):
        """Guarda versión del modelo para futuros oponentes"""
        self.historical_opponents.append(model.copy())

# Uso
self_play = SelfPlayWrapper({
    "obs_type": ObservationType.FEATURE_VECTOR,
    "reward_shaping": "dense"
})

model = PPO("MlpPolicy", ...)

for iteration in range(100):
    # Entrenar contra mix de oponentes pasados
    env = self_play.create_env_vs_past_self(random.choice(range(len(historical))))
    model.learn(total_timesteps=10_000, reset_num_timesteps=False)
    
    # Guardar checkpoint cada 10 iteraciones
    if iteration % 10 == 0:
        self_play.save_checkpoint(model)
```

---

## CRITERIOS DE ÉXITO FINALES

### Funcionalidad ✅
- [ ] Todas las 8 cartas funcionan correctamente
- [ ] Torres se destruyen y detectan victoria
- [ ] Física es realista (velocidades, rangos, colisiones)
- [ ] Compatible 100% con formato State de BuildABot
- [ ] Puede ejecutar 1000+ episodios/hora

### Flexibilidad ✅
- [ ] Soporta Agent vs Agent
- [ ] Soporta Agent vs Bot
- [ ] Soporta Human vs cualquiera
- [ ] Vectorización funcional
- [ ] Múltiples tipos de observación

### Extensibilidad ✅
- [ ] Fácil agregar nuevas cartas
- [ ] Fácil modificar rewards
- [ ] Puede acelerar/desacelerar simulación
- [ ] Preparado para GUI futura

### Calidad de Código ✅
- [ ] Type hints en todo el código
- [ ] Docstrings en todas las clases/métodos
- [ ] Tests unitarios >80% coverage
- [ ] Código formateado con black/isort
- [ ] Mypy pasa sin errores