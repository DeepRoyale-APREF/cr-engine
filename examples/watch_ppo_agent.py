#!/usr/bin/env python
"""
Visualizar cómo juega el agente PPO entrenado.

Modos:
  • Con GUI (pygame): muestra la arena gráficamente.
  • Sin GUI (texto):  imprime jugada-a-jugada en consola.

Uso
----
    # Con GUI (requiere pygame)
    python examples/watch_ppo_agent.py --model runs/ppo_baseline_XXXX/ppo_cr_final.zip

    # Sin GUI (solo texto)
    python examples/watch_ppo_agent.py --model runs/ppo_baseline_XXXX/ppo_cr_final.zip --no-gui

    # Más lento para observar
    python examples/watch_ppo_agent.py --model ... --speed 0.5

    # Acción aleatoria como baseline de comparación
    python examples/watch_ppo_agent.py --random
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from stable_baselines3 import PPO

from clash_royale_engine.core.engine import ClashRoyaleEngine
from clash_royale_engine.env.gymnasium_env import ClashRoyaleEnv, ObservationType
from clash_royale_engine.players.player_interface import HeuristicBot
from clash_royale_engine.utils.constants import CARD_VOCAB, N_HEIGHT_TILES, N_WIDE_TILES


# ══════════════════════════════════════════════════════════════════════════════
# Decodificar acción para mostrarla
# ══════════════════════════════════════════════════════════════════════════════

def decode_action(action: int) -> str:
    """Devuelve string legible de la acción."""
    n_placement = N_WIDE_TILES * N_HEIGHT_TILES * 4
    if action >= n_placement:
        return "NO-OP (esperar)"

    card_idx = action % 4
    remaining = action // 4
    tile_y = remaining % N_HEIGHT_TILES
    tile_x = remaining // N_HEIGHT_TILES
    return f"Carta[{card_idx}] en ({tile_x}, {tile_y})"


def format_state_line(state, step: int, game_time: float) -> str:
    """Resumen compacto de una línea del estado."""
    n = state.numbers
    allies = len(state.allies)
    enemies = len(state.enemies)
    cards = [c.name[:3].upper() for c in state.cards]
    ready = state.ready

    return (
        f"[{game_time:5.1f}s | step {step:>5d}] "
        f"Elixir={n.elixir:4.1f}  "
        f"Aliados={allies} Enemigos={enemies}  "
        f"Torres propias: L={n.left_princess_hp:4.0f} R={n.right_princess_hp:4.0f} K={n.king_hp:4.0f}  "
        f"Torres enemigas: L={n.left_enemy_princess_hp:4.0f} R={n.right_enemy_princess_hp:4.0f} K={n.enemy_king_hp:4.0f}  "
        f"Mano={cards} Ready={ready}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Modo TEXT: jugar y mostrar en consola
# ══════════════════════════════════════════════════════════════════════════════

def watch_text(model, n_episodes: int, time_limit: float, use_random: bool) -> None:
    """Ejecuta episodios mostrando las jugadas en texto."""
    env = ClashRoyaleEnv(
        reward_shaping="dense",
        time_limit=time_limit,
        fog_of_war=True,
    )

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        step = 0
        total_reward = 0.0
        placements = 0
        last_print_step = -999

        print(f"\n{'═' * 80}")
        print(f"  EPISODIO {ep + 1}/{n_episodes}")
        print(f"{'═' * 80}")

        while not done:
            if use_random:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            done = terminated or truncated

            state = info.get("raw_state")
            valid = info.get("action_valid", False)
            game_time = time_limit - state.numbers.time_remaining if state else 0

            # Mostrar cuando el agente coloca una carta (acción válida y no es no-op)
            n_placement = N_WIDE_TILES * N_HEIGHT_TILES * 4
            is_placement = action < n_placement

            if valid and is_placement:
                placements += 1
                card_idx = action % 4
                card_name = state.cards[card_idx].name if state else "?"
                action_str = decode_action(action)
                print(f"  ▶ JUGADA #{placements}: {card_name.upper()} — {action_str}  (reward={reward:+.3f})")
                if state:
                    print(f"    {format_state_line(state, step, game_time)}")

            # Imprimir resumen cada ~5 segundos de juego (150 frames)
            elif state and step - last_print_step >= 150:
                last_print_step = step
                print(f"  · {format_state_line(state, step, game_time)}")

        # Resultado final
        winner = env.engine.get_winner()
        result = "VICTORIA" if winner == 0 else "DERROTA" if winner == 1 else "EMPATE"
        print(f"\n  ── Resultado: {result} ──")
        print(f"  Steps: {step}  |  Cartas jugadas: {placements}  |  Reward total: {total_reward:+.4f}")
        if state:
            n = state.numbers
            print(f"  Torres propias finales:  L={n.left_princess_hp:.0f}  R={n.right_princess_hp:.0f}  K={n.king_hp:.0f}")
            print(f"  Torres enemigas finales: L={n.left_enemy_princess_hp:.0f}  R={n.right_enemy_princess_hp:.0f}  K={n.enemy_king_hp:.0f}")

    env.close()


# ══════════════════════════════════════════════════════════════════════════════
# Modo GUI: jugar con pygame
# ══════════════════════════════════════════════════════════════════════════════

def watch_gui(model, n_episodes: int, time_limit: float, speed: float, use_random: bool) -> None:
    """Ejecuta episodios con visualización pygame."""
    from clash_royale_engine.visualization.renderer import Renderer

    # Crear motor directamente para tener control de frames
    opponent = HeuristicBot(aggression=0.5)

    for ep in range(n_episodes):
        engine = ClashRoyaleEngine(
            player1=HeuristicBot(),  # placeholder, sobreescribimos con acciones PPO
            player2=opponent,
            fps=30,
            time_limit=time_limit,
            speed_multiplier=1.0,
            seed=ep,
        )

        # También crear env wrapper para codificar observaciones
        env = ClashRoyaleEnv(
            reward_shaping="dense",
            time_limit=time_limit,
            fog_of_war=True,
            seed=ep,
        )
        obs, _ = env.reset()

        renderer = Renderer(
            fps=int(30 * speed),
            title=f"PPO Agent vs HeuristicBot — Ep {ep + 1}/{n_episodes}",
        )

        running = True
        done = False
        step = 0
        total_reward = 0.0
        placements = 0

        print(f"\n  Episodio {ep + 1}/{n_episodes} — GUI abierta (ESC para salir)")

        try:
            while running and not done:
                # Obtener acción del agente PPO
                if use_random:
                    action = env.action_space.sample()
                else:
                    action, _ = model.predict(obs, deterministic=True)
                    action = int(action)

                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                step += 1
                done = terminated or truncated

                state = info.get("raw_state")
                valid = info.get("action_valid", False)

                n_placement = N_WIDE_TILES * N_HEIGHT_TILES * 4
                if valid and action < n_placement:
                    placements += 1
                    card_idx = action % 4
                    card_name = state.cards[card_idx].name if state else "?"
                    print(f"    ▶ {card_name.upper()} en {decode_action(action)}")

                # Renderizar
                if state:
                    running = renderer.render(state)

            # Resultado
            winner = env.engine.get_winner()
            result = "VICTORIA" if winner == 0 else "DERROTA" if winner == 1 else "EMPATE"
            print(f"  → {result}  |  Cartas: {placements}  |  Reward: {total_reward:+.4f}")

            # Pausa para ver el resultado final
            if running:
                end_time = time.time() + 3.0
                while time.time() < end_time and running:
                    if state:
                        running = renderer.render(state)

        except KeyboardInterrupt:
            running = False
        finally:
            renderer.close()
            env.close()

        if not running:
            break


# ══════════════════════════════════════════════════════════════════════════════
# Análisis rápido de comportamiento
# ══════════════════════════════════════════════════════════════════════════════

def analyze_behavior(model, n_episodes: int = 10, time_limit: float = 60.0, use_random: bool = False) -> None:
    """Ejecuta N episodios y muestra estadísticas agregadas del comportamiento."""
    env = ClashRoyaleEnv(
        reward_shaping="dense",
        time_limit=time_limit,
        fog_of_war=True,
    )

    card_usage = {name: 0 for name in CARD_VOCAB}
    total_placements = 0
    total_noops = 0
    total_invalid = 0
    total_steps = 0
    wins, losses, draws = 0, 0, 0
    rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            if use_random:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            total_steps += 1
            done = terminated or truncated

            valid = info.get("action_valid", False)
            n_placement = N_WIDE_TILES * N_HEIGHT_TILES * 4

            if action >= n_placement:
                total_noops += 1
            elif valid:
                total_placements += 1
                card_idx = action % 4
                state = info.get("raw_state")
                if state and card_idx < len(state.cards):
                    card_usage[state.cards[card_idx].name] += 1
            else:
                total_invalid += 1

        winner = env.engine.get_winner()
        if winner == 0:
            wins += 1
        elif winner == 1:
            losses += 1
        else:
            draws += 1
        rewards.append(ep_reward)

    env.close()

    # Mostrar análisis
    print(f"\n{'═' * 60}")
    print(f"  ANÁLISIS DE COMPORTAMIENTO ({n_episodes} episodios)")
    print(f"{'═' * 60}")
    print(f"  Resultados:      {wins}W / {losses}L / {draws}D  (WR: {wins/n_episodes:.0%})")
    print(f"  Reward promedio:  {np.mean(rewards):+.4f} ± {np.std(rewards):.4f}")
    print(f"  Steps totales:    {total_steps}")
    print(f"  Steps/episodio:   {total_steps / n_episodes:.0f}")
    print()
    print(f"  Acciones:")
    print(f"    Colocaciones válidas: {total_placements:>6d}  ({total_placements/max(total_steps,1):.1%})")
    print(f"    No-ops (esperar):     {total_noops:>6d}  ({total_noops/max(total_steps,1):.1%})")
    print(f"    Inválidas:            {total_invalid:>6d}  ({total_invalid/max(total_steps,1):.1%})")
    print()
    print(f"  Uso de cartas:")
    for name in sorted(card_usage, key=card_usage.get, reverse=True):
        count = card_usage[name]
        bar = "█" * min(count, 40)
        print(f"    {name:>12s}: {count:>4d}  {bar}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualizar agente PPO jugando Clash Royale")
    p.add_argument("--model", type=str, default=None,
                   help="Ruta al modelo .zip (ej: runs/ppo_baseline_xxx/ppo_cr_final.zip)")
    p.add_argument("--random", action="store_true",
                   help="Usar acciones aleatorias en vez de modelo (baseline de comparación)")
    p.add_argument("--no-gui", action="store_true",
                   help="Solo texto, sin ventana pygame")
    p.add_argument("--episodes", type=int, default=3,
                   help="Número de episodios a jugar (default: 3)")
    p.add_argument("--time-limit", type=float, default=60.0,
                   help="Duración de partida en segundos (default: 60)")
    p.add_argument("--speed", type=float, default=1.0,
                   help="Velocidad de renderizado: 0.5=lento, 1=normal, 2=rápido")
    p.add_argument("--analyze", action="store_true",
                   help="Ejecutar análisis estadístico del comportamiento (10 episodios)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Cargar modelo
    model = None
    use_random = args.random

    if not use_random:
        if args.model is None:
            # Buscar automáticamente el último modelo guardado
            runs_dir = Path("runs")
            if runs_dir.exists():
                candidates = sorted(runs_dir.glob("*/ppo_cr_final.zip"), key=lambda p: p.stat().st_mtime)
                if candidates:
                    args.model = str(candidates[-1])
                    print(f"  Auto-detectado modelo: {args.model}")

        if args.model is None:
            print("  ⚠ No se encontró modelo. Usa --model <ruta> o --random")
            print("  Usando acciones aleatorias como fallback.")
            use_random = True
        else:
            print(f"  Cargando modelo: {args.model}")
            model = PPO.load(args.model)
            print(f"  Modelo cargado OK")

    label = "PPO Agent" if not use_random else "Random Agent"
    print(f"\n  Agente: {label}")
    print(f"  Oponente: HeuristicBot")
    print(f"  Episodios: {args.episodes}")
    print(f"  Tiempo: {args.time_limit}s")

    # Análisis estadístico
    if args.analyze:
        analyze_behavior(model, n_episodes=10, time_limit=args.time_limit, use_random=use_random)
        return

    # Modo de visualización
    if args.no_gui:
        watch_text(model, args.episodes, args.time_limit, use_random)
    else:
        try:
            import pygame  # noqa: F401
            watch_gui(model, args.episodes, args.time_limit, args.speed, use_random)
        except ImportError:
            print("  ⚠ pygame no instalado. Usando modo texto.")
            watch_text(model, args.episodes, args.time_limit, use_random)


if __name__ == "__main__":
    main()
