"""
alife.py
Author: Rishi Urs, Rianna Wadhwani

• Predator–prey artificial‑life simulation on a toroidal grass field
• Animates rabbits (prey) and foxes (predators) eating, reproducing & dying
• Meets requirements 1‑5 (inline #reqX markers)
"""

from __future__ import annotations

# =============================================================================
# Imports
# =============================================================================
import random as rnd
import copy
from typing import List, Dict, Tuple, Iterable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

# =============================================================================
# Model Assumptions & Tunable Parameters   (2a–2d)
# =============================================================================
ARRSIZE = 200          # lattice width × height (cells)
FIGSIZE = 8            # figure size (inches)
GRASS_GROWTH_RATE = 0.05

INIT_RABBITS = 200     # initial rabbit count  (prey)
INIT_FOXES   = 25      # initial fox count     (predator)

RABBIT_MAX_OFFSPRING = 3
FOX_MAX_OFFSPRING   = 2

RABBIT_STARVATION = 1
FOX_STARVATION   = 20

RABBIT_REPRO_LEVEL = 1
FOX_REPRO_LEVEL   = 1

FOX_MOVE_STEPS = 3  # extra moves improve hunting success

# Colour map index → RGB (Requirement 4)
COLORMAP = ListedColormap(["black", "green", "white", "red"])  # 0‑3

# =============================================================================
# Animal Class  (Requirement 1)
# =============================================================================
class Animal:
    """Represents a single rabbit or fox."""

    def __init__(self, species: str, max_offspring: int, starvation_level: int, reproduction_level: int):
        self.species = species  # "rabbit" | "fox"
        self.max_offspring = max_offspring
        self.starvation_level = starvation_level
        self.reproduction_level = reproduction_level

        # random starting location on toroidal grid
        self.x = rnd.randrange(ARRSIZE)
        self.y = rnd.randrange(ARRSIZE)

        self.eaten = 0      # food eaten this generation
        self.hunger = 0     # consecutive generations w/o food
        self.alive  = True

    # ── Movement ──
    def move(self):
        self.x = (self.x + rnd.choice([-1, 0, 1])) % ARRSIZE
        self.y = (self.y + rnd.choice([-1, 0, 1])) % ARRSIZE

    # ── Feeding & Reproduction helpers ──
    def eat(self, amount: int):
        if amount:
            self.eaten += amount

    def ready_to_reproduce(self):
        return self.eaten >= self.reproduction_level

    def make_child(self):
        baby = copy.deepcopy(self)
        baby.eaten = baby.hunger = 0
        return baby

    # ── End‑of‑generation state update ──
    def end_of_generation(self):
        self.hunger = 0 if self.eaten else self.hunger + 1  # req2
        self.alive = self.hunger < self.starvation_level

# =============================================================================
# Field Class – Ecosystem container
# =============================================================================
class Field:
    """Grass grid plus lists of Animal objects."""

    def __init__(self):
        self.grass = np.ones((ARRSIZE, ARRSIZE), dtype=int)
        self.rabbits: List[Animal] = []
        self.foxes:   List[Animal] = []

    # ── Population helpers ──
    def add_animals(self, animals: Iterable[Animal]):
        for a in animals:
            (self.rabbits if a.species == "rabbit" else self.foxes).append(a)

    # ── Generation sub‑steps ──
    def move(self):
        for r in self.rabbits:
            r.move()
        for f in self.foxes:
            for _ in range(FOX_MOVE_STEPS):
                f.move()

    def rabbits_eat_grass(self):
        for r in self.rabbits:
            if self.grass[r.x, r.y]:
                r.eat(1)
                self.grass[r.x, r.y] = 0

    def foxes_eat_rabbits(self):
        # map (x,y)
        loc2r: Dict[Tuple[int, int], List[Animal]] = {}
        for r in self.rabbits:
            loc2r.setdefault((r.x, r.y), []).append(r)
        for f in self.foxes:
            prey = loc2r.get((f.x, f.y))
            if prey:                             # req5
                f.eat(1)
                for r in prey:
                    r.alive = False
                loc2r[(f.x, f.y)] = []

    def update_survival(self):
        self.rabbits = [r for r in self.rabbits if (r.end_of_generation() or r.alive)]
        self.foxes   = [f for f in self.foxes   if (f.end_of_generation() or f.alive)]

    def reproduce(self):
        new_r, new_f = [], []
        for r in self.rabbits:
            if r.ready_to_reproduce():
                for _ in range(rnd.randint(1, r.max_offspring)):
                    new_r.append(r.make_child())  # req3
            r.eaten = 0
        for f in self.foxes:
            if f.ready_to_reproduce():
                for _ in range(rnd.randint(1, f.max_offspring)):
                    new_f.append(f.make_child())
            f.eaten = 0
        self.rabbits.extend(new_r)
        self.foxes.extend(new_f)

    def grow_grass(self):
        grow = (np.random.rand(ARRSIZE, ARRSIZE) < GRASS_GROWTH_RATE).astype(int)
        self.grass = np.maximum(self.grass, grow)

    def generation(self):
        for a in (*self.rabbits, *self.foxes):
            a.eaten = 0
        self.move()
        self.rabbits_eat_grass()
        self.foxes_eat_rabbits()
        self.update_survival()
        self.reproduce()
        self.grow_grass()

    # ── Rendering helper ──
    def display_array(self):
        disp = np.zeros_like(self.grass)
        disp[self.grass == 1] = 1
        for r in self.rabbits:
            disp[r.x, r.y] = max(disp[r.x, r.y], 2)
        for f in self.foxes:
            disp[f.x, f.y] = 3
        return disp

# =============================================================================
# Animation Callback & Stats Collection
# =============================================================================

# Lists grow as long as the animation runs; length ⇒ total generations observed
GEN_RABBITS: list[int] = []
GEN_FOXES:   list[int] = []

def _animate(frame: int, fld: Field, im):
    """Advance ecosystem one generation and collect population sizes."""
    fld.generation()

    # record counts for post‑run time‑series plot
    GEN_RABBITS.append(len(fld.rabbits))
    GEN_FOXES.append(len(fld.foxes))

    # update raster image
    im.set_array(fld.display_array())
    plt.title(
        f"gen {frame}   rabbits: {GEN_RABBITS[-1]}   foxes: {GEN_FOXES[-1]}",
        fontsize=9,
    )
    return im,

# =============================================================================
# Main Entrypoint
# =============================================================================

def main():
    field = Field()

    # populate field with initial animals
    field.add_animals(
        Animal("rabbit", RABBIT_MAX_OFFSPRING, RABBIT_STARVATION, RABBIT_REPRO_LEVEL)
        for _ in range(INIT_RABBITS)
    )
    field.add_animals(
        Animal("fox", FOX_MAX_OFFSPRING, FOX_STARVATION, FOX_REPRO_LEVEL)
        for _ in range(INIT_FOXES)
    )

    fig = plt.figure(figsize=(FIGSIZE, FIGSIZE))
    im = plt.imshow(field.display_array(), cmap=COLORMAP, interpolation="nearest",
                    vmin=0, vmax=3)

    ax = plt.gca()
    ax.set_xlabel("field x‑coordinate (cells)")
    ax.set_ylabel("field y‑coordinate (cells)")
    # show numeric ticks every 25 cells for clarity
    ax.set_xticks(range(0, ARRSIZE + 1, 25))
    ax.set_yticks(range(0, ARRSIZE + 1, 25))

    anim = animation.FuncAnimation(
        fig, _animate, fargs=(field, im), frames=10000, interval=1, blit=False
    )
    plt.show()  # blocks until animation window closed

    # =============================================================================
    # Time‑series plot shown AFTER animation window is closed
    # =============================================================================
    gens = range(len(GEN_RABBITS))
    plt.figure(figsize=(FIGSIZE, FIGSIZE))
    plt.plot(gens, GEN_RABBITS, label="Rabbits")
    plt.plot(gens, GEN_FOXES, label="Foxes")
    plt.xlabel("Generation")
    plt.ylabel("Population size")
    plt.title("Fox & Rabbit populations over time")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
