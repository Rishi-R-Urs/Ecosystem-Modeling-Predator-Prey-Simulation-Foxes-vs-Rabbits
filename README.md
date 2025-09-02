# Ecosystem-Modeling-Predator-Prey-Simulation-Foxes-vs-Rabbits
Predator–prey artificial‑life simulation on a toroidal grass field. Animates rabbits (prey) and foxes (predators) eating, reproducing and dying.

Dependencies

This project was built in a Conda (Python 3.12) environment.
Below are the key packages required to run the simulation:

* numpy – Core numerical backend; manages the 200×200 grass grid, random regrowth, and vectorized operations.
* matplotlib – Visualization and animation of the ecosystem with custom colormaps.
* matplotlib.animation – Used to generate the live predator–prey animation frames.
* pillow – Image I/O support used by Matplotlib rendering.
* tqdm (optional) – Progress bars (useful for debugging long simulations, not critical).
* pytest / pytest-cov (optional) – Testing and coverage tools used during development.

Installation

Using Conda (recommended):
Use the following comand in terminal “conda install numpy matplotlib pillow tqdm pytest pytest-cov”

