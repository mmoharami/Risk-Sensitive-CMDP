from env.gridworld import GridWorld
from env_example.grid import grid_5x5
import numpy as np
from agent.entcmdp import EntCMDP

def main():
    r, u, p, H = grid_5x5()
    np.savez("grid_example_5x5.npz", r=r, u=u, p=p, H=H)
    env = GridWorld(parameters="grid_example_5x5.npz")
    agent = EntCMDP(env, K=15000, alpha=-0.01, B=2.2, delta=0.05, Bon="Relaxed" , scale = 100, seed=42)
    agent.main()

if __name__ == "__main__":
    main()