import numpy as np
from collections import defaultdict
from datetime import date
from collections import deque
import os

class EntCMDP:
    def __init__(self, Env, K, alpha, B, delta, Bon, scale=1, seed=None):
        # MDP enviroment
        self.env = Env

        # Space parameters
        self.H = self.env.H
        self.S = {} # State may depends on horizon
        self.A = {} # action may depends on horizon
        for h in range(self.H):
            self.S[h] = self.env.S(h) if callable(self.env.S) else self.env.S
            self.A[h] = self.env.A(h) if callable(self.env.A) else self.env.A
        self.S[self.H] = 1
        self.K = K

        # Entropy/utility parameters
        if alpha < 0:
            self.alpha = alpha
        else:
            raise ValueError("Alpha must be negative!")
        self.B = B
        self.V_gmax = (1 / np.abs(alpha)) * (np.exp(np.abs(alpha) * self.H) - 1)
        self.tau = 0

        # Rewards and utilities for state-action pairs seen so far
        self.reward = defaultdict(lambda: defaultdict(float))
        self.utility = defaultdict(lambda: defaultdict(float))

        # Horizon-dependent discretization
        self.varepsilon = K ** (-1 / 2)
        self.disc_set = {}
        self.C = {}

        for h in range(self.H+1):
            lower = -self.H
            upper = self.H
            self.disc_set[h] = np.arange(lower, upper, self.varepsilon)
            self.C[h] = len(self.disc_set[h])

        # Dual update parameters
        self.xi = K ** (1 / 4)
        self.lambda_k = 0

        # Learning parameters
        self.scale = np.linspace(scale, 1, K)
        self.eta = K ** (-1 / 4) / self.V_gmax
        self.delta = delta

        # Initialize uniform policy: equal probability for all actions at each state and time step
        self.policy = {}
        for h in range(self.H):
            self.policy[h] = {}
            for s in range(self.S[h]):
                self.policy[h][s] = {}
                num_actions = self.A[h](s) if callable(self.A[h]) else self.A[h] # Support state-dependent action spaces
                for c_hat_idx in range(self.C[h]):
                    self.policy[h][s][c_hat_idx] = np.full(num_actions, 1.0 / num_actions)

        # Initialize history
        self.history = [[] for _ in range(self.H + 1)]


        # Initialize empirical transitions and count functions
        self.N_h = defaultdict(lambda: defaultdict(int))  # Count of (state, action) pairs
        self.P_h = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))  # Transition probabilities

        # Initialize reward and utility value estimates
        self.V_r = {}
        self.Q_r = {}
        self.V_g = {}
        self.Q_g = {}

        for h in range(self.H + 1):
            self.V_r[h] = np.zeros((self.S[h], self.C[h]))
            self.V_g[h] = np.zeros((self.S[h], self.C[h]))

            if h < self.H:
                self.Q_r[h] = {}
                self.Q_g[h] = {}
                for s in range(self.S[h]):
                    num_actions = self.A[h](s) if callable(self.A[h]) else self.A[h] # Support state-dependent action spaces
                    self.Q_r[h][s] = np.full((self.C[h], num_actions), self.H, dtype=float)
                    self.Q_g[h][s] = np.full((self.C[h], num_actions), self.V_gmax, dtype=float)

        # Initialize terminal value: note that the terminal node is unique and fictitious
        self.V_g[self.H] = np.expand_dims(self._u(-self.disc_set[self.H]), axis=0)

        # Track reward, risk, value estimates, and policy matrix at each episode
        self.eps_rew_k = np.zeros(self.K + 1)
        self.eps_risk_k = np.zeros(self.K + 1)
        self.V_g_s0_tau_k = np.zeros(self.K + 1)
        self.V_r_s0_tau_k = np.zeros(self.K + 1)
        self.tau0_k = np.zeros(self.K + 1)
        self.lambda_s0_k = np.zeros(self.K + 1)
        self.policy_k = deque(maxlen=20)  # keeps only the latest 100 episodes


        # Initialize episode counter
        self.k = 0

        # Determine the bonus function to be used
        if Bon not in {"Paper", "Relaxed"}:
            raise ValueError(f"Invalid Bon value: {Bon}. Must be 'Paper' or 'Relaxed'.")
        self.Bon = Bon

        # Fixing the randomness for reproducibility
        self.rng = np.random.default_rng(seed)

    # Utility function
    def _u(self, c): return (1 / self.alpha) * (np.exp(self.alpha * c) - 1)

    # Discretization function: smallest value >= x in disc_set[h]
    def _disc_value(self, x, h):
        disc = self.disc_set[h]
        idx = np.searchsorted(disc, x, side='left')
        return disc[min(idx, len(disc) - 1)]
    
    # Projection function: clamp lambda between 0 and xi
    def _proj_0_xi(self, lamb): return max(0, min(lamb, self.xi))

    def random_argmax(values, rng=None):
        values = np.asarray(values)
        max_val = np.max(values)
        candidates = np.flatnonzero(values == max_val)
        rng = rng or np.random.default_rng()
        return rng.choice(candidates)

    # Forward pass: execute the policy for one episode
    def _run_one_episode(self):
        # Initilization
        s = self.env.s0
        c_hat = self.tau

        eps_rew = 0
        eps_utility = 0
        for h in range(self.H):
            # Find budget index (first index where disc_set[h] >= c)
            c_hat_idx = np.searchsorted(self.disc_set[h], c_hat, side='left')
            c_hat_idx = min(c_hat_idx, len(self.disc_set[h]) - 1)

            # Sample action from current policy
            num_actions = self.A[h](s) if callable(self.A[h]) else self.A[h]
            a = self.rng.choice(range(num_actions), p=self.policy[h][s][c_hat_idx])

            # Environment step
            s_next, r, u = self.env.step(s, a, h)

            self.reward[h][(s,a)] = r
            self.utility[h][(s,a)] = self._disc_value(u,h)
            # Log history
            self.history[h].append((s, a, s_next))

            # Initialize for next step
            s = s_next
            c_hat -= self._disc_value(u,h)

            # Keep Track of utility and reward
            eps_rew += r
            eps_utility += u

        self.eps_rew_k[self.k] = eps_rew
        self.eps_risk_k[self.k] = max(self.disc_set[self.H], key=lambda tau: tau + self._u(eps_utility - tau))

    def _update_model_est(self):
        if not self.history[0]:
            return

        for h in range(self.H):
            s, a, s_next = self.history[h][-1]

            # Count updates
            self.N_h[h][(s, a)] += 1
            self.N_h[h][(s, a, s_next)] += 1

            # Update empirical transition probability
            for s_prime in range(self.S[h+1]):
                self.P_h[h][(s, a)][s_prime] = (self.N_h[h][(s, a, s_prime)] / self.N_h[h][(s, a)])


    def _update_value_est(self):
        # Backward pass
        if not self.history[0]:
            return
        for h in reversed(range(self.H)):
            for c_hat_idx, c_hat in enumerate(self.disc_set[h]):
                for s in range(self.S[h]):
                    num_actions = self.A[h](s) if callable(self.A[h]) else self.A[h]
                    for a in range(num_actions):
                        count = max(1, self.N_h[h][(s, a)])

                        if self.Bon == "Paper":
                            Bon_r = 9 * self.H * np.sqrt(
                                (self.S[h] * self.C[h] * np.log(self.H * self.S[h] * num_actions * self.K * self.C[h] / self.delta)) / count
                            )
                            Bon_g = 6 * self.V_gmax * np.sqrt(
                                (self.S[h] * self.C[h] * np.log(self.H * self.S[h] * num_actions * self.K * self.C[h] * self.V_gmax / self.delta)) / count
                            )
                        else:
                            Bon_r = 0.1 * self.H * (np.log(self.K) / count)
                            Bon_g = 0.005 * self.V_gmax * (np.log(self.K) / count)

                        # Reward and discretized utility associated with this state-action pair
                        r = self.reward[h][(s, a)]
                        u_disc = self.utility[h][(s, a)]

                        # Compute next budget index
                        c_hat_next_idx = np.searchsorted(self.disc_set[h],  c_hat - u_disc, side='left')
                        c_hat_next_idx = min(c_hat_next_idx, len(self.disc_set[h]) - 1)

                        # Compute expected next values
                        next_vals_r = 0
                        next_vals_g = 0
                        for s_next in range(self.S[h + 1]):
                            prob = self.P_h[h][(s, a)][s_next]

                            next_vals_r += prob * self.V_r[h+1][s_next][c_hat_next_idx]
                            next_vals_g += prob * self.V_g[h+1][s_next][c_hat_next_idx]

                        # Q-value updates
                        self.Q_r[h][s][c_hat_idx][a] = min(r + next_vals_r + Bon_r, self.H-h+1)
                        self.Q_g[h][s][c_hat_idx][a] = min(next_vals_g + Bon_g, self.V_gmax)

            for c_hat_idx, c_hat in enumerate(self.disc_set[h]):
                for s in range(self.S[h]):
                    num_actions = self.A[h](s) if callable(self.A[h]) else self.A[h]

                    # Update greedy policy
                    combine_q_values = [self.Q_r[h][s][c_hat_idx][a] + self.lambda_k * self.Q_g[h][s][c_hat_idx][a] for a in range(num_actions)]
                    best_action = np.argmax(combine_q_values)
                    self.policy[h][s][c_hat_idx] = np.eye(num_actions)[best_action]

                    # Value function update
                    self.V_r[h][s][c_hat_idx] = self.Q_r[h][s][c_hat_idx][best_action]
                    self.V_g[h][s][c_hat_idx] = self.Q_g[h][s][c_hat_idx][best_action]


    def main(self):
        # Initialize output file
        # Base directory setup
        parent_dir = os.path.dirname(os.path.dirname(__file__))
        results_root = os.path.join(parent_dir, 'results')
        os.makedirs(results_root, exist_ok=True)

        # Date and formatted result path
        today = date.today().isoformat()  # Format: YYYY-MM-DD
        subfolder = f"K = {self.K}, B = {self.B}, {today}"
        result_path = os.path.join(results_root, subfolder)
        os.makedirs(result_path, exist_ok=True)

        s0 = self.env.s0

        for k in range(self.K+1):
            self.k = k
            self._update_model_est() # update model estimation
            self._update_value_est() # update estimation of Q and V
            tau_idx = max(
                range(self.C[0]), 
                key=lambda c_hat_idx: self.V_r[0][s0][c_hat_idx] + self.lambda_k * (self.disc_set[0][c_hat_idx] + self.V_g[0][s0][c_hat_idx])
                )
            # print("Combined Value Function = ", self.V_r[0][s0] + self.lambda_k * (self.disc_set[0] + self.V_g[0][s0]),"\n")

            self.tau = self.disc_set[0][tau_idx]
            self.lambda_k = self._proj_0_xi(self.lambda_k + self.scale[k-1] * self.eta * (self.B - (self.tau + self.V_g[0][s0][tau_idx])))
            self._run_one_episode()

            # save data
            self.V_r_s0_tau_k[k] = self.V_r[0][s0][tau_idx]
            self.V_g_s0_tau_k[k] = self.V_g[0][s0][tau_idx]
            self.tau0_k[k] = self.tau
            self.lambda_s0_k[k] = self.lambda_k 

            # Save policy data for episode k
            policy_snapshot = {}
            for h in range(self.H):
                policy_snapshot[h] = {}
                for s in range(self.S[h]):
                    policy_snapshot[h][s] = {}
                    for c_hat_idx in range(self.C[h]):
                        policy_snapshot[h][s][c_hat_idx] = self.policy[h][s][c_hat_idx].copy()

            self.policy_k.append(policy_snapshot)

            # Print the iteration summary
            print(f"Iteration {k}: τ = {self.tau:.3f}, λ = {self.lambda_k:.3f}, "
                  f"average_reward = {np.mean(self.eps_rew_k[max(0, k - 20):k + 1]):.3f}, "
                  f"average_risk = {np.mean(self.eps_risk_k[max(0, k - 20):k + 1]):.3f}, "
                  f"V_r = {self.V_r_s0_tau_k[k]:.3f}, "
                  f"V_g = {self.V_g_s0_tau_k[k]:.3f}\n")

            # Inside the loop:
            if k % 1000 == 0:
                filename = os.path.join(result_path, f"run_alpha_{self.alpha:.5f}_Bon_{self.Bon}.npz")
                np.savez(
                    filename,
                    V_r_s0_tau=self.V_r_s0_tau_k[:k+1],
                    V_g_s0_tau=self.V_g_s0_tau_k[:k+1],
                    eps_rew=self.eps_rew_k[:k+1],
                    eps_risk=self.eps_risk_k[:k+1],
                    policy=self.policy_k,
                    tau=self.tau0_k[:k+1],
                    lambda_k=self.lambda_s0_k[:k+1]
                )