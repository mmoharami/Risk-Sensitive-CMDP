import glob
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import re
from collections import defaultdict

# ----------------------------
# Utility Functions
# ----------------------------

def index_to_coord(s, h, rows, cols):
        count = -1
        for row in range(rows):
            col = h - row
            if 0 <= col < cols:
                count += 1
                if count == s:
                    return (row, col)
        raise IndexError(f"Invalid state index {s} at time {h}")

def disc_value(x, h, disc_set):
    """Discretize a value x at horizon h using disc_set."""
    disc = disc_set[h]
    idx = np.searchsorted(disc, x, side='left')
    return disc[min(idx, len(disc) - 1)]

def u_func(c, alpha): return (1 / alpha) * (np.exp(alpha * c) - 1)


def run_one_episode(env, policy, tau, disc_set, A, N_h, alpha, rng):
    """Simulate a single episode using a fixed policy, and update visit counts N_h."""
    s = env.s0
    c_hat = tau

    total_reward = 0
    total_utility = 0
    for h in range(env.H):
        # Find discretized budget index
        c_hat_idx = np.searchsorted(disc_set[h], c_hat, side='left')
        c_hat_idx = min(c_hat_idx, len(disc_set[h]) - 1)

        # Sample action from policy
        num_actions = A[h](s) if callable(A[h]) else A[h]
        a = rng.choice(range(num_actions), p=policy[h][s][c_hat_idx])

        # Environment step
        s_next, r, u = env.step(s, a, h)
        N_h[h][(s, a)] += 1

        s = s_next
        c_hat -= disc_value(u, h, disc_set)

        # Keep Track of utility and reward
        total_utility += u
        total_reward += r


    return total_reward, max(disc_set[env.H], key=lambda tau: tau + u_func(total_utility - tau, alpha))

def recent_avg(arr, window=100):
    """Compute a simple moving average."""
    if arr.ndim != 1:
        raise ValueError("Only 1D arrays are supported")
    return np.convolve(arr, np.ones(window)/window, mode='valid')

# ----------------------------
# Simulation Parameters
# ----------------------------

K = 15000
B = 2.2

# ----------------------------
# Plotting Parameters
# ----------------------------
trials = 1000
ave_window = 20
policy_window = 20

# Set up path to import sibling module
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from env.gridworld import GridWorld

# Load environment
env = GridWorld(parameters="grid_example_5x5.npz")

# Setup state and action spaces for each horizon
H = env.H
S, A = {}, {}
for h in range(H):
    S[h] = env.S(h) if callable(env.S) else env.S
    A[h] = env.A(h) if callable(env.A) else env.A
S[H] = 1  # Terminal step
rows = env.rows
cols = env.cols

# Create discretization sets
varepsilon = K ** (-1 / 2)
disc_set, C = {}, {}
for h in range(H+1):
    disc_set[h] = np.arange(-H, H, varepsilon)
    C[h] = len(disc_set[h])

# ----------------------------
# Load Results
# ----------------------------

# Locate result directory
result_path = os.path.join(parent_dir, 'results')
simulation_dir = glob.glob(os.path.join(result_path, f"K = {K}, B = {B}, *"))[0]

# Regex to extract alpha and method from filenames
filename_pattern = re.compile(r"run_alpha_([-.\d]+)_(Bon_[\w]+)\.npz")
alpha_method_data = defaultdict(dict)

# Load all .npz files
for fname in os.listdir(simulation_dir):
    match = filename_pattern.match(fname)
    if match:
        alpha = float(match.group(1))
        method = match.group(2)
        data = np.load(os.path.join(simulation_dir, fname), allow_pickle=True)
        alpha_method_data[(alpha, method)] = data

# Create output directory for figures
output_dir = os.path.join(simulation_dir, "figures")
os.makedirs(output_dir, exist_ok=True)

# Set up RNG
rng = np.random.default_rng(seed=42)

# Axis labels
ylabel_map = {
    "V_r_s0_tau": r"$V_{r,1}(s_0,\tau)$",
    "eps_rew": r"$\text{Empirical Reward}$",
    "V_g_s0_tau": r"$V_{g,1}(s_0,\tau)$",
    "eps_risk": r"$\text{Empirical Risk}$",
    "lambda_k": r"$\text{Dual Variable } (\lambda)$"
}


# ----------------------------
# Main Loop Over Experiments
# ----------------------------
for (alpha, method), data in sorted(alpha_method_data.items()):
    ave_window = min(ave_window, K)

    print(
        f"For alpha = {alpha}, the final values are:\n"
        f"  V_r_hat[s0] = {np.mean(data['V_r_s0_tau'][-ave_window:]):.4f}, "
        f"V_g_hat[s0] = {np.mean(data['V_g_s0_tau'][-ave_window:] + data['tau'][-ave_window:]):.4f},\n"
        f"  V_r_est[s0] = {np.mean(data['eps_rew'][-ave_window:]):.4f}, "
        f"V_g_est[s0] = {np.mean(data['eps_risk'][-ave_window:]):.4f}"
    )

    fig, axs = plt.subplots(1, 1, figsize=(25, 8))

    # 1. Reward subplot
    # if "V_r_s0_tau" in data and "eps_rew" in data:
    #     axs[0].plot(recent_avg(data["V_r_s0_tau"], ave_window), label=ylabel_map["V_r_s0_tau"])
    #     axs[0].plot(recent_avg(data["eps_rew"], ave_window), '--', label=ylabel_map["eps_rew"])

    #     axs[0].set_title("Reward", fontsize=24)  
    #     axs[0].legend(fontsize=20)              
    #     axs[0].grid(True)

    #     axs[0].tick_params(axis='both', which='major', labelsize=18)  # Increase tick label font size


    # # 2. Risk subplot
    # if "V_g_s0_tau" in data and "eps_risk" in data and "tau" in data:
    #     adjusted_risk = data["V_g_s0_tau"] + data["tau"]
    #     axs[1].plot(recent_avg(adjusted_risk, ave_window), label=ylabel_map["V_g_s0_tau"] + " + τ")
    #     axs[1].plot(recent_avg(data["eps_risk"], ave_window), '--', label=ylabel_map["eps_risk"])

    #     axs[1].set_title("Risk", fontsize=24)
    #     axs[1].legend(fontsize=20)
    #     axs[1].grid(True)

    #     axs[1].tick_params(axis='both', which='major', labelsize=18)  # Increase tick label font size


    # # 3. Lambda subplot
    # if "lambda_k" in data:
    #     axs[2].plot(recent_avg(data["lambda_k"], ave_window), label=ylabel_map["lambda_k"], color='purple')

    #     axs[2].set_title("Dual Variable λ", fontsize=24)
    #     axs[2].legend(fontsize=20)      
    #     axs[2].grid(True)

    #     axs[2].tick_params(axis='both', which='major', labelsize=18)  # Increase tick label font size


    # 4. Policy Heatmap subplot
    if "policy" in data:
        policies = list(data["policy"])
        policy_window = min(policy_window, len(policies))

        # Average last `policy_window` policies
        policy = {}
        for k in range(len(policies) - 1, len(policies) - policy_window - 1, -1):
            for h in range(env.H):
                policy.setdefault(h, {})
                for s in range(S[h]):
                    policy[h].setdefault(s, {})
                    for c_hat_idx in range(C[h]):
                        policy[h][s].setdefault(c_hat_idx, 0.0)
                        policy[h][s][c_hat_idx] += policies[k][h][s][c_hat_idx] / policy_window

        # Run trials and accumulate visit counts
        N_h = defaultdict(lambda: defaultdict(int))
        estimate_rew_tot = 0
        estimate_risk_tot = 0
        for _ in range(trials):
            estimate_rew, estimate_risk = run_one_episode(env, policy, data["tau"][-1], disc_set, A, N_h, alpha, rng)
            estimate_rew_tot += estimate_rew
            estimate_risk_tot += estimate_risk

        print(f"Estiamted reward = {estimate_rew_tot/trials:5f},Estiamted risk = {estimate_risk_tot/trials:5f}\n")

        # Build visit heatmap (split into directional counts)
        policy_right = np.zeros((rows, cols))
        policy_down = np.zeros((rows, cols))
        for h in range(env.H):
            for (s, a), count in N_h[h].items():
                row, col = index_to_coord(s, h, rows, cols)
                if a == 1:
                    policy_down[row][col] += count
                else:
                    policy_right[row][col] += count

        # Plot heatmap
        # ax = axs[3]
        ax = axs

        cmap = plt.cm.Blues
        norm = Normalize(vmin=0, vmax=1)

        for row in range(rows):
            for col in range(cols):
                total = policy_right[row, col] + policy_down[row, col]
                right_intensity = policy_right[row, col] / total if total > 0 else 0
                down_intensity = policy_down[row, col] / total if total > 0 else 0

                x, y = col, row
                ax.add_patch(patches.Polygon(
                    [[x, y+1], [x, y], [x+1, y+1]],
                    color=cmap(norm(down_intensity)),
                    edgecolor='gray'
                ))
                ax.add_patch(patches.Polygon(
                    [[x, y], [x+1, y], [x+1, y+1]],
                    color=cmap(norm(right_intensity)),
                    edgecolor='gray'
                ))

                # ax.text(x + 0.25, y + 0.75, int(policy_down[row, col]), fontsize=6, ha='center', va='center')
                # ax.text(x + 0.75, y + 0.25, int(policy_right[row, col]), fontsize=6, ha='center', va='center')


        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.invert_yaxis()

        # Center ticks in grid cells
        ax.set_xticks(np.arange(cols) + 0.5)
        ax.set_yticks(np.arange(rows) + 0.5)

        # Label them as 0, 1, ..., not 0.5, 1.5, ...
        ax.set_xticklabels(np.arange(cols), fontsize=18)
        ax.set_yticklabels(np.arange(rows), fontsize=18)

        # x-axis ticks and label on top
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.set_xlabel("Column", fontsize=20)

        # y-axis label on left
        ax.set_ylabel("Row", fontsize=20)

        # Keep grid aligned with cell borders (not affected by tick offset)
        ax.set_xticks(np.arange(cols + 1), minor=True)
        ax.set_yticks(np.arange(rows + 1), minor=True)
        ax.grid(True, which='minor', linestyle='--', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=18)

        ax.text(
            0.5, -0.05,  # x=center, y=just below axis (in axis coords)
            "Directional Action Heatmap",
            fontsize=24,
            ha='center',
            va='top',
            transform=ax.transAxes
        )

        # Add shared colorbar for intensity
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        # Add colorbar for the heatmap only
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        sm = ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("Normalized Action Frequency", fontsize=20)
        cbar.ax.tick_params(labelsize=20)
    # if "policy" in data:
    #     fig.suptitle(f"α = {alpha:.3f}, {method.replace('_', ' ')}, Estiamted reward = {estimate_rew_tot/trials:3f},Estiamted risk = {estimate_risk_tot/trials:3f}", fontsize=24)
    # else:
    #     fig.suptitle(f"α = {alpha:.3f}, {method.replace('_', ' ')}", fontsize=24)

    # # Save figure
    # safe_method = method.replace(" ", "_")
    # fig_filename = f"alpha_{alpha:.3f}_{safe_method}.png"
    # fig_path = os.path.join(output_dir, fig_filename)
    # fig.tight_layout(rect=[0, 0, 0.95, 0.95])
    # fig.savefig(fig_path, bbox_inches='tight')
    # plt.close(fig)

    # View Picture
    safe_method = method.replace(" ", "_")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()