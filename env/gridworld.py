import os
import numpy as np

class GridWorld:
    def __init__(self, parameters=None, rows=None, cols=None, H=None, seed=None):
        self.rng = np.random.default_rng(seed)

        # Grid size, rewards, utilities, and transition probabilities
        if parameters is None:
            if rows is None or cols is None:
                raise ValueError("Either 'parameters' or both 'rows' and 'cols' must be provided.")
            self.rows = rows
            self.cols = cols
            self.H = rows + cols - 1 if H is None else H
            self.rewards = self.rng.uniform(0, 1, size=(rows, cols))
            self.utilities = self.rng.uniform(0, 1, size=(rows, cols))
            self.transition_probs = self.rng.uniform(0, 1, size=(rows, cols))
        else:
            parent_dir = os.path.dirname(os.path.dirname(__file__))
            data_path = os.path.join(parent_dir, parameters)
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"File '{parameters}' not found.")
            data = np.load(data_path)
            self.rewards = data['r']
            self.utilities = data['u']
            self.transition_probs = data['p']
            self.rows, self.cols = self.rewards.shape
            self.H = int(data['H']) if 'H' in data else self.rows + self.cols - 1

        self.max_H = self.rows + self.cols - 1
        if self.H > self.max_H:
            raise ValueError(f"H = {self.H} is too long for a {self.rows}x{self.cols} grid.")

        # Starting state index at h = 0 is always (0, 0)
        self.s0 = 0

        # Action space: 0 = right, 1 = down
        self.A = 2

    # State space size at time step h
    def S(self, h):
        return sum(1 for i in range(self.rows) for j in range(self.cols) if i + j == h)

    # Convert state index s to (row, col) at time h
    def _index_to_coord(self, s, h):
        count = -1
        for row in range(self.rows):
            col = h - row
            if 0 <= col < self.cols:
                count += 1
                if count == s:
                    return (row, col)
        
        raise IndexError(f"Invalid state index {s} at time {h}")

    # Convert (row, col) to state index s at time h
    def _coord_to_index(self, row, col, h):

        if row + col != h or not (0 <= row < self.rows) or not (0 <= col < self.cols):
            raise ValueError(f"Invalid coordinate ({row}, {col}) for horizon {h}")
        index = 0
        for i in range(self.rows):
            j = h - i
            if 0 <= j < self.cols:
                if i == row:
                    return index
                index += 1
        raise ValueError(f"Coordinate ({row}, {col}) not found at time {h}")

    # Compute next state index after taking action a from state s at time h
    def _move(self, s, a, h):
        row, col = self._index_to_coord(s, h)

        if a == 0:  # right
            if col < self.cols - 1:
                col += 1
            elif row < self.rows - 1:
                row += 1  # fallback to down

        elif a == 1:  # down
            if row < self.rows - 1:
                row += 1
            elif col < self.cols - 1:
                col += 1  # fallback to right

        if row + col == h+1:
            return self._coord_to_index(row, col, h + 1)
        else:
            return self._coord_to_index(row, col, h)

    
     # Take one step in the environment from (s, h) using action a
    def step(self, s, a, h):
        row, col = self._index_to_coord(s, h)

        alt_a = 1 - a  # alternative action

        # Get possible outcomes and probabilities
        outcomes = [self._move(s, a, h), self._move(s, alt_a, h)]
        probs = [self.transition_probs[row, col], 1 - self.transition_probs[row, col]]
        s_next = self.rng.choice(outcomes, p=probs)

        return s_next, self.rewards[row, col], self.utilities[row, col]