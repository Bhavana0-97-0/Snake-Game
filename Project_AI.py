import numpy as np

# Grid size
rows, cols = 5, 5

# Terminal and blocked states
terminal_states = {(0, 4): 10, (1, 2): -3, (2, 3): 8, (4, 2): 6}
blocked_states = {(1, 1), (1, 3), (1, 4), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4), (4, 3), (4, 4)}

# Constants
actions = ['up', 'down', 'left', 'right']
gamma = 0.9
reward = -0.4
transition_probs = {'intended': 0.8, 'right_angle': 0.1}

# Movement vectors
directions = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1)
}

right_angles = {
    'up': [('left', 0.1), ('right', 0.1)],
    'down': [('left', 0.1), ('right', 0.1)],
    'left': [('up', 0.1), ('down', 0.1)],
    'right': [('up', 0.1), ('down', 0.1)],
}

def is_valid_state(r, c):
    return 0 <= r < rows and 0 <= c < cols and (r, c) not in blocked_states

def get_next_state(r, c, move):
    dr, dc = directions[move]
    nr, nc = r + dr, c + dc
    return (nr, nc) if is_valid_state(nr, nc) else (r, c)

def value_iteration(iter_counts):
    V = np.zeros((rows, cols))
    for (r, c), val in terminal_states.items():
        V[r][c] = val

    history = {}

    for iteration in range(1, max(iter_counts) + 1):
        new_V = np.copy(V)
        for r in range(rows):
            for c in range(cols):
                if (r, c) in terminal_states or (r, c) in blocked_states:
                    continue
                values = []
                for action in actions:
                    val = 0
                    # Intended direction
                    nr, nc = get_next_state(r, c, action)
                    val += transition_probs['intended'] * V[nr][nc]

                    # Right-angle moves
                    for alt_action, prob in right_angles[action]:
                        nr_alt, nc_alt = get_next_state(r, c, alt_action)
                        val += prob * V[nr_alt][nc_alt]

                    values.append(val)
                new_V[r][c] = reward + gamma * max(values)
        V = new_V
        if iteration in iter_counts:
            history[iteration] = np.round(np.copy(V), 8)

    return history


# Run value iteration for required iteration counts
iterations_to_print = [5, 10, 15, 20, 50, 100]
results = value_iteration(iterations_to_print)

# Print the utility values for each iteration
for it in iterations_to_print:
    print(f"\nUtility values after {it} iterations:")
    print(results[it])