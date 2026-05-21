# EX-5: Implementation-of-MC-prediction-for-estimating-the-state-value-function

## Aim:

To implement the Monte Carlo (MC) Prediction algorithm for estimating the state-value function of an agent interacting with an environment and to calculate and plot the estimated state-value function.

---

## Objective:

- To understand Monte Carlo prediction in Reinforcement Learning.
- To estimate the value of each state using sampled episodes.
- To visualize the learned state-value function using a plot.

---

## Theory:

Monte Carlo Prediction is a model-free reinforcement learning method used to estimate the value function of a policy. It learns directly from complete episodes of interaction with the environment.

The state-value function is defined as:

\[
V(s) = E[G_t \mid S_t = s]
\]

Where:

- \(V(s)\) = Value of state \(s\)
- \(G_t\) = Return obtained from time step \(t\)

Monte Carlo methods calculate the average return obtained after visiting a state multiple times.

---

## Algorithm:

### Monte Carlo Prediction Algorithm:

**Step-1:** Initialize:
   - State-value function \(V(s)\)
   - Returns list for each state

**Step-2:** Generate an episode using the given policy.

**Step-3:** For each state appearing in the episode:
   - Calculate the return \(G\)
   - Store the return for that state
   - Update the value function using average return

**Step-4:** Repeat for many episodes.

**Step-5:** Plot the estimated state-value function.

---

## Program:

```python

#Implementation of MC prediction for estimating the state-value function and 
#Calculate and plot the state-value function estimate
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import gym

# Create Environment
env = gym.make("FrozenLake-v1", is_slippery=False)

# Parameters
gamma = 0.9
episodes = 5000

# State value function
V = defaultdict(float)

# Returns storage
returns = defaultdict(list)

# Random policy
def policy(state):
    return env.action_space.sample()

# Generate episode
def generate_episode():
    episode = []

    state, _ = env.reset()
    done = False

    while not done:
        action = policy(state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        episode.append((state, action, reward))

        state = next_state

    return episode

# Monte Carlo Prediction
for ep in range(episodes):

    episode = generate_episode()

    G = 0
    visited_states = set()

    # Traverse backward
    for t in reversed(range(len(episode))):

        state, action, reward = episode[t]

        G = gamma * G + reward

        # First-visit MC
        if state not in visited_states:

            returns[state].append(G)

            V[state] = np.mean(returns[state])

            visited_states.add(state)

# Print Values
print("State Value Function:\n")

for s in range(env.observation_space.n):
    print(f"State {s}: {V[s]:.3f}")

# Convert to 4x4 grid
value_grid = np.zeros((4,4))

for state in range(16):
    row = state // 4
    col = state % 4

    value_grid[row, col] = V[state]

# Plot
plt.figure(figsize=(6,6))

plt.imshow(value_grid)

for i in range(4):
    for j in range(4):
        plt.text(j, i,
                 round(value_grid[i,j],2),
                 ha='center',
                 va='center',
                 color='white',
                 fontsize=12)

plt.title("State Value Function Estimate")
plt.colorbar()
plt.show()

```
## Output:

```text

State Value Function:

State 0: 0.005
State 1: 0.005
State 2: 0.012
State 3: 0.007
State 4: 0.007
State 5: 0.000
State 6: 0.027
State 7: 0.000
State 8: 0.015
State 9: 0.052
State 10: 0.099
State 11: 0.000
State 12: 0.000
State 13: 0.138
State 14: 0.410
State 15: 0.000

```

### Output Graph:

The following heatmap is generated for the estimated state-value function:

- Darker colors represent lower state values.
- Terminal states have value 0.
- States farther from terminal states have larger negative values.

<img width="522" height="490" alt="image" src="https://github.com/user-attachments/assets/092dc491-54e6-4789-930f-eb3351f2b60d" />




---
## Result:

Thus, the Monte Carlo Prediction algorithm was successfully implemented for estimating the state-value function of the environment. The value of each state was calculated using sampled episodes, and the estimated state-value function was plotted successfully using a heatmap.




       

