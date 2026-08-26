# SARSA on Cliff Walking
 
![Architecture](Docs/Architect.png)
 
## Overview
 
This project implements the SARSA (State-Action-Reward-State-Action) RL-algo to solve the Cliff Walking environment. The agent learns a policy to navigate from start to goal while avoiding a cliff, using an epsilon-greedy exploration strategy and a tabular Q-value update.
 
## Reference
 
- Library: [Gymnasium](https://gymnasium.farama.org/)
- Environment: `CliffWalking-v1`
  
## Requirements
 
```bash
pip install gymnasium
pip install "gymnasium[toy-text]"
```

- `SARSA_cliff.ipynb` - main notebook containing the SARSA implementation
  
## How It Works
 
1. Initialize a Q-table of shape `(48, 4)` (48 states, 4 actions).
2. Use an epsilon-greedy policy to select actions (explore vs exploit).
3. For each episode, take a step, observe the next state and action, and update the Q-value using the SARSA update rule:
   
```
   Q(s, a) += alpha * (reward + gamma * Q(s', a') - Q(s, a))
```
 
4. Repeat for a fixed number of episodes to let the agent learn the optimal policy.
5. Run the learned (greedy) policy with rendering to visualize the agent's behavior.

 
## Heads Up - 
 
- Rendering during training is disabled by default since it slows down training significantly.
- After training, the final cell runs the greedy policy with `render_mode="human"` to visually verify the learned behavior.