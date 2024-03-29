We can rephrase the LOB game as a reinforcement learning problem. The states of the MDP contain information on the inventory $s^I_t \in \{ 0, \dots, T \}$, the cash $s^C_t \in \mathbb{R}$ and the price $s^P_t \in \mathbb{R}$. For every action $Q_t \in \{-1, 0, 1\}$ from any state $s_t = (s^I_t, s^C_t, s^P_t)$ we can compute the reward at time $t$ as $r_t(s_{t-1}, Q_t) = \delta_t(Q_t) s^I_{t-1}$ and the next state using the transition function $\Delta_t(s_{t-1}, Q_t) = (s^I_t + Q_t, s^C_t - Q_t(s^P_t + \delta_t(Q_t)), s^P_t + \delta_t(Q_t))$, where both change with time to reflect the change in density and therefore $\delta_t(\cdot)$.

The resulting MDP is structured as a tree, where each node has at most 3 children and we want to find a path from the root to a leaf with the highest cumulative reward.

As a safety check, ensure that the reward of each node is equal to the reward obtained by computing the $C_{t-1} + P_t I_{t-1}$, where $C_{t-1}$ and $I_{t-1}$ refer to the values of the father and $P_t$ is the price of the current node.

## Install

Remember to install the `graphviz` library used to visualize the dot file. Run `pip install .`, optionally inside a virtual environment, to install the required dependencies and then run the main file in the `madtree` subdirectory to generate a tree with Gaussian market densities, otherwise, edit said file.

In macOS, after having installed `graphviz` via Brew, if the environment setup fails due to `pygraphviz` try installing it manually with the command
```bash
python3 -m pip install \
	--config-setting="--global-option=build_ext" \
	--config-setting="--global-option=-I$(brew --prefix graphviz)/include/" \
	--config-setting="--global-option=-L$(brew --prefix graphviz)/lib/" \
	pygraphviz
```
