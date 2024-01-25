from tree_visualization import convert_to_nx, highest_reward_paths, compute_action_distributions
from tree_generation import deltas_factory, gaussian_densities, generate_tree
from market_types import Actions
import random

def interesting_initializations(amount = 1000):
	r = lambda: random.randint(1, 100)

	I = [(r(), 0, 0) for _ in range(amount)]
	C = [(0, r(), 0) for _ in range(amount)]

	# When both cash is zero and inventory is zero the agent is stuck
	# P = [(0, 0, r()) for _ in range(amount)]
	# N = [(0, 0, 0) for _ in range(amount)]

	IC = [(r(), r(), 0) for _ in range(amount)]
	IP = [(r(), 0, r()) for _ in range(amount)]
	CP = [(0, r(), r()) for _ in range(amount)]

	ICP = [(r(), r(), r()) for _ in range(amount)]

	return {
		"I": I,
		"C": C,
		"IC": IC,
		"IP": IP,
		"CP": CP,
		"ICP": ICP
	}

def analize_actions_spread():
	time_horizon = 10

	for name, parameters in interesting_initializations().items():
		print(name, end=" => ")
		deltas = deltas_factory(time_horizon, gaussian_densities)
		
		for par in parameters:
			tree = generate_tree(time_horizon, deltas, *par)

			nx_tree = convert_to_nx(tree, deltas)
			paths = highest_reward_paths(nx_tree)
			distr = []

			for path in paths:
				edgelist = list(zip(path, path[1:]))
				distr.append(compute_action_distributions(nx_tree, edgelist))
				
		for action in [Actions.BUY, Actions.STAY, Actions.SELL]:
			print(action.name, sum(d[action] for d in distr) / len(distr), end=" ")
		print()
