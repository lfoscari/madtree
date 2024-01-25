from tree_visualization import convert_to_nx, highest_reward_paths, compute_action_distributions
from tree_generation import deltas_factory, generate_tree, gaussian_densities, alternating_densities
from market_types import Actions
import random

def interesting_initializations(amount = 100):
	"""
	Build random inputs for all the possible interesting initializations of the tree.
	"""
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

def avg_dict(dicts, keys):
	return {k: sum(d[k] for d in dicts) / len(dicts) for k in keys}

def analize_actions_spread():
	"""
	Using the interesting parameters average the distribution of
	the moves used in the best paths according to the reward of the leaves.
	"""
	print("I = inventory, C = cash, P = price")
	print("(e.g. 'IC' means that only inventory and cash are non-zero)\n")

	time_horizon = 10
	for name, parameters in interesting_initializations().items():
		print(name, end=" => ")
		deltas = deltas_factory(time_horizon, gaussian_densities)
		par_distr = []

		for par in parameters:
			tree = generate_tree(time_horizon, deltas, *par)

			nx_tree = convert_to_nx(tree, deltas)
			paths = highest_reward_paths(nx_tree)

			for path in paths:
				edgelist = list(zip(path, path[1:]))
				par_distr.append(compute_action_distributions(nx_tree, edgelist))

		par_results = avg_dict(par_distr, [Actions.BUY, Actions.STAY, Actions.SELL])
		print(" ".join(f"{a.name}: {p:.3f}" for (a, p) in par_results.items()))