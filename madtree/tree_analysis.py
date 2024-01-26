from tree_visualization import convert_to_nx, highest_reward_paths, compute_action_distributions
from tree_generation import deltas_factory, generate_tree, gaussian_densities, alternating_densities
from market_types import Actions
from typing import Callable
import random

def interesting_initializations(amount: int = 1000):
	"""
	Build random inputs for all the possible interesting initializations of the tree.
	I = inventory, C = cash, P = price (e.g. 'IC' means that only inventory and cash are non-zero)
	"""
	r = lambda: random.randint(1, 100)

	I = [(r(), 0, 0) for _ in range(amount)]
	C = [(0, r(), 0) for _ in range(amount)]
	P = [(0, 0, r()) for _ in range(amount)]

	IC = [(r(), r(), 0) for _ in range(amount)]
	IP = [(r(), 0, r()) for _ in range(amount)]
	CP = [(0, r(), r()) for _ in range(amount)]

	ICP = [(r(), r(), r()) for _ in range(amount)]

	return {
		"I": I,
		"C": C,
		"P": P,
		"IC": IC,
		"IP": IP,
		"CP": CP,
		"ICP": ICP
	}

def avg_dict(dicts, keys):
	return {k: sum(d[k] for d in dicts) / len(dicts) for k in keys}

def analize_actions_spread(time_horizon: int = 5, densities: Callable[[int], tuple[list[float], list[float]]] = gaussian_densities):
	"""
	Using the interesting parameters average the distribution of
	the moves used in the best paths according to the reward of the leaves.
	"""
	init_arguments = interesting_initializations()

	for name, arguments in init_arguments.items():
		deltas = deltas_factory(time_horizon, densities)
		arg_distr = []

		for arg in arguments:
			tree = generate_tree(time_horizon, deltas, *arg)

			nx_tree = convert_to_nx(tree, deltas)
			paths = highest_reward_paths(nx_tree)

			for path in paths:
				edgelist = list(zip(path, path[1:]))
				arg_distr.append(compute_action_distributions(nx_tree, edgelist))

		arg_results = avg_dict(arg_distr, [Actions.BUY, Actions.STAY, Actions.SELL])
		init_arguments[name] = arg_results

	return init_arguments