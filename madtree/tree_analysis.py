from tree_generation import deltas_factory, generate_tree, gaussian_densities, alternating_densities, update_tree
from tree_visualization import convert_to_nx, highest_reward_leaf, path_to_leaf, action_path, compute_action_distributions
from market_types import Action, ACTIONS, MarketTreeNode

from typing import Callable, Dict, List, Tuple, Any
import numpy as np
import random


def nonzero_initializations(amount = 10_000) -> Dict[str, List[Tuple[int, int, float]]]:
	r = lambda: random.randint(1, 20)

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


def proportion_initializations(amount = 1_000, precision = 11) -> Dict[str, List[Tuple[int, int, float]]]:
	"""
	From a starting capital and across a grid of possible distributions, compute
	some random prices and for each split cash and inventory accordingly.
	"""
	capital = 1000
	parameters = {}

	for p in np.linspace(0, 1, precision):
		p_params = []
		for _ in range(amount):
			price = random.random() * capital
			inventory = (capital * p) // price
			cash = capital - inventory * price

			assert (price * inventory + cash) == capital
			p_params.append((inventory, cash, price))
		parameters[f"{int(p * 100)}%"] = p_params

	return parameters


def analize_actions_spread(arguments: Dict[str, List[Tuple[int, int, float]]],
	time_horizon: int = 5, density: Callable[[int], tuple[list[float], list[float]]] = gaussian_densities):
	"""
	Using the non-zero init parameters compute the distribution of the best moves
	on the best path reward-wise and extract mean and variance.
	"""
	tree = MarketTreeNode(1, 1, 1)
	results = {}

	for name, args in arguments.items():
		actions_count = []

		for arg in args:
			deltas = deltas_factory(time_horizon, density)
			tree = update_tree(tree, time_horizon, deltas, *arg)
			path = path_to_leaf(tree, highest_reward_leaf(tree))
			actions = action_path(path)

			actions_count.append({
				action: actions.count(action) / len(actions) 
				for action in ACTIONS
			})

		results_mean = avg_dict(actions_count, ACTIONS)
		results_str = std_dict(actions_count, results_mean, ACTIONS)

		results[name] = {
			"mean": {a.name: r for a, r in results_mean.items()},
			"std": {a.name: r for a, r in results_str.items()}
		}

	return results


def avg_dict(dicts, keys):
	return {k: sum(d[k] for d in dicts) / len(dicts) for k in keys}


def std_dict(dicts, means, keys):
	return {k: np.sqrt(sum((d[k] - means[k]) ** 2 for d in dicts)) / len(dicts) for k in keys}


if __name__ == "__main__":
	time_horizon = 3
	tree = generate_tree(time_horizon, deltas_factory(time_horizon, gaussian_densities), 10, 10, 3)

	best_leaf = highest_reward_leaf(tree)
	path = path_to_leaf(tree, best_leaf)
	tree.print_tree()
	print(best_leaf)
	print([str(n) for n in path])
	print(action_path(path))