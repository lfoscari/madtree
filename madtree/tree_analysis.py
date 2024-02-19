from tree_generation import deltas_factory, generate_tree, gaussian_densities, alternating_densities, update_tree
from tree_visualization import convert_to_nx, highest_reward_leaf, path_to_leaf, action_path, compute_action_distributions
from market_types import Actions, MarketTreeNode

from typing import Callable
import numpy as np
import random

# # TODO: calcola i parametri così:
# 	- estrai un prezzo
# 	- estrai una porzione p in [0, 1]
# 	- distribuisci un capitale iniziale fisso tra cash e inventario in base al prezzo alla proporzione p.

def nonzero_initializations(amount = 1_000) -> dict(str, list(int)):
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


def analize_nonzero_actions_spread(time_horizon: int = 5, density: Callable[[int], tuple[list[float], list[float]]] = gaussian_densities) -> dict(str, dict()):
	"""
	Using the non-zero init parameters compute the distribution of the best moves
	on the best path reward-wise and extract mean and variance.
	"""
	deltas = deltas_factory(time_horizon, density)
	init_arguments = initializations()
	tree = MarketTreeNode(1, 1, 1)

	for name, arguments in init_arguments.items():
		arg_distr = []

		for arg in arguments:
			tree = update_tree(tree, time_horizon, deltas, *arg)
			best_leaf = highest_reward_leaf(tree)
			path = path_to_leaf(tree, best_leaf)
			actions = action_path(path)

			arg_distr.append({
				action.name: actions.count(action) / len(actions) 
				for action in [Actions.BUY, Actions.STAY, Actions.SELL]
			})

		arg_results_mean = avg_dict(arg_distr, [Actions.BUY.name, Actions.STAY.name, Actions.SELL.name])
		arg_results_var = var_dict(arg_distr, arg_results_mean, [Actions.BUY.name, Actions.STAY.name, Actions.SELL.name])

		init_arguments[name] = {
			"mean": arg_results_mean,
			"var": arg_results_var
		}

	return init_arguments

def proportion_initializations(amount = 100, precision = 50) -> tuple((int, int, float)):
	"""
	From a starting capital and across a grid of possible distributions, compute
	some random prices and for each split cash and inventory accordingly.
	"""
	capital = 1000
	parameters = []

	for p in np.linspace(0, 1, precision):
		for _ in range(amount):
			price = random.random() * capital
			inventory = (capital * p) // price
			cash = capital - inventory * price

			assert price * inventory + cash == capital
			parameters.append((inventory, cash, price))

	return parameters


def analize_proportion_actions_spread(time_horizon: int = 5, density: Callable[[int], tuple[list[float], list[float]]] = gaussian_densities):
	"""
	Using the proportion init parameters compute the distribution of the best moves
	on the best path reward-wise and extract mean and variance.
	"""
	pass


def avg_dict(dicts, keys):
	return {k: sum(d[k] for d in dicts) / len(dicts) for k in keys}


def var_dict(dicts, means, keys):
	return {k: sum((d[k] - means[k]) ** 2 for d in dicts) / len(dicts) for k in keys}


if __name__ == "__main__":
	time_horizon = 3
	tree = generate_tree(time_horizon, deltas_factory(time_horizon, gaussian_densities), 10, 10, 3)

	best_leaf = highest_reward_leaf(tree)
	path = path_to_leaf(tree, best_leaf)
	tree.print_tree()
	print(best_leaf)
	print([str(n) for n in path])
	print(action_path(path))