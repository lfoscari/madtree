from market_types import Action, ACTIONS, MarketTreeNode

from typing import Callable
import random, math


def gaussian_densities(time_horizon: int, mu = 0.2, std = 0.1) -> tuple[list[float], list[float]]:
	"""Generate uniform alphas and betas from the given interval"""
	distr = lambda: max(random.normalvariate(mu, std), 0.05)
	alphas = [distr() for _ in range(time_horizon)]
	betas = [distr() for _ in range(time_horizon)]
	return alphas, betas


def uniform_densities(time_horizon: int, interval = (0.05, 0.5)) -> tuple[list[float], list[float]]:
	"""Generate uniform alphas and betas from the given interval"""
	distr = lambda: random.uniform(*interval)
	alphas = [distr() for _ in range(time_horizon)]
	betas = [distr() for _ in range(time_horizon)]
	return alphas, betas


def constant_densities(time_horizon: int, interval = (0.05, 0.5)) -> tuple[list[float], list[float]]:
	"""Generate constant alphas and betas with two values from the given interval"""
	distr = lambda: random.uniform(*interval)
	alpha, beta = distr(), distr()
	alphas = [alpha for _ in range(time_horizon)]
	betas = [beta for _ in range(time_horizon)]
	return alphas, betas


def symmetrical_densities(time_horizon: int) -> tuple[list[float], list[float]]:
	"""Generate gaussian alphas and betas which are symmetrical"""
	alphas, _ = gaussian_densities(time_horizon)
	return alphas, alphas


def alternating_densities(time_horizon: int) -> tuple[list[float], list[float]]:
	"""Generate gaussian alphas and betas, but avoid having a more convenient side two or more times in a row"""
	alphas, betas = gaussian_densities(time_horizon)

	for index in range(1, time_horizon):
		if alphas[index - 1] > betas[index - 1] and alphas[index] > betas[index] or \
		   alphas[index - 1] < betas[index - 1] and alphas[index] < betas[index]:
			alphas[index], betas[index] = betas[index], alphas[index]

	return alphas, betas


def lipschitz_densities(time_horizon: int, start: float = 0.3, constant: float = 0.1) -> tuple[list[float], list[float]]:
	"""Generate lipschitz alphas and betas"""

	def lipschitz_gen(x: float):
		"""Pick a random next value for the lipschitz function"""
		while True:
			x = random.uniform(max(x - constant, 0.), x + constant)
			yield x
			
	alphas = [x for _, x in zip(range(time_horizon), lipschitz_gen(start))]
	betas = [x for _, x in zip(range(time_horizon), lipschitz_gen(start))]
	return alphas, betas


def deltas_factory(time_horizon: int, densities: Callable[[int], tuple[list[float], list[float]]]) -> list[Callable[[int], float]]:
	"""Returns the functions defining the market density, for now it is just gaussian"""

	def delta(alpha: float, beta: float):
		"""Given a market density build the trading cost function"""
		def inner(quantity: int):
			match Action(quantity):
				case Action.BUY:
					price_impact = math.sqrt(2 * quantity / alpha)
				case Action.SELL:
					price_impact = -math.sqrt(-2 * quantity / beta)
				case Action.STAY:
					price_impact = 0
			return price_impact * 2 / 3
		return inner

	alphas, betas = densities(time_horizon)
	return [delta(a, b) for a, b in zip(alphas, betas)]

def flatten(tree: MarketTreeNode):
	"""Convert a tree into a list and remove children links"""
	result = []
	queue = [tree]

	while len(queue) > 0:
		node = queue.pop(0)
		queue.extend(node.children.values())
		result.append(node)
		node.children = dict()

	return result

# Cache used to store leaves to build bigger trees without runtime allocation
unused_leaves = []

def update_tree(root: MarketTreeNode, time_horizon: int, deltas: list[Callable[[int], float]], inventory: int, cash: float, price: float) -> MarketTreeNode:
	"""
		Given a tree and initialization parameters with deltas, update the tree to reflect the parameters.
	"""
	root.inventory, root.cash, root.price = inventory, cash, price
	root.reward = cash + price * inventory
	queue = [root]

	while len(queue) > 0:
		node = queue.pop(0)
		if node.depth >= time_horizon: continue

		delta = deltas[node.depth]
		for action in ACTIONS:
			if node.can_perform(action, delta):
				if action not in node.children:
					# Old tree did not contain the node, use one from the cache
					node.children[action] = unused_leaves.pop() \
						if len(unused_leaves) > 0 else MarketTreeNode()
				node.children[action].inherit(node, action, delta)
				queue.append(node.children[action])
			elif action in node.children:
				# Remove illegal node and store it in cache alongside its children
				unused_leaves.extend(flatten(node.children[action]))
				del node.children[action]

	return root


# def build_complete_tree(deltas, time_horizon = 5):
# 	"""
# 		Build a complete tree with 3^time_horizon leaves.
# 		A bit of a hack, it might not always work.
# 	"""
# 	root = MarketTreeNode(1000, 1000, 100)
# 	queue = [root]
	
# 	while len(queue) > 0:
# 		node = queue.pop(0)
# 		if node.depth >= time_horizon: continue

# 		for action in ACTIONS:
# 			if node.perform(action, deltas[node.depth]):
# 				queue.append(node.children[action])

# 	return root


def generate_tree(time_horizon: int, deltas: list[Callable[[int], float]], inventory = 0, cash = 1, price = 0) -> MarketTreeNode:
	"""Build the market tree up to the given depth and using the given market densities"""
	root = MarketTreeNode(inventory, cash, price)
	queue = [root]

	while len(queue) > 0:
		node = queue.pop(0)
		if node.depth >= time_horizon: break

		for action in ACTIONS:
			if node.perform(action, deltas[node.depth]):
				queue.append(node.children[action])

	return root

if __name__ == "__main__":
	from tree_generation import deltas_factory, gaussian_densities, lipschitz_densities
	from tree_visualization import draw_market_tree

	time_horizon = 3
	deltas = deltas_factory(time_horizon, gaussian_densities)
	complete = MarketTreeNode(1, 1, 1)

	first_update = update_tree(complete, time_horizon, deltas, 10, 10, 1)
	print(first_update.check_tree(deltas, time_horizon))
	draw_market_tree(first_update, deltas)

	second_update = update_tree(complete, time_horizon, deltas, 1000, 0, 100)
	print(second_update.check_tree(deltas, time_horizon))
	draw_market_tree(second_update, deltas)

	# first_update_again = update_tree(complete, (10, 10, 1))
