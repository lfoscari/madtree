from market_types import Actions, MarketTreeNode
from typing import Callable
import random, math

def gaussian_deltas(time_horizon: int):
	"""Returns the functions defining the market density, for now it is just gaussian"""

	def delta(alpha: float, beta: float):
		"""Given a market density build the trading cost function"""
		def inner(quantity: int):
			match Actions(quantity):
				case Actions.BUY:
					price_impact = math.sqrt(2 * quantity / alpha)
				case Actions.SELL:
					price_impact = -math.sqrt(-2 * quantity / beta)
				case Actions.STAY:
					price_impact = 0
			return price_impact * 2 / 3
		return inner

	distr = lambda: max(random.normalvariate(0.2, 0.1), 0.05)
	return [delta(distr(), distr()) for _ in range(time_horizon)]


def generate_tree(time_horizon: int, deltas: list[Callable[int, float]]):
	"""Build the market tree up to the given depth and using the given market densities"""
	root = MarketTreeNode(0, 100, 0)
	queue = [root]

	while len(queue) > 0:
		node = queue.pop(0)
		if node.depth > time_horizon - 1: continue

		for action in [Actions.BUY, Actions.STAY, Actions.SELL]:
			if node.perform(action, deltas[node.depth]):
				queue.append(node.children[action])

	return root


def random_tree(time_horizon = 5):
	time_horizon = 5
	deltas = gaussian_deltas(time_horizon)
	tree = generate_tree(time_horizon, deltas)
	return tree, deltas
