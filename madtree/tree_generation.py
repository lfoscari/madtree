from visualization.tree_visualization import draw_market_tree
from typing import Callable
from enum import Enum
import random, math

class Actions(Enum):
	BUY = 1
	STAY = 0
	SELL = -1

class MarketTreeNode:
	def __init__(self, inventory: int, cash: float, price: float, depth: int = 0, reward: float = 0.0):
		self.inventory = inventory
		self.cash = cash
		self.price = price
		self.depth = depth
		self.reward = reward

		self.children = {
			Actions.BUY: None,
			Actions.STAY: None,
			Actions.SELL: None
		}

	def perform(self, action: Actions, delta: Callable[int, float]):
		"""Execute an action on the tree and create the corresponding child"""
		quantity = action.value
		
		if quantity == 1 and self.price > self.cash or quantity == -1 and self.inventory <= 0:
			return False

		self.children[action] = MarketTreeNode(
			self.inventory + quantity,
			self.cash - quantity * (self.price + delta(quantity)),
			self.price + delta(quantity),
			self.depth + 1,
			self.reward + self.inventory * delta(quantity),
		)

		if self.children[action].inventory < 0 or self.children[action].cash < 0 or self.children[action].price < 0:
			self.children[action] = None
			return False

		return True

	def __str__(self):
		return f"I {self.inventory}, C {self.cash:1.3f}, P {self.price:1.3f}, D {self.depth}, R {self.reward:1.3f}"


def generate_deltas(time_horizon: int):
	"""Returns the functions defining the market density, for it is just random"""

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
	"""With a time horizon build the market tree"""
	root = MarketTreeNode(0, 100, 0)
	queue = [root]

	while len(queue) > 0:
		node = queue.pop(0)
		if node.depth > time_horizon - 1: continue

		for action in [Actions.BUY, Actions.STAY, Actions.SELL]:
			if node.perform(action, deltas[node.depth]):
				queue.append(node.children[action])

	return root

if __name__ == "__main__":
	time_horizon = 10
	deltas = generate_deltas(time_horizon)
	tree = generate_tree(time_horizon, deltas)
	draw_market_tree(tree, deltas)
