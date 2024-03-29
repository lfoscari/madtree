from typing import Callable
from enum import Enum
import math

class Action(Enum):
	BUY = 1
	STAY = 0
	SELL = -1

ACTIONS = [Action.BUY, Action.STAY, Action.SELL]

class MarketTreeNode:
	def __init__(self, inventory: int = 0, cash: float = 0., price: float = 0., depth: int = 0):
		self.inventory = inventory
		self.cash = cash
		self.price = price
		self.depth = depth

		self.reward = cash + price * inventory
		self.children: dict[Action, MarketTreeNode] = dict()

	def can_perform(self, action: Action, delta: Callable[[int], float]) -> bool:
		"""Check that an action can be performed on a node without breaking constraints"""
		quantity = action.value
		price_change = delta(quantity)

		preconditions = action == Action.STAY \
			or (action == Action.BUY and self.price + price_change <= self.cash) \
			or (action == Action.SELL and self.inventory > 0)

		postconditions = self.inventory + quantity >= 0 \
			and self.cash - quantity * (self.price + price_change) >= 0 \
			and self.price + price_change >= 0
		
		return preconditions and postconditions

	def inherit(self, parent, action: int, delta: Callable[[int], float]):
		"""Update the current node to reflect the evolution of the parent node on the given action and delta"""
		quantity = action.value
		price_change = delta(quantity)

		self.inventory = parent.inventory + quantity
		self.cash = parent.cash - quantity * (parent.price + price_change)
		self.price = parent.price + price_change
		self.depth = parent.depth + 1
		self.reward = parent.reward + parent.inventory * price_change

		parent.buy_delta = delta(Action.BUY.value)
		parent.sell_delta = delta(Action.SELL.value)

		assert math.isclose(self.reward, self.cash + self.inventory * self.price), \
			("rewards mismatch", self.reward, self.cash + self.inventory * self.price)

	def perform(self, action: Action, delta: Callable[[int], float]) -> bool:
		"""Execute an action on the tree and create the corresponding child"""
		if not self.can_perform(action, delta):
			return False

		self.children[action] = MarketTreeNode()
		self.children[action].inherit(self, action, delta)
		return True

	def check_tree(self, deltas, time_horizon):
		"""
			Checks the given tree for consistency, making sure that all the
			nodes are valid and that no valid nodes are missing.
		"""
		queue = [self]
		
		while len(queue) > 0:
			node = queue.pop(0)
			if node.depth >= time_horizon: continue
			
			queue.extend(node.children.values())
			delta = deltas[node.depth]
			for action in ACTIONS:
				if node.can_perform(action, delta) and action not in node.children:
					print(action, node)
					return False

		return True

	def print_tree(self):
		queue = [self]
		
		while len(queue) > 0:
			node = queue.pop(0)
			print("- " * node.depth + str(node))
			queue.extend(node.children.values())

	def __str__(self):
		return f"I {self.inventory}, C {self.cash:1.3f}, P {self.price:1.3f}, D {self.depth}, R {self.reward:1.3f}"
