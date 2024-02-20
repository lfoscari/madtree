from typing import Callable
from enum import Enum


class Action(Enum):
	BUY = 1
	STAY = 0
	SELL = -1

ACTIONS = [Action.BUY, Action.STAY, Action.SELL]

class MarketTreeNode:
	def __init__(self, inventory: int, cash: float, price: float, depth: int = 0, reward: float = 0.0):
		self.inventory = inventory
		self.cash = cash
		self.price = price
		self.depth = depth
		self.reward = reward

		self.children: dict[Action, MarketTreeNode] = dict()

	def can_perform(self, action: Action, delta: Callable[[int], float]) -> bool:
		"""Should be checked before performing an action"""
		quantity = action.value
		delta_action = delta(quantity)

		preconditions = action == Action.STAY \
			or (action == Action.BUY and self.price + delta_action <= self.cash) \
			or (action == Action.SELL and self.inventory > 0)

		postconditions = self.inventory + quantity >= 0 \
			and self.cash - quantity * (self.price + delta_action) >= 0 \
			and self.price + delta_action >= 0
		
		return preconditions and postconditions

	def perform(self, action: Action, delta: Callable[[int], float]) -> bool:
		"""Execute an action on the tree and create the corresponding child"""
		quantity = action.value

		self.buy_delta = delta(Action.BUY.value)
		self.sell_delta = delta(Action.SELL.value)
		
		if not self.can_perform(action, delta):
			return False

		delta_action = delta(quantity)
		self.children[action] = MarketTreeNode(
			self.inventory + quantity,
			self.cash - quantity * (self.price + delta_action),
			self.price + delta_action,
			self.depth + 1,
			self.reward + self.inventory * delta_action,
		)

		return True

	def check_tree(self, deltas, time_horizon):
		"""
			Checks the given tree for consistency, making sure that all the
			nodes are valid and that no valid nodes are missing.
		"""
		queue = [self]
		
		while len(queue) > 0:
			node = queue.pop(0)
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
