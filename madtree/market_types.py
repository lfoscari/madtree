from typing import Callable
from enum import Enum


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

		self.children: dict[Actions, MarketTreeNode] = dict()

	def perform(self, action: Actions, delta: Callable[[int], float]):
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
			del self.children[action]
			return False

		return True

	def __str__(self):
		return f"I {self.inventory}, C {self.cash:1.3f}, P {self.price:1.3f}, D {self.depth}, R {self.reward:1.3f}"
