import matplotlib.pyplot as plt
from enum import Enum
import random, math
import graphviz

class Actions(Enum):
	BUY = 1
	STAY = 0
	SELL = -1

class MarketTreeNode:
	def __init__(self, inventory, cash, price, depth=0, reward=0.0):
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

	def perform(self, action, delta):
		"""Execute an action on the tree and create the corresponding child"""
		quantity = action.value
		
		if quantity == 1 and self.price > self.cash or quantity == -1 and self.inventory <= 0:
			return False

		self.children[action] = MarketTreeNode(
			self.inventory + quantity,
			# In the paper, quantity is assumed to be absolute, therefore the sign is changed
			self.cash + quantity * (self.price + delta(quantity)),
			self.price + delta(quantity),
			self.depth + 1,
			self.reward + self.inventory * delta(quantity),
		)

		if self.children[action].inventory < 0 or self.children[action].cash < 0 or self.children[action].price < 0:
			self.children[action] = None
			return False

		return True

	def __str__(self):
		return f"I: {self.inventory} | C: {self.cash:1.3f} | P: {self.price:1.3f} | R: {self.reward:1.3f}"


def generate_deltas(time_horizon):
	"""Returns the functions defining the market density, for it is just random"""

	def delta(alpha, beta):
		"""Given a market density build the trading cost function"""
		def inner(quantity):
			match Actions(quantity):
				case Actions.BUY:
					price_impact = math.sqrt(2 * quantity / alpha)
				case Actions.SELL:
					price_impact = -math.sqrt(2 * -quantity / beta)
				case Actions.STAY:
					price_impact = 0
			return price_impact * 2 / 3
		return inner

	distr = lambda: max(random.normalvariate(0.5, 0.1), 0.05)
	return [delta(distr(), distr()) for _ in range(time_horizon)]


def generate_tree(time_horizon, deltas):
	"""With a time horizon build the market tree"""
	root = MarketTreeNode(0, 1, 0)
	queue = [root]

	while len(queue) > 0 and time_horizon > 0:
		node = queue.pop(0)

		for action in [Actions.BUY, Actions.STAY, Actions.SELL]:
			if node.perform(action, deltas[node.depth]):
				queue.append(node.children[action])

		time_horizon -= 1

	return root

def max_reward(tree):
	"""Return the highest value of the reward among the nodes in the given tree"""
	queue = [tree]
	max_reward = 0

	while len(queue) > 0:
		node = queue.pop(0)
		max_reward = max(max_reward, node.reward)

		for child in node.children.values():
			if child is not None:
				queue.append(child)

	return max_reward

def graph_dot(tree, deltas):
	"""Represent the tree in graphiz dot"""
	graph = graphviz.Digraph(comment="Market configurations tree", node_attr={"ordering": "in", "shape": "record"})
	queue = [(None, tree)]

	max_reward_value = max_reward(tree)
	def node_color(node):
		normalied_reward = node.reward / max_reward_value
		return "#" + "".join(f"{int(255 * c):x}" for c in plt.cm.Blues(normalied_reward))

	while len(queue) > 0:
		_, node = queue.pop(0)
		graph.node(str(id(node)), label=str(node), style="filled", fillcolor=node_color(node))

		for (action, child) in node.children.items():
			if child is None: continue
			queue.append((action, child))
			graph.node(str(id(child)), label=str(child), style="filled", fillcolor=node_color(child))
			graph.edge(str(id(node)), str(id(child)), label=str(action.value))

	return graph

if __name__ == "__main__":
	time_horizon = 100
	deltas = generate_deltas(time_horizon)
	tree = generate_tree(time_horizon, deltas)

	dot = graph_dot(tree, deltas)
	dot.view()
