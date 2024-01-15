import matplotlib.pyplot as plt
from enum import Enum
import random
import graphviz

class Actions(Enum):
	BUY = 1
	STAY = 0
	SELL = -1

class MarketTreeNode:
	def __init__(self, inventory, cash, price, depth=0):
		self.inventory = inventory
		self.cash = cash
		self.price = price
		self.depth = depth

		self.children = {
			Actions.BUY: None,
			Actions.STAY: None,
			Actions.SELL: None
		}

	def perform(self, quantity, delta):
		"""Execute an action on the tree and build the corresponding children"""
		if (quantity.value == 1 and self.price > self.cash) or (quantity.value == -1 and self.inventory <= 0):
			return False

		self.children[quantity] = MarketTreeNode( \
			self.inventory + quantity.value, \
			self.cash - quantity.value * (self.price + delta(quantity.value)), \
			self.price + delta(quantity.value), \
			self.depth + 1
		)

		if self.children[quantity].inventory < 0 or self.children[quantity].cash < 0 or self.children[quantity].price < 0:
			# This can happen because alphas and betas are random
			self.children[quantity] = None
			return False

		return True

	def depth(self): return self.depth

	def reward(self, quantity, delta):
		"""Return the reward obtained executing action {quantity} of this state"""
		return delta(quantity.value) * self.inventory

	def __str__(self):
		return f"I: {self.inventory} | C: {self.cash:1.3f} | P: {self.price:1.3f}"

def delta(alpha, beta):
	"""Given a market density build the trading cost function"""
	return lambda quantity: \
		alpha * quantity if quantity > 0 else beta * quantity

def generate_deltas(time_horizon):
	"""Returns the functions defining the market density, for now just randomly"""
	return [delta(random.random(), random.random()) for _ in range(time_horizon)]

def generate_tree(time_horizon, deltas):
	"""With a time horizon build the market tree"""
	root = MarketTreeNode(0, 1, 0)
	queue = [root]

	while len(queue) > 0 and time_horizon >= 0:
		node = queue.pop(0)

		for action in [Actions.BUY, Actions.STAY, Actions.SELL]:
			if node.perform(action, deltas[node.depth]):
				queue.append(node.children[action])
				
		time_horizon -= 1

	return root

def minmax_reward(tree, deltas):
	queue = [(action, tree) for action in [Actions.BUY, Actions.STAY, Actions.SELL]]
	rewards = []

	while len(queue) > 0:
		action, node = queue.pop(0)
		reward = node.reward(action, deltas[node.depth])
		rewards.append(reward)

		for (action, child) in node.children.items():
			if child is None: continue
			queue.append((action, child))			

	return min(rewards), max(rewards)

def graph_dot(tree, deltas):
	graph = graphviz.Digraph(comment="Market configurations tree", node_attr={"ordering": "in", "shape": "record"})
	queue = [(None, tree)]
	graph.node(str(id(tree)), label=str(tree))

	min_reward, max_reward = minmax_reward(tree, deltas)
	def node_color(node, action):
		normalied_reward = node.reward(action, deltas[node.depth]) / max_reward * min_reward
		return "#" + "".join(f"{int(255 * c):x}" for c in plt.cm.jet(normalied_reward))

	while len(queue) > 0:
		action, node = queue.pop(0)
		if action is not None:
			graph.node(str(id(node)), label=str(node), style="filled", fillcolor=node_color(node, action))

		for (action, child) in node.children.items():
			if child is None: continue
			queue.append((action, child))
			graph.node(str(id(child)), label=str(child), style="filled", fillcolor=node_color(child, action))
			graph.edge(str(id(node)), str(id(child)), label=str(action.value) + f" (r={child.reward(action, deltas[child.depth]):1.4f})")

	return graph

if __name__ == "__main__":
	time_horizon = 100
	deltas = generate_deltas(time_horizon)
	tree = generate_tree(time_horizon, deltas)

	dot = graph_dot(tree, deltas)
	dot.view()
