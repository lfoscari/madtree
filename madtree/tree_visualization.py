from market_types import MarketTreeNode, Action

import matplotlib.pyplot as plt
from typing import Callable
from itertools import chain
import networkx as nx


def highest_reward_leaf(tree: MarketTreeNode) -> list[MarketTreeNode]:
	# I should check for multiple leaves with the highest reward, but it's very unlikely

	queue = [tree]
	max_reward = tree.reward
	leaves = []

	while len(queue) > 0:
		node = queue.pop(0)
		queue.extend(node.children.values())

		if len(node.children) == 0:
			leaves.append(node)
			max_reward = max(max_reward, node.reward)

	return next(leaf for leaf in leaves if leaf.reward == max_reward)


def path_to_leaf(root: MarketTreeNode, leaf: MarketTreeNode) -> list[MarketTreeNode]:
	def rec_path(path: list[MarketTreeNode], leaf: MarketTreeNode) -> bool:
		last_node = path[-1]
		if last_node == leaf:
			return True

		for child in last_node.children.values():
			path.append(child)
			if rec_path(path, leaf):
				return True
			path.pop()

		return False

	path = [root]
	rec_path(path, leaf)
	return path


def action_path(path: list[MarketTreeNode]) -> list[Action]:
	actions = []
	for depth in range(1, len(path)):
		parent, node = path[depth - 1], path[depth]
		action = next(action for action, child in parent.children.items() if child == node)
		actions.append(action)
	return actions

def convert_to_nx(tree: MarketTreeNode, deltas: list[Callable[[int], float]]):
	"""Represent the tree in NetworkX DiGraph"""
	graph = nx.DiGraph()
	queue = [(None, tree)]

	while len(queue) > 0:
		_, node = queue.pop(0)

		graph.add_node(
			id(node),
			inventory=node.inventory,
			cash=node.cash,
			price=node.price,
			depth=node.depth,
			reward=node.reward
		)

		if hasattr(node, "buy_delta"):
			graph.nodes[id(node)]["buy_delta"] = node.buy_delta
			graph.nodes[id(node)]["sell_delta"] = node.sell_delta

		for (action, child) in node.children.items():
			graph.add_edge(
				id(node), id(child),
				action=action.value,
				delta=deltas[node.depth](action.value)
			)
			queue.append((action, child))

	return graph


def nx_highest_reward_paths(graph: nx.DiGraph):
	"""Find the path from the root to the leaves with the highest reward"""
	leaves = [node for node in graph if graph.out_degree(node) == 0]
	max_reward = max(leaves, key=lambda node: graph.nodes[node]["reward"])["reward"]
	best_leaves = [leaf for leaf in leaves if graph.nodes[node]["reward"] == max_reward]

	return chain(*(nx.all_simple_paths(graph, root, leaf) for leaf in best_leaves))


def compute_action_distributions(graph: nx.DiGraph, path_edgelist: list[tuple[int, int]]):
	path_actions = [graph.get_edge_data(u, v)["action"] for (u, v) in path_edgelist]
	return {
		action: path_actions.count(action.value) / len(path_actions) 
		for action in ACTIONS
	}


def draw_nx(graph: nx.DiGraph, title=""):
	fig, ax = plt.subplots()
	position = nx.nx_agraph.graphviz_layout(graph, prog="dot")

	edge_actions = nx.get_edge_attributes(graph, "action")
	edge_deltas = nx.get_edge_attributes(graph, "delta")
	nx.draw_networkx_edges(graph, position, ax=ax)
	nx.draw_networkx_edge_labels(graph, position, edge_labels=edge_actions)

	node_data = nx.get_node_attributes(graph, "data")
	node_rewards = nx.get_node_attributes(graph, "reward")
	nodes = nx.draw_networkx_nodes(graph, position, ax=ax, cmap=plt.cm.Blues, node_color=list(node_rewards.values()))

	action_distributions = []
	for path in nx_highest_reward_paths(graph):
		path_edgelist = list(zip(path, path[1:]))
		action_distributions.append(compute_action_distributions(graph, path_edgelist))

		path_edge_actions = nx.get_edge_attributes(graph, "action")
		nx.draw_networkx_edges(graph, position, ax=ax, edgelist=path_edgelist, edge_color="blue")
		nx.draw_networkx_edge_labels(graph, position, ax=ax, edge_labels=path_edge_actions)

	avg_action_distribution = {
		action: sum(d[action] for d in action_distributions) / len(action_distributions)
		for action in ACTIONS
	}

	plt.title(f"{title if title is not None else ''}\nBest stategy action distribution: " + \
		', '.join(f'{a.name} {int(p * 100)}%' for (a, p) in avg_action_distribution.items()))

	annot = ax.annotate("", xy=(0,0), xytext=(20, 20), textcoords="offset points", 
		bbox=dict(fc="w"), arrowprops=dict(arrowstyle="->"))

	idx_to_node = {}
	for idx, node in enumerate(graph.nodes):
		idx_to_node[idx] = node

	def update_annot(ind):
		idx = ind["ind"][0]
		node = idx_to_node[idx]
		annot.xy = position[node]
		text = "\n".join(f"{k}: {v:.3f}" for k, v in graph.nodes[node].items())
		annot.set_text(text)

	def hover(event):
		if event.inaxes != ax: return

		vis = annot.get_visible()		
		cont, ind = nodes.contains(event)
		if cont:
			update_annot(ind)
			annot.set_visible(True)
			fig.canvas.draw_idle()
		elif vis:
			annot.set_visible(False)
			fig.canvas.draw_idle()

	annot.set_visible(False)
	fig.canvas.mpl_connect("motion_notify_event", hover)

	plt.show()


def draw_market_tree(tree: MarketTreeNode, deltas: list[Callable[[int], float]], title=None):
	graph = convert_to_nx(tree, deltas)
	draw_nx(graph, title)


def bar_plot(ax, data, errors=None, title=None, colors=None, total_width=0.8, single_width=1, legend=True, labels=None):
	"""Draws a bar plot with multiple bars per data point."""
	if title is not None:
		ax.set_title(title)
	
	if colors is None:
		colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

	n_bars = len(data)
	bar_width = total_width / n_bars
	bars = []

	for i, (name, values) in enumerate(data.items()):
		x_offset = (i - n_bars / 2) * bar_width + bar_width / 2
		for x, y in enumerate(values):
			bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])
			if errors and errors[name][x] != 0:
				plt.errorbar(x + x_offset, y, yerr=errors[name][x], color="black", capsize=4)
		bars.append(bar[0])
		

	if legend:
		ax.legend(bars, data.keys())

	if labels is not None:
		plt.xticks(range(len(labels)), labels)