from market_types import MarketTreeNode
import matplotlib.pyplot as plt
from typing import Callable
from itertools import chain
import networkx as nx


def convert_to_nx(tree, deltas: list[Callable[[int], float]]):
	"""Represent the tree in graphiz dot"""
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

def highest_reward_paths(graph: nx.DiGraph):
	"""Find the path from the root to the leaves with the highest reward"""
	leaves = []
	max_reward = 0

	for node in graph:
		if graph.in_degree(node) == 0:
			root = node
			break

	for node in graph:
		if graph.out_degree(node) != 0: continue
		reward = graph.nodes[node]["reward"]

		if reward > max_reward:
			leaves = [node]
			max_reward = reward
		elif reward == max_reward:
			leaves.append(node)

	return chain(*(nx.all_simple_paths(graph, root, leaf) for leaf in leaves))


def draw_nx(graph: nx.DiGraph):
	fig, ax = plt.subplots()
	position = nx.nx_agraph.graphviz_layout(graph, prog="dot")

	edge_actions = nx.get_edge_attributes(graph, "action")
	edge_deltas = nx.get_edge_attributes(graph, "delta")
	nx.draw_networkx_edges(graph, position, ax=ax)
	nx.draw_networkx_edge_labels(graph, position, edge_labels=edge_actions)

	node_data = nx.get_node_attributes(graph, "data")
	node_rewards = nx.get_node_attributes(graph, "reward")
	nodes = nx.draw_networkx_nodes(graph, position, ax=ax, cmap=plt.cm.Blues, node_color=list(node_rewards.values()))

	for path in highest_reward_paths(graph):
		path_edgelist = list(zip(path, path[1:]))
		path_edge_actions = nx.get_edge_attributes(graph, "action")
		nx.draw_networkx_edges(graph, position, ax=ax, edgelist=path_edgelist, edge_color="blue")
		nx.draw_networkx_edge_labels(graph, position, ax=ax, edge_labels=path_edge_actions)

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


def draw_market_tree(tree, deltas: list[Callable[[int], float]]):
	graph = convert_to_nx(tree, deltas)
	draw_nx(graph)