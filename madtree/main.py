from tree_generation import random_tree
from tree_visualization import draw_market_tree

if __name__ == "__main__":
	tree, deltas = random_tree()
	draw_market_tree(tree, deltas)
	