from tree_generation import tree, gaussian_densities, alternating_densities
from tree_visualization import draw_market_tree

if __name__ == "__main__":
	tree, deltas = tree(4, alternating_densities)
	draw_market_tree(tree, deltas)
	