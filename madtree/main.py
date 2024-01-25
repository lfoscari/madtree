from tree_generation import deltas_factory, generate_tree, gaussian_densities
from tree_analysis import analize_actions_spread
from tree_visualization import draw_market_tree

if __name__ == "__main__":
	# time_horizon = 4
	# deltas = deltas_factory(time_horizon, gaussian_densities)
	# tree = generate_tree(time_horizon, deltas, 0, 1, 0)
	# draw_market_tree(tree, deltas)

	analize_actions_spread()
	