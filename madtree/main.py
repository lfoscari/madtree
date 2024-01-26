from tree_visualization import draw_market_tree, bar_plot
from tree_analysis import analize_actions_spread
from market_types import Actions
import matplotlib.pyplot as plt
from tree_generation import *

if __name__ == "__main__":
	time_horizon = 10

	# deltas = deltas_factory(time_horizon, gaussian_densities)
	# tree = generate_tree(time_horizon, deltas, 0, 1, 0)
	# draw_market_tree(tree, deltas)

	for density in [gaussian_densities, alternating_densities, constant_densities, uniform_densities, symmetrical_densities, lipschitz_densities]:
		fig, ax = plt.subplots()

		res = analize_actions_spread(time_horizon, density)
		res_t = {a.name: [v[a] for k, v in res.items()] for a in [Actions.BUY, Actions.STAY, Actions.SELL]}
		bar_plot(ax, res_t, total_width=.8, single_width=1, labels=res.keys())

		plt.savefig(f"results/{density.__name__}.pdf")