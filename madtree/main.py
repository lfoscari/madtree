from tree_visualization import draw_market_tree, bar_plot
from tree_analysis import analize_actions_spread
from market_types import Actions
import matplotlib.pyplot as plt
from tree_generation import *

if __name__ == "__main__":
	time_horizon = 4

	# density = gaussian_densities
	# deltas = deltas_factory(time_horizon, density)
	# tree = generate_tree(time_horizon, deltas, 10, 100, 100)
	# draw_market_tree(tree, deltas, density.__name__.replace("_", " ").capitalize())

	results = open("results/output.txt")

	for density in [symmetrical_densities, lipschitz_densities, gaussian_densities, alternating_densities, constant_densities, uniform_densities]:
		print("Computing", density.__name__, "...")
		fig, ax = plt.subplots()

		res = analize_actions_spread(time_horizon, density)
		res_t = {a.name: [v[a] for k, v in res.items()] for a in [Actions.BUY, Actions.STAY, Actions.SELL]}
		bar_plot(ax, res_t, density.__name__, total_width=.8, single_width=1, labels=res.keys())

		results.write(density.__name__ + "\n" + str(res))
		plt.savefig(f"results/{density.__name__}.pdf")