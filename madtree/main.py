from tree_visualization import draw_market_tree, bar_plot
from tree_analysis import analize_actions_spread
from market_types import Actions
from tree_generation import *

from multiprocessing import Pool
import matplotlib.pyplot as plt
from datetime import datetime
import time

def analyze_density(density, time_horizon = 3):
	start = time.time()
	fig, ax = plt.subplots()

	res = analize_actions_spread(time_horizon, density)
	res_t = {a.name: [v[a] for k, v in res.items()] for a in [Actions.BUY, Actions.STAY, Actions.SELL]}
	bar_plot(ax, res_t, density.__name__, total_width=.8, single_width=1, labels=res.keys())

	plt.savefig(f"results/{density.__name__}.pdf")
	print(density.__name__, time.time() - start, "s")
	return density.__name__ + "\n" + str({k: {a.name: p for (a, p) in v.items()} for (k, v) in res.items()}) + "\n"

if __name__ == "__main__":
	# density = gaussian_densities
	# deltas = deltas_factory(time_horizon, density)
	# tree = generate_tree(time_horizon, deltas, 10, 100, 50)
	# draw_market_tree(tree, deltas, density.__name__.replace("_", " ").capitalize())

	densities = [symmetrical_densities, lipschitz_densities, gaussian_densities, alternating_densities, constant_densities, uniform_densities]

	with open(f"results-{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}.txt", "w") as f:
		with Pool(len(densities)) as pool:
			results = pool.map(analyze_density, densities)
			f.writelines(results)