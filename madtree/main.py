from tree_visualization import draw_market_tree, bar_plot
from tree_analysis import analize_actions_spread
from market_types import Actions, MarketTreeNode
from tree_generation import *

from multiprocessing import Pool
import matplotlib.pyplot as plt
from datetime import datetime
import time
import json

def analyze_density(density, time_horizon = 10):
	start = time.time()
	fig, ax = plt.subplots()
	res = analize_nonzero_actions_spread(time_horizon, density)

	res_mean_t = {a.name: [v["mean"][a.name] for k, v in res.items()] for a in [Actions.BUY, Actions.STAY, Actions.SELL]}
	res_var_t = {a.name: [v["var"][a.name] for k, v in res.items()] for a in [Actions.BUY, Actions.STAY, Actions.SELL]}
	
	bar_plot(ax, res_mean_t, res_var_t, density.__name__, total_width = .8, single_width = 1, labels = res.keys())

	plt.savefig(f"results/{density.__name__}.pdf")
	print(density.__name__, round(time.time() - start), "s")
	return { density.__name__: res }

if __name__ == "__main__":
	# density = gaussian_densities
	# deltas = deltas_factory(time_horizon, density)
	# tree = generate_tree(time_horizon, deltas, 10, 100, 50)
	# draw_market_tree(tree, deltas, density.__name__.replace("_", " ").capitalize())

	densities = [symmetrical_densities, lipschitz_densities, gaussian_densities, alternating_densities, constant_densities, uniform_densities]

	with open(f"results-{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}.txt", "w") as f:
		with Pool(len(densities)) as pool:
			results = pool.map(analyze_density, densities)
			f.writelines(json.dumps({k: v for d in results for k, v in d.items()}, indent = 4))