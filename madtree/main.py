from tree_analysis import analize_actions_spread, nonzero_initializations, proportion_initializations
from tree_visualization import draw_market_tree, bar_plot
from market_types import Action, ACTIONS, MarketTreeNode
from tree_generation import *

from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from functools import partial
from datetime import datetime
import time, os, json

def analyze_density(density, arguments, destination, time_horizon):
	start = time.time()
	fig, ax = plt.subplots()

	results = analize_actions_spread(arguments, time_horizon, density)
	results_mean_t = {a.name: [v["mean"][a.name] for k, v in results.items()] for a in ACTIONS}
	results_std_t = {a.name: [v["std"][a.name] for k, v in results.items()] for a in ACTIONS}
	
	bar_plot(ax, results_mean_t, results_std_t, density.__name__, total_width = .8, single_width = 1, labels = results.keys())

	plt.savefig(f"{destination}/{density.__name__}_{time_horizon}.pdf")
	print(f"({destination.rsplit('/', 1)[1]})", density.__name__, f"{round(time.time() - start)}s")
	return { density.__name__: results }

if __name__ == "__main__":
	time_horizon = 3
	
	density = gaussian_densities
	deltas = deltas_factory(time_horizon, density)
	tree = generate_tree(time_horizon, deltas, 10, 100, 50)
	draw_market_tree(tree, deltas, density.__name__.replace("_", " ").capitalize())

	# destination = "results"

	# densities = [
	# 	symmetrical_densities,
	# 	lipschitz_densities,
	# 	gaussian_densities,
	# 	alternating_densities,
	# 	constant_densities,
	# 	uniform_densities
	# ]

	# if not os.path.exists(f"{destination}/non-zero"): os.makedirs(f"{destination}/non-zero")
	# nonzero = open(f"{destination}/non-zero/{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}_{time_horizon}.json", "w")

	# if not os.path.exists(f"{destination}/proportional"): os.makedirs(f"{destination}/proportional")
	# proportional = open(f"{destination}/proportional/{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}_{time_horizon}.json", "w")

	# with Pool(cpu_count() - 1) as pool:
	# 	analyze_nonzero = partial(analyze_density,
	# 		destination = f"{destination}/non-zero",
	# 		arguments = nonzero_initializations(),
	# 		time_horizon = time_horizon
	# 	)
	# 	results_nonzero = pool.map_async(analyze_nonzero, densities)

	# 	analyze_proportional = partial(analyze_density,
	# 		destination = f"{destination}/proportional",
	# 		arguments = proportion_initializations(),
	# 		time_horizon = time_horizon
	# 	)
	# 	results_proportional = pool.map_async(analyze_proportional, densities)

	# 	nonzero.writelines(json.dumps({k: v for d in results_nonzero.get() for k, v in d.items()}, indent = 4))
	# 	proportional.writelines(json.dumps({k: v for d in results_proportional.get() for k, v in d.items()}, indent = 4))

	# nonzero.close()
	# proportional.close()
