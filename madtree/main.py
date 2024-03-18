from tree_analysis import analize_actions_spread, nonzero_initializations, proportion_initializations
from tree_visualization import draw_market_tree, bar_plot, stacked_bar_plot
from market_types import Action, ACTIONS, MarketTreeNode
from tree_generation import *

from multiprocessing import Pool, RLock, cpu_count
import matplotlib.pyplot as plt
from datetime import datetime
import os, json
from tqdm import tqdm

def analyze_density(density, arguments, destination, time_horizon, pid, description):
	# start = time.time()
	_, ax = plt.subplots()

	frequencies, order = analize_actions_spread(density, arguments, time_horizon, pid, description)
	frequencies_mean_t = {a.name: [r["mean"][a.name] for r in frequencies.values()] for a in ACTIONS}
	frequencies_std_t  = {a.name: [r["std"][a.name]  for r in frequencies.values()] for a in ACTIONS}

	bar_plot(ax, frequencies_mean_t, frequencies_std_t, f"{description} (T: {time_horizon})", total_width = .8, single_width = 1, labels = frequencies.keys())
	plt.savefig(f"{destination}/{density.__name__}_{time_horizon}.pdf")

	_, ax = plt.subplots()
	stacked_bar_plot(ax, order, "Action choices over rounds")
	plt.savefig(f"{destination}/{density.__name__}_{time_horizon}_order.pdf")

	# print(f"({destination.rsplit('/', 1)[1]})", density.__name__, f"{round(time.time() - start)}s")
	return { density.__name__: { "frequencies": frequencies, "order": order }}

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

	# now = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

	# if not os.path.exists(f"{destination}/non-zero"): os.makedirs(f"{destination}/non-zero")
	# nonzero = open(f"{destination}/non-zero/{now}_{time_horizon}.json", "w")

	# if not os.path.exists(f"{destination}/proportional"): os.makedirs(f"{destination}/proportional")
	# proportional = open(f"{destination}/proportional/{now}_{time_horizon}.json", "w")

	# non_zero_arguments = nonzero_initializations()
	# proportional_arguments = proportion_initializations()
		
	# with Pool(cpu_count() - 1, initargs=(RLock(), ), initializer = tqdm.set_lock) as pool:

	# 	nonzero_futures = [
	# 		pool.apply_async(analyze_density,
	# 			(density, non_zero_arguments, f"{destination}/non-zero", time_horizon, index, f"non-zero/{density.__name__}"))
	# 		for index, density in enumerate(densities)]

	# 	proportional_futures = [
	# 		pool.apply_async(analyze_density,
	# 			(density, proportional_arguments, f"{destination}/proportional", time_horizon, index, f"proportional/{density.__name__}"))
	# 		for index, density in enumerate(densities, start=len(densities))]

	# 	nonzero_results = [future.get() for future in nonzero_futures]
	# 	nonzero.writelines(json.dumps({k: v for res in nonzero_results for k, v in res.items()}, indent = 4))

	# 	proportional_results = [future.get() for future in proportional_futures]
	# 	proportional.writelines(json.dumps({k: v for res in proportional_results for k, v in res.items()}, indent = 4))

	# nonzero.close()
	# proportional.close()
