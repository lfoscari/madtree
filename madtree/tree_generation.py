from market_types import Actions, MarketTreeNode
from typing import Callable
import random, math


def gaussian_densities(time_horizon: int, mu = 0.2, std = 0.1):
	"""Generate uniform alphas and betas from the given interval"""
	distr = lambda: max(random.normalvariate(mu, std), 0.05)
	alphas = [distr() for _ in range(time_horizon)]
	betas = [distr() for _ in range(time_horizon)]
	return alphas, betas


def uniform_densities(time_horizon: int, interval = (0.05, 0.5)):
	"""Generate uniform alphas and betas from the given interval"""
	distr = lambda: random.uniform(*interval)
	alphas = [distr() for _ in range(time_horizon)]
	betas = [distr() for _ in range(time_horizon)]
	return alphas, betas


def constant_densities(time_horizon: int, interval = (0.05, 0.5)):
	"""Generate constant alphas and betas with two values from the given interval"""
	distr = lambda: random.uniform(*interval)
	alpha, beta = distr(), distr()
	alphas = [alpha for _ in range(time_horizon)]
	betas = [beta for _ in range(time_horizon)]
	return alphas, betas


def symmetrical_densities(time_horizon: int):
	"""Generate gaussian alphas and betas which are symmetrical"""
	alphas, _ = gaussian_densities(time_horizon)
	return alphas, alphas


def alternating_densities(time_horizon: int):
	"""Generate gaussian alphas and betas, but avoid having a more convenient side two or more times in a row"""
	alphas, betas = gaussian_densities(time_horizon)

	for index in range(1, time_horizon):
		if alphas[index - 1] > betas[index - 1] and alphas[index] > betas[index] or \
		   alphas[index - 1] < betas[index - 1] and alphas[index] < betas[index]:
			alphas[index], betas[index] = betas[index], alphas[index]

	return alphas, betas


def lipschitz_densities(time_horizon: int, start: float = 0.3, constant: float = 0.1):
	"""Generate lipschitz alphas and betas"""

	def lipschitz_gen(x: float):
		"""Pick a random next value for the lipschitz function"""
		while True:
			x = random.uniform(max(x - constant, 0.), x + constant)
			yield x
			
	alphas = [x for _, x in zip(range(time_horizon), lipschitz_gen(start))]
	betas = [x for _, x in zip(range(time_horizon), lipschitz_gen(start))]
	return alphas, betas


def deltas_factory(time_horizon: int, densities: Callable[[int], tuple[list[float], list[float]]]):
	"""Returns the functions defining the market density, for now it is just gaussian"""

	def delta(alpha: float, beta: float):
		"""Given a market density build the trading cost function"""
		def inner(quantity: int):
			match Actions(quantity):
				case Actions.BUY:
					price_impact = math.sqrt(2 * quantity / alpha)
				case Actions.SELL:
					price_impact = -math.sqrt(-2 * quantity / beta)
				case Actions.STAY:
					price_impact = 0
			return price_impact * 2 / 3
		return inner

	alphas, betas = densities(time_horizon)
	return [delta(a, b) for a, b in zip(alphas, betas)]


def generate_tree(time_horizon: int, deltas: list[Callable[[int], float]], inventory = 0, cash = 1, price = 0):
	"""Build the market tree up to the given depth and using the given market densities"""
	root = MarketTreeNode(inventory, cash, price)
	queue = [root]

	while len(queue) > 0:
		node = queue.pop(0)
		if node.depth > time_horizon - 1: continue

		for action in [Actions.BUY, Actions.STAY, Actions.SELL]:
			if node.perform(action, deltas[node.depth]):
				queue.append(node.children[action])

	return root
