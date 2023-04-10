import matplotlib.pyplot as plt
from numpy import log10
from random import choice
from access_database import data_collection
from calculate_log_ratios import log_ratio

def plot(stars, elements, element_1, element_2):
	x = []
	y = []

	for i in range(1000):
		star_selection = choice(stars)

		x1 = 0.0
		x2 = 0.0
		y1 = 0.0
		y2 = 0.0

		for j in range(len(elements)):
			if elements[j] == "h":
				x1 = star_selection.composition[j]
			if elements[j] == "fe":
				x2 = star_selection.composition[j]
			if elements[j] == element_1:
				y1 = star_selection.composition[j]
			if elements[j] == element_2:
				y2 = star_selection.composition[j]

		x.append(log_ratio("h", "fe", x1, x2))
		y.append(log_ratio(element_1, element_2, y1, y2))

	data = data_collection(element_1, element_2)

	x_observations = []
	y_observations = []

	for row in data[1:]:
		row = row.split()
		x_observations.append(float(row[0]))
		y_observations.append(float(row[1]))

	plt.rcParams.update({
	"text.usetex": True,
	"font.family": "serif",
	"font.sans-serif": ["Times New Roman"]})

	fig = plt.figure(figsize=(9, 4.5))
	plt.scatter(x, y, color='red', s=5, label='Model', zorder=1)
	plt.scatter(x_observations, y_observations, s=30, color='k',
				marker='+', linewidths=1.0,
				label='SAGA Database Observations', zorder=0)
	plt.ylim(-1.5, 2.0)
	plt.xlim(-3.0, 0.5)
	plt.xlabel('[Fe/H]')
	plt.ylabel(f'[{element_2.capitalize()}/{element_1.capitalize()}]')
	plt.tight_layout()
	plt.legend()
	plt.show()