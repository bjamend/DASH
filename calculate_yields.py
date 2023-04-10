import numpy as np
import re

cc_mass_list = [9.0, 9.25, 9.5, 9.75, 10.0,
		        10.5, 10.25, 10.75, 11.0, 11.5,
		        11.25, 11.75, 12.0, 12.25, 12.5,
		        12.75, 13.0, 13.1, 13.2, 13.3,
		        13.4, 13.5, 13.6, 13.7, 13.8,
		        13.9, 14.0, 14.1, 14.2, 14.3,
		        14.4, 14.5, 14.6, 14.7, 14.8,
		        14.9, 15.2, 15.7, 15.8, 15.9,
		        16.0, 16.1, 16.2, 16.3, 16.4,
		        16.5, 16.6, 16.7, 16.8, 16.9,
		        17.0, 17.1, 17.3, 17.4, 17.5,
		        17.6, 17.7, 17.9, 18.0, 18.1,
		        18.2, 18.3, 18.4, 18.5, 18.7,
		        18.8, 18.9, 19.0, 19.1, 19.2,
		        19.3, 19.4, 19.7, 19.8, 20.1,
		        20.2, 20.3, 20.4, 20.5, 20.6,
		        20.8, 21.0, 21.1, 21.2, 21.5,
		        21.6, 21.7, 25.2, 25.3, 25.4,
		        25.5, 25.6, 25.7, 25.8, 25.9,
		        26.0, 26.1, 26.2, 26.3, 26.4,
		        26.5, 26.6, 26.7, 26.8, 26.9,
		        27.0, 27.1, 27.2, 27.3, 27.4,
		        29.0, 29.1, 29.2, 29.6]

pn_mass_list = [1.00, 1.25, 1.50, 1.75, 1.90, 2.00,
				2.25, 2.50, 3.00, 3.50, 4.00, 4.50,
				5.00, 5.50, 6.00, 6.50]


def identify_cc_mass(m):
	arr = np.asarray(cc_mass_list)
	i = (np.abs(arr - m)).argmin()
	return arr[i]


def identify_pn_mass(m):
	arr = np.asarray(pn_mass_list)
	i = (np.abs(arr - m)).argmin()
	return arr[i]


def core_collapse_supernova_yields(m, elements):
	mass = identify_cc_mass(m)
	species = np.zeros(len(elements)+1)
	combined = []
	file = open(f"yields/core_collapse_supernovae/s{mass}.yield_table", 'r')
	next(file)
	for row in file:
	    row = row.split()
	    combined.append(float(row[1]) + float(row[2]))
	    match = re.match(R"([a-z]+)([0-9]+)", row[0], re.I)
	    if match:
	    	items = match.groups()
	    for element in elements:
		    if element == items[0]:
		    	species[elements.index(element)] += float(row[1]) + float(row[2])
	species[-1] = sum(combined) - sum(species)
	
	return species / sum(combined)


def planetary_nebula_yields(m, elements):
	mass = identify_pn_mass(m)
	species = np.zeros(len(elements)+1)
	combined = []
	file = open(f"yields/planetary_nebulae/{mass:.2f}.dat", 'r')
	next(file)
	for row in file:
	    row = row.split()
	    combined.append(float(row[8]))
	    match = re.match(R"([a-z]+)([0-9]+)", row[3], re.I)
	    if match:
	    	items = match.groups()
	    for element in elements:
		    if element == items[0]:
		    	species[elements.index(element)] += float(row[8])
	species[-1] = 1.0 - sum(species)
	
	return species


def type_ia_supernova_yields(elements):
	remnant_mass = 1.3779094971180308
	species = np.zeros(len(elements)+1)
	file = open(f"yields/type_ia_supernovae/typeia.dat", 'r')
	for row in file:
		row = row.split()
		match = re.match(R"([a-z]+)([0-9]+)", row[0], re.I)
		if match:
			items = match.groups()
		for element in elements:
			if element == items[0]:
				species[elements.index(element)] += float(row[1])
		species[-1] = remnant_mass - sum(species)
	return species / remnant_mass


def binary_neutron_star_merger_yields(elements):
	remnant_mass = 1.4
	species = np.zeros(len(elements)+1)
	file = open(f"yields/binary_neutron_star_mergers/bnsyields.dat", 'r')
	for row in file:
		row = row.split()
		match = re.match(R"([a-z]+)([0-9]+)", row[0], re.I)
		if match:
			items = match.groups()
		for element in elements:
			if element == items[0]:
				species[elements.index(element)] += float(row[1])
		species[-1] = remnant_mass - sum(species)
	return species / remnant_mass


def gas_infall_abundances(elements):
	abundances = np.zeros(len(elements)+1)
	abundances[0] = 1
	return abundances