import matplotlib.pyplot as plt
from numpy import log10
from random import choice
from access_database import data_collection
from calculate_log_ratios import log_ratio


def abundance_pattern(stars, elements, element_1, element_2, num_stars):
    """
    Produce an abundance pattern plot of [element_2/element_1] vs. [Fe/H].

    Inputs:
            stars: class instance array
                    list of stars still alive at the end of the simulation
            elements: string tuple
                    list of symbols for tracked chemical elements
            element_1, element_2: string
                    names of chemical elements being plotted
            num_stars: int
                    number of stars to pull chemical abundances from
    """
    x = []
    y = []

    counter = 0

    while counter < num_stars:
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

        if x1 != 0.0:
            if x2 != 0.0:
                if y1 != 0.0:
                    if y2 != 0.0:
                        x.append(log_ratio("h", "fe", x1, x2))
                        y.append(log_ratio(element_1, element_2, y1, y2))
                        counter += 1

    data = data_collection(element_1, element_2)

    x_observations = []
    y_observations = []

    for row in data[1:]:
        row = row.split()
        x_observations.append(float(row[0]))
        y_observations.append(float(row[1]))

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.sans-serif": ["Times New Roman"],
        }
    )

    fig = plt.figure(figsize=(9, 4.5))
    plt.scatter(x, y, color="red", s=5, label="Model", zorder=1)
    plt.scatter(
        x_observations,
        y_observations,
        s=30,
        color="k",
        marker="+",
        linewidths=1.0,
        label="SAGA Database Observations",
        zorder=0,
    )
    plt.ylim(-1.0, 1.5)
    plt.xlim(-3.0, 1.0)
    plt.xlabel("[Fe/H]")
    plt.ylabel(f"[{element_2.capitalize()}/{element_1.capitalize()}]")
    plt.tight_layout()
    plt.legend()
    plt.show()


def age_metallicity(stars, elements, num_stars):
    """
    Plot the age-metallicity relation ([Fe/H] vs. elapsed time).
    """

    times = []
    x = []

    counter = 0

    while counter < num_stars:
        star_selection = choice(stars)

        x1, x2 = 0.0, 0.0

        for i in range(len(elements)):
            if elements[i] == "h":
                x1 = star_selection.composition[i]
            if elements[i] == "fe":
                x2 = star_selection.composition[i]

        if x1 != 0.0:
            if x2 != 0.0:
                x.append(log_ratio("h", "fe", x1, x2))
                times.append(13.6 - star_selection.age)
                counter += 1

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.sans-serif": ["Times New Roman"],
        }
    )

    fig = plt.figure(figsize=(9, 4.5))
    plt.scatter(times, x, color="k", s=15)
    plt.xlim(0.0, 13.6)
    plt.ylim(-4.0, 1.0)
    plt.xlabel("Time (Gyr)")
    plt.ylabel("[Fe/H]")
    plt.show()
