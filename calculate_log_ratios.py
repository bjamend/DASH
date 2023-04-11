from numpy import log10


def log_ratio(element_name_1, element_name_2, element_abundance_1, element_abundance_2):
    """
    Calculate the log ratio of the abundances of two chemical elements: [element_2 / element_1]

    Inputs:
            element_name_1, element_name_2: strings
                    symbols representing the chemical elements
            element_abundance_1, element_abundance_2: floats
                    chemical abundances of each element
    """
    solar_abundance_1 = 0
    solar_abundance_2 = 0
    f1 = open(f"solar_abundances.txt", "r")
    for row in f1:
        row = row.split()
        if row[0] == element_name_1:
            solar_abundance_1 = float(row[1])
        if row[0] == element_name_2:
            solar_abundance_2 = float(row[1])

    solar_log_ratio = solar_abundance_2 - solar_abundance_1

    element_mass_1 = 0
    element_mass_2 = 0

    f2 = open(f"atomic_masses.txt", "r")
    for row in f2:
        row = row.split()
        if row[0] == element_name_1:
            element_mass_1 = float(row[1])
        if row[0] == element_name_2:
            element_mass_2 = float(row[1])

    model_log_ratio = (
        log10(
            (element_abundance_2 / element_abundance_1)
            * (element_mass_1 / element_mass_2)
        )
        - solar_log_ratio
    )

    return model_log_ratio
