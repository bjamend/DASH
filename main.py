from random import randint, seed, uniform
from numpy import array, exp, zeros
import matplotlib.pyplot as plt
from calculate_yields import (
    core_collapse_supernova_yields,
    gas_infall_abundances,
    type_ia_supernova_yields,
    planetary_nebula_yields,
    binary_neutron_star_merger_yields,
    gas_outflow_abundances,
)
from plot_abundance_pattern import abundance_pattern, age_metallicity


# Random Seed for Repeatable Runs
seed(487)

# Elements to Evolve
elements = ("h", "fe", "o", "mg", "eu")

# Milky Way Parameters
present_day_total_surface_mass_density = 54.0  # [M. / pc^2]
galaxy_age = 13.6  # [Gyr]

# Initial Mass Function Parameters
lower_mass_bound = 0.1  # [M.]
upper_mass_bound = 100.0  # [M.]
kroupa_threshold_mass = 0.5  # [M.]
salpeter_power = -2.35
kroupa_power_1 = -1.3
kroupa_power_2 = -2.3

# Gas Infall Parameters
gas_infall_timescale = 7.0  # [Gyr]

# Gas Outflow Parameters
mass_loading_factor = 0.72

# Star Formation Parameters (Kennicutt-Schmidt)
ks_star_formation_rate_power = 1.4
unit_surface_density = 1.0  # [M. / pc^2]
star_formation_efficiency = 2.0  # [M. / Gyr / pc^2]


class starParticle:
    """
    A class to represent a star particle, itself representative of a population of identical stars.

    Attributes:
            age: float
                    current age of the star [Gyr]
            mass: float
                    mass of the star as sampled from the IMF [M.]
            statistical_weight: float
                    weighting of the intrinsic mass relative to the gas density available for star formation [1 / pc^2]
            classification: string
                    main-sequence, white-dwarf, black-hole, or neutron-star
            composition: float array
                    fractional abundances of all atomic species in the star, normalized to 1 [M. / pc^2]

    Methods:
            stellar_lifetime():
                    Calculates the lifetime of a newborn star. [Gyr]
    """

    def __init__(self, age, mass, statistical_weight, classification, composition):
        self.age = age
        self.mass = mass
        self.statistical_weight = statistical_weight
        self.classification = classification
        self.composition = composition

    def stellar_lifetime(self):
        """
        Return the lifetime of a star given its mass. [Gyr]
        """
        lifetime = 10.0 * self.mass ** (-2.5)
        return lifetime


def sample_mass(imf):
    """
    Return a stellar mass randomly sampled from an initial mass
    function (IMF), hardcoded here as the Salpeter IMF. [M.]

    Inputs:
            imf: string
                    name of initial mass function
    """
    m_l = lower_mass_bound
    m_u = upper_mass_bound
    u = uniform(0, 1)

    if imf == "salpeter":
        p = salpeter_power
        mass = (u * (m_u ** (p + 1) - m_l ** (p + 1)) + m_l ** (p + 1)) ** (1 / (p + 1))

    if imf == "kroupa":
        m_t = kroupa_threshold_mass
        alpha_1 = kroupa_power_1
        alpha_2 = kroupa_power_2
        A = 1.0 / (
            (m_t ** (alpha_1 + 1) - m_l ** (alpha_1 + 1)) / (alpha_1 + 1)
            + (m_u ** (alpha_2 + 1) - m_t ** (alpha_2 + 1))
            * m_t ** (alpha_1 - alpha_2)
            / (alpha_2 + 1)
        )
        u_threshold = A / (alpha_1 + 1) * (m_t ** (alpha_1 + 1) - m_l ** (alpha_1 + 1))
        if u <= u_threshold:
            mass = (u * (alpha_1 + 1) / A + m_l ** (alpha_1 + 1)) ** (1 / (alpha_1 + 1))
        else:
            mass = (
                (u - u_threshold) * (alpha_2 + 1) / A / m_t ** (alpha_1 - alpha_2)
                + m_t ** (alpha_2 + 1)
            ) ** (1 / (alpha_2 + 1))

    return mass


def gas_infall_rate(t):
    """
    Return the surface rate density at which gas falls into the galaxy
    from the circumgalactic medium (CGM). [M. / Gyr / pc^2]

    Inputs:
            t: float
                    current simulation time [Gyr]
    """
    sigma_tot = present_day_total_surface_mass_density
    tau = gas_infall_timescale
    age = galaxy_age
    infall = exp(-t / tau) * sigma_tot / tau / (1.0 - exp(-age / tau))

    return infall


def gas_outflow_rate(t, star_formation_rate):
    """
    Return the surface rate density at which gas flows out of the
    galaxy. [M. / Gyr / pc^2]

    Inputs:
            t: float
                    current simulation time [Gyr]
            current_stellar_mass: float
                    mass of all stars in the galaxy at the current time [M.]
    """
    outflow = mass_loading_factor * star_formation_rate

    return outflow


def star_formation_rate(sigma_gas):
    """
    Return the surface rate density at which stars form from the gas
    comprising the interstellar medium, hardcoded here according to
    the Kennicutt-Schmidt SFR law. [M. / Gyr / pc^2]

    Inputs:
            sigma_gas: float
                    surface gas density [M. / pc^2]
    """
    k = ks_star_formation_rate_power
    nu = star_formation_efficiency
    sigma_0 = unit_surface_density
    sfr = nu * (sigma_gas / sigma_0) ** k

    return sfr


def form_stars(num_stars, sigma_gas, galaxy_age):
    """
    Return array of 'stars' (starParticle class instances).

    Inputs:
            num_stars: float
                    number of stars to be formed
            gas_mass: float array
                    mass by species of interstellar gas [M.]
            galaxy_age: float
                    age of the galaxy [Gyr]
    """

    stars = []

    for i in range(num_stars):
        star_age = 0.0
        star_mass = sample_mass("kroupa")
        star_statistical_weight = 1.0
        star_classification = "main-sequence"
        star_composition = sigma_gas / sum(sigma_gas)
        star = starParticle(
            star_age,
            star_mass,
            star_statistical_weight,
            star_classification,
            star_composition,
        )
        stars.append(star)

    return stars


def total_stellar_mass(stars):
    """
    Return sum of sampled masses of newborn stars. [M.]

    Inputs:
            stars: class instance array
                    list of starParticle objects
    """
    stellar_mass = 0.0

    for i in range(len(stars)):
        stellar_mass += stars[i].mass

    return stellar_mass


def sort_stars(stars, mortal_stars, immortal_stars, statistical_weight, t):
    """
    Classify stars as either mortal or immortal based on their
    lifetimes and the remaining simulation time.

    Inputs:
            stars: class instance array
                    list of starParticle objects
            mortal_stars: class instance array
                    list of stars that will die during the simulation
            immortal_stars: class instance array
                    list of stars that will live longer than the simulation
            statistical_weight: float
                    the ratio of the starParticle mass to the representative
                    stellar mass density
            t: float
                    current simulation time [Gyr]
    """
    for i in range(len(stars)):
        stars[i].statistical_weight = statistical_weight

        if stars[i].stellar_lifetime() > (galaxy_age - t):
            immortal_stars.append(stars[i])

        else:
            mortal_stars.append(stars[i])


def type_ia_delay(dtd):
    """
    Return delay time after which a Type Ia may occur [Gyr].

    Inputs:
            dtd: string
                    name of delay time distribution
    """
    if dtd == "constant":
        return 2.5

    if dtd == "power_law":
        u = uniform(0, 1)
        return 10**u + 0.85

    else:
        return 0.0


def core_collapse_supernova(star):
    """
    Return ejecta from a core-collapse supernova. [M.]

    Inputs:
            star: class instance
                    single instance of the starParticle class
    """
    remnant_mass = 1.8
    ejecta = (
        ((star.mass - remnant_mass) * star.statistical_weight)
        * core_collapse_supernova_yields(star.mass, elements)
        * array([1.0, 0.35, 0.35, 0.35, 0.35, 1.0])
    )
    star.mass = remnant_mass

    return ejecta


def planetary_nebula(star):
    """
    Return ejecta from a planetary nebula. [M.]

    Inputs:
            star: class instance
                    a single star nearing the end of its life
    """
    ejecta = (
        star.statistical_weight
        * (star.mass - 0.6)
        * planetary_nebula_yields(star.mass, elements)
    )
    star.mass = 0.6

    return ejecta


def type_ia_supernova(wd_remnant):
    """
    Return ejecta from a type Ia supernova. [M.]

    Inputs:
            wd_remnant: class instance
                    a single white dwarf remnant
    """
    ejecta = (
        wd_remnant.mass
        * wd_remnant.statistical_weight
        * type_ia_supernova_yields(elements)
        * array([1.0, 0.35, 0.35, 0.35, 0.35, 1.0])
    )

    return ejecta


def binary_neutron_star_merger(ns_remnant):
    """
    Return ejecta from a binary neutron star merger. [M.]

    Inputs:
            ns_remnant: class instance
                    a single neutron star remnant
    """
    ejecta = (
        ns_remnant.mass
        * ns_remnant.statistical_weight
        * binary_neutron_star_merger_yields(elements)
    )

    return ejecta


def explode_stars(mortal_stars, black_holes, white_dwarfs, neutron_stars):
    """
    Return ejecta from explosions at the end of stars' lives. [M.]

    Inputs:
            mortal_stars: class instance array
                    list of stars that will die during the simulation
            black_holes: class instance array
                    list of black hole remnants
            white_dwarfs: class instance array
                    list of white dwarf remnants
            neutron_stars: class instance array
                    list of neutron star remnants
    """
    i = 0

    ejecta = zeros(len(elements) + 1)

    while i < len(mortal_stars):
        star = mortal_stars[i]

        if star.age > star.stellar_lifetime():
            if star.mass > 8.0:
                ejecta += core_collapse_supernova(star)
                if star.mass > 2.5:
                    star.classification = "black-hole"
                    star.age = 0.0
                    black_holes.append(mortal_stars.pop(i))
                else:
                    star.classification = "neutron-star"
                    star.age = 0.0
                    neutron_stars.append(mortal_stars.pop(i))

            else:
                ejecta += planetary_nebula(star)
                star.classification = "white-dwarf"
                star.age = 0.0
                white_dwarfs.append(mortal_stars.pop(i))

        else:
            i += 1

    return ejecta


def explode_remnants(immortal_stars, white_dwarfs, neutron_stars):
    """
    Return ejecta from remnant explosions. [M.]

    Inputs:
            immortal_stars: class instance array
                    list of stars that will live longer than the simulation
            white_dwarfs: class instance array
                    list of white dwarf remnants
            neutron_stars: class instance array
                    list of neutron star remnants
    """
    i = 0
    j = 0

    ejecta = zeros(len(elements) + 1)

    while i < len(white_dwarfs):
        wd_remnant = white_dwarfs[i]

        if wd_remnant.age > type_ia_delay("power_law"):
            u = randint(1, 24)

            if u == 1:
                ejecta += type_ia_supernova(wd_remnant)
                white_dwarfs.pop(i)

            else:
                immortal_stars.append(white_dwarfs.pop(i))

        else:
            i += 1

    while j < len(neutron_stars):
        ns_remnant = neutron_stars[j]

        if ns_remnant.age > 0.5:
            u = randint(1, 30)

            if u == 1:
                ejecta += binary_neutron_star_merger(ns_remnant)
                neutron_stars.pop(j)

            else:
                immortal_stars.append(neutron_stars.pop(j))

        else:
            j += 1

    return ejecta


def evolve_stars(
    immortal_stars, mortal_stars, black_holes, white_dwarfs, neutron_stars, dt
):
    """
    Evolve the ages of the stars.

    Inputs:
            mortal_stars: class instance array
                    list of stars that will die during the simulation
            black_holes: class instance array
                    list of black hole remnants
            white_dwarfs: class instance array
                    list of white dwarf remnants
            neutron_stars: class instance array
                    list of neutron star remnants
            dt: float
                    simulation timestep [Gyr]
    """
    for i in range(len(immortal_stars)):
        immortal_stars[i].age += dt

    for i in range(len(mortal_stars)):
        mortal_stars[i].age += dt

    for i in range(len(black_holes)):
        black_holes[i].age += dt

    for i in range(len(white_dwarfs)):
        white_dwarfs[i].age += dt

    for i in range(len(neutron_stars)):
        neutron_stars[i].age += dt


def advance_state(
    t,
    dt,
    num_stars,
    sigma_gas,
    immortal_stars,
    mortal_stars,
    black_holes,
    white_dwarfs,
    neutron_stars,
):
    """
    Advance the state of the simulation by one timestep.

    Inputs:
            t: float
                    current simulation time [Gyr]
            dt: float
                    simulation timestep [Gyr]
            num_stars: int
                    number of stars to be formed
            gas_mass: float array
                    mass by species of interstellar gas [M.]
            immortal_stars: class instance array
                    list of stars that will live longer than the simulation
            mortal_stars: class instance array
                    list of stars that will die during the simulation
            black_holes: class instance array
                    list of black hole remnants
            white_dwarfs: class instance array
                    list of white dwarf remnants
            neutron_stars: class instance array
                    list of neutron star remnants
    """
    # gas is set aside for forming stars
    star_forming_gas_density = (
        dt * star_formation_rate(sigma_gas) * sigma_gas / sum(sigma_gas)
    )

    # enriched material is returned to the ISM from stellar winds/explosive enrichment events
    stellar_return_gas_density = explode_stars(
        mortal_stars, black_holes, white_dwarfs, neutron_stars
    ) + explode_remnants(immortal_stars, white_dwarfs, neutron_stars)

    # gas falls into the ISM from the CGM
    infalling_gas_density = dt * gas_infall_rate(t) * gas_infall_abundances(elements)

    # gas flows out of the galaxy in winds
    outflowing_gas_density = (
        dt
        * gas_outflow_rate(t, star_formation_rate(sum(sigma_gas)))
        * sigma_gas
        / sum(sigma_gas)
    )

    # stars are formed from the gas set aside for star formation
    stars = form_stars(num_stars, sigma_gas, galaxy_age)

    # a statistical weight is computed to normalize the masses of the
    # new stars to the gas density available for star formation
    statistical_weight = sum(star_forming_gas_density) / total_stellar_mass(stars)

    # stars are classified as 'mortal' or 'immortal' based on their
    # lifetimes relative to the current time and the age of the galaxy
    sort_stars(stars, mortal_stars, immortal_stars, statistical_weight, t)

    # the ISM mass by species is evolved in time
    sigma_gas += (
        infalling_gas_density
        - star_forming_gas_density
        + stellar_return_gas_density
        - outflowing_gas_density
    )

    # stars are evolved in time
    evolve_stars(
        immortal_stars, mortal_stars, black_holes, white_dwarfs, neutron_stars, dt
    )


def main():
    immortal_stars = []
    mortal_stars = []
    black_holes = []
    white_dwarfs = []
    neutron_stars = []

    dt = 0.01
    t = dt
    t_max = 13.6

    # gas is injected into the galaxy over one timestep to initialize everything
    sigma_gas = dt * gas_infall_rate(dt) * gas_infall_abundances(elements)
    element_list = dict([(element, 0.0) for element in elements])
    for i in range(len(elements)):
        element_list[elements[i]] += sigma_gas[i]

    time_series = [t]

    counter = 0
    num_stars = 1000  # number of stars produced for each timestep

    while t < t_max:
        advance_state(
            t,
            dt,
            num_stars,
            sigma_gas,
            immortal_stars,
            mortal_stars,
            black_holes,
            white_dwarfs,
            neutron_stars,
        )
        for i in range(len(elements)):
            element_list[elements[i]] += sigma_gas[i]
        if (counter % 10) == 0:
            print(f"Time: {t:.2f}Gyr")
        t += dt
        counter += 1
        time_series.append(t)

    age_metallicity(immortal_stars, elements, num_stars)
    # abundance_pattern(immortal_stars, elements, "fe", "o", num_stars)


main()
