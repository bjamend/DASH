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


def gas_infall_rate(t, elements):
    """
    Return the surface rate density at which gas falls into the galaxy
    from the circumgalactic medium (CGM). [M. / Gyr / pc^2]

    Inputs:
            t: float
                    current simulation time [Gyr]
            elements: string list
                    list of species being tracked
    """
    sigma_tot = present_day_total_surface_mass_density
    tau = gas_infall_timescale
    age = galaxy_age
    Lambda = sigma_tot / tau / (1.0 - exp(-age / tau))
    infall = Lambda * exp(-t / tau) * gas_infall_abundances(elements)

    return infall


def gas_outflow_rate(sigma_gas):
    """
    Return the surface rate density at which gas flows out of the
    galaxy. [M. / Gyr / pc^2]

    Inputs:
            sigma_gas: float
                    mass density of ISM by species [M.]
    """
    outflow = (
        mass_loading_factor
        * star_formation_rate(sum(sigma_gas))
        * (sigma_gas / sum(sigma_gas))
    )

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


def form_stars(
    num_stars,
    sigma_gas,
    star_forming_gas_density,
    mortal_stars,
    immortal_stars,
    t,
):
    """
    Form star particles, determine their statistical weights, and classify them as mortal or immortal.

    Inputs:
            num_stars: int
                number of star particles to be formed
            sigma_gas: float
                surface gas density [M. / pc^2]
            star_forming_gas_density: float array
                mass density of star forming gas by species [M. / pc^2]
            mortal_stars: class instance array
                list of stars that will die during the simulation
            immortal_stars: class instance array
                list of stars that will live longer than the simulation
            t: float
                current simulation time [Gyr]
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

    stellar_mass = 0.0

    for i in range(len(stars)):
        stellar_mass += stars[i].mass

    for i in range(len(stars)):
        stars[i].statistical_weight = sum(star_forming_gas_density) / stellar_mass

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


def planetary_nebulae(mortal_stars, white_dwarfs):
    """
    Return ejecta from planetary nebulae. [M.]

    Inputs:
            mortal_stars: class instance array
                list of stars that will die during the simulation
            white_dwarfs: class instance array
                list of white dwarf remnants
    """
    i = 0

    ejecta = zeros(len(elements) + 1)

    while i < len(mortal_stars):
        star = mortal_stars[i]
        if star.age > star.stellar_lifetime():
            if star.mass <= 8.0:
                ejecta += (
                    star.statistical_weight
                    * (star.mass - 0.6)
                    * planetary_nebula_yields(star.mass, elements)
                )
                star.mass = 0.6
                star.classification = "white-dwarf"
                star.age = 0.0
                white_dwarfs.append(mortal_stars.pop(i))
            else:
                i += 1
        else:
            i += 1

    return ejecta


def core_collapse_supernovae(mortal_stars, neutron_stars, black_holes):
    """
    Return ejecta from core-collapse supernovae. [M.]

    Inputs:
            mortal_stars: class instance array
                list of stars that will die during the simulation
            neutron_stars: class instance array
                list of neutron star remnants
            black_holes: class instance array
                list of black hole remnants
    """
    i = 0

    ejecta = zeros(len(elements) + 1)

    remnant_mass = 1.8

    while i < len(mortal_stars):
        star = mortal_stars[i]
        if star.age > star.stellar_lifetime():
            if star.mass > 8.0:
                ejecta += (
                    (star.mass - remnant_mass) * star.statistical_weight
                ) * core_collapse_supernova_yields(star.mass, elements)
                if star.mass > 20.0:
                    star.classification = "black-hole"
                    star.mass = remnant_mass
                    star.age = 0.0
                    black_holes.append(mortal_stars.pop(i))
                else:
                    star.classification = "neutron-star"
                    star.mass = remnant_mass
                    star.age = 0.0
                    neutron_stars.append(mortal_stars.pop(i))
            else:
                i += 1
        else:
            i += 1

    return ejecta


def type_ia_supernovae(white_dwarfs, immortal_stars):
    """
    Return ejecta from type Ia supernovae. [M.]

    Inputs:
            white_dwarfs: class instance array
                list of white dwarf remnants
            immortal_stars: class instance array
                list of stars that will live longer than the simulation
    """
    i = 0

    ejecta = zeros(len(elements) + 1)

    while i < len(white_dwarfs):
        wd_remnant = white_dwarfs[i]

        if wd_remnant.age > type_ia_delay("power_law"):
            u = randint(1, 24)

            if u == 1:
                ejecta += (
                    wd_remnant.mass
                    * wd_remnant.statistical_weight
                    * type_ia_supernova_yields(elements)
                )
                white_dwarfs.pop(i)

            else:
                immortal_stars.append(white_dwarfs.pop(i))

        else:
            i += 1

    return ejecta


def binary_neutron_star_mergers(neutron_stars, immortal_stars):
    """
    Return ejecta from binary neutron star mergers. [M.]

    Inputs:
            neutron_stars: class instance array
                list of neutron star remnants
            immortal_stars: class instance array
                list of stars that will live longer than the simulation
    """
    i = 0

    ejecta = zeros(len(elements) + 1)

    while i < len(neutron_stars):
        ns_remnant = neutron_stars[i]

        if ns_remnant.age > 0.5:
            u = randint(1, 30)

            if u == 1:
                ejecta += (
                    ns_remnant.mass
                    * ns_remnant.statistical_weight
                    * binary_neutron_star_merger_yields(elements)
                )
                neutron_stars.pop(i)

            else:
                immortal_stars.append(neutron_stars.pop(i))

        else:
            i += 1

    return ejecta


def evolve_stars(
    immortal_stars, mortal_stars, black_holes, white_dwarfs, neutron_stars, dt
):
    """
    Evolve the ages of the stars.

    Inputs:
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
    mortal_stars,
    immortal_stars,
    white_dwarfs,
    neutron_stars,
    black_holes,
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
            mortal_stars: class instance array
                    list of stars that will die during the simulation
            immortal_stars: class instance array
                    list of stars that will live longer than the simulation
            white_dwarfs: class instance array
                    list of white dwarf remnants
            neutron_stars: class instance array
                    list of neutron star remnants
            black_holes: class instance array
                    list of black hole remnants
    """
    # gas is extracted from the ISM and set aside for star formation
    star_forming_gas_density = (
        dt * star_formation_rate(sigma_gas) * sigma_gas / sum(sigma_gas)
    )

    # form star particle objects
    form_stars(
        num_stars,
        sigma_gas,
        star_forming_gas_density,
        mortal_stars,
        immortal_stars,
        t,
    )

    # enriched material is returned to the ISM from stellar winds/explosive enrichment events
    stellar_return_gas_density = (
        planetary_nebulae(mortal_stars, white_dwarfs)
        + core_collapse_supernovae(mortal_stars, neutron_stars, black_holes)
        + type_ia_supernovae(white_dwarfs, immortal_stars)
        + binary_neutron_star_mergers(neutron_stars, immortal_stars)
    )

    # gas falls into the ISM from the CGM
    infalling_gas_density = dt * gas_infall_rate(t, elements)

    # gas flows out of the galaxy in winds
    outflowing_gas_density = dt * gas_outflow_rate(sigma_gas)

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

    dt = 0.05  # [Gyr]
    t = dt  # [Gyr]
    t_max = galaxy_age  # [Gyr]

    # gas is injected into the galaxy over one timestep to initialize everything
    sigma_gas = dt * gas_infall_rate(dt, elements)
    element_list = dict([(element, 0.0) for element in elements])
    for i in range(len(elements)):
        element_list[elements[i]] += sigma_gas[i]

    # time is initialized at one timestep
    time_series = [t]

    counter = 0
    num_stars = 100  # number of stars produced for each timestep

    while t < t_max:
        advance_state(
            t,
            dt,
            num_stars,
            sigma_gas,
            mortal_stars,
            immortal_stars,
            white_dwarfs,
            neutron_stars,
            black_holes,
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
