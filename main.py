from random import randint, seed, uniform
from numpy import array, exp, zeros
from calculate_yields import (
    core_collapse_supernova_yields,
    gas_infall_abundances,
    type_ia_supernova_yields,
    planetary_nebula_yields,
    binary_neutron_star_merger_yields,
)
from plot_abundance_pattern import abundance_pattern, age_metallicity


# Random Seed for Repeatable Runs
seed(487)

# Initial Mass Function Parameters
lower_mass_bound = 0.1  # [M.]
upper_mass_bound = 100.0  # [M.]
imf_power = -2.35

# Elements to Evolve
elements = ("h", "fe", "o", "mg", "eu")

# Milky Way Parameters
present_day_total_surface_mass_density = 41.0  # [M. / pc^2]
gas_infall_timescale = 5.0  # [Gyr]
galaxy_age = 13.6  # [Gyr]
ks_star_formation_rate_power = 1.4
star_formation_efficiency = 0.25  # [M.^(1-k) / Gyr / pc^2]
galaxy_surface_area = 8.25e8  # [pc^2]


class starParticle:
    """
    A class to represent a star, which in turn represents a group
    of identical stars.

    Attributes:
            age: float
                    current age of the star [Gyr]
            intrinsic_mass: float
                    mass of the star as sampled from the IMF [M.]
            scaled_mass: float
                    mass of the star normalized to the amount of available
                    star-forming gas
            kind: string
                    main-sequence, white-dwarf, black-hole, or neutron-star
            composition: float array
                    masses of all atomic species in the star [M.]

    Methods:
            stellar_lifetime():
                    Calculates the lifetime of a newborn star. [Gyr]
    """

    def __init__(self, age, mass, mass_scaling_factor, kind, composition):
        self.age = age
        self.mass = mass
        self.mass_ratio = mass_scaling_factor
        self.kind = kind
        self.composition = composition

    def stellar_lifetime(self):
        """
        Return the lifetime of a star given its mass. [Gyr]
        """
        lifetime = 10.0 * self.mass ** (-2.5)
        return lifetime


def sample_mass():
    """
    Return a stellar mass randomly sampled from an initial mass
    function (IMF), hardcoded here as the Salpeter IMF. [M.]
    """
    m_l = lower_mass_bound
    m_u = upper_mass_bound
    p = imf_power
    u = uniform(0, 1)
    mass = (u * (m_u ** (p + 1) - m_l ** (p + 1)) + m_l ** (p + 1)) ** (1 / (p + 1))

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
    sfr = nu * sigma_gas**k

    return sfr


def form_stars(num_stars, gas_mass, galaxy_age, t):
    """
    Return array of 'stars' (starParticle class instances).

    Inputs:
            num_stars: float
                    number of stars to be formed
            gas_mass: float array
                    mass by species of interstellar gas [M.]
            galaxy_age: float
                    age of the galaxy [Gyr]
            t: float
                    current simulation time [Gyr]
    """

    stars = []

    for i in range(num_stars):
        star_age = 0.0
        star_mass = sample_mass()
        star_mass_scaling_factor = 1.0
        star_kind = "main-sequence"
        star_composition = gas_mass / sum(gas_mass)
        star = starParticle(
            star_age, star_mass, star_mass_scaling_factor, star_kind, star_composition
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


def sort_stars(stars, mortal_stars, immortal_stars, mass_scaling_factor, t):
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
            mass_scaling_factor: float
                    the ratio of the starParticle mass to the representative
                    stellar mass
            t: float
                    current simulation time [Gyr]
    """
    for i in range(len(stars)):
        stars[i].mass_scaling_factor = mass_scaling_factor

        if stars[i].stellar_lifetime() > (galaxy_age - t):
            immortal_stars.append(stars[i])

        else:
            mortal_stars.append(stars[i])


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
                    black_holes.append(mortal_stars.pop(i))
                else:
                    neutron_stars.append(mortal_stars.pop(i))

            else:
                ejecta += planetary_nebula(star)
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

        if wd_remnant.age > 2.0:
            u = randint(1, 3)

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


def core_collapse_supernova(star):
    """
    Return ejecta from a core-collapse supernova. [M.]

    Inputs:
            star: class instance
                    single instance of the starParticle class
    """
    star.kind = "black-hole"
    star.age = 0.0
    remnant_mass = 0.01 * star.mass * star.mass - 0.1 * star.mass + 1.0
    ejecta = (
        (star.mass - remnant_mass) * star.mass_scaling_factor
    ) * core_collapse_supernova_yields(star.mass, elements)
    star.mass = remnant_mass

    return ejecta


def planetary_nebula(star):
    """
    Return ejecta from a planetary nebula. [M.]

    Inputs:
            star: class instance
                    a single star nearing the end of its life
    """
    star.kind = "white-dwarf"
    star.age = 0.0
    ejecta = (
        star.mass_scaling_factor
        * (star.mass - 0.6)
        * planetary_nebula_yields(star.mass, elements)
    )
    star.mass -= ejecta / star.mass_scaling_factor

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
        * wd_remnant.mass_scaling_factor
        * type_ia_supernova_yields(elements)
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
        * ns_remnant.mass_scaling_factor
        * binary_neutron_star_merger_yields(elements)
    )

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
    galactic_area,
    gas_mass,
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
            galactic_area: float
                    surface area of the galaxy [pc^2]
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
    # gas falls into the ISM from the CGM
    infalling_gas_density = dt * gas_infall_rate(t) * gas_infall_abundances(elements)
    infalling_gas_mass = infalling_gas_density * galactic_area
    gas_mass += infalling_gas_mass

    # gas is removed from the ISM for star formation
    sigma_gas = sum(gas_mass) / galactic_area
    star_forming_gas_density = (
        dt * star_formation_rate(sigma_gas) * gas_mass / sum(gas_mass)
    )
    star_forming_mass = star_forming_gas_density * galactic_area
    gas_mass -= star_forming_mass

    # stars are formed from the gas set aside for star formation
    stars = form_stars(num_stars, gas_mass, galaxy_age, t)

    # a mass scaling factor is computed to normalize the masses of the
    # new stars to the gas mass available for star formation
    mass_scaling_factor = sum(star_forming_mass) / total_stellar_mass(stars)

    # stars are classified as 'mortal' or 'immortal' based on their
    # lifetimes relative to the current time and the age of the galaxy
    sort_stars(stars, mortal_stars, immortal_stars, mass_scaling_factor, t)

    # stars eject content back into the ISM and their remnants are
    # classified as black holes, neutron stars, or white dwarves
    stellar_ejecta = explode_stars(
        mortal_stars, black_holes, white_dwarfs, neutron_stars
    )

    # merging remnants eject material back into the ISM and are
    # completely destroyed
    remnant_ejecta = explode_remnants(immortal_stars, white_dwarfs, neutron_stars)

    # ejected material is reincorporated into the ISM
    gas_mass += stellar_ejecta + remnant_ejecta

    # stars are evolved in time
    evolve_stars(
        immortal_stars, mortal_stars, black_holes, white_dwarfs, neutron_stars, dt
    )


def main():
    num_stars = 1000  # number of stars produced for each timestep
    galactic_area = 8.25e8  # surface area of the galaxy [pc^2]

    immortal_stars = []
    mortal_stars = []
    black_holes = []
    white_dwarfs = []
    neutron_stars = []

    gas_mass = zeros(len(elements) + 1)

    t = 0.0
    dt = 0.05
    t_max = 13.6

    element_list = dict([(element, 0.0) for element in elements])

    time_series = [t]

    counter = 0
    num_stars = 1000

    while t < t_max:
        advance_state(
            t,
            dt,
            num_stars,
            galactic_area,
            gas_mass,
            immortal_stars,
            mortal_stars,
            black_holes,
            white_dwarfs,
            neutron_stars,
        )
        for i in range(len(elements)):
            element_list[elements[i]] += gas_mass[i]
        if (counter % 10) == 0:
            print(f"Time: {t:.2f}Gyr")
        t += dt
        counter += 1
        time_series.append(t)

    # abundance_pattern(immortal_stars, elements, "fe", "o", num_stars)
    age_metallicity(immortal_stars, elements, num_stars)


main()
