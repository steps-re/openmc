import math

import openmc


def test_photoelectron_direction_follows_photon(run_in_tmpdir):
    """Photoelectrons are emitted relative to the incident photon (#4222).

    An isotropic source at the centre of a lead sphere is symmetric, so photons
    heading +x and -x must agree. With thick-target bremsstrahlung the
    bremsstrahlung is born along the electron direction, so photoelectrons
    sampled around the fixed +x axis made the +x bin about 13% high.
    """
    openmc.reset_auto_ids()

    m = openmc.Material()
    m.add_element('Pb', 1.0)
    m.set_density('g/cm3', 11.35)

    surf = openmc.Sphere(r=5.0, boundary_type='vacuum')
    cell = openmc.Cell(fill=m, region=-surf)

    model = openmc.Model()
    model.geometry = openmc.Geometry([cell])
    model.settings.run_mode = 'fixed source'
    model.settings.source = openmc.IndependentSource(
        space=openmc.stats.Point(),
        angle=openmc.stats.Isotropic(),
        energy=openmc.stats.Discrete([2.6145e6], [1.0]),
        particle='photon',
    )
    model.settings.particles = 20000
    model.settings.batches = 5
    model.settings.photon_transport = True
    model.settings.electron_treatment = 'ttb'
    model.settings.seed = 1

    # Bins of photon direction in the x-y plane: +x is |phi| < pi/4, and -x is
    # the two quarter-width bins either side of phi = +-pi
    edges = [-math.pi, -0.75*math.pi, -0.25*math.pi, 0.25*math.pi,
             0.75*math.pi, math.pi]
    tally = openmc.Tally()
    tally.filters = [
        openmc.ParticleFilter(['photon']),
        openmc.EnergyFilter([1.0e5, 1.0e6]),
        openmc.AzimuthalFilter(edges),
    ]
    tally.scores = ['flux']
    model.tallies = openmc.Tallies([tally])
    model.run(apply_tally_results=True)

    flux = tally.mean.ravel()
    plus_x = flux[2]
    minus_x = flux[0] + flux[4]
    assert abs(plus_x / minus_x - 1.0) < 0.04
