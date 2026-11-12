"""Size the +x anisotropy from photoelectric TTB bremsstrahlung, for choosing a test.

Isotropic 2614 keV point source in a lead shell (r 1-6 cm). Six void spheres at 90 cm on
+-x, +-y, +-z. By symmetry all six photon fluxes must agree. Prints each sphere's flux, its
relative error, and z = (+x - mean of the other five) / combined sigma.
    python calibrate.py PARTICLES BATCHES
"""
import math, sys, tempfile
import numpy as np
import openmc

particles, batches = int(sys.argv[1]), int(sys.argv[2])
pb = openmc.Material(); pb.add_element("Pb", 1.0); pb.set_density("g/cm3", 11.35)
inner, outer = openmc.Sphere(r=1.0), openmc.Sphere(r=6.0)
world = openmc.Sphere(r=100.0, boundary_type="vacuum")
dirs = [(90, 0, 0), (-90, 0, 0), (0, 90, 0), (0, -90, 0), (0, 0, 90), (0, 0, -90)]
balls = [openmc.Sphere(x0=x, y0=y, z0=z, r=3.0) for x, y, z in dirs]
ball_cells = [openmc.Cell(region=-b) for b in balls]
gap = +outer & -world
for b in balls:
    gap &= +b
cells = [openmc.Cell(region=-inner), openmc.Cell(fill=pb, region=+inner & -outer), *ball_cells,
         openmc.Cell(region=gap)]
s = openmc.Settings()
s.run_mode = "fixed source"; s.particles = particles; s.batches = batches
s.photon_transport = True; s.electron_treatment = "ttb"; s.seed = 1
s.source = openmc.IndependentSource(space=openmc.stats.Point(), angle=openmc.stats.Isotropic(),
                                    energy=openmc.stats.Discrete([2.6145e6], [1.0]), particle="photon")
t = openmc.Tally(name="balls")
t.filters = [openmc.CellFilter(ball_cells), openmc.ParticleFilter(["photon"])]
t.scores = ["flux"]
model = openmc.Model(openmc.Geometry(cells), openmc.Materials([pb]), s, openmc.Tallies([t]))
with tempfile.TemporaryDirectory() as d:
    sp_path = model.run(cwd=d, output=False)
    with openmc.StatePoint(sp_path) as sp:
        tt = sp.get_tally(name="balls")
        m, e = tt.mean.ravel(), tt.std_dev.ravel()
rest = np.mean(m[1:]); rest_e = math.sqrt(np.sum(e[1:] ** 2)) / 5
z = (m[0] - rest) / math.hypot(e[0], rest_e)
for lab, a, b in zip(["+x", "-x", "+y", "-y", "+z", "-z"], m, e):
    print(f"  {lab}: {a:.5e} +- {100 * b / a:.2f}%")
print(f"RESULT particles={particles} batches={batches} +x/rest={m[0] / rest:.4f} z={z:.1f}")
