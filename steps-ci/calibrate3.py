"""Direction-binned check for #4222: photon flux inside a lead sphere, binned by the azimuthal
angle of the photon's direction in the x-y plane. Isotropic 2614 keV source at the centre, so
every direction bin must agree. With the bug, TTB bremsstrahlung is born heading +x (phi ~ 0).
Prints the ratio of the +x bin to the -x bin for 100 keV-1 MeV photons and its z-score.
    python calibrate3.py PARTICLES BATCHES
"""
import math, sys, tempfile
import numpy as np
import openmc

particles, batches = int(sys.argv[1]), int(sys.argv[2])
pb = openmc.Material(); pb.add_element("Pb", 1.0); pb.set_density("g/cm3", 11.35)
outer = openmc.Sphere(r=5.0, boundary_type="vacuum")
lead = openmc.Cell(fill=pb, region=-outer)
s = openmc.Settings()
s.run_mode = "fixed source"; s.particles = particles; s.batches = batches
s.photon_transport = True; s.electron_treatment = "ttb"; s.seed = 1
s.source = openmc.IndependentSource(space=openmc.stats.Point(), angle=openmc.stats.Isotropic(),
                                    energy=openmc.stats.Discrete([2.6145e6], [1.0]), particle="photon")
# edges inside [-pi, pi]: the -x group is the two quarter-width bins either side of +-pi
edges = [-math.pi, -0.75 * math.pi, -0.25 * math.pi, 0.25 * math.pi, 0.75 * math.pi, math.pi]
t = openmc.Tally(name="dir")
t.filters = [openmc.ParticleFilter(["photon"]), openmc.EnergyFilter([1e5, 1e6]),
             openmc.AzimuthalFilter(edges)]
t.scores = ["flux"]
model = openmc.Model(openmc.Geometry([lead]), openmc.Materials([pb]), s, openmc.Tallies([t]))
with tempfile.TemporaryDirectory() as d:
    sp_path = model.run(cwd=d, output=False)
    with openmc.StatePoint(sp_path) as sp:
        tt = sp.get_tally(name="dir")
        m, e = tt.mean.ravel(), tt.std_dev.ravel()
plus, minus = m[2], m[0] + m[4]          # +x: |phi| < pi/4 ; -x: |phi| > 3pi/4
sp_, sm_ = e[2], math.hypot(e[0], e[4])
r = plus / minus; sr = r * math.hypot(sp_ / plus, sm_ / minus)
side = (m[1] + m[3]) / 2                  # +-y quarter, same width as +x
print("  bins:", " ".join(f"{x / m.mean():.3f}" for x in m))
print(f"RESULT particles={particles} batches={batches} +x/-x={r:.4f} +- {sr:.4f} z={(r - 1) / sr:.1f}  +x/+-y={plus / side:.4f}")
