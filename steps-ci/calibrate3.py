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
nb = 8
t = openmc.Tally(name="dir")
t.filters = [openmc.ParticleFilter(["photon"]), openmc.EnergyFilter([1e5, 1e6]),
             openmc.AzimuthalFilter(np.linspace(-math.pi, math.pi, nb + 1) + math.pi / nb)]
t.scores = ["flux"]
model = openmc.Model(openmc.Geometry([lead]), openmc.Materials([pb]), s, openmc.Tallies([t]))
with tempfile.TemporaryDirectory() as d:
    sp_path = model.run(cwd=d, output=False)
    with openmc.StatePoint(sp_path) as sp:
        tt = sp.get_tally(name="dir")
        m, e = tt.mean.ravel(), tt.std_dev.ravel()
# bin edges are shifted by pi/nb, so bin nb-1 wraps through phi = pi... find bins by centre
centres = (np.linspace(-math.pi, math.pi, nb + 1) + math.pi / nb)[:-1] + math.pi / nb
ip = int(np.argmin(np.abs(np.angle(np.exp(1j * centres)))))           # centre at phi = 0 (+x)
im = int(np.argmin(np.abs(np.angle(np.exp(1j * (centres - math.pi))))))  # centre at phi = pi (-x)
r = m[ip] / m[im]; sr = r * math.hypot(e[ip] / m[ip], e[im] / m[im])
others = np.delete(m, [ip]).mean()
print("  bins:", " ".join(f"{x / m.mean():.3f}" for x in m))
print(f"RESULT particles={particles} batches={batches} +x/-x={r:.4f} +- {sr:.4f} z={(r - 1) / sr:.1f}  +x/others={m[ip] / others:.4f}")
