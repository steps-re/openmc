"""Sharper #4222 check: compare photon flux in the +x and -x halves of a shell just outside a
lead sphere, split by energy. Isotropic 2614 keV point source at the centre, so by symmetry
+x/-x must be 1. Prints the ratio and its z-score per energy bin.
    python calibrate2.py PARTICLES BATCHES
"""
import math, sys, tempfile
import openmc

particles, batches = int(sys.argv[1]), int(sys.argv[2])
pb = openmc.Material(); pb.add_element("Pb", 1.0); pb.set_density("g/cm3", 11.35)
inner, outer, shell = openmc.Sphere(r=1.0), openmc.Sphere(r=6.0), openmc.Sphere(r=10.0, boundary_type="vacuum")
xp = openmc.XPlane(0.0)
plus, minus = openmc.Cell(region=+outer & -shell & +xp), openmc.Cell(region=+outer & -shell & -xp)
cells = [openmc.Cell(region=-inner), openmc.Cell(fill=pb, region=+inner & -outer), plus, minus]
s = openmc.Settings()
s.run_mode = "fixed source"; s.particles = particles; s.batches = batches
s.photon_transport = True; s.electron_treatment = "ttb"; s.seed = 1
s.source = openmc.IndependentSource(space=openmc.stats.Point(), angle=openmc.stats.Isotropic(),
                                    energy=openmc.stats.Discrete([2.6145e6], [1.0]), particle="photon")
edges = [1e3, 1e5, 3e5, 6e5, 1e6, 2.6e6, 3e6]
t = openmc.Tally(name="halves")
t.filters = [openmc.CellFilter([plus, minus]), openmc.ParticleFilter(["photon"]), openmc.EnergyFilter(edges)]
t.scores = ["flux"]
model = openmc.Model(openmc.Geometry(cells), openmc.Materials([pb]), s, openmc.Tallies([t]))
with tempfile.TemporaryDirectory() as d:
    sp_path = model.run(cwd=d, output=False)
    with openmc.StatePoint(sp_path) as sp:
        tt = sp.get_tally(name="halves")
        m = tt.mean.reshape(2, len(edges) - 1); e = tt.std_dev.reshape(2, len(edges) - 1)
for k in range(len(edges) - 1):
    r = m[0, k] / m[1, k]
    sr = r * math.hypot(e[0, k] / m[0, k], e[1, k] / m[1, k])
    print(f"  {edges[k]/1e3:7.0f}-{edges[k+1]/1e3:5.0f} keV  +x/-x = {r:.4f} +- {sr:.4f}  z={(r - 1) / sr:5.1f}")
P, M = m[0].sum(), m[1].sum(); sP, sM = math.sqrt((e[0] ** 2).sum()), math.sqrt((e[1] ** 2).sum())
r = P / M; sr = r * math.hypot(sP / P, sM / M)
print(f"RESULT particles={particles} batches={batches} all energies +x/-x={r:.4f} +- {sr:.4f} z={(r - 1) / sr:.1f}")
