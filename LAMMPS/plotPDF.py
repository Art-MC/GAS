from diffpy.Structure import loadStructure
from diffpy.srreal.pdfcalculator import PDFCalculator, DebyePDFCalculator
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt

material = "SiO2"

# Load your structure
structure = []
name = ["crystal", "12p_center_1000K", "12p_nn_1000K", "12p_all_1000K", "amorphous", "melt"]
for n in range(len(name)):
    structure.append(loadStructure("%s_%s" %(material, name[n])))

# calculate PDF by real-space summation
# pc = PDFCalculator()


# calculate PDF using Debye formula
cfg = { "qmax" : 25,
        "rmin" : 1.4,
        "rmax" : 6.0,
        "rstep" : 0.05,
}
pc = DebyePDFCalculator(**cfg)

rc, gc = pc(structure[0])
ra, ga = pc(structure[-1])

plt.clf()
for n in range(len(name)):
    r, g = pc(structure[n])
    emd_c = wasserstein_distance(g, gc)
    emd_a = wasserstein_distance(g, ga)
    plt.plot(r, g, label=f"{name[n]} ({emd_c:.3f}, {emd_a:.3f})")

plt.legend()
plt.xlabel("r(Å)", fontsize=14)
plt.ylabel("G(r)", fontsize=14)
plt.grid(True)
plt.show()