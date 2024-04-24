import numpy as np
import math
import random
from mp_api.client import MPRester
from pymatgen.io.lammps.data import LammpsData
from pymatgen.io.cif import CifParser
import os

api_key = "LBS9z1OVSRL3SlJmgjY5c1eTC55ZAmWQ"

# Get structure from MP
struc_dic = {"SiO2":"mp-6930", "Al2O3":"mp-1143", "BN":"mp-984",\
"C":"mp-66"}

mat = "C" 
# with MPRester(api_key) as mpr:
#     structure = mpr.get_structure_by_material_id(struc_dic[mat])

# Get structure from cif
parser = CifParser('%s_orth.cif' %mat)
structure = parser.parse_structures()[0]

# Scale to target in A
# target = [60, 60, 160] #45%
# target = [50, 50, 125] #25%
# target = [40, 40, 100] #12.5%
target = [80, 80, 200]
print(structure.num_sites, structure.lattice)
# a, b, c = target[0]/structure.lattice.a, target[1]/structure.lattice.b, target[2]/structure.lattice.c
a, b, c = math.ceil(target[0]/structure.lattice.a), math.ceil(target[1]/structure.lattice.b), math.ceil(target[2]/structure.lattice.c)
print(a, b, c, structure.num_sites*a*b*c)

# Create the supercell
structure.make_supercell([a, b, c])
print(structure.lattice)
print(structure.lattice.a, structure.lattice.b, structure.lattice.c)

# # Add random perturbations if neccessary
magnitude = 0.05  # in Angstroms
for site in structure:
    perturbation = np.random.normal(scale=magnitude, size=(3,))
    site.coords += perturbation

# # Add vacancy defect if neccessary
# atoms_to_delete = int(0.2 * structure.num_sites)
# indices_to_delete = random.sample(range(structure.num_sites), atoms_to_delete)
# for index in sorted(indices_to_delete, reverse=True):
#     structure.remove_sites([index])

# Generate input structure file
LammpsData.from_structure(structure).write_file("%s.lmp" %struc_dic[mat])
