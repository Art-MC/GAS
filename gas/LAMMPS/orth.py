import abtem
import ase.io
from ase.visualize import view

abtem.config.set({"device": "cpu"})

org_sys = ase.io.read('BN.cif')

orth_sys, strain = abtem.orthogonalize_cell(org_sys, return_transform=True)
print(strain)

view(orth_sys)
orth_sys.write('BN_orth.cif')