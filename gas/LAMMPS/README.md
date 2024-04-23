
orth.py: generate orthogonal unit cell
    - abTEM not compatible with Python 3.12, need an earlier version
    - check strain after the transformation

lmp_struc_gen.py: generate structure input for LAMMPS
    - scale to the desired dimension
    - add perturbation/defect if applicable

plotPDF.py: plot PDF from .xyz
    - diffpy requires Python 3.7
    - similarity evaluation EMD 

SiO2_rate13_in: sample LAMMPS input of melt-quench method in obataining amorphous silica

SiO2_partial_crystal_in: sample LAMMPS input of replacing center amorphous atoms to crystal, and the heat-quench to obatain partially ordered silica