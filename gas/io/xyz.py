import numpy as np
import h5py

from ase import Atoms
from ase import io as aio


from warnings import simplefilter
simplefilter(action='ignore', category=DeprecationWarning)

def load_as_ase_atoms(fp, dkey):
    with h5py.File(fp, mode='r') as f:
        data = np.copy(f[dkey])
        s = f[dkey].attrs['shape']

    return Atoms(positions=data[:, 1:], numbers=data[:, 0], cell=s)

def load_xyz(f, pbcs=None, filter_pos=[None,None,None], pbc_pad=0, v=1):
    if v >= 1:
        print(f"loading file: {f.name}")
    atoms = aio.read(f)
    with open(f, 'r') as ff:
        tag = ff.readlines()[1]
        if pbcs is None:
            pbcs = tag.split("pbc")[1][2:-2]
            pbcs = [i=="T" for i in pbcs[::2]]

        cell = tag.split('"')[1]
        cell = np.fromstring(cell, sep=' ').reshape((3,3))
        if not np.any(cell):
            print("unable to find cell, generating from positions")
            cell = np.max(atoms.positions, axis=0)
    # wrap positions in x and y but not z
    for i in range(len(pbcs)):
        if pbcs[i]:
            atoms.positions[:,i] = atoms.positions[:,i] % cell[i,i]
        else:
            atoms.positions[:,i] -= atoms.positions[:,i].min()
            cell[i,i] = atoms.positions[:,i].max()+1e-9

    atoms.set_cell(cell)
    atoms.set_pbc(pbcs)

    if np.any(filter_pos):
        atoms = filter_atoms_positions(atoms, filter_pos, pbc_pad)

    if v >= 1:
        print(f"# atoms: {len(atoms)}")
        print(atoms)
    return atoms


def filter_atoms_positions(atoms, ncell, pbc_pad=0):
    pbcs = atoms.get_pbc()
    if np.any(atoms.cell.array):
        cell = np.diag(atoms.cell.array).copy()
    else:
        cell = atoms.positions.max(axis=0)
    filt = atoms.positions[:,0] >= -1*np.inf # all
    for i in range(len(pbcs)):
        if ncell[i] is not None:
            if atoms.positions[:,i].max() < ncell[i]:
                continue
            dim = ncell[i] - pbc_pad if pbcs[i] else ncell[i]
            filt = filt & (atoms.positions[:,i] < dim)
            cell[i] = ncell[i]

    bads = np.where(~filt)
    del atoms[bads]
    atoms.set_cell(cell)
    return atoms