import h5py
from ase import Atoms
import numpy as np
import ase.io

from scipy.spatial import Voronoi

from czone.volume import Volume, makeRectPrism, Plane
from czone.generator import AmorphousGenerator
from czone.transform import Rotation, rot_vtv, rot_v

import fileinput

def get_cell_info(fp):
    with open(fp, 'r') as f:
        _ = f.readline()
        cell_line = f.readline()

    fields = cell_line.split()
    idx = [i for i, field in enumerate(fields) if 'cell' in field]
    cell_data = {}
    for i in idx:
        cell_data[fields[i]] = np.array([float(x) for x in fields[i+1:i+4]])

    pbc_idx = [i for i, field in enumerate(fields) if 'pbc' in field][0]
    cell_data['pbc'] = np.array([bool(x) for x in fields[pbc_idx+1:pbc_idx+4]])

    return cell_data

def load_and_prepare_xyz(fp, fold_positions=True):
    cell_data = get_cell_info(fp)
    # if not np.array_equal(cell_data['cell_orig'], np.zeros(3)):
    #     raise NotImplementedError("Handling non-zero origin is not yet implemented.")
    
    ## Grab cell vectors 
    cell = np.array([cell_data['cell_vec1'],
                    cell_data['cell_vec2'],
                    cell_data['cell_vec3']])
    
    ## Read atoms and set cell domain
    atoms = ase.io.read(fp, format='extxyz')
    cell_data = {}
    cell = atoms.get_cell()
    cell_data['cell_vec1'] = cell[0]
    cell_data['cell_vec2'] = cell[1]
    cell_data['cell_vec3'] = cell[2]
    cell_data['cell_orig'] = [0, 0, 0]
    cell_data['pbc'] = [1, 1, 1]

    if fold_positions:
        ## Fold atoms into periodic cell
        # TODO: check semantic equivalence with PeriodicScene wrapping--ASE has a target to move atoms to a target (fractional) center
        atoms.wrap(eps=0.0)
        atoms.set_positions(atoms.get_positions() + cell_data['cell_orig'])

    return atoms, cell_data

def fix_xyz_header(fp, source_fp):
    """Replace ASE's xyz header from nc_seed output with LAMMPS-generated header, which includes origin information"""
    with open(source_fp, 'r') as source_f:
        source_header = source_f.readlines()[1]

    ## Replace just the header info
    with fileinput.input(fp, inplace=True) as target_f:
        for i, line in enumerate(target_f):
            if i == 1:
                # Print redirects here to the file stream, within the context block
                print(source_header, end='')
            else:
                print(line, end='')

def write_dataset_to_h5(fp, arr, dataset_key, metadata, chunks=True):
    with h5py.File(fp, mode="w") as f:
        if chunks is not True:
            max_shape = arr.shape[0]
            chunk_shape = True if max_shape < chunks[0] else chunks
        else:
            chunk_shape = True

        # Write dataset
        dset = f.require_dataset(
            dataset_key,
            (arr.shape),
            dtype=arr.dtype,
            chunks=chunk_shape,
            compression="gzip",
        )
        dset[:] = arr[:]

        for k, v in metadata.items():
            dset.attrs[k] = v

def load_as_ase_atoms(fp, dkey):
    with h5py.File(fp, mode='r') as f:
        data = np.copy(f[dkey])
        s = f[dkey].attrs['shape']

    return Atoms(positions=data[:, 1:], numbers=data[:, 0], cell=s)

def get_periodic_images(ipoints, domain, buffer_size):
    ## creates set of points which is periodic X, Y over the domain
    buffer_size = np.min([buffer_size] + list(domain[:2])) # in case domain is < buffer size

    # get faces
    fx0 = ipoints[:, 0] < buffer_size
    fx1 = ipoints[:, 0] >= (domain[0] - buffer_size)
    fy0 = ipoints[:, 1] < buffer_size
    fy1 = ipoints[:, 1] >= (domain[1] - buffer_size)


    bx0 = ipoints[fx0, :] # from x = 0-buffer to x = domain_size + buffer
    bx1 = ipoints[fx1, :] # from x = d - buffer -> d to x = -buffer -> 0
    by0 = ipoints[fy0, :] # from x = 0-buffer to x = domain_size + buffer
    by1 = ipoints[fy1, :] # from x = d - buffer -> d to x = -buffer -> 0

    bx0[:,0] += domain[0]
    bx1[:,0] -= domain[0]
    by0[:,1] += domain[1]
    by1[:,1] -= domain[1]

    faces = [bx0, bx1, by0, by1]

    # get corners
    cxy00 = np.logical_and(fx0, fy0)
    cxy01 = np.logical_and(fx0, fy1)
    cxy10 = np.logical_and(fx1, fy0)
    cxy11 = np.logical_and(fx1, fy1)

    bxy00 = ipoints[cxy00, :]
    bxy01 = ipoints[cxy01, :]
    bxy10 = ipoints[cxy10, :]
    bxy11 = ipoints[cxy11, :]

    bxy00[:, 0] += domain[0]
    bxy00[:, 1] += domain[1]

    bxy01[:, 0] += domain[0]
    bxy01[:, 1] -= domain[1]

    bxy10[:, 0] -= domain[0]
    bxy10[:, 1] += domain[1]

    bxy11[:, 1] -= domain[0]
    bxy11[:, 1] -= domain[1]

    corners = [bxy00, bxy01, bxy10, bxy11]

    return faces, corners


def get_voronoi_cells(min_dist, density, domain, rng, buffer_size=20):
    ## Sample a low density set of points in the target domain
    # We use these points to calculate a feasible Voronoi region
    # which we use as the grains of the nanocrystallites
    box = makeRectPrism(**domain)
    vor_gen = AmorphousGenerator(min_dist=min_dist, density=density, rng=rng)
    vor_obj = Volume(points=box, generator=vor_gen)
    vor_obj.populate_atoms(print_progress=False)

    vor_points = vor_obj.atoms

    ## Augment points with periodic images
    faces, corners = get_periodic_images(vor_points, [domain[k] for k in 'abc'], buffer_size)
    all_points = np.vstack([vor_points] + faces + corners)

    ## Calculate Voronoi tesselation
    vor = Voronoi(all_points)

    return vor, vor_points.shape[0]

def get_voronoi_bisectors(cells, interior):
    ## Initialize dict of interior regions corresponding to interior points
    region_bisectors = {}
    for idx in interior:
        region_bisectors[idx] = []

    ## Get all bisectors by interior point
    # Voronoi objects store an array of tuples K : (i, j), where K is a facet of a voronoi cell
    # and (i, j) are the points in the input domain which the facet bisects
    for k in cells.ridge_points:
        for kk in tuple(k):
            if kk in region_bisectors:
                bisect_idx = tuple(set(k).difference(set([kk])))[0] # gross!
                region_bisectors[kk].append(bisect_idx)

    return region_bisectors

def get_planes_from_bisectors(cells, bisectors):
    ## For each interior region, return all facets as Planes, defined by bisectors
    plane_sets = {}
    for k, v in bisectors.items():
        ipoint = cells.points[k, :] # save interior point
        plane_sets[k] = []

        # iterate over exterior points
        for epoint in [cells.points[j,:] for j in v]:
            ppoint = (ipoint + epoint) /2.0
            pnormal = epoint - ipoint # will be normalized on Plane init
            plane_sets[k].append(Plane(pnormal, ppoint))

    return plane_sets


def orient_and_shift_grain(generator, rng, lattice_param):
    ## Apply random rotation
    # rotate target zone-axis to +Z
    za = rng.normal(size=(3,1))
    za /= np.linalg.norm(za) # Since Si is cubic, no need to put into basis of reciprocal lattice

    za_rot = Rotation(matrix=rot_vtv(za.ravel(), [0,0,1]))
    generator.transform(za_rot)

    # rotate about ZA random amount
    theta = rng.uniform(0, 2*np.pi)
    rot_001 = Rotation(matrix=rot_v([0,0,1], theta))
    generator.transform(rot_001)

    ## Apply random shift
    shift = rng.uniform(0, lattice_param, (3,))
    generator.origin = shift

    return generator

def get_nanocrystalline_grains(min_dist, density, domain, rng):

    cells, N_interior = get_voronoi_cells(min_dist, density, domain, rng)

    ## Get bisectors for all facets of each cell and change into planes by region
    bisectors = get_voronoi_bisectors(cells, np.arange(N_interior))
    plane_sets = get_planes_from_bisectors(cells, bisectors)

    ## Create a set of volumes utilizing each facets by Voronoi region
    # we have to add the top/bottom faces, since we are only periodic in x/y
    origin = domain['center'] - np.array([domain['a'], domain['b'], domain['c']])/2 
    topbottom = [Plane((0,0,1), origin+np.array([0,0,domain['c']])), Plane((0,0,-1), origin)]

    return [Volume(alg_objects=p + topbottom) for p in plane_sets.values()]