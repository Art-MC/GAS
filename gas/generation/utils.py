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
    atoms = ase.io.read(fp, format='xyz')
    atoms.set_cell(cell)
    atoms.set_pbc(cell_data['pbc'])

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
    # all_points = vor_points     

    # all_points = np.vstack([vor_points, box])

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

def get_nanocrystalline_grains(min_dist, density, domain, rng)-> list[Volume]:
    
    vor, N_interior = get_voronoi_cells(min_dist, density, domain, rng)
    ## Get bisectors for all facets of each cell and change into planes by region
    bisectors = get_voronoi_bisectors(vor, np.arange(N_interior))
    plane_sets = get_planes_from_bisectors(vor, bisectors)

    ## Create a set of volumes utilizing each facets by Voronoi region
    # we have to add the top/bottom faces, since we are only periodic in x/y
    origin = domain['center'] - np.array([domain['a'], domain['b'], domain['c']])/2 
    topbottom = [Plane((0,0,1), origin+np.array([0,0,domain['c']])), Plane((0,0,-1), origin)]

    return [Volume(alg_objects=p + topbottom) for p in plane_sets.values()]



    # volumes = []
    
    # vor.vertices[:,0] = np.clip(vor.vertices[:,0], 0, domain['a'])
    # vor.vertices[:,1] = np.clip(vor.vertices[:,1], 0, domain['b'])
    # vor.vertices[:,2] = np.clip(vor.vertices[:,2], 0, domain['c'])
    
    # # for i, region_idx in enumerate(vor.point_region):
    # #     inds = np.array(vor.regions[region_idx])
    # #     vertices = vor.vertices[inds]
    # #     cpoint = vor.points[np.argwhere(vor.point_region==region_idx)].squeeze()
    # #     vertices[inds==-1] = cpoint
    # #     # if -1 in vor.regions[region_idx]:  # Ignore infinite regions
    # #     #     continue
        
    # #     # Clip to bounding box
    # #     # clipped_vertices = np.clip(vertices, [0, 0, 0], [domain['a'], domain['b'], domain['c']])
        
    # #     volume = Volume(vertices)  # Assuming cz.Volume takes a list of vertices
    # #     volumes.append(volume)
    # #     print("\nvolume: \n", volume)
    
    # center = vor.points.mean(axis=0)
    # ptp_bound = np.ptp(vor.points, axis=0)

    # central_vertices = [] 
    # edge_vertices = [] 
    
    # for i in range(len(vor.points)): 
    #     cpoint = vor.points[i] 
    #     region = np.array(vor.regions[vor.point_region[i]])
    #     # print(cpoint, region)

    #     if np.all(region>=0): 
    #         vertices = vor.vertices[region]
    #         central_vertices.append(vertices)
    #     else: 
    #         ridge_inds = np.where(vor.ridge_points == i)[0]
    #         ridge_vertices = np.array(vor.ridge_vertices)[ridge_inds]
    #         ridge_points = np.array(vor.ridge_points)[ridge_inds]
    #         # print(ridge_vertices)
    #         vertices = vor.vertices[np.unique(ridge_vertices[ridge_vertices>=0])].tolist()
    #         vertices.append(cpoint.tolist()) 
    #         for pointidx, simplex in zip(ridge_points, ridge_vertices): 
    #             if np.any(simplex < 0): 
    #                 i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

    #                 t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
    #                 t /= np.linalg.norm(t)
    #                 print("t: ", t)
    #                 n = np.array([-t[1], t[0]])  # normal

    #                 midpoint = vor.points[pointidx].mean(axis=0)
    #                 print("midpont c: ", midpoint, center, n)
    #                 direction = np.sign(np.dot(midpoint - center, n)) * n
                    
    #                 far_point = vor.vertices[i] + direction * ptp_bound.max()
    #                 far_point[0] = np.clip(far_point[0], 0, domain['a'])
    #                 far_point[1] = np.clip(far_point[1], 0, domain['b'])
    #                 if far_point.tolist() not in vertices: 
    #                     vertices.append(far_point.tolist()) 
    #         edge_vertices.append(np.array(vertices))
            
    #     # print(vertices)
    #     # print()
    #     volume = Volume(vertices)  # Assuming cz.Volume takes a list of vertices
    #     volumes.append(volume)

    
    
    # return volumes
    
#     bounding_box = np.array([[0, domain['a']], [0, domain['b']], [0, domain['c']]])  # Cube from (0,0,0) to (10,10,10)
#     voronoi_cells = get_voronoi_cells2(vor, bounding_box)
#     volumes = []
#     for c, verts in voronoi_cells.items(): 
#         volumes.append(
#             Volume(verts)
#         )
#     return volumes 
    
    
    
# def plane_box_intersection(plane_point, plane_normal, bbox):
#     """
#     Compute intersections of a plane (defined by a point and normal) with a 3D bounding box.
#     Returns a list of intersection points.
#     """
#     intersections = []
#     min_x, max_x = bbox[0]
#     min_y, max_y = bbox[1]
#     min_z, max_z = bbox[2]

#     # Define 6 bounding box planes
#     box_planes = [
#         ((min_x, 0, 0), (1, 0, 0)),  # X = min_x
#         ((max_x, 0, 0), (-1, 0, 0)), # X = max_x
#         ((0, min_y, 0), (0, 1, 0)),  # Y = min_y
#         ((0, max_y, 0), (0, -1, 0)), # Y = max_y
#         ((0, 0, min_z), (0, 0, 1)),  # Z = min_z
#         ((0, 0, max_z), (0, 0, -1))  # Z = max_z
#     ]

#     for (p0, n) in box_planes:
#         # Solve for intersection t where (p - p0) ⋅ n = 0
#         denom = np.dot(plane_normal, n)
#         if abs(denom) > 1e-6:  # Avoid division by zero (parallel planes)
#             t = np.dot(np.array(p0) - plane_point, n) / denom
#             intersection = plane_point + t * plane_normal
            
#             # Check if the intersection is within the bounding box
#             if (min_x <= intersection[0] <= max_x and 
#                 min_y <= intersection[1] <= max_y and 
#                 min_z <= intersection[2] <= max_z):
#                 intersections.append(intersection)

#     return intersections

# def get_voronoi_cells2(vor, bbox):
#     """
#     Compute the Voronoi cells clipped to the bounding box.
#     Returns a dictionary mapping each input point to its bounding polyhedron.
#     """
#     voronoi_cells = {}

#     for i, point in enumerate(vor.points):
#         region_index = vor.point_region[i]
#         region = vor.regions[region_index]

#         if -1 in region:  # Region has infinite ridges
#             finite_vertices = [vor.vertices[v] for v in region if v != -1]
#             for (p1, p2), ridge_vertices in zip(vor.ridge_points, vor.ridge_vertices):
#                 if -1 in ridge_vertices and (p1 == i or p2 == i):
#                     # Find finite vertex and compute direction
#                     finite_vertex = [vor.vertices[v] for v in ridge_vertices if v != -1][0]
#                     voronoi_center = vor.points[[p1, p2]].mean(axis=0)
#                     direction = finite_vertex - voronoi_center
#                     direction /= np.linalg.norm(direction)  # Normalize

#                     # Compute intersections with bounding box
#                     clipped_points = plane_box_intersection(finite_vertex, direction, bbox)
#                     finite_vertices.extend(clipped_points)

#             voronoi_cells[tuple(point)] = np.array(finite_vertices)
#         else:  # Fully finite cell
#             finite_vertices = [vor.vertices[v] for v in region]
#             voronoi_cells[tuple(point)] = np.array(finite_vertices)

#     return voronoi_cells