from scipy.spatial.transform import Rotation as R 
from czone.transform import Rotation as CZRotation
from czone.scene import PeriodicScene
from czone.util.voxel import Voxel
from czone.molecule import Molecule
from czone.volume import MultiVolume, Plane, Volume, get_bounding_box, makeRectPrism
from gas.generation.utils import get_nanocrystalline_grains, get_spherical_grains
from czone.generator import Generator, NullGenerator

from ase import Atoms 
import numpy as np 
from tqdm import tqdm 
from scipy.spatial import KDTree

# def modify_volume(starting_atoms: Atoms, config: dict, rng:np.random.Generator | None=None, region_type="vor") -> Atoms: 
#     if rng is None: 
#         rng = np.random.default_rng() 
#     if "region_type" in config.keys(): 
#         region_type = config["region_type"]
    
#     domain = {k:v for k, v in zip(('a', 'b', 'c'), tuple(np.diag(starting_atoms.get_cell())))}
#     block_domain = makeRectPrism(**domain)
#     domain['center'] = np.mean(block_domain, axis=0)
    
    
#     starting_molecule = Molecule.from_ase_atoms(starting_atoms)

#     if region_type.lower() in ["vor", "voronoi"]:     
#         all_grains = get_nanocrystalline_grains(
#             min_dist=config["min_dist"], density=config["density"], domain=domain, rng=rng
#         )
        
#         random_inds = np.arange(len(all_grains))
#         rng.shuffle(random_inds)
#         if "N_seeds" in config.keys(): 
#             n_seeds = config["N_seeds"]
#             if n_seeds == "all": 
#                 n_seeds = len(all_grains) 
#         else: 
#             seeds_frac = config.get("seeds_frac", 0.5)
#             n_seeds = int(len(all_grains) * seeds_frac)

#         # Get seed grains 
#         modified_grains: list[Volume] = [all_grains[i] for i in random_inds[:n_seeds]]
#         static_grains: list[Volume] = [all_grains[i] for i in random_inds[n_seeds:]]
#         print(f"{len(modified_grains)} / {len(all_grains)} Voronoi seeds selected")
        
#     elif region_type.lower() == "sphere": 
#         # get modified grains as list of sphere Volumes
#         # and do a static which is opposite of that
#         modified_grains = get_spherical_grains(
#             domain=domain, rng=rng, config=config
#         )
#         static_grains = [] 

#     tolerance = config.get("tolerance", 1e-10)
#     for seed in modified_grains + static_grains:
#         for p in seed.alg_objects:
#             p.tol = tolerance

#     modified_seeds = []
#     tot_atoms = 0 
#     bad_atoms = np.ones(len(starting_atoms), dtype=bool) 
#     for cell in tqdm(modified_grains, desc="modifying grains"): 
#         goods = cell.checkIfInterior(starting_molecule.atoms)
#         bad_atoms[goods] = 0 
#         tot_atoms += np.sum(goods)
#         m = Molecule(species=starting_molecule.species[goods], positions=starting_molecule._atoms[goods], origin=starting_molecule.origin)
#         theta = (np.random.rand()*2-1) * config["theta_max"] 
#         vec = np.random.rand(3)
#         vec /= np.linalg.norm(vec)
#         rotmat = R.from_rotvec(vec*theta, degrees=True).as_matrix()
#         rot = CZRotation(matrix=rotmat, origin=m._atoms.mean(axis=0))
#         m.transform(rot)
#         cell._generator = m 
#         if region_type.lower() != "sphere": 
#             new_ex_grain = Volume(generator=NullGenerator(), priority=2)
#             new_ex_grain.add_alg_object([Plane(p.normal, p.point + p.normal*tolerance ) for p in cell.alg_objects])
#             seed_with_buffer = MultiVolume(volumes=[cell, new_ex_grain], priority=0)
#         else: 
#             seed_with_buffer = cell 
#         modified_seeds.append(seed_with_buffer)
    
#     static_seeds = [] 
#     for cell in tqdm(static_grains, desc="static grains"): 
#         goods = cell.checkIfInterior(starting_molecule.atoms)
#         bad_atoms[goods] = 0 
#         m = Molecule(species=starting_molecule.species[goods], positions=starting_molecule._atoms[goods], origin=starting_molecule.origin)
#         cell._generator = m 
#         new_ex_grain = Volume(generator=NullGenerator(), priority=2)
#         new_ex_grain.add_alg_object([Plane(p.normal, p.point + p.normal*tolerance ) for p in cell.alg_objects])
#         seed_with_buffer = MultiVolume(volumes=[cell, new_ex_grain], priority=1)
#         static_seeds.append(seed_with_buffer)
    
#     print(f"total atoms: mod {tot_atoms} = {tot_atoms / len(starting_molecule.atoms) * 100:.1f}% | edge: {bad_atoms.sum()} | sum {bad_atoms.sum() + tot_atoms} | true total {len(starting_molecule.atoms)} | ") 
    
#     # bad_seed = 
#     m = Molecule(species=starting_molecule.species[bad_atoms], positions=starting_molecule._atoms[bad_atoms], origin=starting_molecule.origin)
#     extra_atoms = Volume(generator=NullGenerator(), priority=0)
#     extra_atoms._generator = m
#     extra_atoms.points = block_domain
#     # new_ex_grain.add_alg_object([Plane(p.normal, p.point + p.normal*tolerance ) for p in cell.alg_objects])
#     # seed_with_buffer = MultiVolume(volumes=[cell, new_ex_grain], priority=1)
#     # static_seeds.append(new_ex_grain)
    
    
#     print('making scene')
#     scene = PeriodicScene(
#         domain=Voxel(starting_atoms.get_cell(), origin=[0,0,0]),
#         objects=static_seeds + modified_seeds + [extra_atoms],
#         pbc=starting_atoms.pbc,
#     )
    
#     print('populating')
#     scene.populate_no_collisions()
#     print('done')    
#     atoms = scene.ase_atoms
#     print("# atoms1: ", len(atoms))
#     atoms.set_pbc(starting_atoms.pbc)
#     print("# atoms2: ", len(atoms))
#     atoms.wrap()
#     return atoms


def modify_volume(starting_atoms: Atoms, config: dict, rng:np.random.Generator | None=None) -> Atoms: 
    if rng is None: 
        rng = np.random.default_rng() 
        
    atoms = starting_atoms.copy() 
    pbcs = [True,True,True]
    assert np.all(atoms.pbc)
    bbox = np.diag(atoms.cell)
    
    if "gaussian_sigma" in config.keys(): 
        sigma = config["gaussian_sigma"] 
        if sigma >0: 
            atoms.positions += rng.normal(0, sigma, atoms.positions.shape)
            atoms.wrap(eps=1e-10)
    
    for a0 in tqdm(range(config["N_iterations"])): 
        _pick_point_tries = 0 
        cpoints = [] 
        while len(cpoints) < config["N_points_per_iter"] and _pick_point_tries < config["_pick_point_max"]: 
            _pick_point_tries += 1 
            npoint = rng.random(3) * bbox 
            dists = get_dists_pbcs(cpoints, npoint, bbox, pbcs=pbcs)
            if np.all(dists >= config["radius"]*2):
                cpoints.append(npoint)
                _pick_point_tries = 0 
        if _pick_point_tries >= config["_pick_point_max"]: 
            print(f"_pick_point_tries failed after {len(cpoints)} / {config["N_points_per_iter"]} points found")
        cpoints = np.array(cpoints)
        
        tree = KDTree(atoms.positions, boxsize = bbox) 
        spheres = tree.query_ball_point(cpoints, config["radius"])        
        
        theta = (rng.random()*2-1) * config["theta_max"] 
        vec = rng.random(3)
        vec /= np.linalg.norm(vec)
        rotmat = R.from_rotvec(vec*theta, degrees=True).as_matrix()
        rotmat
        
        for sphere_inds, cpoint in zip(spheres, cpoints): 
            shifted_pos = atoms.positions[sphere_inds] - cpoint
            shifted_pos = np.mod(shifted_pos+bbox/2, bbox) - bbox/2 
            rot_pos = shifted_pos @ rotmat 
            rot_pos += cpoint 
            rot_pos = np.mod(rot_pos, bbox)
            atoms.positions[sphere_inds] = rot_pos      
    atoms = push_close_atoms(atoms, threshold=config["threshold"])
    return atoms 

def remove_close_atoms(starting_atoms:Atoms, threshold:float):
    atoms = starting_atoms.copy() 
    positions = starting_atoms.positions
    tree = KDTree(positions, boxsize=np.diag(atoms.cell.array))
    to_remove = set()

    # Find all neighbors within the threshold distance
    for i, neighbors in enumerate(tree.query_ball_tree(tree, threshold)):
        if i in to_remove:
            continue  
        for j in neighbors: 
            if i != j:
                to_remove.add(j)  

    # Keep only unmarked positions
    mask = np.array([i in to_remove for i in range(len(positions))])
    del atoms[mask]
    return atoms 


def push_close_atoms(starting_atoms: Atoms, threshold: float, max_iterations=100, push_factor=0.75):
    """
    Adjusts atom positions in the system such that no two atoms are closer than `threshold`.
    
    Args:
        starting_atoms (Atoms): Initial atomic configuration.
        threshold (float): Minimum allowed distance between atoms.
        max_iterations (int): Maximum iterations to avoid infinite loops.
        push_factor (float): Fraction of threshold used to push atoms apart.
    
    Returns:
        Atoms: Modified atomic configuration with adjusted positions.
    """
    atoms = starting_atoms.copy()
    positions = atoms.positions
    cell = atoms.cell.array
    pbc = atoms.get_pbc()

    for a0 in tqdm(range(max_iterations), desc="pushing atoms"):
        tree = KDTree(positions, boxsize=np.diag(cell) if np.any(pbc) else None)
        moved = False  # Track if any atoms are adjusted
        
        for i, neighbors in enumerate(tree.query_ball_tree(tree, threshold)):
            for j in neighbors:
                if i >= j:
                    continue  
                
                # Compute distance vector with periodic boundary handling
                displacement = positions[j] - positions[i]
                if np.any(pbc):  # Apply minimum image convention
                    displacement -= np.round(displacement / cell.diagonal()) * cell.diagonal()

                distance = np.linalg.norm(displacement)
                if distance < threshold:
                    moved = True
                    # Normalize and scale displacement to push atoms apart
                    push_distance = (threshold - distance) * push_factor
                    direction = displacement / distance  # Unit vector
                    
                    positions[i] -= push_distance * direction
                    positions[j] += push_distance * direction  # Move in opposite direction
        
        positions = np.mod(positions, np.diag(cell))
        
        if not moved:
            # print("done after iter: ", a0)
            break  
    
    print("\rDone")
    atoms.set_positions(positions)
    return atoms

def get_dists_pbcs(points, cpointslists, bbox, pbcs=[1, 1, 1]):
    points = np.array(points, ndmin=2) 
    cpointslists = np.array(cpointslists, ndmin=2) 
    if not np.any(points): 
        return np.array([np.inf])
        
    if np.any(pbcs):
        assert np.all(np.min(cpointslists, axis=-2) >= 0)
        assert np.all(
            np.max(cpointslists, axis=-2) <= bbox
        ), f"bbox: {bbox}, pointslist max: {np.max(cpointslists, axis=(0,1))}"
    assert "float" in str(
        cpointslists.dtype
    ), f"cpointslists type: {cpointslists.dtype}"
    abs = np.abs(cpointslists - points[:, None])
    for i, ind in enumerate(pbcs):
        if ind:
            abs[:, :, i] = np.minimum(abs[:, :, i], bbox[i] - abs[:, :, i])
    return np.sqrt(np.sum(abs**2, axis=-1))