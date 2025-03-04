from scipy.spatial.transform import Rotation as R 
from czone.transform import Rotation as CZRotation
from czone.scene import PeriodicScene
from czone.util.voxel import Voxel
from czone.molecule import Molecule
from czone.volume import MultiVolume, Plane, Volume, get_bounding_box, makeRectPrism
from gas.generation.utils import get_nanocrystalline_grains, get_spherical_grains
from czone.generator import Generator, NullGenerator
from scipy.interpolate import RegularGridInterpolator

from ase import Atoms 
import numpy as np 
from tqdm import tqdm 
from scipy.spatial import KDTree


## TODO make this a class, with vprint and params dict and rng and such 
## and then a wrapper class for generating lots of them, with ranges of values to choose from, etc.

def modify_volume(starting_atoms: Atoms, config: dict, rng:np.random.Generator | None=None, v:int=1) -> Atoms: 
    if rng is None: 
        rng = np.random.default_rng() 
        
    atoms = starting_atoms.copy() 

    atoms = _jitter_atoms(atoms, config, rng) 

    atoms = _rotate_atoms(atoms, config, rng, v) 

    atoms = _shift_atoms(atoms, config, rng)
            
    atoms = _push_close_atoms(atoms, threshold=config["threshold"], v=v)
    return atoms 


def _rotate_atoms(atoms: Atoms, config:dict, rng:np.random.Generator, v:int=1) -> Atoms: 
    bbox = np.diag(atoms.cell)
    rad = config['radius']
    if config["N_iterations_rot"] >= 1: 
        assert rad <= bbox.min()/2, f"Sphere radius {rad} must be < bbox/2: {bbox/2}"
    
    # assert np.all(atoms.pbc), "need to check _rotate_atoms if not having pbcs"
    # pbcs = [True,True,True]
    pbcs = atoms.pbc 
    
    for a0 in tqdm(range(config["N_iterations_rot"]), desc="applying rotations", disable=v<1): 
        _pick_point_tries = 0 
        cpoints = [] 
        while len(cpoints) < config["N_points_per_iter"] and _pick_point_tries < config["_pick_point_max"]: 
            _pick_point_tries += 1 
            npoint = [] 
            for a1 in range(3): 
                if pbcs[a1]: 
                    _p = rng.random() * bbox[a1] 
                else: 
                    _p = rng.random() * (bbox[a1] - 2 * rad) + rad
                npoint.append(_p)
                
            dists = _get_dists_pbcs(cpoints, npoint, bbox, pbcs=pbcs)
            if np.all(dists >= rad*2):
                cpoints.append(npoint)
                _pick_point_tries = 0 
        if _pick_point_tries >= config["_pick_point_max"] and v > 1: 
            print(f"_pick_point_tries failed after {len(cpoints)} / {config["N_points_per_iter"]} points found")
        cpoints = np.array(cpoints)
        
        positions = atoms.positions 
        positions = np.mod(positions, bbox)
        tree = KDTree(positions, boxsize = bbox) 
        spheres = tree.query_ball_point(cpoints, rad)        
        
        for sphere_inds, cpoint in zip(spheres, cpoints): 
        
            theta = (rng.random()*2-1) * config["theta_max"] 
            # vec = [0,0,1]
            vec = rng.random(3)
            vec /= np.linalg.norm(vec)
            rotmat = R.from_rotvec(vec*theta, degrees=True).as_matrix()
            rotmat
            
            shifted_pos = atoms.positions[sphere_inds] - cpoint
            shifted_pos = np.mod(shifted_pos+bbox/2, bbox) - bbox/2 
            rot_pos = shifted_pos @ rotmat 
            rot_pos += cpoint 
            rot_pos = np.mod(rot_pos, bbox)
            atoms.positions[sphere_inds] = rot_pos      
            
    return atoms         
    

def _remove_close_atoms(starting_atoms:Atoms, threshold:float, v:int=1):
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
    if v: 
        rm = np.sum(mask)
        tot = len(positions)
        print(f"Deleting {rm} / {tot} atoms = {rm/tot*100:.2f}%")
    del atoms[mask]
    return atoms 


def _push_close_atoms(starting_atoms: Atoms, threshold: float, max_iterations=50, push_factor=1, v:int=1):
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
    
    pbar = tqdm(range(max_iterations), desc=f"pushing atoms (max iters = {max_iterations})", leave=True, disable=v<1, total=np.inf)
    for a0 in pbar:
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
                    direction = displacement / distance  # Unit vectorf
                    
                    positions[i] -= push_distance * direction
                    positions[j] += push_distance * direction  # Move in opposite direction
        push_factor *= 0.99        
        positions = np.mod(positions, np.diag(cell))
        
        if not moved:
            # print("done after iter: ", a0)
            break  
    atoms.set_positions(positions)
    
    if moved: 
        _remove_close_atoms(atoms, threshold)

    
    return atoms

def _get_dists_pbcs(points, cpointslists, bbox, pbcs=[1, 1, 1]):
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


def _shift_atoms(atoms: Atoms, config: dict, rng:np.random.Generator, v:int=1) -> Atoms: 
    if "shift_sigma" not in config.keys():
        return atoms  
    if config["shift_sigma"] == 0: 
        return atoms 
    
    bbox = np.diag(atoms.cell) 
    N_points = np.ceil(bbox/config["shift_point_spacing"]).astype('int')
    N_points = np.maximum(N_points, 4) # interpolator requires min 4 points each direction


    for a0 in tqdm(range(config["N_iterations_shift"]), desc="applying shifts", disable=v<1): 
        
        w_positions = atoms.positions 
        w_positions = np.mod(w_positions, bbox)
        
        xgrid = np.linspace(0, bbox[0], N_points[0])
        ygrid = np.linspace(0, bbox[1], N_points[1])
        zgrid = np.linspace(0, bbox[2], N_points[2])

        shifts = rng.normal(0, config["shift_sigma"], (3, *N_points))
        # shifts = rng.normal(0, config["shift_sigma"], (3, *N_points))
        shifts[:,-1] = shifts[:, 0] # pbcs
        shifts[:,:,-1] = shifts[:,:,0] # pbcs
        shifts[:,:,:,-1] = shifts[:,:,:,0] # pbcs
        xinterp = RegularGridInterpolator((xgrid,ygrid,zgrid), shifts[0], method='cubic')
        yinterp = RegularGridInterpolator((xgrid,ygrid,zgrid), shifts[1], method='cubic')
        zinterp = RegularGridInterpolator((xgrid,ygrid,zgrid), shifts[2], method='cubic')

        xshifts = xinterp(w_positions)
        yshifts = yinterp(w_positions)
        zshifts = zinterp(w_positions)
        shifts = np.stack([xshifts,yshifts,zshifts]).T 
        
        atoms.positions += shifts 

        atoms.wrap(eps=1e-10)      
    
    return atoms 


def _jitter_atoms(atoms: Atoms, config:dict, rng:np.random.Generator) -> Atoms: 
    if "gaussian_sigma" in config.keys(): 
        sigma = config["gaussian_sigma"] 
        if sigma >0: 
            atoms.positions += rng.normal(0, sigma, atoms.positions.shape)
            atoms.wrap(eps=1e-10)
    
    return atoms 