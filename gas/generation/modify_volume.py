from scipy.spatial.transform import Rotation as R 
from czone.transform import Rotation as CZRotation
from czone.scene import PeriodicScene
from czone.util.voxel import Voxel
from czone.molecule import Molecule
from czone.volume import MultiVolume, Plane, Volume, get_bounding_box, makeRectPrism
from gas.generation.utils import get_nanocrystalline_grains
from czone.generator import Generator, NullGenerator

from ase import Atoms 
import numpy as np 
from tqdm import tqdm 
from scipy.spatial import KDTree

def modify_volume(starting_atoms: Atoms, config: dict, rng:np.random.Generator | None=None) -> Atoms: 
    if rng is None: 
        rng = np.random.default_rng() 
        
    domain = {k:v for k, v in zip(('a', 'b', 'c'), tuple(np.diag(starting_atoms.get_cell())))}
    block_domain = makeRectPrism(**domain)
    domain['center'] = np.mean(block_domain, axis=0)
    
    
    starting_molecule = Molecule.from_ase_atoms(starting_atoms)
    # starting_block_block = Volume(points=block_domain, generator=starting_molecule)
    # starting_block_block.priority = 1
    
    # Divide the volume into voronoi grains
    # print("getting grains")
    all_grains = get_nanocrystalline_grains(
        min_dist=config["min_dist"], density=config["density"], domain=domain, rng=rng
    )
    
    # tot_atoms = 0 
    # all_atoms = np.zeros(len(starting_atoms), dtype=bool) 
    # for grain in all_grains: 
    #     goods = grain.checkIfInterior(starting_molecule.atoms)
    #     all_atoms = all_atoms | goods
    #     tot_atoms+= np.sum(goods)
    # print(f"total atoms included: {tot_atoms} / {len(starting_atoms)} | frac: {tot_atoms / len(starting_molecule.atoms)*100:.2f}")
    # bad_atoms = starting_atoms.positions[~all_atoms]
    # print(f"Len bad atoms: {len(bad_atoms)}")
    # print("bad_atoms positions: \n", bad_atoms)
    # print("block_domain: \n", block_domain)
    
    # scene = PeriodicScene(
    #     domain=Voxel(starting_atoms.get_cell(), origin=[0,0,0]),
    #     objects=all_grains,
    #     pbc=[True,True,True],#starting_atoms.pbc,
    # )
    # scene = PeriodicScene(domain=Voxel(starting_atoms.get_cell(), origin=[0,0,0]), 
    #                       objects=[g.from_volume(generator=starting_molecule) for g in all_grains], 
    #                       pbc=(False,False,False))
    # scene.populate_no_collisions()
    # print("num scene atoms: ", len(scene.ase_atoms))
    
    # Get seed grains 
    random_inds = np.arange(len(all_grains))
    rng.shuffle(random_inds)
    if "N_seeds" in config.keys(): 
        n_seeds = config["N_seeds"]
    else: 
        seeds_frac = config.get("seeds_frac", 0.5)
        n_seeds = int(len(all_grains) * seeds_frac)
    modified_grains: list[Volume] = [all_grains[i] for i in random_inds[:n_seeds]]
    static_grains: list[Volume] = [all_grains[i] for i in random_inds[n_seeds:]]

    tolerance = config.get("tolerance", 1e-10)
    for seed in modified_grains + static_grains:
        for p in seed.alg_objects:
            p.tol = tolerance
            
    print(f"{len(modified_grains)} / {len(all_grains)} seeds selected")

    modified_seeds = []
    tot_atoms = 0 
    bad_atoms = np.ones(len(starting_atoms), dtype=bool) 
    for cell in tqdm(modified_grains, desc="modifying grains"): 
        goods = cell.checkIfInterior(starting_molecule.atoms)
        bad_atoms[goods] = 0 
        tot_atoms += np.sum(goods)
        m = Molecule(species=starting_molecule.species[goods], positions=starting_molecule._atoms[goods], origin=starting_molecule.origin)
        theta = (np.random.rand()*2-1) * config["theta_max"] 
        vec = np.random.rand(3)
        vec /= np.linalg.norm(vec)
        rotmat = R.from_rotvec(vec*theta, degrees=True).as_matrix()
        rot = CZRotation(matrix=rotmat, origin=m._atoms.mean(axis=0))
        m.transform(rot)
        cell._generator = m 
        new_ex_grain = Volume(generator=NullGenerator(), priority=2)
        new_ex_grain.add_alg_object([Plane(p.normal, p.point + p.normal*tolerance ) for p in cell.alg_objects])
        seed_with_buffer = MultiVolume(volumes=[cell, new_ex_grain], priority=0)
        modified_seeds.append(seed_with_buffer)
    
    static_seeds = [] 
    for cell in tqdm(static_grains, desc="static grains"): 
        goods = cell.checkIfInterior(starting_molecule.atoms)
        bad_atoms[goods] = 0 
        m = Molecule(species=starting_molecule.species[goods], positions=starting_molecule._atoms[goods], origin=starting_molecule.origin)
        cell._generator = m 
        new_ex_grain = Volume(generator=NullGenerator(), priority=2)
        new_ex_grain.add_alg_object([Plane(p.normal, p.point + p.normal*tolerance ) for p in cell.alg_objects])
        seed_with_buffer = MultiVolume(volumes=[cell, new_ex_grain], priority=1)
        static_seeds.append(seed_with_buffer)
    
    print(f"total atoms: mod {tot_atoms} | edge: {bad_atoms.sum()} | sum {bad_atoms.sum() + tot_atoms} | true total {len(starting_molecule.atoms)} | ") 
    
    # bad_seed = 
    m = Molecule(species=starting_molecule.species[bad_atoms], positions=starting_molecule._atoms[bad_atoms], origin=starting_molecule.origin)
    extra_atoms = Volume(generator=NullGenerator(), priority=0)
    extra_atoms._generator = m
    extra_atoms.points = block_domain
    # new_ex_grain.add_alg_object([Plane(p.normal, p.point + p.normal*tolerance ) for p in cell.alg_objects])
    # seed_with_buffer = MultiVolume(volumes=[cell, new_ex_grain], priority=1)
    # static_seeds.append(new_ex_grain)
    
    
    print('making scene')
    scene = PeriodicScene(
        domain=Voxel(starting_atoms.get_cell(), origin=[0,0,0]),
        objects=static_seeds + modified_seeds + [extra_atoms],
        pbc=starting_atoms.pbc,
    )
    
    print('populating')
    scene.populate_no_collisions()
    print('done')    
    atoms = scene.ase_atoms
    print("# atoms1: ", len(atoms))
    atoms.set_pbc(starting_atoms.pbc)
    print("# atoms2: ", len(atoms))
    atoms.wrap()
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
