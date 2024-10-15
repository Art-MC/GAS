from pathlib import Path

import numpy as np
from czone.generator import Generator, NullGenerator
from czone.molecule import Molecule
from czone.scene import PeriodicScene
from czone.util.voxel import Voxel
from czone.volume import MultiVolume, Plane, Volume, get_bounding_box, makeRectPrism
from pymatgen.core import Structure
from utils import (
    fix_xyz_header,
    get_nanocrystalline_grains,
    load_and_prepare_xyz,
    orient_and_shift_grain,
)


def sample_grains(grains, min_dist, N, rng=np.random.default_rng(), max_attempts=32):
    """Sample seed grains with rejection sampling, to ensure minimum spacing"""
    centers = [
        np.mean(get_bounding_box(grain.alg_objects)[0], axis=0) for grain in grains
    ]

    all_indices = set(np.arange(len(grains)))
    chosen_indices = set([])
    cur_attempt = 0
    while len(chosen_indices) < N and cur_attempt < max_attempts:
        # Choose a random remaining grain
        next_trial = rng.choice(list(all_indices))
        trial_success = True

        # Check if chosen grain is too close to previously chosen grains
        for i in chosen_indices:
            if np.linalg.norm(centers[i] - centers[next_trial]) < min_dist:
                trial_success = False
                break

        cur_attempt += 1

        # Update list of grain indices and remove chosen index from canditates
        if trial_success:
            chosen_indices.update([next_trial])
            all_indices.difference_update([next_trial])
            cur_attempt = 0

    return [grain for i, grain in enumerate(grains) if i in chosen_indices]


def get_nanocrystal_seeds(
    domain,
    N_seeds,
    seed_crystal,
    min_dist,
    density,
    rng=np.random.default_rng(),
    **kwargs,
):
    # Get volumes for seed grains
    all_grains = get_nanocrystalline_grains(
        min_dist=min_dist, density=density, domain=domain, rng=rng
    )
    seed_grains = sample_grains(all_grains, min_dist=min_dist, N=N_seeds, rng=rng)

    # Attach a randomly oriented lattice generator for each seed
    crystal = Structure.from_file(seed_crystal)
    base_crystal_generator = Generator(structure=crystal)
    for grain in seed_grains:
        grain.add_generator(
            orient_and_shift_grain(
                base_crystal_generator.from_generator(), rng, crystal.lattice.a
            )
        )

    return seed_grains


def add_seeds_to_amorphous_block(
    atoms,
    cell_data,
    pbc,
    rng=np.random.default_rng(),
    return_seeds=False,
    tolerance=1e-5,
    **kwargs,
):
    """Starting with a given amorphous block, add in crystalline seeds."""

    ## Create a molecular generator with a box domain for the starting block
    domain = {k:v for k, v in zip(('a', 'b', 'c'), tuple(np.diag(atoms.get_cell())))}

    block_domain = makeRectPrism(**domain)
    block_domain += cell_data['cell_orig']

    domain['center'] = np.mean(block_domain, axis=0)

    amor_atoms = Molecule.from_ase_atoms(atoms)
    amor_block = Volume(points=block_domain, generator=amor_atoms)
    amor_block.priority = 1


    ## get nanocrystalline seeds
    print("Sampling nanocrystalline seeds")
    seed_grains = get_nanocrystal_seeds(domain, rng=rng, **kwargs)
    for seed in seed_grains:
        for p in seed.alg_objects:
            p.tol = tolerance

    final_seeds = []
    for seed in seed_grains:
        new_ex_grain = Volume(generator=NullGenerator(), priority=1)
        new_ex_grain.add_alg_object([Plane(p.normal, p.point + p.normal*tolerance ) for p in seed.alg_objects])
        seed_with_buffer = MultiVolume(volumes=[seed, new_ex_grain], priority=0)
        final_seeds.append(seed_with_buffer)

    ## Combine into scene and generate atoms
    scene = PeriodicScene(
        domain_cell=Voxel(atoms.get_cell(), origin=cell_data['cell_orig']),
        objects=[amor_block] + final_seeds,
        pbc=pbc,
    )

    scene.populate()

    if return_seeds:
        grain_scene = PeriodicScene(
            domain_cell=scene.domain_cell,
            objects=final_seeds,
            pbc=pbc,
        )
        grain_scene.populate()
        return scene, grain_scene
    else:
        return scene


def main():
    ## This will throw an error until in_file is updated
    in_file = Path("SiO2_1200K_amp.xyz")
    out_file = Path("SiO_seed_test.xyz")
    starting_block, cell_data = load_and_prepare_xyz(in_file)

    ## Set up the seed generation
    crystal_cif = Path("example_files/mp-149_as_Au.cif")
    config = {
        "seed_crystal": crystal_cif,  # which crystal to use as a seed grain
        "pbc": (True, True, False),  # pbc of scene
        "min_dist": 8,  # minimum distance between seed centers, in angstroms
        "N_seeds": 8,  # maximum number of seeds to include
        "density": 1e-3,  # number density of seed centers, in N * angstrom ^ -3
        # "tolerance": 1.0,  # amount of excess space to exclude orig liquid atoms from
        "return_seeds": (
            return_seeds := True
        ),  # also return a scene with just the seed grains
    }

    if return_seeds:
        block_with_seeds, seeds = add_seeds_to_amorphous_block(starting_block, cell_data, **config)
        seeds.to_file("SiO_seeds.xyz")
    else:
        block_with_seeds = add_seeds_to_amorphous_block(starting_block, cell_data, **config)
    block_with_seeds.to_file("SiO_seed_test.xyz", )

    fix_xyz_header(out_file, in_file)




if __name__ == "__main__":
    main()
