from scipy.spatial.transform import Rotation as R
from czone.transform import Rotation as CZRotation
from czone.scene import PeriodicScene
from czone.util.voxel import Voxel
from czone.molecule import Molecule
from czone.volume import MultiVolume, Plane, Volume, get_bounding_box, makeRectPrism
from gas.generation.utils import get_nanocrystalline_grains, get_spherical_grains
from czone.generator import Generator, NullGenerator
from scipy.interpolate import RegularGridInterpolator
import sys 
import os 

from ase import Atoms
from ase import units as aunits
import numpy as np
from tqdm import tqdm
from scipy.spatial import KDTree
from mace.calculators import mace_mp
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin


## TODO make this a class, with vprint and params dict and rng and such
## and then a wrapper class for generating lots of them, with ranges of values to choose from, etc.


class TransformVolume(object):
    DEFAULT_CONFIG_MD = {
        "run_md": False,
        "chunk_size": (40, 40, 40),
        "sim_time_fs": 100,
        "max_step_fs": 10,
        "min_step_fs": 0.5,
        "temp_k": 0,
        "temp_k_reset": 300,
        "friction": 1e-3,
        "total_run_time": 0,  # will be updated as sims run
        "pbc": False,  # requires more memory if True
    }

    DEFAULT_CONFIG_RANDOMIZE = {
        "run_randomize": True,
        "N_iterations_rot": 1,
        "rot_radius": 10,  # A
        "theta_max": 180,
        "jitter_gaussian_sigma": 0.05,
        "N_iterations_shift": 1,  # max 1 if doing MD?
        "shift_point_spacing": 25,  # A -- max used
        "shift_sigma": 0.25,  # A
        "push_threshold": 0,  # A - this should really be set each time
        "pbc": True,  # rotations are less useful if unable to rotate across boundaries
        "N_points_per_iter": 1e3,  # will pick up to this number
        "pick_point_max": 1e3,  # number of tries
        "push_iter_max": 50,
        "region_type": "sphere",
    }

    def __init__(
        self,
        config_randomize: dict = {},
        config_md: dict = {},
        rng: np.random.Generator | None = None,
        v: int = 1,
    ):
        self.config_randomize = config_randomize
        self.config_md = config_md
        self.rng = rng if rng is not None else np.random.default_rng()
        self.verbose = v
        with Suppressor(): 
            self.calc = mace_mp(model="small", default_dtype="float32", device="cuda")

    @property
    def config_randomize(self) -> dict:
        return self._config_randomize

    @config_randomize.setter
    def config_randomize(self, config: dict):
        for key in config.keys():
            if key not in self.DEFAULT_CONFIG_RANDOMIZE.keys():
                raise KeyError(f"{key} not a recognized randomize config key")
        if hasattr(self, "_config_randomize"):
            self._config_randomize = self._config_randomize | config  # keep all old settings
        else:
            self._config_randomize = self.DEFAULT_CONFIG_RANDOMIZE | config  # use defaults

    @property
    def config_md(self) -> dict:
        return self._config_md

    @config_md.setter
    def config_md(self, config: dict):
        for key in config.keys():
            if key not in self.DEFAULT_CONFIG_MD.keys():
                raise KeyError(f"{key} not a recognized md config key")
        if hasattr(self, "_config_md"):
            self._config_md = self._config_md | config  # keep all old settings
        else:
            self._config_md = self.DEFAULT_CONFIG_MD | config  # use defaults

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    @rng.setter
    def rng(self, gen: np.random.Generator):
        if not isinstance(gen, np.random.Generator):
            raise TypeError(f"rng should be a np.random.Generator, got {type(rng)}")
        self._rng = gen

    @property
    def verbose(self) -> int:
        return self._verbose

    @verbose.setter
    def verbose(self, v: bool | int | float):
        if isinstance(v, float):
            if v.is_integer():
                v = int(v)
        if isinstance(v, bool):
            self._verbose = int(v)
        elif isinstance(v, int):
            self._verbose = max(0, v)
        else:
            raise TypeError(f"verbose should be integer or boolean, got {type(v)} v = {v}")

    def vprint(self, *args, **kwargs):
        """Print messages if verbose is enabled."""
        if self.verbose:
            print(*args, **kwargs)

    def apply(
        self,
        starting_atoms: Atoms,
        rng: np.random.Generator | None = None,
        v: int | None = None,
        config_randomize: dict | None = None,
        config_md: dict | None = None,
    ) -> Atoms:
        if v is not None:
            self.verbose = v
        if rng is not None:
            self.rng = rng
        if config_randomize is not None:
            self.config_randomize = config_randomize
        if config_md is not None:
            self.config_md = config_md

        if self.config_randomize["run_randomize"]:
            atoms = starting_atoms.copy()
            atoms.set_pbc(self.config_randomize["pbc"])

            atoms = self._jitter_atoms(atoms)
            atoms = self._rotate_atoms(atoms)
            atoms = self._shift_atoms(atoms)
            atoms = self._push_close_atoms(atoms)
        self._pre_md_atoms = atoms.copy() 
        if self.config_md["run_md"]:
            atoms = self.run_md(atoms)

        return atoms

    def run_md(
        self,
        starting_atoms: Atoms,
        reset:bool=True,
        config_md: dict | None = None,
        v: int | None = None,
        rng: np.random.Generator | None = None,
    ):

        if v is not None:
            self.verbose = v
        if rng is not None:
            self.rng = rng
        if config_md is not None:
            self.config_md = config_md
        assert self.config_md["temp_k_reset"] >= self.config_md["temp_k"]
        
        atoms = starting_atoms.copy()
        atoms_cell = np.diag(atoms.get_cell())
        max_chunk = np.array(self.config_md["chunk_size"])
        if np.any(atoms_cell > max_chunk): 
            # for each chunk # atoms.set_pbc(False)
            if not reset: 
               print("hard to keep track of sim times when chunking") 
            reset = True 
            chunks = self._chunk_volume(atoms, self.config_md["chunk_size"])
            self.vprint(f"chunking MD sim into {len(chunks)} parts")
            for a0 in tqdm(range(len(chunks)), desc="MD chunks"): 
                # print(f"chunk {a0+1}/{len(chunks)}")
                ch = chunks[a0]
                ch.set_pbc(self.config_md["pbc"])
                chunks[a0] = self._run_md_chunk(ch, reset=True, v=0)
            md_atoms = self.recombine_chunks(chunks, atoms.get_cell(), self.config_md["chunk_size"])
            # md_atoms = self._push_close_atoms(md_atoms)
            md_atoms = self._remove_close_atoms(md_atoms)
            
        else: 
            atoms.set_pbc(self.config_md["pbc"])
            md_atoms = self._run_md_chunk(atoms, reset, v=self.verbose)
                    
        self.vprint(f"total sim time: {self.config_md["total_run_time"] / 1e3:.3f} ps -- DONE")
        return md_atoms 

    def _run_md_chunk(self, atoms: Atoms, reset:bool=True, v:int|None=None) -> Atoms:
        if reset:
            self.config_md["total_run_time"] = 0
            tot_time = 0
            ckp_time = 0
        else:
            tot_time = self.config_md["total_run_time"]
            ckp_time = self.config_md["total_run_time"]
        start_time = tot_time 

        if v is None:
            v = self.verbose

        temp_k = self.config_md["temp_k"]

        atoms.calc = self.calc 
        MaxwellBoltzmannDistribution(
            atoms, temperature_K=temp_k, rng=self.rng
        )  # init velocities at high T for melting
        md_step_fs = self.config_md["max_step_fs"]
        dyn = Langevin(
            atoms,
            md_step_fs * aunits.fs,
            temperature_K=temp_k,
            friction=self.config_md["friction"],
            rng=self.rng,
        )

        a0 = 0
        if v > 1:
            self.printenergy(atoms, a0, tot_time)
        ckp_pos = atoms.positions.copy()
        nsteps_since_temp_reset = 0
        pbar = tqdm(
            total=self.config_md["sim_time_fs"],
            desc=f"Etot {999:.3f} eV | T {999:03.0f} K | rst {0} | fs",
            bar_format="{desc}{n:.1f}/{total_fmt} |{bar}| {percentage:.2f}% | {rate_fmt} | {remaining}",
            unit="fs",
            disable=v<1, 
        )
        while tot_time < start_time + self.config_md["sim_time_fs"]:
            curr_temp = atoms.get_kinetic_energy() / len(atoms) / (1.5 * aunits.kB)
            if curr_temp > self.config_md["temp_k_reset"]:  # reset temperature to
                nsteps_since_temp_reset = 0
                if curr_temp > 1e4:
                    # atoms = remove_ejected_atoms(atoms)

                    # Revert back to checkpoint and decrease timestep
                    if md_step_fs <= self.config_md["min_step_fs"]:
                        if v>=1: 
                            print(
                                f'step size {md_step_fs} fs <= min step of {self.config_md["min_step_fs"]} and temp is {curr_temp} K -- aborting'
                            )
                        atoms.positions = ckp_pos
                        tot_time = ckp_time
                        break
                    pbar.update((ckp_time - tot_time))
                    tot_time = ckp_time
                    atoms.positions = ckp_pos
                    md_step_fs = max(md_step_fs / 2, self.config_md["min_step_fs"])
                    dyn = Langevin(
                        atoms,
                        md_step_fs * aunits.fs,
                        temperature_K=temp_k,
                        friction=self.config_md["friction"],
                        rng=self.rng,
                    )
                    if v > 1:
                        print("new time step: ", md_step_fs)
                    # atoms.rattle(0.05)
                MaxwellBoltzmannDistribution(atoms, temperature_K=temp_k, rng=self.rng)
            else:
                if nsteps_since_temp_reset > 5 and md_step_fs < self.config_md["max_step_fs"]:
                    nsteps_since_temp_reset = 0 
                    md_step_fs = min(self.config_md["max_step_fs"], md_step_fs * 2)
                    if v > 1:
                        print("energy back, new time step: ", md_step_fs)
                    dyn = Langevin(
                        atoms,
                        md_step_fs * aunits.fs,
                        temperature_K=temp_k,
                        friction=self.config_md["friction"],
                        rng=self.rng,
                    )

                ckp_pos = atoms.positions.copy()
                ckp_time = tot_time

            dyn.run(1)
            nsteps_since_temp_reset += 1 
            tot_time += md_step_fs
            pbar.update(md_step_fs)
            pbar.set_description(
                f"Etot {atoms.get_total_energy()/len(atoms):.3f} eV | T {atoms.get_temperature():03.0f} K | rst {nsteps_since_temp_reset} | fs"
            )
            if v > 1:
                self.printenergy(atoms, a0, tot_time)
            a0 += 1

        pbar.close()
        self.config_md["total_run_time"] = tot_time
        # atoms.set_cell(atoms.cell)
        # atoms.center()
        # atoms.wrap(eps=1e-10)

        # if no pbcs, should remove the atoms outside of the volume here
        if not self.config_md["pbc"]:
            dimx, dimy, dimz = atoms.cell.array.diagonal()
            goodsx = (0 <= atoms.positions[:, 0]) & (atoms.positions[:, 0] <= dimx)
            goodsy = (0 <= atoms.positions[:, 1]) & (atoms.positions[:, 1] <= dimy)
            goodsz = (0 <= atoms.positions[:, 2]) & (atoms.positions[:, 2] <= dimz)
            goods = goodsx & goodsy & goodsz
            del atoms[~goods]

        return atoms

    def _rotate_atoms(self, atoms: Atoms) -> Atoms:
        bbox = np.diag(atoms.cell)
        rad = self.config_randomize["rot_radius"]
        if self.config_randomize["N_iterations_rot"] <= 0:
            return atoms
        else:
            assert rad <= bbox.min() / 2, f"Sphere radius {rad} must be < bbox/2 = {bbox/2}"

        pbcs = atoms.pbc

        for a0 in tqdm(
            range(self.config_randomize["N_iterations_rot"]),
            desc="applying rotations",
            disable=self.verbose < 1,
        ):
            _pick_point_tries = 0
            cpoints = []
            while (
                len(cpoints) < self.config_randomize["N_points_per_iter"]
                and _pick_point_tries < self.config_randomize["pick_point_max"]
            ):
                _pick_point_tries += 1
                npoint = []
                for a1 in range(3):
                    if pbcs[a1]:
                        _p = self.rng.random() * bbox[a1]
                    else:
                        _p = self.rng.random() * (bbox[a1] - 2 * rad) + rad
                    npoint.append(_p)

                dists = self._get_dists_pbcs(cpoints, npoint, bbox, pbcs=pbcs)
                if np.all(dists >= rad * 2):
                    cpoints.append(npoint)
                    _pick_point_tries = 0
            if _pick_point_tries >= self.config_randomize["pick_point_max"] and self.verbose > 1:
                print(
                    f"_pick_point_tries failed after {len(cpoints)} / {self.config_randomize["N_points_per_iter"]} points found"
                )
            cpoints = np.array(cpoints)

            positions = atoms.positions
            positions = np.mod(positions, bbox)
            tree = KDTree(positions, boxsize=bbox)
            spheres = tree.query_ball_point(cpoints, rad)

            for sphere_inds, cpoint in zip(spheres, cpoints):

                theta = (self.rng.random() * 2 - 1) * self.config_randomize["theta_max"]
                vec = self.rng.random(3)
                vec /= np.linalg.norm(vec)
                rotmat = R.from_rotvec(vec * theta, degrees=True).as_matrix()
                rotmat

                shifted_pos = atoms.positions[sphere_inds] - cpoint
                shifted_pos = np.mod(shifted_pos + bbox / 2, bbox) - bbox / 2
                rot_pos = shifted_pos @ rotmat
                rot_pos += cpoint
                rot_pos = np.mod(rot_pos, bbox)
                atoms.positions[sphere_inds] = rot_pos

        return atoms

    def _shift_atoms(self, atoms: Atoms) -> Atoms:
        if "shift_sigma" not in self.config_randomize.keys():
            return atoms
        if (
            self.config_randomize["shift_sigma"] == 0
            or self.config_randomize["N_iterations_shift"] == 0
        ):
            return atoms

        bbox = np.diag(atoms.cell)
        N_points = np.ceil(bbox / self.config_randomize["shift_point_spacing"]).astype("int")
        N_points = np.maximum(N_points, 4)  # interpolator requires min 4 points each direction
        if self.verbose > 1:
            print("N_points = ", N_points)

        for a0 in tqdm(
            range(self.config_randomize["N_iterations_shift"]),
            desc="applying shifts",
            disable=self.verbose < 1,
        ):

            w_positions = atoms.positions
            w_positions = np.mod(w_positions, bbox)

            xgrid = np.linspace(0, bbox[0], N_points[0])
            ygrid = np.linspace(0, bbox[1], N_points[1])
            zgrid = np.linspace(0, bbox[2], N_points[2])

            shifts = self.rng.normal(0, self.config_randomize["shift_sigma"], (3, *N_points))
            shifts[:, -1] = shifts[:, 0]  # pbcs
            shifts[:, :, -1] = shifts[:, :, 0]  # pbcs
            shifts[:, :, :, -1] = shifts[:, :, :, 0]  # pbcs
            xinterp = RegularGridInterpolator((xgrid, ygrid, zgrid), shifts[0], method="cubic")
            yinterp = RegularGridInterpolator((xgrid, ygrid, zgrid), shifts[1], method="cubic")
            zinterp = RegularGridInterpolator((xgrid, ygrid, zgrid), shifts[2], method="cubic")

            xshifts = xinterp(w_positions)
            yshifts = yinterp(w_positions)
            zshifts = zinterp(w_positions)
            shifts = np.stack([xshifts, yshifts, zshifts]).T

            atoms.positions += shifts

            atoms.wrap(eps=1e-10)

        return atoms

    def _remove_close_atoms(self, starting_atoms: Atoms) -> Atoms:
        atoms = starting_atoms.copy()
        positions = starting_atoms.positions
        tree = KDTree(positions, boxsize=np.diag(atoms.cell.array))
        to_remove = set()

        if self.config_randomize["push_threshold"] == 0:
            print("Push threshold is 0 -- continuing")
            return atoms

        # Find all neighbors within the threshold distance
        for i, neighbors in enumerate(
            tree.query_ball_tree(tree, self.config_randomize["push_threshold"])
        ):
            if i in to_remove:
                continue
            for j in neighbors:
                if i != j:
                    to_remove.add(j)

        # Keep only unmarked positions
        mask = np.array([i in to_remove for i in range(len(positions))])
        rm = np.sum(mask)
        tot = len(positions)
        if rm/tot > 0.1: 
            print(f"\nWarning -- Deleting {rm} / {tot} atoms = {rm/tot*100:.2f}% | Atoms {atoms.symbols[:15]} | Threshold: {self.config_randomize["push_threshold"]:.2f}\n")
        elif self.verbose:
            print(f"Deleting {rm} / {tot} atoms = {rm/tot*100:.2f}%")
        del atoms[mask]
        return atoms

    def _push_close_atoms(self, starting_atoms: Atoms) -> Atoms:
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

        pbar = tqdm(
            range(self.config_randomize["push_iter_max"]),
            desc=f"pushing atoms (max iters = {self.config_randomize["push_iter_max"]})",
            leave=True,
            disable=self.verbose < 1,
            total=np.inf,
        )
        push_factor = 1
        for a0 in pbar:
            tree = KDTree(positions, boxsize=np.diag(cell) if np.any(pbc) else None)
            moved = False  # Track if any atoms are adjusted

            for i, neighbors in enumerate(
                tree.query_ball_tree(tree, self.config_randomize["push_threshold"])
            ):
                for j in neighbors:
                    if i >= j:
                        continue

                    # Compute distance vector with periodic boundary handling
                    displacement = positions[j] - positions[i]
                    if np.any(pbc):  # Apply minimum image convention
                        displacement -= np.round(displacement / cell.diagonal()) * cell.diagonal()

                    distance = np.linalg.norm(displacement)
                    if distance < self.config_randomize["push_threshold"]:
                        moved = True
                        # Normalize and scale displacement to push atoms apart
                        push_distance = (
                            self.config_randomize["push_threshold"] - distance
                        ) * push_factor
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
            self._remove_close_atoms(atoms)

        return atoms

    def _jitter_atoms(self, atoms: Atoms) -> Atoms:
        sigma = self.config_randomize["jitter_gaussian_sigma"]
        if sigma > 0:
            atoms.positions += self.rng.normal(0, sigma, atoms.positions.shape)
            atoms.wrap(eps=1e-10)
        return atoms

    def _get_dists_pbcs(self, points, cpointslists, bbox, pbcs=[1, 1, 1]):
        points = np.array(points, ndmin=2)
        cpointslists = np.array(cpointslists, ndmin=2)
        if not np.any(points):
            return np.array([np.inf])

        if np.any(pbcs):
            assert np.all(np.min(cpointslists, axis=-2) >= 0)
            assert np.all(
                np.max(cpointslists, axis=-2) <= bbox
            ), f"bbox: {bbox}, pointslist max: {np.max(cpointslists, axis=(0,1))}"
        assert "float" in str(cpointslists.dtype), f"cpointslists type: {cpointslists.dtype}"
        abs = np.abs(cpointslists - points[:, None])
        for i, ind in enumerate(pbcs):
            if ind:
                abs[:, :, i] = np.minimum(abs[:, :, i], bbox[i] - abs[:, :, i])
        return np.sqrt(np.sum(abs**2, axis=-1))

    def printenergy(self, atoms: Atoms, iteration=None, time_fs=None) -> None:
        """Function to print the potential, kinetic and total energy"""
        epot = atoms.get_potential_energy() / len(atoms)
        ekin = atoms.get_kinetic_energy() / len(atoms)
        temp_k = ekin / (1.5 * aunits.kB)
        s = ""
        if iteration is not None:
            s += f"iter {iteration} | "
        if time_fs is not None:
            s += f"{time_fs:.1f} fs | "
        s += f"E/atom: Epot = {epot:.3f}eV Ekin = {ekin:.3f}eV (T = {temp_k:.2f}) Etot = {epot+ekin:.3f}eV"
        print(s)

    def remove_ejected_atoms(self, atoms: Atoms, threshold_distance=None, calc=None) -> Atoms:
        if threshold_distance is None:
            threshold_distance = atoms.cell.max() * np.sqrt(3) * 1.25
        center = atoms.get_center_of_mass()
        distances = np.linalg.norm(atoms.positions - center, axis=1)

        ejected_indices = np.where(distances > threshold_distance)[0]

        if len(ejected_indices) > 0:
            print(f"Removing {len(ejected_indices)} ejected atoms.")
            atoms = atoms[[i for i in range(len(atoms)) if i not in ejected_indices]]
            atoms.calc = calc

        return atoms

    @staticmethod
    def _chunk_volume(
        atoms: Atoms, max_chunk_size: tuple[float] = (20.0, 20.0, 20.0)
    ) -> list[Atoms]:
        """
        Splits a large atomic structure into smaller rectangular chunks.

        Parameters:
            atoms (ASE Atoms): The input atomic structure.
            chunk_size (tuple): (dimx, dimy, dimz) chunk sizes in Å.

        Returns:
            List of Atoms objects, each representing a chunk.
        """
        cell = atoms.get_cell()
        lx, ly, lz = cell.lengths()

        # Number of chunks along each dimension
        nx = int(np.ceil(lx / max_chunk_size[0]))
        ny = int(np.ceil(ly / max_chunk_size[1]))
        nz = int(np.ceil(lz / max_chunk_size[2]))

        chunks = []
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    # Calculate the bounds of the current chunk
                    x1 = ix * max_chunk_size[0]
                    x2 = min((ix + 1) * max_chunk_size[0], lx)
                    y1 = iy * max_chunk_size[1]
                    y2 = min((iy + 1) * max_chunk_size[1], ly)
                    z1 = iz * max_chunk_size[2]
                    z2 = min((iz + 1) * max_chunk_size[2], lz)

                    # Slice the atoms within the chunk bounds
                    mask = (
                        (atoms.positions[:, 0] >= x1)
                        & (atoms.positions[:, 0] < x2)
                        & (atoms.positions[:, 1] >= y1)
                        & (atoms.positions[:, 1] < y2)
                        & (atoms.positions[:, 2] >= z1)
                        & (atoms.positions[:, 2] < z2)
                    )
                    chunk = atoms[mask]

                    # Update the cell of the chunk
                    chunk.set_cell([x2 - x1, y2 - y1, z2 - z1])
                    chunk.center()
                    chunks.append(chunk)

        return chunks

    @staticmethod
    def recombine_chunks(
        chunks: list[Atoms], original_cell: tuple[float], max_chunk_size=(20, 20, 20)
    ):
        """
        Recombines processed chunks back into a single volume.

        Parameters:
            chunks (list of ASE Atoms): The list of chunked structures.
            chunk_size (tuple): The original (dimx, dimy, dimz) chunk sizes.

        Returns:
            ASE Atoms: The recombined atomic structure.
        """
        combined_positions = []
        combined_numbers = []
        current_pos = [0, 0, 0]
        
        # Reassemble the chunks into their original positions
        # for chunk in chunks:
        #     pos = chunk.get_positions()
        #     combined_positions.append(pos)
        #     combined_numbers.append(chunk.get_atomic_numbers())

        # # Create a new Atoms object with the recombined positions and atomic numbers
        # combined_positions = np.concatenate(combined_positions, axis=0)
        # combined_numbers = np.concatenate(combined_numbers, axis=0)
        # combined_atoms = Atoms(
        #     numbers=combined_numbers, positions=combined_positions, cell=original_cell
        # )

        # return combined_atoms
        for chunk in chunks:
            pos = chunk.get_positions()
            pos += current_pos
            combined_positions.append(pos)
            combined_numbers.append(chunk.get_atomic_numbers())
            
            # Update the current position
            lx, ly, lz = chunk.get_cell().lengths()
            current_pos[2] += lz
            if current_pos[2] >= original_cell[2][2]:
                current_pos[2] = 0
                current_pos[1] += ly
            if current_pos[1] >= original_cell[1][1]:
                current_pos[1] = 0
                current_pos[0] += lx

        # Create a new Atoms object with the recombined positions and atomic numbers
        combined_positions = np.concatenate(combined_positions, axis=0)
        combined_numbers = np.concatenate(combined_numbers, axis=0)
        combined_atoms = Atoms(numbers=combined_numbers, positions=combined_positions, cell=original_cell)
        
        return combined_atoms
    
    
    
class Suppressor:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

