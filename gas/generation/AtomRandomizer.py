import numpy as np 
from ase import Atoms 
from gas.generation.modify_volume import TransformVolume
from gas.io.xyz import load_xyz
from ase import io as aio
from gas.PDFs.RDFs import RDF
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import ndimage as ndi
import torch 


def extract_region_around_com(atoms:Atoms, box_size:tuple):
    """
    Extracts a rectangular region of specified size around the center of mass of an ASE Atoms object.
    """
    com = atoms.get_center_of_mass()
    x_size, y_size, z_size = box_size
    x_min, x_max = com[0] - x_size / 2, com[0] + x_size / 2
    y_min, y_max = com[1] - y_size / 2, com[1] + y_size / 2
    z_min, z_max = com[2] - z_size / 2, com[2] + z_size / 2

    selected_indices = [
        i for i, pos in enumerate(atoms.positions)
        if (x_min <= pos[0] <= x_max) and (y_min <= pos[1] <= y_max) and (z_min <= pos[2] <= z_max)
    ]
    sub_atoms = atoms[selected_indices].copy()
    
    new_cell = np.array([
        [x_size, 0, 0],  
        [0, y_size, 0],  
        [0, 0, z_size]   
    ])
    sub_atoms.positions -= [x_min, y_min, z_min]
    
    sub_atoms.set_cell(new_cell)
    sub_atoms.wrap(eps=1e-10)
    sub_atoms.center()
    # sub_atoms.set_pbc(False) 
    return sub_atoms

class AtomsRandomizer(object):

    def __init__(
        self,
        orig_files: list,
        config_randomize: dict,
        config_md: dict,
        config_rdf: dict,
        rng: np.random.Generator | None = None,
        device: str = "cpu",
        v: int = 1,
    ):
        if rng is None:
            rng = np.random.default_rng()
        self.orig_files = sorted(orig_files)
        self.config_randomize = config_randomize
        self.config_md = config_md
        self.rdf_gaussian = config_rdf.pop("rdf_gaussian", 0)
        self.rdf_config = config_rdf
        self.rng = rng
        self.device = device
        self.v = v 
        
        self.rdf = RDF(device=self.device, v=0)

        self._load_orig_files(show=v>=2)
        self.generate_new_params()
        self._num_chars_start = 6
        self.files = []
        
    def _load_cif_and_tile(self, f) -> Atoms: 
        unitcell = aio.read(f) 
        # unitcell = abtem.structures.orthogonalize_cell(unitcell)
        target_size = self.config_randomize["cell_size"]
        cell_lengths = unitcell.get_cell().lengths()
        repeats = [2*int(np.ceil(np.max(target_size) / cell_lengths[i])) for i in range(3)]
        bigatoms:Atoms = unitcell.repeat(repeats)
        v = self.rng.random(3)
        v /= np.linalg.norm(v)
        theta = (self.rng.random()-0.5) * 360
        bigatoms.rotate(theta, v, rotate_cell=True)
        atoms = extract_region_around_com(bigatoms, target_size)
        return atoms 

    def _load_orig_files(self, show=True):
        orig_grs = [] 
        gr_thresholds = [] 
        
        if show: 
            fig, axs = plt.subplots(ncols=len(self.orig_files), figsize=(5*len(self.orig_files), 5))
            if len(self.orig_files) == 1: 
                axs = [axs]

        for i, f in enumerate(self.orig_files): 
            if f.suffix == ".cif": 
                atoms = self._load_cif_and_tile(f)
            elif f.suffix == ".xyz": 
                atoms = load_xyz(f)
            
            r, gr = self.rdf.rdf(
                atoms,
                pbcs=[0,0,0], 
                **self.rdf_config, 
            )
            if self.device != "cpu": 
                r = r.get()
                gr = gr.get()
            
            orig_grs.append(gr)
            gr_thresholds.append(self.config_randomize['threshold_frac'] * r[np.min(np.where(gr>1))])
            # gr_thresholds.append(self.config_randomize['threshold_frac'] * r[np.argmax(gr[r<3])])
            if show: 
                axs[i].plot(r, ndi.gaussian_filter(gr, self.rdf_gaussian))
                axs[i].set_title(f.stem)

        self._orig_grs = np.array(orig_grs) 
        self._gr_thresholds = np.array(gr_thresholds) 
        self._gr_r = r 
        
        if show: 
            plt.show() 
        
        return
    
    def generate(self, num:int, outdir:str|None=None, return_atoms:bool = False, v:int=1, show=False):
        if outdir is not None: 
            assert outdir.exists()
            fs = sorted(list(outdir.glob("*.xyz")))
            snum = len(fs) 
        else:
            snum = 0 
        
        new_atoms = [] 
        
        for a0 in tqdm(range(snum, snum+num), disable=v<1): 
            torch.cuda.empty_cache() 
            self.generate_new_params()
            atoms_index = a0%len(self.orig_files)
            oname = self.orig_files[atoms_index].stem
            oname = "-".join(oname.split("_"))
            
            _running_mod = True 
            orig_chunk_size = self.config_md["chunk_size"]
            current_chunk_size = list(orig_chunk_size)
            while _running_mod: 
                try: 
                    mod_atoms = self._modify_volume(atoms_index, v=v-1)
                    _running_mod = False 
                except RuntimeError as e: 
                    print(f"\nAvoiding a Runtime Error for {oname} - chunkz = {current_chunk_size[2]}")
                    if current_chunk_size[2] > 4.01: 
                        current_chunk_size[2] = max(4, current_chunk_size[2] * 3/4)
                    else: 
                        current_chunk_size[0] = current_chunk_size[0]/2
                        current_chunk_size[1] = current_chunk_size[1]/2
                    
                    self.config_md["chunk_size"] = (current_chunk_size[0], current_chunk_size[1], current_chunk_size[2]/2)
                    print(f"Decreasing chunk size temporarily from {orig_chunk_size} to {self.config_md["chunk_size"]}\n")
                    
            self.config_md["chunk_size"] = orig_chunk_size
            
            if return_atoms: 
                new_atoms.append(mod_atoms)
            
            if outdir is not None: 
                n0 = len(list(outdir.glob("*.xyz")))
                fname = outdir / f"{n0:0{self._num_chars_start}}_mod_{oname}.xyz" 
                # print(fname)
                aio.write(fname, mod_atoms)
                self.files.append(fname) 
            
            if show: 
                r, gr = self.rdf.rdf(
                    mod_atoms,
                    pbcs=[0,0,0], # config_md pbc if run_md else config_randomize pbc 
                    **self.rdf_config, 
                )            
                gr = ndi.gaussian_filter(gr.get(), self.rdf_gaussian)
                fig, ax = plt.subplots(figsize=(4,2))
                ax.plot(r.get(), gr)
                ax.set_title(oname)
                plt.show()
                
        if return_atoms:
            return new_atoms
        else: 
            return 

    def _modify_volume(self, atoms_index:int, v=None):
        if v is None: 
            v = self.v 
        
        f = self.orig_files[atoms_index] 
        assert f.suffix == ".cif"
        atoms = self._load_cif_and_tile(f) # reload so new random orientation
        
        config = {
            "N_iterations_rot": self.c_N_iterations_rot, 
            "rot_radius": self.c_radius, 
            "jitter_gaussian_sigma": self.c_gaussian_sigma, 
            "N_iterations_shift": self.c_N_iterations_shift, 
            "shift_point_spacing": self.c_shift_point_spacing,
            "shift_sigma": self.c_shift_sigma,  
            "push_threshold": self.gr_thresholds[atoms_index], 
            
            "pbc": True, 
            "N_points_per_iter": self.c_N_points_per_iter, 
            "theta_max": self.c_theta_max, 
        }
        
        modifier = TransformVolume(
            config_randomize=config, 
            config_md=self.config_md, 
            rng=self.rng,
            v=v,
        )      
        mod_atoms = modifier.apply(atoms)
        mod_atoms = modifier._push_close_atoms(mod_atoms)
          
        return mod_atoms 


    def show_max_min_ranges(self, atoms_index:int, return_atoms=False, rng_seed=42):
        # pick the min and max values from each range and calculate rdf and display for (each?) orig file
        atoms = self._load_cif_and_tile(self.orig_files[atoms_index]) 
        
        config_min = {
            "N_iterations_rot": np.min(self.config_randomize["N_iterations_rot"]), 
            "N_iterations_shift": np.min(self.config_randomize["N_iterations_shift"]), 
            "N_points_per_iter": np.min(self.config_randomize["N_points_per_iter"]), 
            "rot_radius": np.min(self.config_randomize["rot_radius"]), 
            "theta_max": np.min(self.config_randomize["theta_max"]), 
            "region_type": "sphere", 
            "push_threshold": self.gr_thresholds[atoms_index], 
            "jitter_gaussian_sigma": np.min(self.config_randomize["jitter_gaussian_sigma"]), 
            "shift_point_spacing": self.config_randomize["shift_point_spacing"][0], # min order is larger number
            "shift_sigma": np.min(self.config_randomize["shift_sigma"]),  
        }

        rng = np.random.default_rng(rng_seed)
        modifier_min = TransformVolume(
            config_randomize=config_min, 
            config_md=self.config_md, 
            rng=rng,
            v=self.v - 1,
        )      
        mod_atoms_min = modifier_min.apply(atoms)
        mod_atoms_min_premd = modifier_min._pre_md_atoms.copy()
        mod_atoms_min = modifier_min._push_close_atoms(mod_atoms_min)
                
        config_max = {
            "N_iterations_rot": np.max(self.config_randomize["N_iterations_rot"]), 
            "N_iterations_shift": np.max(self.config_randomize["N_iterations_shift"]), 
            "N_points_per_iter": np.max(self.config_randomize["N_points_per_iter"]), 
            "rot_radius": np.max(self.config_randomize["rot_radius"]), 
            "theta_max": np.max(self.config_randomize["theta_max"]), 
            "region_type": "sphere", 
            "push_threshold": self.gr_thresholds[atoms_index], 
            "jitter_gaussian_sigma": np.max(self.config_randomize["jitter_gaussian_sigma"]), 
            "shift_point_spacing": self.config_randomize["shift_point_spacing"][1],
            "shift_sigma": np.max(self.config_randomize["shift_sigma"]),  
        }
        

        rng = np.random.default_rng(rng_seed)
        modifier_max = TransformVolume(
            config_randomize=config_max, 
            config_md=self.config_md, 
            rng=rng,
            v=self.v-1,
        )      
        mod_atoms_max = modifier_max.apply(atoms)
        mod_atoms_max_premd = modifier_max._pre_md_atoms.copy()
        mod_atoms_max = modifier_max._push_close_atoms(mod_atoms_max)

        r, gr_min = self.rdf.rdf(
            mod_atoms_min,
            pbcs=[0,0,0], 
            **self.rdf_config, 
        )
        r, gr_min_premd = self.rdf.rdf(
            mod_atoms_min_premd,
            pbcs=[0,0,0], 
            **self.rdf_config, 
        )
        r, gr_max = self.rdf.rdf(
            mod_atoms_max,
            pbcs=[0,0,0], 
            **self.rdf_config, 
        )        
        r, gr_max_premd = self.rdf.rdf(
            mod_atoms_max_premd,
            pbcs=[0,0,0], 
            **self.rdf_config, 
        )
        if self.device != "cpu": 
            r = r.get()
            gr_min = gr_min.get()
            gr_min_premd = gr_min_premd.get()
            gr_max = gr_max.get()
            gr_max_premd = gr_max_premd.get()
        
        gr_min = ndi.gaussian_filter(gr_min, self.rdf_gaussian)
        gr_min_premd = ndi.gaussian_filter(gr_min_premd, self.rdf_gaussian)
        gr_max = ndi.gaussian_filter(gr_max, self.rdf_gaussian)
        gr_max_premd = ndi.gaussian_filter(gr_max_premd, self.rdf_gaussian)


        fig, ax = plt.subplots()
        ax.plot(r, self.orig_grs[atoms_index], label="orig")
        if self.config_md["run_md"]:
            ax.plot(r, gr_min_premd, label="min values premd", lw=2)
            ax.plot(r, gr_max_premd, label="max values premd", lw=2)
        ax.plot(r, gr_min, label="min values", lw=2)
        ax.plot(r, gr_max, label='max values', lw=2)
        ax.legend()
        ax.set_xlabel("r (A)")
        ax.set_ylabel("gr")
        ax.set_title(self.orig_files[atoms_index].stem)
        ax.set_ylim([-0.05, gr_min.max()+1])
        ax.hlines([1], 1,r.max(), colors='k')
        plt.show()
            
        if return_atoms: 
            return mod_atoms_min, mod_atoms_max
        else: 
            return
    
    
    
    def _show_max_ranges(self, atoms_index:int, return_atoms=False, rng_seed=42):
        # pick the min and max values from each range and calculate rdf and display for (each?) orig file
        atoms = self._load_cif_and_tile(self.orig_files[atoms_index]) 
        config_max = {
            "N_iterations_rot": np.max(self.config_randomize["N_iterations_rot"]), 
            "N_iterations_shift": np.max(self.config_randomize["N_iterations_shift"]), 
            "N_points_per_iter": np.max(self.config_randomize["N_points_per_iter"]), 
            "rot_radius": np.max(self.config_randomize["rot_radius"]), 
            "theta_max": np.max(self.config_randomize["theta_max"]), 
            "region_type": "sphere", 
            "push_threshold": self.gr_thresholds[atoms_index], 
            "jitter_gaussian_sigma": np.max(self.config_randomize["jitter_gaussian_sigma"]), 
            "shift_point_spacing": self.config_randomize["shift_point_spacing"][1],
            "shift_sigma": np.max(self.config_randomize["shift_sigma"]),  
        }

        rng = np.random.default_rng(rng_seed)
        modifier_max = TransformVolume(
            config_randomize=config_max, 
            config_md=self.config_md, 
            rng=rng,
            v=self.v-1,
        )      
        mod_atoms_max = modifier_max.apply(atoms)
        mod_atoms_max_premd = modifier_max._pre_md_atoms.copy()
        mod_atoms_max = modifier_max._push_close_atoms(mod_atoms_max)

        r, gr_max = self.rdf.rdf(
            mod_atoms_max,
            pbcs=[0,0,0], 
            **self.rdf_config, 
        )        
        r, gr_max_premd = self.rdf.rdf(
            mod_atoms_max_premd,
            pbcs=[0,0,0], 
            **self.rdf_config, 
        )
        if self.device != "cpu": 
            r = r.get()
            gr_max = gr_max.get()
            gr_max_premd = gr_max_premd.get()
        
        gr_max = ndi.gaussian_filter(gr_max, self.rdf_gaussian)
        gr_max_premd = ndi.gaussian_filter(gr_max_premd, self.rdf_gaussian)
        
        fig, ax = plt.subplots()
        ax.plot(r, self.orig_grs[atoms_index], label="orig")
        if self.config_md["run_md"]:
            ax.plot(r, gr_max_premd, label="max values premd", lw=2)
        ax.plot(r, gr_max, label='max values', lw=2)
        ax.legend()
        ax.set_xlabel("r (A)")
        ax.set_ylabel("gr")
        ax.set_title(self.orig_files[atoms_index].stem)
        ax.set_ylim([-0.05, gr_max.max()+1])
        ax.hlines([1], 1,r.max(), colors='k')
        plt.show()
            
        if return_atoms: 
            return mod_atoms_max
        else: 
            return
    
        
    
    def generate_new_params(self): 
        iter_range = self.config_randomize["N_iterations_rot"]
        self._c_N_iterations_rot = self.rng.integers(iter_range[0], iter_range[1], endpoint=True) 

        iter_range = self.config_randomize["N_iterations_shift"]
        self._c_N_iterations_shift = self.rng.integers(iter_range[0], iter_range[1], endpoint=True) 
    
        ppi_range = self.config_randomize["N_points_per_iter"]
        self._c_N_points_per_iter = self.rng.integers(ppi_range[0], ppi_range[1], endpoint=True) 
        
        radius_range = self.config_randomize["rot_radius"]
        self._c_radius = self.rng.random() * np.ptp(radius_range) + np.min(radius_range)

        theta_max_range = self.config_randomize["theta_max"]
        self._c_theta_max = self.rng.random() * np.ptp(theta_max_range) + np.min(theta_max_range)
        
        gaussian_sigma_range = self.config_randomize["jitter_gaussian_sigma"]
        self._c_gaussian_sigma = self.rng.random() * np.ptp(gaussian_sigma_range) + np.min(gaussian_sigma_range)
                
        shift_points_range = self.config_randomize["shift_point_spacing"]
        self._c_shift_point_spacing = self.rng.random() * np.ptp(shift_points_range) + np.min(shift_points_range)
                 
        shift_sigma_range = self.config_randomize["shift_sigma"]
        self._c_shift_sigma = self.rng.random() * np.ptp(shift_sigma_range) + np.min(shift_sigma_range)

        return 
    
    @property
    def c_N_iterations_rot(self) -> float: 
        return self._c_N_iterations_rot

    @property
    def c_N_iterations_shift(self) -> float: 
        return self._c_N_iterations_shift
    
    @property
    def c_N_points_per_iter(self) -> float: 
        return self._c_N_points_per_iter

    
    @property
    def c_radius(self) -> float: 
        return self._c_radius
    
    @property
    def c_theta_max(self) -> float:
        return self._c_theta_max 
    
    
    @property
    def c_gaussian_sigma(self) -> float: 
        return self._c_gaussian_sigma
    
    
    @property
    def c_shift_point_spacing(self) -> float: 
        return self._c_shift_point_spacing
        
    
    @property
    def c_shift_sigma(self) -> float: 
        return self._c_shift_sigma
        
    @property
    def orig_grs(self) -> np.ndarray: 
        return self._orig_grs 
    
    @property
    def gr_thresholds(self) -> list[float]: 
        
        return self._gr_thresholds 
    
    @property 
    def gr_r(self) -> np.ndarray: 
        return self._gr_r