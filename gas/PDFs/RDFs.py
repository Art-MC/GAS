import os
import time
from datetime import timedelta
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree

try:
    import cupy as cp
    from .PDFs_utils.kernels import kernels
except (ImportError, ModuleNotFoundError) as e:
    print("Cupy is required")
    raise e
    cp = np


class RDF(object):

    def __init__(self, device="cpu", v=1):
        if isinstance(device, int):
            self._xp = cp
            cp.cuda.runtime.setDevice(device)
        elif device.lower() == "cpu":
            self._xp = np
        self.device = device
        self._v = v
        return

    def rdf(
        self,
        atoms,
        dr=0.05,
        hist_r_max=20,
        skip=1,
        dr_ind=1,
        batch_size=25_000,
        v=None,
        pbcs=None,
        reduced=True,
    ):
        xp = self._xp
        stime = time.perf_counter()
        v = self._v if v is None else v
        vprint = print if v >= 1 else lambda *a, **k: None
        if skip == None:
            skip = 1
        if pbcs is None:
            pbcs = atoms.get_pbc()
        self.pbcs = pbcs

        bbox_np = np.array(atoms.cell[(0, 1, 2), (0, 1, 2)])
        pbcs_np = np.copy(pbcs)
        pbcs = xp.array(pbcs).astype("bool")
        positions = xp.array(atoms.positions, dtype=cp.float64)
        positions_np = np.array(atoms.positions)  # for KD tree
        assert (
            positions.shape[-1] == 3
        ), "should be generalizable but havent checked with non 3D vectors"
        Dim = 3
        assert hist_r_max < bbox_np.min() / 2, f" (should check first for pbcs) cell {atoms.cell}, hist_r_max {hist_r_max}"


        vprint(f"Cell size (A): {bbox_np}")
        # vprint(f"max dist possible between two points with full pbcs: {np.sqrt(3*(bbox_np.max()/2)**2):.2f} A")
        vprint(f"Using a max radius of {hist_r_max} A")

        ### get center positions that aren't within r_max of a boundary without pbcs
        goods = np.ones(len(positions_np)).astype("bool")
        for i, p in enumerate(pbcs_np):
            if p:
                continue
            goods *= (positions_np[:, i] > hist_r_max) & (
                positions_np[:, i] < (bbox_np[i] - hist_r_max)
            )
        center_inds = np.where(goods)[0]
        center_inds_skip = center_inds[::skip]
        num_centers = len(center_inds_skip)

        ### Histogram coordinates
        rr = xp.arange(0.0, hist_r_max + dr, dr).astype("float")
        numbins = len(rr)
        histo_height = 4096*4 # int(tot_num_neighbors)
        hist_sig = xp.zeros((histo_height, numbins), dtype=xp.float64)

        ### Density
        cell = xp.array(atoms.cell)
        volume = xp.abs(xp.sum(cell[:, 0] * xp.cross(cell[:, 1], cell[:, 2])))
        dens = positions.shape[0] / volume
        r_vol = 4 * np.pi * hist_r_max**3
        ave_neighbors = dens * r_vol

        ### setup volume
        vol_size = np.round(bbox_np / dr_ind).astype('int')
        scaled_dr_ind = bbox_np / vol_size
        x_inds = np.mod(np.round(atoms.positions[:,0] / scaled_dr_ind[0]).astype('int'), vol_size[0])
        y_inds = np.mod(np.round(atoms.positions[:,1] / scaled_dr_ind[1]).astype('int'), vol_size[1])
        z_inds = np.mod(np.round(atoms.positions[:,2] / scaled_dr_ind[2]).astype('int'), vol_size[2])

        vol = -1*np.ones(vol_size,dtype='int')
        vol[x_inds,y_inds,z_inds] = np.arange(atoms.positions.shape[0])
        assert(len(np.unique(vol)) == len(atoms.positions)+1), f"Try reducing dr_ind"

        vprint(f"Num total atoms in sim = {len(positions)}")
        if not np.all(pbcs):
            vprint(f"Num non-edge atoms = {len(center_inds)}")
        if skip != 1:
            vprint(
                f"Skip = {skip}, so calculating using {num_centers} atoms as centers"
            )
        vprint(f'atomic density = {dens:.3} atoms / A^3')

        ### array setup
        if batch_size <= 0:
            batch_size = num_centers
        else:
            batch_size = min(batch_size, num_centers)
        positions_round_cp = cp.array(np.round(positions_np / scaled_dr_ind), dtype=cp.int32)
        volume_inds_cp = cp.array(vol, dtype=cp.int32)
        volume_shape_cp = cp.array(volume_inds_cp.shape, dtype=cp.int32)
        if len(positions) > 1000:
            N_max_neighbors = int(min(np.round(ave_neighbors)*1.2, len(positions)))  # calculate from density
        else:
            N_max_neighbors = len(positions)
        neighbors_inds_cp = -1*cp.ones((batch_size, N_max_neighbors), dtype=cp.int64)
        neighbors_num_cp = cp.zeros(batch_size, dtype=cp.int64)
        Dim = 3
        bbox_cp = cp.array(bbox_np, dtype=cp.float32)
        pbcs_cp = xp.array(pbcs_np).astype("bool")
        search_rad = int(np.ceil(hist_r_max / dr_ind))

        positions_cp = cp.array(positions, dtype=cp.float64)
        neighbor_poslist_shape = cp.array(cp.shape(neighbors_inds_cp), dtype=cp.int64)

        ### kernels
        # Currently have two seperate kernels, can/should be combined into one
        # will happen with weighted iteration
        kernel_NN = kernels["find_neighbors"]
        kernel_RDF = kernels["rdf"]
        # surprisingly faster than using all threads by ~10%
        threads_NN = (kernel_NN.max_threads_per_block//2,)
        blocks_NN = (batch_size // threads_NN[0] + 1,)
        threads_RDF = (kernel_RDF.max_threads_per_block,)
        blocks_RDF = (cp.size(neighbors_inds_cp) // threads_RDF[0] + 1,)

        for a0 in range(0, num_centers, batch_size):
            b0 = min(a0 + batch_size, num_centers)
            batch_center_inds = cp.array(center_inds_skip[a0:b0], dtype=cp.int64)
            cbatch_size = b0 - a0

            ### get nearest neighbors for this batch
            kernel_NN(
                blocks_NN,
                threads_NN,
                (
                    positions_round_cp,
                    volume_inds_cp,
                    volume_shape_cp,
                    search_rad,
                    batch_center_inds,
                    neighbors_inds_cp,
                    neighbors_num_cp,
                    int(N_max_neighbors),
                    int(cbatch_size),
                    int(Dim),
                    pbcs_cp,
                ),
            )

            ### calculate RDF contribution for batch
            kernel_RDF(
                blocks_RDF,
                threads_RDF,
                (
                    positions_cp,
                    batch_center_inds,
                    neighbors_inds_cp,
                    neighbor_poslist_shape,
                    cbatch_size,
                    int(Dim),
                    hist_sig,
                    int(histo_height),
                    int(numbins),
                    float(dr),
                    bbox_cp,
                    pbcs_cp,
                ),
            )

        cp.cuda.Stream.null.synchronize()

        gr = hist_sig.sum(axis=0)
        if reduced:
            gr /= num_centers
            gr[0] = 0
            gr[1:] /= (dr * dens * 4 * np.pi) * rr[1:] ** 2

        vprint("-- done --")
        ttime = time.perf_counter() - stime
        vprint(
            f"Total time (h:m:s) {str(timedelta(seconds=round(ttime,3))).rstrip('0')}"
        )
        vprint(f"Center atoms per sec: {len(center_inds_skip) / ttime:_.2f}\n")
        return rr, gr

    def _rdf_old(
        self,
        atoms,
        dr=0.1,
        hist_r_max=20,
        skip=1,
        batch_size=1,
        v=None,
        pbcs=None,
        reduced=True,
    ):
        xp = self._xp
        stime = time.perf_counter()
        v = self._v if v is None else v
        vprint = print if v >= 1 else lambda *a, **k: None
        if skip == None:
            skip = 1
        if pbcs is None:
            pbcs = atoms.get_pbc()
        self.pbcs = pbcs

        bbox_np = np.array(atoms.cell[(0, 1, 2), (0, 1, 2)])
        bbox = xp.array(bbox_np, dtype=cp.float64)
        pbcs_np = np.copy(pbcs)
        pbcs = xp.array(pbcs).astype("bool")
        positions = xp.array(atoms.positions)
        positions_np = np.array(atoms.positions)  # for KD tree

        vprint(f"Cell size (A): {bbox_np}")
        # vprint(f"max dist possible between two points with full pbcs: {np.sqrt(3*(bbox_np.max()/2)**2):.2f} A")
        vprint(f"Using a max radius of {hist_r_max} A")

        ### get center positions that aren't within r_max of a boundary without pbcs
        goods = np.ones(len(positions_np)).astype("bool")
        for i, p in enumerate(pbcs_np):
            if p:
                continue
            goods *= (positions_np[:, i] > hist_r_max) & (
                positions_np[:, i] < (bbox_np[i] - hist_r_max)
            )
        center_inds = np.where(goods)[0]

        ### KD tree for getting neighbors, cpu cuz faster
        start_KD = time.perf_counter()
        bbox_np_pbcs = bbox_np * pbcs_np
        if np.any(
            np.round(positions_np.max(axis=0), 9) >= np.round(bbox_np, 9)
        ):
            # cant have atom right on cell edge
            for i in range(len(pbcs)):
                if bbox_np_pbcs[i] != 0:
                    bbox_np_pbcs[i] += 1e-9
                    bbox[i] += 1e-9
        tree_all = KDTree(positions_np, copy_data=True, boxsize=bbox_np_pbcs)
        center_inds_skip = center_inds[::skip]
        tree_centers = KDTree(
            positions_np[center_inds_skip], copy_data=True, boxsize=bbox_np_pbcs
        )
        inds_all = tree_centers.query_ball_tree(tree_all, hist_r_max)
        inds_all = [xp.array(i, dtype=xp.float32) for i in inds_all]
        end_KD = time.perf_counter()
        vprint(f"KDTree calc time: {end_KD - start_KD:.1e} s")

        ### Histogram coordinates
        rr = xp.arange(0.0, hist_r_max + dr, dr).astype("float")
        hist_sig = xp.zeros_like(rr)

        ### Density
        cell = xp.array(atoms.cell)
        volume = xp.abs(xp.sum(cell[:, 0] * xp.cross(cell[:, 1], cell[:, 2])))
        dens = positions.shape[0] / volume

        ### making big arrays that will be filled
        natoms_dens = (4 / 3 * xp.pi * hist_r_max**3) * dens
        maxval_n = int(1.25 * natoms_dens**2)
        vals = xp.zeros(maxval_n * batch_size, dtype=cp.int32)
        maxinds = max([len(i) for i in inds_all])
        inds2 = xp.zeros((batch_size, maxinds), dtype=cp.int32)
        drs = xp.zeros((batch_size, maxinds), dtype=cp.float64)

        vprint(f"Num total atoms in sim = {len(positions)}")
        num_centers = len(center_inds_skip)
        if not np.all(pbcs):
            vprint(f"Num non-edge atoms = {len(center_inds)}")
        if skip != 1:
            vprint(
                f"Skip = {skip}, so calculating using {num_centers} atoms as centers"
            )
        # vprint(f'atomic density = {dens:.3} atoms / A^3')
        num_bins = hist_sig.shape[0]
        vprint("Final shape will be: ", hist_sig.shape)

        for a0 in range(0, num_centers, batch_size):
            b0 = min(a0 + batch_size, num_centers)
            batch_center_inds = center_inds_skip[a0:b0]
            bsize = len(batch_center_inds)
            bnum_inds = xp.array(
                [len(inds) for inds in inds_all[a0:b0]], dtype=cp.int32
            )
            for i in range(bsize):
                inds2[i, : bnum_inds[i]] = inds_all[a0 + i]

            drs = self._get_dists_pbcs(
                positions[batch_center_inds],
                positions[inds2[:bsize]],
                bbox=bbox,
                pbcs=pbcs,
            )

            for i in range(bsize):
                drs[i, bnum_inds[i] :] = 0

            drs2 = drs[:bsize].ravel()

            r_inds = drs2 / dr
            r_floor = xp.floor(r_inds).astype("int")
            r_weights = r_inds - r_floor

            good_vals = (r_floor > 0) & (r_floor < num_bins)
            nvals = good_vals.sum()
            weights = r_weights[good_vals]

            hist_sig += xp.bincount(
                r_floor[good_vals],
                weights=(1 - weights),
                minlength=num_bins,
            )

            good_vals = (r_floor > 0) & (r_floor + 1 < num_bins)
            nvals = good_vals.sum()
            vals[:nvals] = r_floor[good_vals]
            weights = r_weights[good_vals]

            hist_sig += xp.bincount(
                r_floor[good_vals] + 1,
                weights=weights,
                minlength=num_bins,
            )

        gr = xp.copy(hist_sig[:-1])
        rr = rr[:-1]

        if reduced:
            gr /= num_centers
            gr[0] = 0
            gr[1:] /= (dr * dens * 4 * np.pi) * rr[1:] ** 2

        vprint("-- done --")
        ttime = time.perf_counter() - stime
        vprint(
            f"Total time (h:m:s) {str(timedelta(seconds=round(ttime,3))).rstrip('0')}"
        )
        vprint(f"Center atoms per sec: {num_centers / ttime:.1f}\n")
        return rr, gr

    def _get_dists_pbcs(self, points, pointslists, bbox, pbcs=[1, 1, 1]):
        xp = self._xp
        if np.any(pbcs):
            assert xp.all(xp.min(pointslists, axis=1) >= 0)
            assert xp.all(
                xp.max(pointslists, axis=1) <= bbox
            ), f"bbox: {bbox}, pointslist max: {xp.max(pointslists, axis=(0,1))}"
        assert "float" in str(
            pointslists.dtype
        ), f"pointslists type: {pointslists.dtype}"
        abs = xp.abs(pointslists - points[:, None])
        for i, ind in enumerate(pbcs):
            if ind:
                abs[:, :, i] = xp.minimum(abs[:, :, i], bbox[i] - abs[:, :, i])
        return np.sqrt(np.sum(abs**2, axis=-1))


def quick_rdf(
    atoms,
    dr_ind=1.5,
    skip=1,
    hist_r_max=10.0,
    hist_dr=0.05,
    renormalize_reduced_pdf=True,
    plot_result=True,
    use_pbcs=[1, 1, 0],
):
    """
    Calculate the RDF in a two-stage process:
    1 - write indices into a volume to find neighbors within some range.
    2 - Compute the distribution
    """

    # Construct index volume
    bbox = atoms.cell[(0, 1, 2), (0, 1, 2)]
    vol_size = np.round(bbox / dr_ind).astype("int")
    dr_ind = bbox / vol_size

    # Histogram coordinates
    hist_r = np.arange(0.0, hist_r_max, hist_dr)
    hist_sig = np.zeros_like(hist_r)
    num_bins = hist_r.shape[0]

    ### KD tree for getting neighbors, cpu cuz faster
    bbox_np_pbcs = bbox * np.array(use_pbcs)
    if np.any(
        np.round(atoms.positions.max(axis=0), 9) >= np.round(bbox, 9)
    ):  # cant have atom right on cell edge
        for i in range(len(use_pbcs)):
            if bbox_np_pbcs[i] != 0:
                bbox_np_pbcs += 1e-9
    tree_all = KDTree(atoms.positions, copy_data=True, boxsize=bbox_np_pbcs)
    center_inds_skip = np.arange(len(atoms.positions))[::skip]
    tree_centers = KDTree(
        atoms.positions[center_inds_skip], copy_data=True, boxsize=bbox_np_pbcs
    )
    inds_all = tree_centers.query_ball_tree(tree_all, hist_r_max)
    inds_all = [np.array(i) for i in inds_all]

    # Density
    volume = np.abs(
        np.sum(atoms.cell[:, 0] * np.cross(atoms.cell[:, 1], atoms.cell[:, 2]))
    )
    dens = atoms.positions.shape[0] / volume
    print("atomic density = " + str(np.round(dens, 5)))

    ### making big arrays that will be filled
    batch_size = 1
    natoms_dens = (4 / 3 * np.pi * hist_r_max**3) * dens
    maxval_n = int(1.25 * natoms_dens**2)
    vals = np.zeros(maxval_n * batch_size).astype("int")
    maxinds = max([len(i) for i in inds_all])
    inds2 = np.zeros((batch_size, maxinds)).astype("int")

    # loop over all atoms
    count = 0
    # for a0 in range(0,atoms.positions.shape[0],skip):

    for a0 in range(len(center_inds_skip)):
        count += 1

        inds = inds_all[a0]
        if np.any(use_pbcs):
            dr = get_dists_pbcs(
                atoms.positions[inds, :],
                atoms.positions[center_inds_skip[a0], :],
                bbox,
                use_pbcs,
            )
        else:
            dr = np.sqrt(
                np.sum(
                    (
                        atoms.positions[inds, :]
                        - atoms.positions[center_inds_skip[a0], :]
                    )
                    ** 2,
                    axis=1,
                )
            )

        r_ind = dr / hist_dr
        r_floor = np.floor(r_ind).astype("int")
        dr_ind = r_ind - r_floor

        sub = r_floor < num_bins

        good_vals = r_floor > 0
        nvals = good_vals.sum()
        vals[:nvals] = r_floor[good_vals]

        hist_sig += np.bincount(
            r_floor[sub], weights=(1 - dr_ind[sub]), minlength=num_bins
        )

        sub = r_floor < num_bins - 1
        hist_sig += np.bincount(
            r_floor[sub] + 1, weights=dr_ind[sub], minlength=num_bins
        )

    hist_sig /= count

    hist_sig[0] = 0

    gr = hist_sig.copy()
    gr[1:] /= (hist_dr * dens * 4 * np.pi) * hist_r[1:] ** 2

    if renormalize_reduced_pdf:
        scale = np.sum(gr * hist_r) / np.sum(hist_r)
        gr /= scale

    if plot_result:
        fig, ax = plt.subplots(figsize=(8, 3))

        ax.plot(
            hist_r,
            gr,
        )
        ax.set_ylabel("g(r)")
        ax.set_xlabel("r (A)")

    return hist_r, hist_sig, gr

def get_dists_pbcs(point, pointslist, bbox, pbcs=[1, 1, 1]):
    # pbcs [x,y,z] to match with ase, but really just [-3,-2-1] axes
    abs = np.abs(pointslist - point)
    for i, ind in enumerate(pbcs):
        if ind:
            abs[:, i] = np.minimum(abs[:, i], bbox[i] - abs[:, i])
    return np.sqrt(np.sum(abs**2, axis=1))
