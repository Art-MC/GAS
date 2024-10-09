import os
import time
from datetime import timedelta
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm, trange

try:
    import cupy as cp
    from .PDFs_utils.kernels import kernels
except (ImportError, ModuleNotFoundError) as e:
    cp = np


class RDF_2D(object):

    def __init__(self, device="cpu", v=1):
        if isinstance(device, int):
            self._xp = cp
            cp.cuda.runtime.setDevice(device)
        elif device.lower() == "cpu":
            self._xp = np
        elif device.lower() == "gpu":
            self._xp = cp
        else:
            raise ValueError(f"Unknown device {device}")
        self.device = device
        self._v = v
        return

    def rdf_cpu(
        self,
        atoms,
        dr=0.1,
        hist_r_max=20,
        skip=1,
        batch_size=1,
        v=None,
        pbcs=None,
        reduced=True,
        force_GPU = False,
    ):
        stime = time.perf_counter()
        v = self._v if v is None else v
        vprint = print if v >= 1 else lambda *a, **k: None
        if self.device != "cpu" and not force_GPU:
            vprint("Will run with CPU because this is normally faster.")
            vprint("can override with force_GPU=True")
            xp = np
        else:
            xp = self._xp
        if skip == None:
            skip = 1
        if pbcs is None:
            pbcs = atoms.get_pbc()
        self.pbcs = pbcs

        bbox_np = np.array(atoms.cell[(0, 1, 2), (0, 1, 2)])[:2]
        bbox = xp.array(bbox_np, dtype=cp.float64)[:2]
        pbcs_np = np.copy(pbcs)[:2]
        pbcs = xp.array(pbcs).astype("bool")[:2]
        positions = xp.array(atoms.positions)[:,:2]
        positions_np = np.array(atoms.positions)[:,:2]  # for KD tree

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
        vprint("starting KD Tree")
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
        volume = xp.abs(xp.sum(xp.cross(cell[:, 0], cell[:, 1])))
        dens = positions.shape[0] / volume

        ### making big arrays that will be filled
        # natoms_dens = (4 / 3 * xp.pi * hist_r_max**3) * dens
        # maxval_n = int(1.25 * natoms_dens**2)
        natoms_dens = (xp.pi * hist_r_max**2) * dens
        maxval_n = max(int(2 * natoms_dens), 100)
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
        # vprint("Final shape will be: ", hist_sig[:-1].shape)

        for a0 in tqdm(range(0, num_centers, batch_size), disable=v<=0):
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
                xp=xp,
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
        gr /= num_centers

        if reduced:
            gr[0] = 0
            gr[1:] /= (dr * dens * 2 * np.pi) * rr[1:]

        vprint("-- done --")
        ttime = time.perf_counter() - stime
        vprint(
            f"Total time (h:m:s) {str(timedelta(seconds=round(ttime,3))).rstrip('0')}"
        )
        vprint(f"Center atoms per sec: {num_centers / ttime:.1f}\n")

        return rr, gr

    def rdf_slices(
        self,
        atoms,
        dr=0.1,
        hist_r_max=20,
        slice_thickness = 2,
        skip=1,
        batch_size=1,
        v=None,
        pbcs=None,
        reduced=True,
        force_GPU = True,
    ):
        gr_2D_all = []
        gr_2D = None
        stime = time.perf_counter()
        v = self._v if v is None else v
        vprint = print if v >= 1 else lambda *a, **k: None
        self.slice_thickness = slice_thickness

        positions = atoms.positions.copy()

        num_slices = int(np.ceil(positions[:,2].max()/slice_thickness))
        for a0 in trange(num_slices, disable=v<1):
            z1, z2 = a0 * slice_thickness, (a0+1) * slice_thickness
            goods = (positions[:,2] >= z1) & (positions[:,2] <= z2)
            shifted_pos = positions[goods]
            shifted_pos[:,2] -= a0*slice_thickness
            atoms.positions = shifted_pos
            atoms.cell[2,2] = np.max(shifted_pos[:,2])+0.001

            rr, gr = self.rdf_cpu(
                atoms,
                dr=dr,
                hist_r_max=hist_r_max,
                skip=skip,
                pbcs=pbcs,
                reduced=True,
                v=v-1,
                force_GPU = force_GPU,
            )

            gr_2D_all.append(gr)
        atoms.positions = positions

        gr_2D_all = self._xp.array(gr_2D_all)

        # corrected
        self._2D_to_3D_arr = self.gen_2D_to_3D_arr(rr, slice_thickness)
        gr_2D_corrected = self.convert_2D_to_3D(gr_2D_all, self._2D_to_3D_arr)

        vprint("-- done --")
        ttime = time.perf_counter() - stime
        vprint(
            f"Total time (h:m:s) {str(timedelta(seconds=round(ttime,3))).rstrip('0')}"
        )
        vprint(f"Center atoms per sec: {len(positions) / ttime:.1f}")
        vprint(f"Slices atoms per sec: {num_slices / ttime:.1f}\n")

        return rr, gr_2D_all, gr_2D_corrected

    # @staticmethod
    def non_uniform_pdf_transform(self, r, xarr, thk):
        """
        r: radius value (2D) [A]
        xarr: array of r values on the final RDF [A]
        thk: slice thickness [A]
        """
        xarr = np.copy(xarr)
        if r == 0:
            return np.zeros_like(xarr)
        maxrad = np.sqrt(r**2 + thk**2)
        dx = xarr[1] - xarr[0]
        xarr += dx/2 # integrated value heavily dependent on this offset
        if (maxrad - r) < dx/2: # becomes delta function
            out = np.zeros_like(xarr)
            closest = np.argmin(np.abs(xarr-r-dx/2))
            out[closest] = 1
            return out
        xarr[xarr==0] = 1e9
        xarr[xarr==r] = 1e9
        zval = np.sqrt(np.abs(xarr**2 - r**2) )
        original_pdf = (thk - zval)/thk
        transform = r/(xarr**2 * np.sqrt(abs(1-r**2/xarr**2)))
        norm = (2 * thk * np.arccos(r/np.sqrt(thk**2 + r**2)) - r * np.log(1 + thk**2 / r**2)) / (2 * thk)
        out = transform * original_pdf / norm
        out[xarr<=r] = 0
        out[xarr>maxrad] = 0
        return out

    # @staticmethod
    def gen_2D_to_3D_arr(self, xarr, thk, v=0):
        """
        xarr: x/r values of the rdf
        thk: slice thickness used for the 2D RDF
        """
        dx = xarr[1] - xarr[0]
        # padding so don't have normalization blowups at the large r edge
        xarr_larger = np.linspace(0, xarr[-1]+10*dx, len(xarr)+10)
        rdf_2D_to_3D = []

        for rval in tqdm(xarr_larger, disable=v<=0):
            shifts = self.non_uniform_pdf_transform(rval, xarr_larger, thk)
            # normalization is heavily dependent on the precise x values used due to integrating over
            # the asymptote. So correcting manually here.
            # as is, integrates correctly but will be off by factor of dx
            norm = self._xp.trapz(shifts)
            norm = norm if norm != 0 else 1
            rdf_2D_to_3D.append(shifts/norm)

        rdf_2D_to_3D = self._xp.array(rdf_2D_to_3D)[:len(xarr), :len(xarr)]
        return rdf_2D_to_3D

    # @staticmethod
    def convert_2D_to_3D(self, gr, conv_arr):
        fixed = gr[..., None] * conv_arr
        return fixed.sum(axis=-2)

    def _get_dists_pbcs(self, points, pointslists, bbox, pbcs=[1, 1, 1], xp=None):
        if xp is None:
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