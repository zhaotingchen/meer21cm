"""Legacy field→sky deposits vs the unified NGP path in ``grid.py``.

The two functions below are the pre-unification implementations
(``_grid_field_to_sky_map_wcs`` / ``_grid_field_to_sky_map_healpix``).
They are kept only to check that :meth:`grid_field_to_sky_map` matches
them on a small box.
"""

import healpy as hp
import numpy as np
import pytest

from meer21cm import PowerSpectrum
from meer21cm.grid import project_particle_to_regular_grid
from meer21cm.util import find_ch_id, radec_to_indx, redshift_to_freq


def _legacy_grid_field_to_sky_map_wcs(
    ps,
    field,
    average=True,
    mask=True,
    wproj=None,
    num_pix_x=None,
    num_pix_y=None,
    los_sel=None,
):
    """Copy of the former ``LightconeGriddingMixin._grid_field_to_sky_map_wcs``."""
    if wproj is None:
        wproj = ps.wproj
    if num_pix_x is None:
        num_pix_x = ps.num_pix_x
    if num_pix_y is None:
        num_pix_y = ps.num_pix_y
    los_sel = (
        np.arange(ps.box_ndim[2], dtype=int)
        if los_sel is None
        else np.asarray(los_sel, dtype=int)
    )
    expected_shape = (ps.box_ndim[0], ps.box_ndim[1], los_sel.size)
    if field.shape != expected_shape:
        raise ValueError(
            f"field shape {field.shape} does not match expected shape "
            f"{expected_shape} for los_sel size {los_sel.size}"
        )
    x_vec = ps.x_vec[0]
    y_vec = ps.x_vec[1]
    z_vec = ps.x_vec[2][los_sel]
    nx = x_vec.size
    ny = y_vec.size
    nz = z_vec.size
    nxyz = nx * ny * nz
    pos_xyz = np.empty((nxyz, 3), dtype=ps.real_dtype)
    pos_xyz[:, 0] = np.repeat(x_vec, ny * nz)
    pos_xyz[:, 1] = np.tile(np.repeat(y_vec, nz), nx)
    pos_xyz[:, 2] = np.tile(z_vec, nx * ny)
    pos_ra, pos_dec, pos_z, _ = ps.ra_dec_z_for_coord_in_box(pos_xyz)
    pos_indx_1, pos_indx_2 = radec_to_indx(pos_ra, pos_dec, wproj, to_int=False)
    pos_indx_z = redshift_to_freq(pos_z) - ps.nu.min()
    pos_indx_array = np.empty((nxyz, 3), dtype=ps.real_dtype)
    pos_indx_array[:, 0] = pos_indx_1
    pos_indx_array[:, 1] = pos_indx_2
    pos_indx_array[:, 2] = pos_indx_z
    map_bin, _, count_bin = project_particle_to_regular_grid(
        pos_indx_array,
        np.array([num_pix_x, num_pix_y, ps.nu.max() - ps.nu.min()]),
        np.array([num_pix_x, num_pix_y, ps.nu.size]),
        particle_mass=field.ravel(),
        average=average,
        compensate=False,
        grid_scheme="nnb",
    )
    if mask:
        map_bin *= ps.W_HI
    return map_bin, count_bin


def _legacy_grid_field_to_sky_map_healpix(
    ps,
    field,
    average=True,
    mask=True,
    los_sel=None,
):
    """Copy of the former ``LightconeGriddingMixin._grid_field_to_sky_map_healpix``."""
    los_sel = (
        np.arange(ps.box_ndim[2], dtype=int)
        if los_sel is None
        else np.asarray(los_sel, dtype=int)
    )
    expected_shape = (ps.box_ndim[0], ps.box_ndim[1], los_sel.size)
    if field.shape != expected_shape:
        raise ValueError(
            f"field shape {field.shape} does not match expected shape "
            f"{expected_shape} for los_sel size {los_sel.size}"
        )
    nside = int(ps.hp_nside)
    pixel_id = np.asarray(ps.pixel_id, dtype=np.int64)
    n_out = pixel_id.size
    n_ch = int(ps.nu.size)
    order = np.argsort(pixel_id, kind="mergesort")
    pix_sorted = pixel_id[order]

    x_vec = ps.x_vec[0]
    y_vec = ps.x_vec[1]
    z_vec = ps.x_vec[2][los_sel]
    nx = x_vec.size
    ny = y_vec.size
    nz = z_vec.size
    nxyz = nx * ny * nz
    pos_xyz = np.empty((nxyz, 3), dtype=ps.real_dtype)
    pos_xyz[:, 0] = np.repeat(x_vec, ny * nz)
    pos_xyz[:, 1] = np.tile(np.repeat(y_vec, nz), nx)
    pos_xyz[:, 2] = np.tile(z_vec, nx * ny)
    pos_ra, pos_dec, pos_z, _ = ps.ra_dec_z_for_coord_in_box(pos_xyz)
    hpix = hp.ang2pix(nside, pos_ra, pos_dec, lonlat=True).astype(np.int64)
    pos_nu = np.asarray(redshift_to_freq(pos_z), dtype=np.float64)
    ch_idx = find_ch_id(pos_nu, ps.nu)
    valid_ch = (ch_idx >= 0) & (ch_idx < n_ch)
    hpix = hpix[valid_ch]
    ch_idx = ch_idx[valid_ch]
    mass = np.asarray(field, dtype=ps.real_dtype).ravel()[valid_ch]

    row_s = np.searchsorted(pix_sorted, hpix)
    in_bounds = row_s < n_out
    in_survey = np.zeros(hpix.shape, dtype=bool)
    in_survey[in_bounds] = pix_sorted[row_s[in_bounds]] == hpix[in_bounds]
    row = order[row_s[in_survey]]
    ch_idx = ch_idx[in_survey]
    mass = mass[in_survey]

    map_sum = np.zeros((n_out, n_ch), dtype=ps.real_dtype)
    cnt = np.zeros((n_out, n_ch), dtype=ps.real_dtype)
    np.add.at(map_sum, (row, ch_idx), mass)
    np.add.at(cnt, (row, ch_idx), 1.0)
    if average:
        with np.errstate(divide="ignore", invalid="ignore"):
            map_bin = np.where(cnt > 0, map_sum / cnt, 0.0)
    else:
        map_bin = map_sum
    count_bin = cnt
    if mask:
        map_bin *= ps.W_HI
    return map_bin, count_bin


def _lowres_healpix():
    nu = np.linspace(redshift_to_freq(0.8), redshift_to_freq(0.6), 12)
    ps = PowerSpectrum(
        nu=nu,
        hp_nside=32,
        ra_range=(200.0, 210.0),
        dec_range=(-5.0, 5.0),
        downres_factor_radial=2.0,
        downres_factor_transverse=2.0,
    )
    ps.get_enclosing_box()
    return ps


def _lowres_wcs():
    ps = PowerSpectrum(
        survey="meerklass_2021",
        band="L",
        ra_range=(339.0, 345.0),
        dec_range=(-35.0, -32.0),
        downres_factor_radial=4.0,
        downres_factor_transverse=4.0,
    )
    ps.get_enclosing_box()
    return ps


@pytest.mark.parametrize("fmt", ["healpix", "wcs"])
@pytest.mark.parametrize("average", [False, True])
def test_grid_field_to_sky_map_matches_legacy(fmt, average):
    ps = _lowres_healpix() if fmt == "healpix" else _lowres_wcs()
    field = np.arange(np.prod(ps.box_ndim), dtype=ps.real_dtype).reshape(ps.box_ndim)
    new_m, new_c = ps.grid_field_to_sky_map(field, average=average, mask=False)
    if fmt == "healpix":
        old_m, old_c = _legacy_grid_field_to_sky_map_healpix(
            ps, field, average=average, mask=False
        )
    else:
        old_m, old_c = _legacy_grid_field_to_sky_map_wcs(
            ps, field, average=average, mask=False
        )
    assert np.array_equal(new_m, old_m)
    assert np.array_equal(new_c, old_c)
