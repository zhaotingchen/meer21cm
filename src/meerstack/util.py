import numpy as np
from astropy import constants, units
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import healpy as hp
from astropy.io import fits
from hiimtool.basic_util import check_unit_equiv, jy_to_kelvin, f_21
from astropy.cosmology import Planck18
import inspect


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def read_healpix_fits(file):
    hp_map = hp.read_map(file)
    hp_nside = hp.get_nside(hp_map)
    with fits.open(file) as hdul:
        header = hdul[1].header
        map_unit = units.Unit(header["TUNIT1"])
        map_freq = units.Quantity(header["FREQ"]).to("Hz").value
    return hp_map, hp_nside, map_unit, map_freq


def get_wcs_coor(wcs, xx, yy):
    assert wcs.naxis == 2, "input wcs must be 2-dimensional."
    coor = wcs.pixel_to_world(xx, yy)
    ra = coor.ra.deg
    dec = coor.dec.deg
    return ra, dec


def PCAclean(
    M,
    N_fg,
    w=None,
    W=None,
    returnAnalysis=False,
    MeanCentre=False,
    los_axis=-1,
    return_A=False,
):
    """
    Performs PCA cleaning of the map data.
    """
    assert len(M.shape) == 3, "map must be 3D."
    if los_axis < 0:
        # change -1 to 2
        los_axis = 3 + los_axis
    # make sure los is the fist axis
    axes = [0, 1, 2]
    axes.remove(los_axis)
    axes = [
        los_axis,
    ] + axes
    # transpose map data
    M = np.transpose(M, axes=axes)
    nz, nx, ny = M.shape
    M = M.reshape((len(M), -1))
    if W is not None:
        W = np.transpose(W, axes=axes)
        W = W.reshape((len(M), -1))
    if w is not None:
        w = np.transpose(w, axes=axes)
        w = W.reshape((len(M), -1))
    # this is weird. Why are there two weights?
    if MeanCentre:
        if W is None:
            M = M - np.mean(M, 1)[:, None]  # Mean centre data
        else:
            M = M - np.sum(M * W, 1)[:, None] / np.sum(W, 1)[:, None]
    ### Covariance calculation:
    if w is None:
        w = 1.0
    C = np.cov(w * M)  # include weight in frequency covariance estimate
    if returnAnalysis == True:
        eigenval = np.linalg.eigh(C)[0]
        eignumb = np.linspace(1, len(eigenval), len(eigenval))
        eigenval = eigenval[::-1]  # Put largest eigenvals first
        V = np.linalg.eigh(C)[1][
            :, ::-1
        ]  # Eigenvectors from covariance matrix with most dominant first
        return C, eignumb, eigenval, V
    ### Remove dominant modes:
    V = np.linalg.eigh(C)[1][
        :, ::-1
    ]  # Eigenvectors from covariance matrix with most dominant first
    A = V[:, :N_fg]  # Mixing matrix, first N_fg most dominant modes of eigenvectors
    S = np.dot(
        A.T, M
    )  # not including weights in mode subtraction (as per Yi-Chao's approach)
    Residual = M - np.dot(A, S)
    Residual = np.reshape(Residual, (nz, nx, ny))
    Residual = np.transpose(Residual, axes=np.argsort(axes))
    if return_A:
        return Residual, A
    return Residual


def plot_map(
    map_in,
    wproj,
    W=None,
    title=None,
    cbar_label=None,
    cbarshrink=1,
    ZeroCentre=False,
    vmin=None,
    vmax=None,
    cmap="magma",
):
    """
    Stolen from meerpower
    """
    plt.figure()
    plt.subplot(projection=wproj)
    ax = plt.gca()
    lon = ax.coords[0]
    lat = ax.coords[1]
    lon.set_major_formatter("d")
    lon.set_ticks_position("b")
    lat.set_ticks_position("l")
    plt.grid(True, color="grey", ls="solid", lw=0.5)
    if len(np.shape(map_in)) == 3:
        map_in = np.mean(
            map_in, 2
        )  # Average along 3rd dimention (LoS) as default if 3D map given
        if W is not None:
            W = np.mean(W, 2)
    if vmax is not None:
        map_in[map_in > vmax] = vmax
    if vmin is not None:
        map_in[map_in < vmin] = vmin
    if ZeroCentre == True:
        divnorm = colors.TwoSlopeNorm(
            vmin=np.min(map_in), vcenter=0, vmax=np.max(map_in)
        )
        cmap = copy.copy(matplotlib.cm.get_cmap("seismic"))
        cmap.set_bad(color="grey")
    else:
        divnorm = None
    if W is not None:
        map_in[W == 0] = np.nan
    plt.imshow(map_in.T, cmap=cmap, norm=divnorm)
    if vmax is not None or vmin is not None:
        plt.clim(vmin, vmax)
    cbar = plt.colorbar(orientation="horizontal", shrink=cbarshrink, pad=0.2)
    if cbar_label is None:
        cbar.set_label("mK")
    else:
        cbar.set_label(cbar_label)
    ax.invert_xaxis()
    plt.xlabel("R.A [deg]")
    plt.ylabel("Dec. [deg]")
    plt.title(title, fontsize=18)


def radec_to_indx(ra_arr, dec_arr, wproj, to_int=True):
    coor = SkyCoord(ra_arr, dec_arr, unit="deg")
    indx_1, indx_2 = wproj.world_to_pixel(coor)
    if to_int:
        indx_1 = np.round(indx_1).astype("int")
        indx_2 = np.round(indx_2).astype("int")
    return indx_1, indx_2


def convert_hpmap_in_jy_to_temp(hp_map, freq):
    nside = hp.get_nside(hp_map)
    hp_map = jy_to_kelvin(hp_map, hp.nside2resol(nside), freq)
    return hp_map


def healpix_to_wcs(hp_map, xx, yy, wcs):
    """
    Project the healpix map onto a 2-D wcs grid.
    Map has to be in temperature unit.
    """
    nside = hp.get_nside(hp_map)
    ra_map, dec_map = get_wcs_coor(wcs, xx, yy)
    pix_indx = hp.ang2pix(nside, ra_map, dec_map, lonlat=True)
    output_map = hp_map[pix_indx]
    return output_map


def rebin_spectrum(spectrum, rebin_width=3, mode="avg"):
    assert spectrum.size % 2 == 1
    assert rebin_width % 2 == 1
    rebin_pad = spectrum.size % rebin_width
    if rebin_pad % 2 == 1:
        rebin_pad += rebin_width
    rebin_pad = rebin_pad // 2
    spectrum_rebin = (
        spectrum[rebin_pad:-rebin_pad].reshape((-1, rebin_width)).mean(axis=-1)
    )
    if mode == "sum":
        spectrum_rebin *= rebin_width
    return spectrum_rebin


def find_rotation_matrix(vec):
    """
    find the rotation matrix to rotate the input vector to (0,0,1).
    """
    theta_rot = np.arctan2(vec[1], vec[0])
    rot_mat_1 = np.array(
        [
            [np.cos(-theta_rot), -np.sin(-theta_rot), 0],
            [np.sin(-theta_rot), np.cos(-theta_rot), 0],
            [0, 0, 1],
        ]
    )
    inter_vec = rot_mat_1 @ vec
    phi_rot = -np.arctan2(inter_vec[0], inter_vec[2])
    rot_mat_2 = np.array(
        [
            [np.cos(-phi_rot), 0, -np.sin(-phi_rot)],
            [0, 1, 0],
            [np.sin(-phi_rot), 0, np.cos(-phi_rot)],
        ]
    )
    return rot_mat_2 @ rot_mat_1


def minimum_enclosing_box_of_lightcone(ra_arr, dec_arr, freq, cosmo=Planck18):
    """
    not really minimum but should be okay.
    """
    ra_temp = ra_arr.copy()
    ra_temp[ra_temp > 180] -= 360
    ra_mean = ra_temp.mean()
    dec_mean = dec_arr.mean()
    mean_vec = hp.ang2vec(ra_mean, dec_mean, lonlat=True)
    rot_mat = find_rotation_matrix(mean_vec)
    z_arr = f_21 / freq - 1
    vec_arr = hp.ang2vec(ra_arr, dec_arr, lonlat=True)
    # rotate so that centre of field is the line-of-sight [0,0,1]
    vec_arr = np.einsum("ab,ib->ia", rot_mat, vec_arr)
    comov_dist_arr = cosmo.comoving_distance(z_arr)
    pos_arr = vec_arr[:, None, :] * comov_dist_arr[None, :, None]
    pos_arr = pos_arr.reshape((-1, 3))
    x_min, y_min, z_min = pos_arr.min(axis=0)
    x_max, y_max, z_max = pos_arr.max(axis=0)
    inv_rot = np.linalg.inv(rot_mat)
    return (x_min, y_min, z_min, x_max - x_min, y_max - y_min, z_max - z_min, inv_rot)
