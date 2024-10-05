import numpy as np
import pmesh
import healpy as hp
from astropy.cosmology import Planck18
from astropy import units
from .util import f_21

allowed_window_scheme = ("nnb", "cic", "tsc", "pcs")


def minimum_enclosing_box_of_lightcone(
    ra_arr,
    dec_arr,
    freq,
    cosmo=Planck18,
    ang_unit="deg",
    tile=True,
    return_coord=False,
    buffkick=0.0,
    rot_mat=None,
):
    """
    This functions finds a rotational axis to rotate the sky vectors of input coordinates so that the (crude) mean of the coordinates is at (0,0,1), and then finds the enclosing cuboid box for the coordinates. The box is not really minimum but should be quite optimal.

    The function also returns a rotational matrix for rotating the coordinates in the cuboid back to the sky positions. For any point in the box ``pos = np.array([x,y,z])``, you can find its RA and Dec by performing the rotation

    .. highlight:: python
    .. code-block:: python

        vec = inv_rot @ pos
        vec /= np.sqrt(np.sum(vec**2))
        ra_pos, dec_pos = hp.vec2ang(vec,lonlat=True)


    Parameters
    ----------
        ra_arr: ``numpy`` array.
            The RA of the coordinates
        dec_arr: ``numpy`` array.
            The Dec of the coordinates
        freq: ``numpy`` array.
            The frequencies of the coordinates.
        cosmo: :class:`astropy.cosmology.Cosmology` object, default `astropy.cosmology.Planck18`.
            The input cosmology for converting frequencies to los length.
        ang_unit: str or :class:`astropy.units.Unit`
            The unit of the input angular coordinates.
        tile: bool, default True.
            Whether to tile the input cooridnates so that the output is a meshgrid of input angular coordinates and frequencies.
        return_coord: bool, default True.
            If True, also returns the corrosponding (x,y,z) coordinate of the input coordinates.
        buffkick: float, default 0.0.
            The box is extended by ``buffkick`` on each end of each dimension.
        rot_mat: ``numpy`` array, default None.
            If specified, override the rotation matrix calculated from the mean cooridnate.



    Returns
    -------
        x_min: float.
            The origin of the box along x-axis.
        y_min: float.
            The origin of the box along y-axis.
        z_min: float.
            The origin of the box along z-axis.
        x_len: float.
            The length of the box along x-axis.
        y_len: float.
            The length of the box along y-axis.
        z_len: float.
            The length of the box along z-axis.
        inv_rot: ``numpy`` array.
            The rotational matrix to rotate the box back to the sky positions.
        pos_arr: ``numpy'' array.
            Only returns if ``return_coord = True''.
            The Cartesian coordinates of the input ra and dec.

    """
    ra_arr = (ra_arr.ravel() * units.Unit(ang_unit)).to("deg").value
    dec_arr = (dec_arr.ravel() * units.Unit(ang_unit)).to("deg").value
    ra_temp = ra_arr.copy()
    ra_temp[ra_temp > 180] -= 360
    ra_mean = ra_temp.mean()
    dec_mean = dec_arr.mean()
    mean_vec = hp.ang2vec(ra_mean, dec_mean, lonlat=True)
    if rot_mat is None:
        rot_mat = find_rotation_matrix(mean_vec)
    z_arr = f_21 / freq.ravel() - 1
    vec_arr = hp.ang2vec(ra_arr, dec_arr, lonlat=True)
    # rotate so that centre of field is the line-of-sight [0,0,1]
    vec_arr = np.einsum("ab,ib->ia", rot_mat, vec_arr)
    comov_dist_arr = cosmo.comoving_distance(z_arr).value
    if tile:
        pos_arr = vec_arr[:, None, :] * comov_dist_arr[None, :, None]
    else:
        pos_arr = vec_arr * comov_dist_arr[:, None]
    pos_arr = pos_arr.reshape((-1, 3))
    x_min, y_min, z_min = pos_arr.min(axis=0) - buffkick
    x_max, y_max, z_max = pos_arr.max(axis=0) + buffkick
    inv_rot = np.linalg.inv(rot_mat)
    result = (x_min, y_min, z_min, x_max - x_min, y_max - y_min, z_max - z_min, inv_rot)
    if return_coord:
        result += (pos_arr,)
    return result


def find_rotation_matrix(vec):
    r"""
    find the rotation matrix to rotate the input vector to (0,0,1).

    Note that in 3D space, the rotation is not unique. For simplicity, this function first finds the rotational matrix so that the vector (x,y,z) is first rotated to :math:`(\sqrt{x^2+y^2},0,z)`, and then find another matrix to rotate the vector to (0,0,1).

    Parameters
    ----------
        vec: ``numpy`` array.
            The input unit vector

    Returns
    -------
        rot_mat: ``numpy`` array.
            The rotational matrix so that ``rot_mat @ vec`` is ``np.array([0,0,1])``.
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


def fourier_window_for_assignment(
    num_mesh,
    window="nnb",
):
    r"""
    Calculate the effective window function in Fourier space from mass assignment scheme
    that sample continueous fields to discrete grids.

    The window function can be written as [1]

    .. math::
        W(k_x,k_y,k_z) = \Bigg({\rm sinc}\bigg(\frac{k_x H_x}{2}\bigg)
        {\rm sinc}\bigg(\frac{k_y H_y}{2}\bigg)
        {\rm sinc}\bigg(\frac{k_z H_z}{2}\bigg)\Bigg)^p,


    where :math:`k_{x,y,z}` is the wavenumber of the grid in Fourier space
    and :math:`H_{x,y,z}` is the length of the grid in real space.
    :math:`p` is the power index related to the mass assignment scheme, and
    is equal to [1,2,3,4] for [nnb,cic,tsc,pcs]

    Parameters
    ----------
        num_mesh: list
            The number of grids on each side
        window: str, default "nnb".
            The mass assignment scheme

    Returns
    -------
        window_in_fourier: ``numpy`` array
            The window function in Fourier space

    References
    ----------
    .. [1] Sefusatti, E. et al.,
        "Accurate Estimators of Correlation Functions in Fourier Space",
        https://ui.adsabs.harvard.edu/abs/2016MNRAS.460.3624S.
    """
    p = allowed_window_scheme.index(window) + 1
    wx, wy, wz = [np.sinc(np.fft.fftfreq(num_mesh[i])) for i in range(3)]
    window_in_fourier = (wx[:, None, None] * wy[None, :, None] * wz[None, None, :]) ** p
    return window_in_fourier


def compensate_grid_window_effects(
    field_in_real_space,
    num_mesh,
    window="nnb",
    pmesh_pos_k=True,
):
    """
    Apply correction to cancel the windowing effects from
    discretization of fields into grids.

    Parameters
    ----------
        field_in_real_space: :class:`pmesh.pm.RealField`
            The meshed field in real space.
        num_mesh: list
            The number of grids on each side
        window: str, default "nnb".
            The mass assignment scheme
        pmesh_pos_k: bool, default True.
            For `pmesh`, only positive k is saved.
            For potential future update of Fourier convention.

    Returns
    -------
        field_in_real_space: :class:`pmesh.pm.RealField`
            The compensated field in real space.
    """
    assign_window = fourier_window_for_assignment(
        num_mesh,
        window=window,
    )
    if pmesh_pos_k:
        assign_window = assign_window[:, :, : num_mesh[-1] // 2 + 1]
    field_in_fourier_space = field_in_real_space.r2c()
    field_in_fourier_space /= assign_window
    field_in_fourier_space.c2r(field_in_real_space)
    return field_in_real_space


def project_particle_to_regular_grid(
    particle_pos,
    box_size,
    num_mesh,
    window="nnb",
    particle_value=None,
    particle_weights=None,
    compensate=False,
    shift=0.0,
    average=True,
):
    """
    Project particles into 3D regular grids with a certain assignment scheme.
    This function allows mass assignment scheme compensation and interlacing for
    more accurate estimation of power spectrum. The details of the techniques
    can be found in Cunnington & Wolz 2024 [1].

    Parameters
    ----------
        particle_pos: ``numpy`` array.
            The coordinates of the particles.
            Last axis must have a length of 3 corresponding to (x,y,z).
        box_size: ``numpy`` array.
            The length of the box on each side
        num_mesh: list
            The number of grids on each side
        window: str, default "nnb".
            The mass assignment scheme
        particle_value: ``numpy`` array, default None.
            The mass of each particle.
        particle_weights: ``numpy`` array, default None.
            The weights of each particle.
        compensate: bool, default False.
            Whether to apply the correction to deconvolve the
            sampling window function.
        shift: float, default 0.0.
            The shift of the field when performing Fourier transform, in the unit of cell size.
        average: bool, default True.
            The grid values are weighted averages of the particles if True
            and weighted sums of the particles if False.

    Returns
    -------
        projected_map: :class:`pmesh.pm.RealField`
            The projected field.
        projected_weights: :class:`pmesh.pm.RealField`
            The projected weights.
        particle_counts: :class:`pmesh.pm.RealField`
            The number counts of particles in each grid.

    References
    ----------
    .. [1] Cunnington S. and Wolz L., "Accurate Fourier-space statistics for line intensity mapping: Cartesian grid sampling without aliased power", https://ui.adsabs.harvard.edu/abs/2024MNRAS.528.5586C
    """
    particle_pos = particle_pos.reshape((-1, 3))
    if particle_value is None:
        particle_value = np.ones(len(particle_pos))
    if particle_weights is None:
        particle_weights = np.ones(len(particle_pos))
    pm = pmesh.pm.ParticleMesh(BoxSize=box_size, Nmesh=num_mesh)
    shifted = pm.affine.shift(shift)
    particle_counts = pm.paint(particle_pos, transform=shifted)
    projected_map = pm.paint(
        particle_pos,
        mass=(particle_value * particle_weights).ravel(),
        resampler=window,
        transform=shifted,
    )
    projected_weights = pm.paint(
        particle_pos,
        mass=particle_weights.ravel(),
        resampler=window,
        transform=shifted,
    )
    if average:
        projected_map[projected_weights > 0] = (
            projected_map[projected_weights > 0]
            / projected_weights[projected_weights > 0]
        )
    if compensate:
        projected_map = compensate_grid_window_effects(
            projected_map,
            num_mesh,
            window=window,
        )
    return projected_map, projected_weights, particle_counts


def interlace_two_fields(
    real_field_1,
    real_field_2,
    shift,
    box_resol,
):
    """
    Interlacing two fields, where one is the shifted version of the other.

    Parameters
    ----------
        real_field_1: :class:`pmesh.pm.RealField`
            The first field for interlacing
        real_field_2: :class:`pmesh.pm.RealField`
            The second field for interlacing, which should be a shifted version of the first.
        shift: float.
            The shift of the field when performing Fourier transform, in the unit of cell size.
        box_resol: list of float.
            The length of the cell along each direction.

    Returns
    -------
        interlaced_field: :class:`pmesh.pm.RealField`
            The interlaced field.
    """
    c1 = real_field_1.r2c()
    c2 = real_field_2.r2c()
    for k, s1, s2 in zip(c1.slabs.x, c1.slabs, c2.slabs):
        kH = sum(k[i] * box_resol[i] for i in range(3))
        s1[...] = s1[...] * 0.5 + s2[...] * 0.5 * np.exp(shift * 1j * kH)
    interlaced_field = c1.c2r()
    return interlaced_field
