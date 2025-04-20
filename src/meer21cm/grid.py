import numpy as np
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
    p = float(allowed_window_scheme.index(window) + 1)
    wx, wy, wz = [np.sinc(np.fft.fftfreq(num_mesh[i])) for i in range(3)]
    window_in_fourier = (wx[:, None, None] * wy[None, :, None] * wz[None, None, :]) ** p
    return window_in_fourier


def compensate_grid_window_effects(
    field_in_real_space,
    grid_scheme="nnb",
):
    """
    Apply correction to cancel the windowing effects from
    discretization of fields into grids.
    """
    num_mesh = field_in_real_space.shape
    window = fourier_window_for_assignment(
        num_mesh,
        grid_scheme,
    )
    field_in_fourier_space = np.fft.fftn(field_in_real_space)
    field_in_fourier_space /= window
    field_compensated = np.fft.ifftn(field_in_fourier_space).real
    return field_compensated


def interlace_two_fields(
    real_field_1,
    real_field_2,
    shift,
):
    """
    Interlacing two fields, where one is the shifted version of the other.

    Parameters
    ----------
        real_field_1: array-like.
            The first field for interlacing
        real_field_2: array-like.
            The second field for interlacing, which should be a shifted version of the first.
        shift: float.
            The shift of the field when performing Fourier transform, in the unit of cell size.

    Returns
    -------
        interlaced_field: array-like.
            The interlaced field.
    """
    box_ndim = real_field_1.shape
    fourier_field_1 = np.fft.fftn(real_field_1)
    fourier_field_2 = np.fft.fftn(real_field_2)
    kH_2 = [np.fft.fftfreq(box_ndim[i]) for i in range(3)]
    kH_2 = np.array(np.meshgrid(*kH_2, indexing="ij"))
    exp_term = np.prod(np.exp(-2 * 1j * shift * kH_2), axis=0)
    fourier_field_1 = (fourier_field_1 + exp_term * fourier_field_2) / 2
    return np.fft.ifftn(fourier_field_1).real


def project_function(
    s_arr,
    grid_scheme="nnb",
):
    """
    Return the weighting function for the given mass assignment scheme and input 1D distance.
    The distance is in the unit of the cell size.

    Parameters
    ----------
        s_arr: float array.
            The distance in the unit of the cell size.
        grid_scheme: str, default 'nnb'
            The mass assignment scheme.

    Returns
    -------
        weight_arr: float array.
            The weighting function.
    """
    s_arr = np.abs(s_arr)
    p = allowed_window_scheme.index(grid_scheme)
    if p == 0:
        return (s_arr <= 0.5).astype("float")
    elif p == 1:
        return (1 - s_arr) * (s_arr <= 1)
    elif p == 2:
        result = (3 / 4 - s_arr**2) * (s_arr <= 0.5) + (0.5 * (1.5 - s_arr) ** 2) * (
            s_arr < 1.5
        ) * (s_arr > 0.5)
        return result
    elif p == 3:
        result = (4 - 6 * s_arr**2 + 3 * s_arr**3) / 6 * (s_arr <= 1) + (
            2 - s_arr
        ) ** 3 / 6 * (s_arr < 2) * (s_arr > 1)
        return result


def particle_to_mesh_distance(
    particle_pos,
    box_len,
    box_ndim,
):
    """
    Calculate the distance between particles and the nearest mesh center.
    The distance is in the unit of the cell size.
    For particles outside the box, the nearest mesh center is the one on the boundary.

    Parameters
    ----------
        particle_pos: array.
            The coordinates of the particles.
        box_len: array.
            The length of the box on each side
        box_ndim: array.
            The number of grids on each side

    Returns
    -------
        dist: array.
            The distance between the particles and the nearest mesh center.
        indx_grid: array.
            The index of the nearest mesh center.
    """
    box_resol = box_len / box_ndim
    mesh_edges = [np.linspace(0, box_len[i], box_ndim[i] + 1) for i in range(3)]
    mesh_cen = [(mesh_edges[i][:-1] + mesh_edges[i][1:]) / 2 for i in range(3)]
    indx_grid = []
    for i in range(3):
        indx_i = np.digitize(particle_pos[:, i], mesh_edges[i]) - 1
        # if particle is outside the box, set to 0 or box_ndim-1 as appropriate
        indx_i[indx_i < 0] = 0
        indx_i[indx_i >= box_ndim[i]] = box_ndim[i] - 1
        indx_grid += [
            indx_i,
        ]
    mesh_cen = [
        (np.linspace(0, box_ndim[i] - 1, box_ndim[i]) + 0.5) * box_resol[i]
        for i in range(3)
    ]
    particle_pos_mesh = [mesh_cen[i][indx_grid[i]] for i in range(3)]
    particle_pos_mesh = np.array(particle_pos_mesh).T
    return (particle_pos - particle_pos_mesh) / box_resol[None, :], indx_grid


def project_particle_to_regular_grid(
    particle_pos,
    box_len,
    box_ndim,
    grid_scheme="nnb",
    particle_mass=None,
    particle_weights=None,
    average=True,
    shift=0.0,
    compensate=False,
):
    """
    Project particles into a regular grid with a certain mass assignment scheme.

    Parameters
    ----------
        particle_pos: array.
            The coordinates of the particles.
        box_len: array.
            The length of the box on each side
        box_ndim: array.
            The number of grids on each side
        grid_scheme: str, default 'nnb'
            The mass assignment scheme.
        particle_mass: array, default None.
            The mass of each particle.
        particle_weights: array, default None.
            The weights of each particle.
        average: bool, default True.
            The grid values are weighted averages of the particles if True
            and weighted sums of the particles if False.
        shift: float, default 0.0.
            Shift the position of the particles by the same amount in all directions,
            in the unit of cell size.

    Returns
    -------
        mesh_mass: array.
            The mass of each grid.
        mesh_weights: array.
            The weights of each grid.
        mesh_counts: array.
            The effective number of particles in each grid.
    """
    p = allowed_window_scheme.index(grid_scheme)
    if particle_mass is None:
        particle_mass = np.ones(len(particle_pos))
    if particle_weights is None:
        particle_weights = np.ones(len(particle_pos))
    box_resol = box_len / box_ndim
    mesh_mass = np.zeros(box_ndim)
    mesh_weights = np.zeros(box_ndim)
    mesh_counts = np.zeros(box_ndim)
    par_pos = particle_pos + shift * box_resol[None, :]
    particle_s, indx_grid = particle_to_mesh_distance(par_pos, box_len, box_ndim)
    indx_grid = np.array(indx_grid).T
    shift_limit = np.floor(p / 2 + 0.5)
    shift_mat = np.meshgrid(
        np.arange(-shift_limit, shift_limit + 1),
        np.arange(-shift_limit, shift_limit + 1),
        np.arange(-shift_limit, shift_limit + 1),
        indexing="ij",
    )
    shift_mat = np.array([shift_mat[i].ravel() for i in range(3)]).T
    for shift in shift_mat:
        s_shift = particle_s + shift[None, :]
        grid_func_shift = project_function(s_shift, grid_scheme)
        indx_shift = (indx_grid - shift[None, :]).astype("int")
        indx_sel = np.prod(indx_shift >= 0, axis=1)
        indx_sel *= np.prod(indx_shift < box_ndim[None, :], axis=1)
        indx_sel = indx_sel.astype("bool")
        np.add.at(
            mesh_mass,
            tuple(indx_shift[indx_sel].T),
            (particle_mass * particle_weights * np.prod(grid_func_shift, axis=1))[
                indx_sel
            ],
        )
        np.add.at(
            mesh_weights,
            tuple(indx_shift[indx_sel].T),
            (particle_weights * np.prod(grid_func_shift, axis=1))[indx_sel],
        )
        np.add.at(
            mesh_counts,
            tuple(indx_shift[indx_sel].T),
            np.prod(grid_func_shift, axis=1)[indx_sel],
        )
    if average:
        mesh_mass = np.where(mesh_weights > 0, mesh_mass / mesh_weights, 0)
    if compensate:
        mesh_mass = compensate_grid_window_effects(
            mesh_mass,
            grid_scheme,
        )
    return mesh_mass, mesh_weights, mesh_counts
