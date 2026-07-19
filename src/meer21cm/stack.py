import numpy as np
from .util import radec_to_indx, find_ch_id, redshift_to_freq


def stack_cubelet(
    map_in,
    w_map_in,
    indx_0_g,
    indx_1_g,
    indx_z_g,
    weights_gal=None,
    weighting="conventional",
    stack_angular_num_nearby_pix=10,
    symmetrize=False,
):
    r"""
    Workhorse routine that builds the 3D stacked cubelet from an intensity map and a set of
    source pixel positions. This function does not depend on any :class:`meer21cm.Specification`
    object; all inputs are plain arrays.

    Following the stacking formalism, the 3D cubelet around the :math:`i`-th source is

    .. math::
        \bm{I}_{\bm{s}\bm{x}_i} = \sum_{\bm{x}} \mathcal{S}^{\bm{s}}_{\bm{x}_i\bm{x}}\, \bm{L}_{\bm{x}},
        \qquad
        \mathcal{S}^{\bm{s}}_{\bm{x}_i\bm{x}} = \delta^{\rm K}_{(\bm{x}-\bm{x}_i)\bm{s}}\, w_{\bm{x}},

    i.e. the cubelet voxel at separation :math:`\bm{s}` is the map value :math:`L_{\bm{x}_i+\bm{s}}`
    weighted by the map weight :math:`w_{\bm{x}_i+\bm{s}}`. The 3D stacked signal is the
    galaxy-weighted sum over all sources, normalised by a factor :math:`Q_0`,

    .. math::
        \bm{I}_{\bm{s}} = \frac{1}{Q_0} \sum_i \bm{w}^{\rm gal}_i\, \bm{I}_{\bm{s}\bm{x}_i}
        = \frac{1}{Q_0}\sum_i \bm{w}^{\rm gal}_i\, w_{\bm{x}_i+\bm{s}}\, L_{\bm{x}_i+\bm{s}}.

    The galaxy weight :math:`\bm{w}^{\rm gal}_i` is a single constant per source and is **not** a
    function of the separation :math:`\bm{s}`; for example it can be a binary weight that selects a
    particular subsample of the galaxy catalogue.

    In both cases the numerator is the same galaxy-weighted sum
    :math:`\sum_i \bm{w}^{\rm gal}_i\, w_{\bm{x}_i+\bm{s}}\, L_{\bm{x}_i+\bm{s}}`; the two
    normalisation schemes differ only in :math:`Q_0`:

    - ``"conventional"``: :math:`Q_0` is the per-separation sum of the effective weights, using the
      map pixel weight evaluated at each separation coordinate :math:`\bm{x}_i+\bm{s}`, so that the
      cubelet is the weighted **average** over the contributing sources,

      .. math::
          \bm{I}_{\bm{s}} = \frac{\sum_i \bm{w}^{\rm gal}_i\, w_{\bm{x}_i+\bm{s}}\, L_{\bm{x}_i+\bm{s}}}
               {\sum_i \bm{w}^{\rm gal}_i\, w_{\bm{x}_i+\bm{s}}}.

    - ``"quadratic"``: :math:`Q_0` is the single scalar normalisation of the quadratic estimator,
      i.e. the sum over sources of the galaxy weight times the map pixel weight evaluated at the
      **centre pixel** :math:`\bm{x}_i` (Eq. 30 of the formalism),

      .. math::
          Q_0 = \sum_i \bm{w}^{\rm gal}_i\, w_{\bm{x}_i}.

      Unlike the conventional scheme, :math:`Q_0` is independent of the separation :math:`\bm{s}`
      because the map pixel weight is taken at the source centre rather than at each separation.

    The cubelet extends over the entire frequency range of the map so the spectral separation is
    sampled at [:math:`-N_{\rm ch}\delta\nu`,...,0,..., :math:`N_{\rm ch}\delta\nu`].
    The angular sampling of the cubelet corresponds to the map pixels, and the size of the angular
    plane is set by ``stack_angular_num_nearby_pix``. Note that ``stack_angular_num_nearby_pix`` is
    the number of pixels **each side of the centre** so the size of the angular plane is
    ``(2 * stack_angular_num_nearby_pix + 1)**2``.

    If ``symmetrize``, a mirroring of the individual cubelets is performed along :math:`\Delta\nu=0`.
    This corresponds to the 180deg rotation along the spectral axis described in Sinigaglia et al.
    (2022) [1] and is the only symmetry that single-dish IM stacking is sensitive to.

    Parameters
    ----------
        map_in: array.
            The intensity map data cube, with shape ``(n_ra, n_dec, n_ch)``.
        w_map_in: array.
            The per-voxel map weights :math:`w_{\bm{x}}`, with the same shape as ``map_in``.
        indx_0_g: array.
            The first angular pixel index of each source centre.
        indx_1_g: array.
            The second angular pixel index of each source centre.
        indx_z_g: array.
            The frequency channel index of each source centre.
        weights_gal: array, optional, default None.
            The per-source weights :math:`\bm{w}^{\rm gal}_i`. If None, uniform weights are used.
        weighting: str, optional, default "conventional".
            The normalisation scheme, either ``"conventional"`` (per-separation weighted average,
            map pixel weight at each separation) or ``"quadratic"`` (scalar quadratic-estimator
            normalisation :math:`\sum_i w^{\rm gal}_i w_{\bm{x}_i}`, map pixel weight at the centre pixel).
        stack_angular_num_nearby_pix: optional, default 10.
            The number of map pixels sampled on each side relative to the source centre.
        symmetrize: optional, default False.
            Whether to symmetrize the stacking.

    Returns
    -------
        stack_3D_map: array.
            The normalised cubelet for the stacking.
        stack_3D_weight: array.
            The accumulated per-voxel effective weights :math:`\sum_i \bm{w}^{\rm gal}_i w_{\bm{x}_i+\bm{s}}`.
            This is the per-voxel normalisation used by the ``"conventional"`` scheme.

    References
    ----------
    .. [1] Sinigaglia, F. et al., "Optimizing spectral stacking for 21-cm observations of galaxies: accuracy assessment and symmetrized stacking", https://ui.adsabs.harvard.edu/abs/2022MNRAS.514.4205S.

    """
    if weighting not in ("conventional", "quadratic"):
        raise ValueError(
            f"weighting must be 'conventional' or 'quadratic', got '{weighting}'"
        )
    map_in = np.asarray(map_in)
    w_map_in = np.asarray(w_map_in)
    num_ch = map_in.shape[-1]
    # copy so that the in-place padding shift does not mutate the caller's arrays
    indx_0_g = np.array(indx_0_g)
    indx_1_g = np.array(indx_1_g)
    indx_z_g = np.asarray(indx_z_g)
    num_g = indx_0_g.size
    if weights_gal is None:
        weights_gal = np.ones(num_g)
    weights_gal = np.asarray(weights_gal, dtype=float)
    # check if some galaxies are outside the range
    sel = (
        (indx_0_g < 0)
        + (indx_0_g >= map_in.shape[0])
        + (indx_1_g < 0)
        + (indx_1_g >= map_in.shape[1])
        + (indx_z_g == num_ch)
    )
    if sel.sum() > 0:
        raise ValueError("some galaxies are outside survey area or frequency range")
    # zero pad the sky map and the weights
    map_stack = np.zeros(
        (
            np.array(map_in.shape)
            + np.array(
                [2 * stack_angular_num_nearby_pix, 2 * stack_angular_num_nearby_pix, 0]
            )
        )
    )
    map_stack[
        stack_angular_num_nearby_pix:-stack_angular_num_nearby_pix,
        stack_angular_num_nearby_pix:-stack_angular_num_nearby_pix,
    ] = map_in.copy()
    w_stack = np.zeros(
        (
            np.array(map_in.shape)
            + np.array(
                [2 * stack_angular_num_nearby_pix, 2 * stack_angular_num_nearby_pix, 0]
            )
        )
    )
    w_stack[
        stack_angular_num_nearby_pix:-stack_angular_num_nearby_pix,
        stack_angular_num_nearby_pix:-stack_angular_num_nearby_pix,
    ] = w_map_in.copy()
    # indices are shifted by zero-padding
    indx_0_g += stack_angular_num_nearby_pix
    indx_1_g += stack_angular_num_nearby_pix

    num_angular_bin = 2 * stack_angular_num_nearby_pix + 1
    # take a nearby area around each source
    indx_xx, indx_yy = np.meshgrid(
        *(
            (
                np.arange(
                    -stack_angular_num_nearby_pix,
                    stack_angular_num_nearby_pix + 1,
                ),
            )
            * 2
        ),
        indexing="ij",
    )
    indx_0_sample = indx_0_g[None, None, :] + indx_xx[:, :, None]
    indx_1_sample = indx_1_g[None, None, :] + indx_yy[:, :, None]
    # the results to be stacked
    stack_3D_map = np.zeros((num_angular_bin, num_angular_bin, 2 * num_ch - 1))
    stack_3D_weight = np.zeros((num_angular_bin, num_angular_bin, 2 * num_ch - 1))
    # loop over frequency channel should be a good balance between speed and memory
    for ch_id in range(num_ch):
        # the centre image around each source in channel i
        map_source_i = map_stack[
            indx_0_sample.ravel(), indx_1_sample.ravel(), ch_id
        ].reshape(indx_0_sample.shape)
        weight_source_i = w_stack[
            indx_0_sample.ravel(), indx_1_sample.ravel(), ch_id
        ].reshape(indx_0_sample.shape)
        # fold the per-source galaxy weight into the effective voxel weight so that both the
        # numerator and the normalisation are weighted by w_gal_i (see Eq. 6 of the formalism)
        weight_source_i = weight_source_i * weights_gal[None, None, :]
        # each source is added to a different channel in the final cube
        # this is wrong because repeating indices are only added in the last occurance
        # stack_3D_map[:, :, ch_id - indx_z_g + num_ch - 1] += (
        #    map_source_i * weight_source_i
        # )
        # stack_3D_weight[:, :, ch_id - indx_z_g + num_ch - 1] += weight_source_i
        add_id = ch_id - indx_z_g + num_ch - 1
        if symmetrize:
            add_id = np.append(add_id, 2 * num_ch - 2 - add_id)
            weight_source_i = np.concatenate(
                [weight_source_i, weight_source_i], axis=-1
            )
            map_source_i = np.concatenate([map_source_i, map_source_i], axis=-1)
        # some new np black magic
        np.add.at(stack_3D_weight, (slice(None), slice(None), add_id), weight_source_i)
        np.add.at(
            stack_3D_map,
            (slice(None), slice(None), add_id),
            weight_source_i * map_source_i,
        )

    # normalise
    if weighting == "conventional":
        # per-separation weighted average: divide by the accumulated weights at each separation,
        # i.e. the map pixel weight evaluated at each separation coordinate x_i + s
        stack_3D_map[stack_3D_weight > 0] = (
            stack_3D_map[stack_3D_weight > 0] / stack_3D_weight[stack_3D_weight > 0]
        )
    else:
        # quadratic estimator normalisation Q0 = sum_i w_gal_i w_L(x_i) (Eq. 30 of the formalism):
        # the sum over sources of the galaxy weight times the map pixel weight at the centre pixel
        # (indx_0_g, indx_1_g are the zero-padding-shifted centre indices of w_stack)
        centre_w = w_stack[indx_0_g, indx_1_g, indx_z_g]
        q0 = np.sum(weights_gal * centre_w)
        if symmetrize:
            # each cubelet is mirrored and added twice, so the numerator is doubled accordingly
            q0 = 2 * q0
        if q0 != 0:
            stack_3D_map = stack_3D_map / q0
    return stack_3D_map, stack_3D_weight


def stack(
    sp,
    weights_gal=None,
    weighting="conventional",
    stack_angular_num_nearby_pix=10,
    symmetrize=False,
):
    r"""
    Calculate a stacked 3D cubelet using the intensity maps and source positions stored in ``sp``.

    This is a thin wrapper around :func:`stack_cubelet`. It extracts the intensity map
    (``sp.data``), the map weights (``sp.w_HI``) and the source positions
    (``sp.ra_gal``, ``sp.dec_gal``, ``sp.z_gal``) from the input object, converts the source sky
    positions into map pixel/channel indices, and delegates the actual stacking to
    :func:`stack_cubelet`.

    Parameters
    ----------
        sp: :class:`meer21cm.Specification` object.
            The data used for stacking.
        weights_gal: array, optional, default None.
            The per-source weights :math:`\bm{w}^{\rm gal}_i`. If None, uniform weights are used.
        weighting: str, optional, default "conventional".
            The normalisation scheme, either ``"conventional"`` (per-separation weighted average,
            map pixel weight at each separation) or ``"quadratic"`` (scalar quadratic-estimator
            normalisation :math:`\sum_i w^{\rm gal}_i w_{\bm{x}_i}`, map pixel weight at the centre pixel).
        stack_angular_num_nearby_pix: optional, default 10.
            The number of map pixels sampled on each side relative to the source centre.
        symmetrize: optional, default False.
            Whether to symmetrize the stacking.

    Returns
    -------
        stack_3D_map: array.
            The normalised cubelet for the stacking.
        stack_3D_weight: array.
            The accumulated per-voxel effective weights in the cubelet.

    See Also
    --------
    stack_cubelet : The underlying ``sp``-independent stacking routine.
    """
    map_in = sp.data.copy()
    w_map_in = sp.w_HI.copy()
    ra_g_in = sp.ra_gal.copy()
    dec_g_in = sp.dec_gal.copy()
    z_g_in = sp.z_gal.copy()
    wproj = sp.wproj
    # retrive the centre pixel positions
    indx_0_g, indx_1_g = radec_to_indx(ra_g_in, dec_g_in, wproj)
    indx_z_g = find_ch_id(redshift_to_freq(z_g_in), sp.nu)
    return stack_cubelet(
        map_in,
        w_map_in,
        indx_0_g,
        indx_1_g,
        indx_z_g,
        weights_gal=weights_gal,
        weighting=weighting,
        stack_angular_num_nearby_pix=stack_angular_num_nearby_pix,
        symmetrize=symmetrize,
    )


def sum_3d_stack(stack_3D_map, vel_ch_avg=5, ang_sum_dist=3.0):
    """
    Collapse a stacked cubelet into stacked image and stacked spectrum.

    Note that for stacked image, `vel_ch_avg` is the number of channels that go into
    the summation on each side of the centre channel so that the total number of
    channels that are summed is `(2 * vel_ch_avg + 1)`.

    For stacked spectrum, the angular pixels that go into the summation are determined
    by the distance to the center pixel. Note that the distance is in cell length not physical angular unit.


    Parameters
    ----------
        stack_3D_map: array.
            The stacked cubelet.
        vel_ch_avg: optional, default 5.
            How many channels on each side of the center to sum into stacked image.
        ang_sum_dist: optional, default 3.0.
            The distance within which the angular pixels are summed to stacked spectrum

    Returns
    -------
        angular_stack_map: array.
            The stacked image.
        spectral_stack_map: array.
            The stacked spectrum.
    """
    mid_point = stack_3D_map.shape[-1] // 2
    ang_centre = stack_3D_map.shape[0] // 2
    xx, yy = np.meshgrid(
        np.linspace(-ang_centre, ang_centre, stack_3D_map.shape[0]),
        np.linspace(-ang_centre, ang_centre, stack_3D_map.shape[0]),
    )
    pix_dist = np.sqrt(xx**2 + yy**2)
    pix_sel = pix_dist <= (ang_sum_dist)
    angular_stack_map = stack_3D_map[
        :, :, mid_point - vel_ch_avg : mid_point + vel_ch_avg + 1
    ].sum(axis=-1)
    spectral_stack_map = stack_3D_map[pix_sel].sum(axis=0)
    return angular_stack_map, spectral_stack_map
