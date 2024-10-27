import numpy as np
import matplotlib.pyplot as plt
from astropy import constants, units
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import proj_plane_pixel_area
from scipy.ndimage import gaussian_filter
from meer21cm.util import (
    check_unit_equiv,
    get_wcs_coor,
    radec_to_indx,
    f_21,
    center_to_edges,
)


def weight_source_peaks(
    map_in,
    wproj,
    ra_g_in,
    dec_g_in,
    z_g_in,
    nu,
    W_map_in=None,
    w_map_in=None,
    velocity_width_halfmax=50,
    velocity_profile="gaussian",
    sigma_beam_in=None,
    no_vel=False,
    internal_step=2000,
    verbose=False,
    ignore_double_counting=False,
    project_mat=None,
    gal_sel_indx=None,
    no_sel_weight=False,
    ang_unit="deg",
):
    if W_map_in is None:
        W_map_in = np.ones_like(map_in)
    if w_map_in is None:
        w_map_in = np.ones_like(map_in)
    num_ch = map_in.shape[-1]
    num_g = ra_g_in.size
    ra_g_in = (ra_g_in * units.Unit(ang_unit)).to("deg").value
    dec_g_in = (dec_g_in * units.Unit(ang_unit)).to("deg").value
    if ignore_double_counting:
        return np.zeros(map_in.shape, dtype="int") - 1, W_map_in
    xx, yy = np.meshgrid(
        np.arange(map_in.shape[0]), np.arange(map_in.shape[1]), indexing="ij"
    )
    # the coordinates of each pixel in the map
    ra, dec = get_wcs_coor(wproj, xx, yy)
    # in deg^2
    pix_area = proj_plane_pixel_area(wproj)
    # in km/s/frequency
    dvdf = (constants.c / nu).to("km/s").value.mean()
    # in km/s
    vel_resol = dvdf * np.diff(nu).mean()
    if no_vel:
        num_ch_vel = 0
    else:
        num_ch_vel = int(4 * velocity_width_halfmax // vel_resol) + 1
    # this profile is only for assigning pixels to sources, not for actual weighting
    if velocity_profile == "gaussian":
        profile_func = (
            lambda x: np.exp(-(x**2 / 2 / velocity_width_halfmax**2))
            / np.sqrt(2 * np.pi)
            / velocity_width_halfmax
        )
    elif velocity_profile == "step":
        profile_func = (
            lambda x: ((np.abs(x) < velocity_width_halfmax).astype("float"))
            / 2
            / velocity_width_halfmax
        )
    else:
        raise ValueError("Unrecognised velocity profile: " + str(velocity_profile))

    gal_freq = f_21 / (1 + z_g_in)
    # which channel each source centre belongs to
    gal_which_ch = np.argmin(np.abs(gal_freq[None, :] - nu[:, None]), axis=0)

    # some internal integrals for calculating HI flux density per channel
    if not no_vel:
        step_size = (velocity_width_halfmax * 4 + 2 * vel_resol) / (internal_step - 1)
        vel_max_int = step_size * internal_step
        # vel_int_arr centres at the galaxy position along los
        vel_int_arr = np.linspace(-vel_max_int, vel_max_int, num=internal_step)
        vel_int_edges = center_to_edges(vel_int_arr)
        ## how the profile looks like across the velocities
        # if no_vel:
        #    hiprofile_g = np.zeros_like(vel_int_edges)
        #    hiprofile_g[vel_int_edges == 0] = 1
        hiprofile_g = profile_func(vel_int_edges)
        hiprofile_g /= np.sum(hiprofile_g)

        hicumflux_g = np.cumsum(hiprofile_g, axis=0)
        # deviation from the channel centre in velocity
        vel_start_pos = (gal_freq - nu[gal_which_ch]) * dvdf
        # zero is centre channel
        vel_ch_arr = (
            np.linspace(-num_ch_vel, num_ch_vel, 2 * num_ch_vel + 1) * vel_resol
        )
        vel_ch_arr = center_to_edges(vel_ch_arr)
        # zero should correspond to the deviation so shifting
        vel_gal_arr = vel_ch_arr[:, None] + vel_start_pos[None, :]
        # which integration step each channel belongs to
        vel_indx = (
            ((vel_gal_arr[:, :, None] - vel_int_edges[None, None, :]) > 0).reshape(
                (-1, len(vel_int_edges))
            )
        ).sum(axis=-1) - 1
        vel_indx = vel_indx.reshape(vel_gal_arr.shape)

        # this is not the real flux density, just a normalised weight for determining the assignment
        hifluxd_ch = np.zeros(vel_indx.shape)
        for i in range(num_g):
            hifluxd_ch[:, i] = hicumflux_g[:][vel_indx[:, i]]
        hifluxd_ch = np.diff(hifluxd_ch, axis=0)
    else:
        hifluxd_ch = np.ones((1, num_g))

    # find the pixels each source belongs to
    # coor_g = SkyCoord(ra_g_in, dec_g_in, unit="deg")
    # indx_1_g, indx_2_g = wproj.world_to_pixel(coor_g)
    # indx_1_g = np.round(indx_1_g).astype("int")
    # indx_2_g = np.round(indx_2_g).astype("int")
    indx_1_g, indx_2_g = radec_to_indx(ra_g_in, dec_g_in, wproj)

    # assign each pixel with a source index, -1 is any source
    map_gal_indx = np.zeros(map_in.shape, dtype="int") - 1
    # this weight is only used for comparing the "distance"
    map_gal_weight = np.zeros_like(map_in)
    gal_arr = range(num_g)
    # only for debugging, never use in production
    if gal_sel_indx is not None:
        gal_arr = gal_arr[gal_sel_indx[0] : gal_sel_indx[1]]
    for gal_i in gal_arr:
        # weight of source i on the entire map
        map_weight_i = np.zeros_like(map_in)
        # zero-pad the map for out-of-frequency-range profiles
        if not no_vel:
            map_weight_i = np.concatenate(
                (
                    map_weight_i[:, :, :num_ch_vel],
                    map_weight_i,
                    map_weight_i[:, :, :num_ch_vel],
                ),
                axis=-1,
            )
        map_weight_i[
            indx_1_g[gal_i],
            indx_2_g[gal_i],
            gal_which_ch[gal_i] : gal_which_ch[gal_i] + 2 * num_ch_vel + 1,
        ] = hifluxd_ch[:, gal_i]
        # only take the frequency range
        if not no_vel:
            map_weight_i = map_weight_i[:, :, num_ch_vel:-(num_ch_vel)]
        # smooth to the beam
        if sigma_beam_in is not None:
            for i in range(len(nu)):
                map_weight_i[:, :, i] = gaussian_filter(
                    map_weight_i[:, :, i], sigma_beam_in[i] / np.sqrt(pix_area)
                )
        if project_mat is not None:
            map_weight_i = np.einsum("ij,abj->abi", project_mat, map_weight_i)
        if no_sel_weight:
            map_gal_weight += map_weight_i
        else:
            # if a pixel has weight for source i, and it's bigger than the previous weight caused by other sources
            search_indx = (
                (map_weight_i > 0)
                * (map_weight_i > (1e-3 * map_weight_i))
                * (map_weight_i > map_gal_weight)
            )
            # then the weight gets updated
            map_gal_weight[search_indx] = map_weight_i[search_indx]
            # these pixels are assigned to source i
            map_gal_indx[search_indx] = gal_i

    return map_gal_indx, map_gal_weight


def stack(
    map_in,
    wproj,
    ra_g_in,
    dec_g_in,
    z_g_in,
    nu,
    W_map_in=None,
    w_map_in=None,
    velocity_width_halfmax=250,
    velocity_profile="gaussian",
    sigma_beam_in=None,
    no_vel=False,
    internal_step=2000,
    verbose=False,
    ignore_double_counting=False,
    project_mat=None,
    gal_sel_indx=None,
    no_sel_weight=False,
    stack_angular_num_nearby_pix=10,
    x_unit=units.km / units.s,
    return_indx_and_weight=False,
    ang_unit="deg",
):
    # calculate what pixels to stack
    map_gal_indx, map_gal_weight = weight_source_peaks(
        map_in,
        wproj,
        ra_g_in,
        dec_g_in,
        z_g_in,
        nu,
        W_map_in=W_map_in,
        w_map_in=w_map_in,
        velocity_width_halfmax=velocity_width_halfmax,
        velocity_profile=velocity_profile,
        sigma_beam_in=sigma_beam_in,
        no_vel=no_vel,
        internal_step=internal_step,
        verbose=verbose,
        ignore_double_counting=ignore_double_counting,
        project_mat=project_mat,
        gal_sel_indx=gal_sel_indx,
        no_sel_weight=no_sel_weight,
        ang_unit=ang_unit,
    )
    if W_map_in is None:
        W_map_in = np.ones_like(map_in)
    if w_map_in is None:
        w_map_in = np.ones_like(map_in)
    num_ch = map_in.shape[-1]
    num_g = ra_g_in.size
    ra_g_in = (ra_g_in * units.Unit(ang_unit)).to("deg").value
    dec_g_in = (dec_g_in * units.Unit(ang_unit)).to("deg").value
    xx, yy = np.meshgrid(
        np.arange(map_in.shape[0]), np.arange(map_in.shape[1]), indexing="ij"
    )
    # the coordinates of each pixel in the map
    ra, dec = get_wcs_coor(wproj, xx, yy)

    gal_freq = f_21 / (1 + z_g_in)
    # which channel each source centre belongs to
    gal_which_ch = np.argmin(np.abs(gal_freq[None, :] - nu[:, None]), axis=0)
    # in deg^2
    pix_area = proj_plane_pixel_area(wproj)
    # in km/s/frequency
    dvdf = (constants.c / nu).to("km/s").value.mean()
    # in km/s
    vel_resol = dvdf * np.diff(nu).mean()
    # if x_unit is frequency based
    if check_unit_equiv(x_unit, units.Hz):
        freq_stack_arr = (
            np.linspace(-(len(nu) - 1), (len(nu) - 1), 2 * len(nu) - 1)
            * np.diff(nu).mean()
        )
        x_edges = center_to_edges(freq_stack_arr * units.MHz).to(x_unit).value
    elif check_unit_equiv(x_unit, constants.c):
        vel_stack_arr = (
            np.linspace(-(len(nu) - 1), (len(nu) - 1), 2 * len(nu) - 1) * vel_resol
        )
        x_edges = center_to_edges(vel_stack_arr)
    else:
        raise ValueError("x_unit must be either frequency or velocity")

    # find the pixels each source belongs to
    indx_1_g, indx_2_g = radec_to_indx(ra_g_in, dec_g_in, wproj)
    # coor_g = SkyCoord(ra_g_in, dec_g_in, unit="deg")
    # indx_1_g, indx_2_g = wproj.world_to_pixel(coor_g)
    # indx_1_g = np.round(indx_1_g).astype("int")
    # indx_2_g = np.round(indx_2_g).astype("int")
    if no_sel_weight:
        w_map_in *= map_gal_weight
    # zero-pad the map and the weights
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
    # default is -1
    sel_stack = -np.ones(
        (
            np.array(map_in.shape)
            + np.array(
                [2 * stack_angular_num_nearby_pix, 2 * stack_angular_num_nearby_pix, 0]
            )
        ),
        dtype="int",
    )
    sel_stack[
        stack_angular_num_nearby_pix:-stack_angular_num_nearby_pix,
        stack_angular_num_nearby_pix:-stack_angular_num_nearby_pix,
    ] = map_gal_indx.copy()

    # prepare the empty cube for stacking
    num_angular_bin = 2 * stack_angular_num_nearby_pix + 1
    stack_3D_map = np.zeros((num_angular_bin, num_angular_bin, 2 * len(nu) - 1))
    stack_3D_weight = stack_3D_map.copy()

    for gal_i in range(num_g):
        gal_ch_centre = gal_which_ch[gal_i]
        if ignore_double_counting:
            gal_sel_indx = 1
        else:
            gal_sel_indx = (
                ((sel_stack == gal_i) + (sel_stack == -1)).astype("bool")
            ).astype("float")
        ra_gal_i = ra_g_in[gal_i]
        dec_gal_i = dec_g_in[gal_i]
        coor_gal_i = SkyCoord(ra_gal_i, dec_gal_i, unit="deg")
        indx_1_i, indx_2_i = wproj.world_to_pixel(coor_gal_i)
        # account for padding
        indx_1_i = np.round(indx_1_i).astype("int") + stack_angular_num_nearby_pix
        indx_2_i = np.round(indx_2_i).astype("int") + stack_angular_num_nearby_pix
        stack_3D_map[
            :, :, len(nu) - 1 - gal_ch_centre : 2 * len(nu) - gal_ch_centre - 1
        ] += (map_stack * w_stack * gal_sel_indx)[
            indx_1_i
            - stack_angular_num_nearby_pix : indx_1_i
            + stack_angular_num_nearby_pix
            + 1,
            indx_2_i
            - stack_angular_num_nearby_pix : indx_2_i
            + stack_angular_num_nearby_pix
            + 1,
        ]
        stack_3D_weight[
            :, :, len(nu) - 1 - gal_ch_centre : 2 * len(nu) - gal_ch_centre - 1
        ] += (w_stack * gal_sel_indx)[
            indx_1_i
            - stack_angular_num_nearby_pix : indx_1_i
            + stack_angular_num_nearby_pix
            + 1,
            indx_2_i
            - stack_angular_num_nearby_pix : indx_2_i
            + stack_angular_num_nearby_pix
            + 1,
        ]
    with np.errstate(divide="ignore", invalid="ignore"):
        stack_3D_map /= stack_3D_weight
    stack_3D_map[stack_3D_weight == 0] = 0.0
    # in deg
    ang_resol = np.sqrt(pix_area)
    ang_edges = (
        np.linspace(
            -stack_angular_num_nearby_pix, stack_angular_num_nearby_pix, num_angular_bin
        )
        * ang_resol
    )
    ang_edges = center_to_edges(ang_edges)
    if verbose:
        stack_2D_map = np.nan_to_num(stack_3D_map).sum(axis=(-1))
        plt.pcolormesh(ang_edges, ang_edges, stack_2D_map.T)
        plt.colorbar()
        plt.xlabel(r"$\Delta_x$ [deg]")
        plt.ylabel(r"$\Delta_y$ [deg]")
        plt.show()
        stack_1D_map = np.nan_to_num(stack_3D_map).sum(axis=(0, 1))
        plt.stairs(stack_1D_map, x_edges)
        plt.xlabel(r"$\Delta _z$ [" + f"{x_unit:latex}" + "]")
        plt.show()
    if return_indx_and_weight:
        return (
            stack_3D_map,
            stack_3D_weight,
            x_edges,
            ang_edges,
            map_gal_indx,
            map_gal_weight,
        )
    else:
        return stack_3D_map, stack_3D_weight, x_edges, ang_edges


def sum_3d_stack(stack_3D_map, vel_ch_avg=5, ang_sum_dist=3.0):
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


# def _stack(
#    map_in,wproj,ra_g_in,dec_g_in,z_g_in,nu,
#    W_map_in=None,
#    w_map_in=None,
#    velocity_width_halfmax = 250,
#    velocity_profile='gaussian',
#    sigma_beam_in=None,
#    no_vel=False,
#    internal_step=2000,
#    verbose=False,
#    stack_angular_num_nearby_pix=10,
#    x_unit=units.km/units.s,
#    ignore_double_counting=False,
#    project_mat=None,
# ):
#    if W_map_in is None:
#        W_map_in = np.ones_like(map_in)
#    if w_map_in is None:
#        w_map_in = np.ones_like(map_in)
#    num_ch = map_in.shape[-1]
#    num_g = ra_g_in.size
#    xx,yy = np.meshgrid(np.arange(map_in.shape[0]),np.arange(map_in.shape[1]))
#    # the coordinates of each pixel in the map
#    ra,dec = get_wcs_coor(wproj,xx,yy)
#    # in deg^2
#    pix_area = proj_plane_pixel_area(wproj)
#    # in km/s/frequency
#    dvdf = (constants.c/nu).to('km/s').value.mean()
#    # in km/s
#    vel_resol = dvdf*np.diff(nu).mean()
#    num_ch_vel = int(4*velocity_width_halfmax//vel_resol)+1
#    # if x_unit is frequency based
#    if check_unit_equiv(x_unit,units.Hz):
#        freq_stack_arr = np.linspace(-(len(nu)-1),(len(nu)-1),2*len(nu)-1)*np.diff(nu).mean()
#        x_edges = centre_to_edges(freq_stack_arr*units.MHz).to(x_unit).value
#    elif check_unit_equiv(x_unit,constants.c):
#        vel_stack_arr = np.linspace(-(len(nu)-1),(len(nu)-1),2*len(nu)-1)*vel_resol
#        x_edges = centre_to_edges(vel_stack_arr)
#    else:
#        raise ValueError('x_unit must be either frequency or velocity')
#    #this profile is only for assigning pixels to sources, not for actual weighting
#    if velocity_profile == 'gaussian':
#        profile_func = lambda x: np.exp(
#            -(x**2/2/velocity_width_halfmax**2)
#        )/np.sqrt(2*np.pi)/velocity_width_halfmax
#    elif velocity_profile == 'step':
#        profile_func = lambda x: (
#            (np.abs(x)<velocity_width_halfmax).astype('float')
#        )/2/velocity_width_halfmax
#    else:
#        raise ValueError('Unrecognised velocity profile: '+str(velocity_profile))
#    # some internal integrals for calculating HI flux density per channel
#    step_size = (velocity_width_halfmax*4+2*vel_resol)/(internal_step-1)
#    vel_max_int = step_size*internal_step
#    # vel_int_arr centres at the galaxy position along los
#    vel_int_arr = np.linspace(-vel_max_int,vel_max_int,num=internal_step)
#    vel_int_edges = centre_to_edges(vel_int_arr)
#    # how the profile looks like across the velocities
#    if no_vel:
#        hiprofile_g = np.zeros_like(vel_int_edges)
#        hiprofile_g[vel_int_edges==0] = 1
#    else:
#        hiprofile_g = profile_func(vel_int_edges)
#        hiprofile_g /= np.sum(hiprofile_g)
#
#    gal_freq = f_21/(1+z_g_in)
#    # which channel each source centre belongs to
#    gal_which_ch = np.argmin(np.abs(gal_freq[None,:]-nu[:,None]),axis=0)
#
#    hicumflux_g = np.cumsum(hiprofile_g,axis=0)
#
#    #deviation from the channel centre in velocity
#    vel_start_pos = (gal_freq-nu[gal_which_ch])*dvdf
#    # zero is centre channel
#    vel_ch_arr = np.linspace(-num_ch_vel,num_ch_vel,2*num_ch_vel+1)*vel_resol
#    vel_ch_arr = centre_to_edges(vel_ch_arr)
#    #zero should correspond to the deviation so shifting
#    vel_gal_arr = vel_ch_arr[:,None]+vel_start_pos[None,:]
#    # which integration step each channel belongs to
#    vel_indx = (((vel_gal_arr[:,:,None]-vel_int_edges[None,None,:])>0).reshape((-1,len(vel_int_edges)))).sum(axis=-1)-1
#    vel_indx = vel_indx.reshape(vel_gal_arr.shape)
#
#    # this is not the real flux density, just a normalised weight for determining the assignment
#    hifluxd_ch = np.zeros(vel_indx.shape)
#    for i in range(num_g):
#        hifluxd_ch[:,i] = hicumflux_g[:][vel_indx[:,i]]
#    hifluxd_ch = np.diff(hifluxd_ch,axis=0)
#
#    # find the pixels each source belongs to
#    coor_g = SkyCoord(ra_g_in,dec_g_in,unit='deg')
#    indx_1_g,indx_2_g = wproj.world_to_pixel(coor_g)
#    indx_1_g = np.round(indx_1_g).astype('int')
#    indx_2_g = np.round(indx_2_g).astype('int')
#
#    # assign each pixel with a source index, -1 is no source
#    map_gal_indx = np.zeros(map_in.shape,dtype='int')-1
#    # this weight is only used for comparing the "distance"
#    map_gal_weight = np.zeros_like(map_in)
#    gal_arr = range(num_g)
#    if ignore_double_counting:
#        pass
#    else:
#        for gal_i in gal_arr:
#            # weight of source i on the entire map
#            map_weight_i = np.zeros_like(map_in)
#            # zero-pad the map for out-of-frequency-range profiles
#            map_weight_i = np.concatenate((map_weight_i[:,:,:num_ch_vel],map_weight_i,map_weight_i[:,:,:num_ch_vel]),axis=-1)
#            map_weight_i[
#                indx_1_g[gal_i],
#                indx_2_g[gal_i],
#                gal_which_ch[gal_i]:gal_which_ch[gal_i]+2*num_ch_vel+1
#            ]=hifluxd_ch[:,gal_i]
#            #only take the frequency range
#            map_weight_i = map_weight_i[:,:,num_ch_vel:-(num_ch_vel)]
#            #smooth to the beam
#            if sigma_beam_in is not None:
#                for i in range(len(nu)):
#                    map_weight_i[:,:,i] = gaussian_filter(map_weight_i[:,:,i],sigma_beam_in[i]/np.sqrt(pix_area))
#            if project_mat is not None:
#                map_weight_i = np.einsum(
#                    'ij,abj->abi',project_mat,map_weight_i
#                )
#            #if a pixel has weight for source i, and it's bigger than the previous weight caused by other sources
#            search_indx = (map_weight_i>0)*(map_weight_i>map_gal_weight)
#            # then the weight gets updated
#            map_gal_weight[search_indx] = map_weight_i[search_indx]
#            # these pixels are assigned to source i
#            map_gal_indx[search_indx] = gal_i
#
#    # zero-pad the map and the weights
#    map_stack = np.zeros(
#        (np.array(map_in.shape)+
#         np.array(
#             [2*stack_angular_num_nearby_pix,
#              2*stack_angular_num_nearby_pix,
#              0]
#         ))
#    )
#    map_stack[
#        stack_angular_num_nearby_pix:-stack_angular_num_nearby_pix,
#        stack_angular_num_nearby_pix:-stack_angular_num_nearby_pix
#    ] = map_in.copy()
#    w_stack = np.zeros(
#        (np.array(map_in.shape)+
#         np.array(
#             [2*stack_angular_num_nearby_pix,
#              2*stack_angular_num_nearby_pix,
#              0]
#         ))
#    )
#    w_stack[
#        stack_angular_num_nearby_pix:-stack_angular_num_nearby_pix,
#        stack_angular_num_nearby_pix:-stack_angular_num_nearby_pix
#    ] = w_map_in.copy()
#    #default is -1
#    sel_stack = -np.ones(
#        (np.array(map_in.shape)+
#         np.array([
#             2*stack_angular_num_nearby_pix,
#             2*stack_angular_num_nearby_pix,
#             0
#         ])),
#        dtype='int',
#    )
#    sel_stack[
#        stack_angular_num_nearby_pix:-stack_angular_num_nearby_pix,
#        stack_angular_num_nearby_pix:-stack_angular_num_nearby_pix
#    ] = map_gal_indx.copy()
#
#    #prepare the empty cube for stacking
#    num_angular_bin = 2*stack_angular_num_nearby_pix+1
#    stack_3D_map = np.zeros((num_angular_bin,num_angular_bin,2*len(nu)-1))
#    stack_3D_weight = stack_3D_map.copy()
#
#    for gal_i in range(num_g):
#        gal_ch_centre = gal_which_ch[gal_i]
#        if ignore_double_counting:
#            gal_sel_indx = 1
#        else:
#            gal_sel_indx = (sel_stack==gal_i).astype('float')
#        ra_gal_i = ra_g_in[gal_i]
#        dec_gal_i = dec_g_in[gal_i]
#        coor_gal_i = SkyCoord(ra_gal_i,dec_gal_i,unit='deg')
#        indx_1_i,indx_2_i = wproj.world_to_pixel(coor_gal_i)
#        #account for padding
#        indx_1_i = np.round(indx_1_i).astype('int')+stack_angular_num_nearby_pix
#        indx_2_i = np.round(indx_2_i).astype('int')+stack_angular_num_nearby_pix
#        stack_3D_map[
#            :,:,
#            len(nu)-1-gal_ch_centre:2*len(nu)-gal_ch_centre-1
#        ] += (
#            map_stack*w_stack*gal_sel_indx
#        )[
#            indx_1_i-stack_angular_num_nearby_pix:indx_1_i+stack_angular_num_nearby_pix+1,
#            indx_2_i-stack_angular_num_nearby_pix:indx_2_i+stack_angular_num_nearby_pix+1
#        ]
#        stack_3D_weight[
#            :,:,len(nu)-1-gal_ch_centre:2*len(nu)-gal_ch_centre-1
#        ] += (
#            w_stack*gal_sel_indx
#        )[
#            indx_1_i-stack_angular_num_nearby_pix:indx_1_i+stack_angular_num_nearby_pix+1,
#            indx_2_i-stack_angular_num_nearby_pix:indx_2_i+stack_angular_num_nearby_pix+1
#        ]
#    stack_3D_map /= stack_3D_weight
#    # in deg
#    ang_resol = np.sqrt(pix_area)
#    ang_edges = np.linspace(
#        -stack_angular_num_nearby_pix,
#        stack_angular_num_nearby_pix,
#        num_angular_bin
#    )*ang_resol
#    ang_edges = centre_to_edges(ang_edges)
#    if verbose:
#        stack_2D_map = (
#            np.nan_to_num(stack_3D_map*stack_3D_weight).sum(axis=(-1))/
#            np.nan_to_num(stack_3D_weight).sum(axis=(-1))
#        )
#        plt.pcolormesh(ang_edges,ang_edges,stack_2D_map.T)
#        plt.colorbar()
#        plt.xlabel(r'$\Delta_x$ [deg]')
#        plt.ylabel(r'$\Delta_y$ [deg]')
#        plt.show()
#        stack_1D_map = (
#            np.nan_to_num(stack_3D_map*stack_3D_weight).sum(axis=(0,1))/
#            np.nan_to_num(stack_3D_weight).sum(axis=(0,1))
#        )
#        plt.stairs(stack_1D_map,x_edges)
#        plt.xlabel(r'$\Delta _z$ ['+f"{x_unit:latex}"+']')
#        plt.show()
#    return stack_3D_map,stack_3D_weight,x_edges,ang_edges
