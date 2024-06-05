import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import os
matplotlib.rcParams['figure.figsize'] = (18, 9)
from astropy.cosmology import Planck18
from numpy.random import default_rng
from hiimtool.basic_util import busy_function_simple
from hiimtool.basic_util import himf,himf_pars_jones18,cal_himf,sample_from_dist,tully_fisher
from astropy import constants,units
from hiimtool.basic_util import f_21,centre_to_edges
from astropy.wcs.utils import proj_plane_pixel_area
from scipy.ndimage import gaussian_filter
from .stack import stack
from .util import PCAclean,check_unit_equiv,get_wcs_coor,plot_map,radec_to_indx

lamb_21 = (constants.c/f_21*units.s).to('m')

def run_poisson_mock(
    nu,num_g,himf_pars,wproj,
    base_map=None,
    verbose=False,
    seed=None,
    N_fg=0,
    mmin=11.7,
    mmax=14.0,
    no_vel=True,
    W_HI=None,
    w_HI=None,
    tf_slope=None,
    tf_zero=None,
    cosmo=Planck18,
    map_unit=units.Jy,
    sigma_beam_ch=None,
    mycmap='bwr',
    do_stack=True,
    dndz=None,
    zmin=None,
    zmax=None,
    internal_step=1001,
    return_key=None,
    meerpower=None,
    x_dim=None,
    y_dim=None,
    stack_angular_num_nearby_pix=10,
    x_unit=units.km/units.s,
    ignore_double_counting=False,
    return_indx_and_weight=False,
    ra_range=(-np.inf,np.inf),
    dec_range=(-400,400),
    velocity_width_halfmax=50,
    fix_ra_dec = None,
    fix_z = None,
):
    def mock_stack(verbose):
        stack_result = stack(
            himap_g,wproj,
            ra_g_mock[inside_range],dec_g_mock[inside_range],
            z_g_mock[inside_range],nu,
            W_map_in=W_HI,w_map_in=w_HI,no_vel=no_vel,
            sigma_beam_in=sigma_beam_ch,
            velocity_width_halfmax = velocity_width_halfmax,
            stack_angular_num_nearby_pix=stack_angular_num_nearby_pix,
            ignore_double_counting=ignore_double_counting,
            x_unit=x_unit,
            verbose=verbose,
            return_indx_and_weight=return_indx_and_weight,
        )
        return stack_result
    # in deg^2
    pix_area = proj_plane_pixel_area(wproj)
    if base_map is not None:
        xx,yy = np.meshgrid(
            np.arange(base_map.shape[0]),
            np.arange(base_map.shape[1])
        )
    elif x_dim is not None and y_dim is not None:
        xx,yy = np.meshgrid(
            np.arange(x_dim),
            np.arange(y_dim)
        )
    else:
        raise ValueError('either base_map or (x_dim,y_dim) is neededd')
    # the coordinates of each pixel in the map
    ra,dec = get_wcs_coor(wproj,xx,yy)
    if verbose and meerpower is None:
        raise ValueError('must have meerpower for plotting')
    if zmin is None:
        zmin = f_21/nu[-1]/1e6-1
    if zmax is None:
        zmax = f_21/nu[0]/1e6-1
    if W_HI is None:
        W_HI = np.ones(ra.shape+nu.shape)
    if dndz is None:
        dndz = lambda x: np.ones_like(x)
    if base_map is None:
        base_map = np.zeros(ra.shape+nu.shape)
    rng = default_rng(seed=seed)
    if W_HI[:,:,0].sum() == 0:
        raise ValueError('all pixels are masked by W_HI')
    indx_1,indx_2= np.where(W_HI[:,:,0].astype('bool'))
    indx_1_g = np.array([])
    indx_2_g = np.array([])
    if ra_range is not None:
        ra_range = np.array(ra_range)
        ra_range[ra_range>180] -= 360
    # randomly select positions until enough are unmasked
    num_gal_in_range = 0
    while num_gal_in_range<num_g:
        indx_1_rand = rng.uniform(indx_1.min(),indx_1.max(),size=num_g)
        indx_2_rand = rng.uniform(indx_2.min(),indx_2.max(),size=num_g)
        coor_rand = wproj.pixel_to_world(indx_1_rand,indx_2_rand)
        ra_rand = coor_rand.ra.degree
        dec_rand = coor_rand.dec.degree
        ra_rand[ra_rand>180] -= 360
        sel_rand_indx = (
            W_HI[:,:,0][
            np.round(indx_1_rand).astype('int'),
            np.round(indx_2_rand).astype('int')]
        ).astype('bool')
        indx_1_g = np.append(indx_1_g,indx_1_rand[sel_rand_indx])
        indx_2_g = np.append(indx_2_g,indx_2_rand[sel_rand_indx])
        inside_range = (
            sel_rand_indx*
            (ra_rand>ra_range[0])*(ra_rand<ra_range[1])*
            (dec_rand>dec_range[0])*(dec_rand<dec_range[1])
        )
        num_gal_in_range += inside_range.sum()
    # convert to angular coor
    coor_g = wproj.pixel_to_world(indx_1_g,indx_2_g)
    ra_g_mock = coor_g.ra.degree
    dec_g_mock = coor_g.dec.degree
    indx_1_g,indx_2_g = radec_to_indx(ra_g_mock,dec_g_mock,wproj)
    #indx_1_g = np.round(indx_1_g).astype('int')
    #indx_2_g = np.round(indx_2_g).astype('int')
    ra_g_temp = ra_g_mock.copy()
    ra_g_temp[ra_g_temp>180] -= 360
    inside_range = (
        (ra_g_temp>ra_range[0])*(ra_g_temp<ra_range[1])*
        (dec_g_mock>dec_range[0])*(dec_g_mock<dec_range[1])
    )
    # only need num_g sources inside the range
    # this if is for avoiding num_g exceeding the maximum index
    if inside_range.mean()==1:
        indx_stop = num_g
    else:
        indx_stop = np.where(inside_range)[0][num_g]
    ra_g_mock = ra_g_mock[:indx_stop]
    dec_g_mock = dec_g_mock[:indx_stop]
    indx_1_g = indx_1_g[:indx_stop]
    indx_2_g = indx_2_g[:indx_stop]
    inside_range = inside_range[:indx_stop]
    if fix_ra_dec is not None:
        ra_g_mock[inside_range] = fix_ra_dec[0]
        dec_g_mock[inside_range] = fix_ra_dec[1]
        fix_indx = radec_to_indx(fix_ra_dec[0],fix_ra_dec[1],wproj)
        indx_1_g[inside_range] = fix_indx[0]
        indx_2_g[inside_range] = fix_indx[1]
    # update num_g to include sources outside the range
    num_g = ra_g_mock.size
    # samples from himf distribution
    himass_g = sample_from_dist(
        lambda x: himf(10**x,himf_pars[0],10**himf_pars[1],himf_pars[2]),
        mmin,mmax,size=num_g,seed=seed
    )
    rand_z = sample_from_dist(dndz,zmin,zmax,size=num_g,seed=seed)
    if fix_z is not None:
        rand_z[inside_range] = fix_z
    if verbose:
        plt.hist(rand_z,bins=20,density=True)
        plt.title('redshift distribution')
        plt.xlabel('z')
        plt.ylabel('dn/dz')
        plt.show()

        plt.subplot(projection=wproj)
        ax = plt.gca()
        lon = ax.coords[0]
        lat = ax.coords[1]
        lon.set_major_formatter('d')
        contours = plt.contour(W_HI[:,:,0].T, levels=[0.5], colors='black')  
        if inside_range.mean()<1: 
            plt.scatter(
                ra_g_mock[(1-inside_range).astype('bool')],
                dec_g_mock[(1-inside_range).astype('bool')],
                transform=ax.get_transform('world'),s=1,
                label='galaxy outside the range but in the map',color='tab:blue'
            )
        plt.scatter(ra_g_mock[inside_range],dec_g_mock[inside_range],transform=ax.get_transform('world'),s=1,label='galaxy positions',color='tab:red')
        lon = ax.coords[0]
        lat = ax.coords[1]
        #lon.set_ticks(spacing=np.sqrt(pix_area) * units.degree)
        #lat.set_ticks(spacing=np.sqrt(pix_area) * units.degree)
        ax.coords.grid(True, color='black', ls='solid')
        plt.xlabel('R.A [deg]',fontsize=18)
        plt.ylabel('Dec. [deg]',fontsize=18)
        plt.legend()
        plt.show()
    
    # in km/s/freq
    dvdf = (constants.c/nu).to('km/s').value.mean()
    # in km/s
    vel_resol = dvdf*np.diff(nu).mean()
    # get velocity for the sources
    if no_vel:
        num_ch_vel = 0
    else:
        hivel_g = tully_fisher(10**himass_g/1.4,tf_slope,tf_zero,inv=True)
        incli_g = np.abs(np.sin(rng.uniform(0,2*np.pi,size=num_g)))
        hiwidth_g = incli_g*hivel_g
        num_ch_vel = (int(hiwidth_g.max()/vel_resol))//2+2
    
    comov_dist_g = cosmo.comoving_distance(rand_z).value
    lumi_dist_g = (1+rand_z)*comov_dist_g
    # convert to flux. from 1705.04210
    # in Jy km s-1
    hiintflux_g = 10**himass_g*(1+rand_z)/2.356/1e5/(lumi_dist_g)**2
    # random busy functions
    busy_c = 10**(rng.uniform(-3,-2,size=num_g))
    busy_b = 10**(rng.uniform(-2,0,size=num_g))
    # zero is the centre of source along los
    vel_ch_arr = np.linspace(-num_ch_vel,num_ch_vel,2*num_ch_vel+1)*vel_resol
    if no_vel:
        hiprofile_g = hiintflux_g[None,:]
    else:
        vel_int_arr = np.linspace(-hiwidth_g.max(),hiwidth_g.max(),num=internal_step)
        hiprofile_g = busy_function_simple(vel_int_arr[:,None],1,busy_b,(busy_c/hiwidth_g)[None,:]*2,hiwidth_g[None,:]/2)
    # the integral over velocity should give the flux
    hiprofile_g = hiprofile_g/(np.sum(hiprofile_g,axis=0))[None,:]*hiintflux_g[None,:]
    
    gal_freq = f_21/(1+rand_z)/1e6
    # which channel the galaxies belong to
    gal_which_ch = np.argmin(np.abs(gal_freq[None,:]-nu[:,None]),axis=0)
    # obtain the emission line profile for each galaxy
    if no_vel:
        hifluxd_ch = hiprofile_g
    else:
        hicumflux_g = np.cumsum(hiprofile_g,axis=0)
        vel_start_pos = (nu[gal_which_ch]-gal_freq)*dvdf
        vel_gal_arr = vel_ch_arr[:,None]-vel_start_pos[None,:]
        vel_indx = np.argmin(np.abs(vel_gal_arr[:,:,None]-vel_int_arr[None,None,:]).reshape((-1,len(vel_int_arr))),axis=1)
        vel_indx = vel_indx.reshape(vel_gal_arr.shape)
        hifluxd_ch = np.zeros(vel_indx.shape)
        for i in range(num_g):
            hifluxd_ch[:,i] = hicumflux_g[:,i][vel_indx[:,i]]
        hifluxd_ch = np.diff(hifluxd_ch,axis=0)
        hifluxd_ch = np.concatenate((np.zeros(num_g)[None,:],hifluxd_ch),axis=0)
    hifluxd_ch /= vel_resol
    himap_g = np.zeros(ra.shape+nu.shape)
    # add a filler channel for both start and end of frequency
    himap_g = np.concatenate((himap_g[:,:,0][:,:,None],himap_g,himap_g[:,:,0][:,:,None]),axis=-1)
    for i,indx_diff in enumerate(np.linspace(-num_ch_vel,num_ch_vel,2*num_ch_vel+1).astype('int')):
        # note the start filler thus +1
        indx_z = gal_which_ch + 1 + indx_diff
        # throw away edge bits
        indx_z[indx_z<0] = 0
        indx_z[indx_z>=himap_g.shape[-1]] = himap_g.shape[-1]-1
        himap_g[indx_1_g,indx_2_g,indx_z] += hifluxd_ch[i]
    himap_g = himap_g[:,:,1:-1]
    
    #smooth to beam
    if sigma_beam_ch is not None:
        for i in range(himap_g.shape[-1]):
            himap_g[:,:,i] = gaussian_filter(himap_g[:,:,i],sigma_beam_ch[i]/np.sqrt(pix_area))
    # convert to temp if needed
    if check_unit_equiv(map_unit,units.K):
        z_ch = f_21/nu/1e6-1
        himap_g = (himap_g*units.Jy/(2*constants.k_B/(lamb_21*(1+(z_ch)))**2)/(pix_area*np.pi**2/180**2)).to(map_unit).value
    elif check_unit_equiv(map_unit,units.Jy):
        himap_g = (himap_g*units.Jy).to(map_unit).value
    else:
        raise ValueError("map unit must be either temperature or flux density")

    if verbose:
        sys.path.insert(1,meerpower)
        import Init
        import plot
        plot_map(himap_g,W=W_HI,map_ra=ra,map_dec=dec,wproj=wproj,
         title='mock HI signal',ZeroCentre=False,
         cbar_label=f"{map_unit:latex}",
        )
        plt.show()
    # overlay mock on the original map
    himap_g += base_map
    # if PCA is needed
    if N_fg > 0:
        himap_g = PCAclean(himap_g,N_fg=N_fg,W=W_HI,w=w_HI)
    z_g_mock = rand_z
    if not do_stack:
        return himap_g,ra_g_mock,dec_g_mock,z_g_mock,indx_1_g,indx_2_g,gal_which_ch,hifluxd_ch,inside_range
    # run stacking
    stack_result = mock_stack(verbose)
    return (himap_g,ra_g_mock,dec_g_mock,z_g_mock,indx_1_g,indx_2_g,gal_which_ch,hifluxd_ch,inside_range)+stack_result