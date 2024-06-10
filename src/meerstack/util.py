import numpy as np
from astropy import constants,units
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def get_wcs_coor(wcs,xx,yy):
    assert wcs.naxis==2, 'input wcs must be 2-dimensional.'
    coor = wcs.pixel_to_world(xx,yy)
    ra = coor.ra.deg.T
    dec = coor.dec.deg.T
    return ra,dec

def PCAclean(M,N_fg,w=None,W=None,returnAnalysis=False,MeanCentre=False,los_axis=-1,return_A=False):
    '''
    Performs PCA cleaning of the map data.
    '''
    assert len(M.shape) == 3, "map must be 3D."
    if los_axis < 0:
        # change -1 to 2
        los_axis = 3+los_axis
    # make sure los is the fist axis
    axes = [0,1,2]
    axes.remove(los_axis)
    axes = [los_axis,]+axes
    # transpose map data
    M = np.transpose(M,axes=axes)
    nz,nx,ny = M.shape
    M = M.reshape((len(M),-1))
    if W is not None:
        W = np.transpose(W,axes=axes)
        W = W.reshape((len(M),-1))
    if w is not None:
        w = np.transpose(w,axes=axes)
        w = W.reshape((len(M),-1))
    # this is weird. Why are there two weights?
    if MeanCentre:
        if W is None:
            M = M - np.mean(M,1)[:,None] # Mean centre data
        else:
            M = M - np.sum(M*W,1)[:,None]/np.sum(W,1)[:,None]
    ### Covariance calculation:
    if w is None:
        w = 1.0
    C = np.cov(w*M) # include weight in frequency covariance estimate
    if returnAnalysis==True:
        eigenval = np.linalg.eigh(C)[0]
        eignumb = np.linspace(1,len(eigenval),len(eigenval))
        eigenval = eigenval[::-1] #Put largest eigenvals first
        V = np.linalg.eigh(C)[1][:,::-1] # Eigenvectors from covariance matrix with most dominant first
        return C,eignumb,eigenval,V
    ### Remove dominant modes:
    V = np.linalg.eigh(C)[1][:,::-1] # Eigenvectors from covariance matrix with most dominant first
    A = V[:,:N_fg] # Mixing matrix, first N_fg most dominant modes of eigenvectors
    S = np.dot(A.T,M) # not including weights in mode subtraction (as per Yi-Chao's approach)
    Residual = (M - np.dot(A,S))
    Residual = np.reshape(Residual,(nz,nx,ny))
    Residual = np.transpose(Residual,axes=np.argsort(axes))
    if return_A:
        return Residual,A
    return Residual

def check_unit_equiv(u1,u2):
    """
    Check if two units are equivelant
    """
    return ((1*u1/u2).si.unit == units.dimensionless_unscaled)
    
    
def plot_map(map_in,wproj,W=None,title=None,cbar_label=None,cbarshrink=1,ZeroCentre=False,vmin=None,vmax=None,cmap='magma'):
    '''
    Stolen from meerpower
    '''
    plt.figure()
    plt.subplot(projection=wproj)
    ax = plt.gca()
    lon = ax.coords[0]
    lat = ax.coords[1]
    lon.set_major_formatter('d')
    lon.set_ticks_position('b')
    lat.set_ticks_position('l')
    plt.grid(True, color='grey', ls='solid',lw=0.5)
    if len(np.shape(map_in))==3:
        map_in = np.mean(map_in,2) # Average along 3rd dimention (LoS) as default if 3D map given
        if W is not None: W = np.mean(W,2)
    if vmax is not None: map_in[map_in>vmax] = vmax
    if vmin is not None: map_in[map_in<vmin] = vmin
    if ZeroCentre==True:
        divnorm = colors.TwoSlopeNorm(vmin=np.min(map_in), vcenter=0, vmax=np.max(map_in))
        cmap = copy.copy(matplotlib.cm.get_cmap("seismic"))
        cmap.set_bad(color='grey')
    else: divnorm = None
    if W is not None: map_in[W==0] = np.nan
    plt.imshow(map_in.T,cmap=cmap,norm=divnorm)
    if vmax is not None or vmin is not None: plt.clim(vmin,vmax)
    cbar = plt.colorbar(orientation='horizontal',shrink=cbarshrink,pad=0.2)
    if cbar_label is None:
        cbar.set_label('mK')
    else: cbar.set_label(cbar_label)
    ax.invert_xaxis()
    plt.xlabel('R.A [deg]')
    plt.ylabel('Dec. [deg]')
    plt.title(title,fontsize=18)
    
def radec_to_indx(ra_arr,dec_arr,wproj,to_int=True):
    coor = SkyCoord(ra_arr,dec_arr,unit='deg')
    indx_1,indx_2 = wproj.world_to_pixel(coor)
    if to_int:
        indx_1 = np.round(indx_1).astype('int')
        indx_2 = np.round(indx_2).astype('int')
    return indx_1,indx_2