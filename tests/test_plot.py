import matplotlib.pyplot as plt
from meer21cm.plot import *


def test_plt(test_W, test_nu, test_wproj):
    plt.switch_backend("Agg")
    plot_pixels_along_los(test_W, test_W, zaxis=test_nu[:1])
    plot_eigenspectrum(np.array([1, 2]))
    plot_map(test_W, test_wproj, W=test_W, vmin=0, vmax=1)
    plt.close("all")
