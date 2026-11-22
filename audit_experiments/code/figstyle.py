"""Shared figure style for the manuscript.
Palette (validated with the dataviz six-check validator, light mode, all pairs: lightness band, chroma floor,
CVD separation >= 8.6, normal-vision >= 20.6, contrast >= 3:1 on white):
  cobalt #4062b0  burnt copper #c47321  jade #16996c
Neutrals are warm greys. Type: TeX Gyre Heros (Helvetica metrics), 8 pt body, 7 pt ticks, at final print size
(single column 3.5 in, double column 7.2 in). Score maps use the perceptually uniform 'inferno' colormap."""
import matplotlib.pyplot as plt

COBALT, COPPER, JADE = '#4062b0', '#c47321', '#16996c'
INK, INK2, MUTED = '#1c1b19', '#4a4842', '#8a877e'
GRID, BAND, SAND = '#e8e5de', '#ece8df', '#d6d0c2'
DET = {'patchcore': COBALT, 'gauss': COPPER, 'dino': JADE}
DET_NAME = {'patchcore': 'PatchCore-style (primary)', 'gauss': 'Shared Gaussian', 'dino': 'DINOv2-token kNN'}
CMAP = 'inferno'
SINGLE, DOUBLE = 3.5, 7.2


def apply():
    plt.rcParams.update({
        'font.family': 'TeX Gyre Heros', 'font.size': 8, 'mathtext.fontset': 'stixsans',
        'axes.titlesize': 8.5, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'axes.titlepad': 6,
        'axes.labelsize': 8, 'axes.labelcolor': INK, 'axes.edgecolor': INK2, 'axes.linewidth': 0.6,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.axisbelow': True,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'xtick.color': INK2, 'ytick.color': INK2,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'grid.color': GRID, 'grid.linewidth': 0.5, 'legend.fontsize': 7, 'legend.frameon': False,
        'text.color': INK, 'savefig.dpi': 600, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
        'pdf.fonttype': 42, 'ps.fonttype': 42})


def panel(ax, letter, x=-0.12, y=1.06):
    """Bold panel letter, journal style: (a), (b), ..."""
    ax.text(x, y, f'({letter})', transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='left')
