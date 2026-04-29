import matplotlib as mpl

def fancy_plotting(use_tex):
    if use_tex:
        mpl.rcParams.update({
            # Enable LaTeX for all text rendering
            'text.usetex': True,

            # Set the font to Computer Modern (the default LaTeX font)
            'font.family': 'serif',
            'font.serif': ['Computer Modern Roman'],

            # Figure and font sizes
            'font.size': 18,             # Base font size
            'axes.labelsize': 15,        # Axis labels
            'axes.titlesize': 16,        # Axes titles
            'xtick.labelsize': 15,       # X tick labels
            'ytick.labelsize': 15,       # Y tick labels
            'legend.fontsize': 13,       # Legend text size

            # Figure properties
            'figure.figsize': [6, 5],  # Default figure size in inches
            'figure.dpi': 100,             # Default figure resolution

            # Save figures with high resolution
            'savefig.dpi': 300,

            # Make sure negative signs are rendered correctly
            'axes.unicode_minus': False,

            # Adjust the layout automatically to fit labels
            'figure.autolayout': True,

            # LaTeX preamble: load extra packages such as amsmath and amssymb
            'text.latex.preamble': r'\usepackage{amsmath,amssymb}'
        })
