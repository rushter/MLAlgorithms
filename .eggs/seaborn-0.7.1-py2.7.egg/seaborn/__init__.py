# Capture the original matplotlib rcParams
import matplotlib as mpl
_orig_rc_params = mpl.rcParams.copy()

# Import seaborn objects
from .rcmod import *
from .utils import *
from .palettes import *
from .linearmodels import *
from .categorical import *
from .distributions import *
from .timeseries import *
from .matrix import *
from .miscplot import *
from .axisgrid import *
from .widgets import *
from .xkcd_rgb import xkcd_rgb
from .crayons import crayons

# Set default aesthetics
set()

__version__ = "0.7.1"
