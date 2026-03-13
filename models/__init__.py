# models/__init__.py
from .mf   import MF
from .sf   import SF
from .mb   import MB
from .mfp  import MFP
from .sfp  import SFP
from .rrmf import RRMF
from .rrsf import RRSF
from .rrmb import RRMB

# Ready-to-use instances — import this dict in notebooks and fitting code
ALL_MODELS = {
    'MF':   MF(),
    'MFP':  MFP(),
    'SF':   SF(),
    'SFP':  SFP(),
    'MB':   MB(),
    'RRMF': RRMF(),
    'RRSF': RRSF(),
    'RRMB': RRMB(),
}
