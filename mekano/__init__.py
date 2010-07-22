"""\
Mekano
======

Python building blocks for IR and machine learning research.

Additional documentation is available in the docstrings.

alad@cs.cmu.edu
"""

__version__ = "1.3"

# Misc
import Errors as errors
import Logging as logging

# Atoms
from atoms import *

import textual

# IO
import IO as io
from Dataset import Dataset
#import Collections as collections

# Indexing
from InvertedIndex import InvertedIndex
from WeightVectors import WeightVectors
from CorpusStats import CorpusStats
import indri


import ml
