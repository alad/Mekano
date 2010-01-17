"""\
Mekano
======

Python building blocks for IR and machine learning research.

Additional documentation is available in the docstrings.

alad@cs.cmu.edu
"""

__version__ = "1.1"

# Misc
import Errors as errors
import Logging as logging

# Atoms
import AtomFactory as af
from AtomVector import AtomVector
from AtomFactory import AtomFactory
from AtomVectorStore import AtomVectorStore

# Text processing
import Textual as textual

# IO
import IO as io
from Dataset import Dataset

# Indexing
from InvertedIndex import InvertedIndex
from WeightVectors import WeightVectors
from CorpusStats import CorpusStats

# Classifiers
from SVMClassifier import SVMClassifier, SVMMultiClassifier
from KNNClassifier import KNNClassifier
from MultiClassifier import MultiClassifier

from ClassifierDriver import *

# Evaluation
import Evaluator as eval
import Thresholder

