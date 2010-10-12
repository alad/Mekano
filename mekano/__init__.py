"""\
Mekano
======
Provides low-level building blocks for information retrieval and machine learning,
with a special focus on text processing.

Features
========
* Representing text documents as sparse vectors
* Representing a collection of documents as a dataset, which can be subsetted for cross-validation etc.
* Evaluation using various metrics
* Reading various common input formats like SMART and TREC
* Parsing and tokenizing text
* Maintaining corpus statistics (term frequecies), creating inverted indexes
* Creating weighted document vectors (TF--IDF) based on corpus statistics

Most of the code is in Python, with some crucial functions implemented in C++.

Detailed documentation is available in the docstrings.

"""

__version__ = "2.0"

# Misc
import logging

# Atoms
from atoms import *

# IO
import io
from dataset import Dataset
import textual

import indri

from ml import *
