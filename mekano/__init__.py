"""\
Mekano
======
Provides low-level building blocks for information retrieval and machine learning,
with a special focus on text processing.

Features
========
  - Representing text documents as sparse vectors
  - Representing a collection of documents as a dataset, which can be subsetted for cross-validation etc.
  - Evaluation using various metrics
  - Reading various common input formats like SMART and TREC
  - Parsing and tokenizing text
  - Maintaining corpus statistics (term frequecies), creating inverted indexes
  - Creating weighted document vectors (TF--IDF) based on corpus statistics
  
Most of the code is in Python, with some crucial functions implemented in C++.

Getting started
===============
The L{atoms} sub-package provides all functionality related to representing text documents as numbers.
This is a good place to start using the Mekano package.

The L{ml} sub-package provides access to classifiers and related utilities.

The L{dataset} module provides a handy class for representing and working with datasets (collections of documents).

The L{io} module contains several functions for reading common file formats and working with Python pickles.

The L{evaluator} module contains evaluation tools.

The L{textual} module contains functions for parsing and tokenizing text.

The L{indri} module provides a simple interface to the U{Indri<http://www.lemurproject.org/indri/>} binaries.

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
