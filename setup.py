from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_AtomVector = Extension("mekano.AtomVector",
                             ["mekano/AtomVector.pyx", "mekano/AtomVector.pxd", "support/CUtils.cpp"],
                             language="c++")

ext_AtomVectorStore = Extension("mekano.AtomVectorStore",
                             ["mekano/AtomVectorStore.pyx", "support/CUtils.cpp"],
                             language="c++")

ext_CS = Extension("mekano.CorpusStats",
                        ["mekano/CorpusStats.pyx", "mekano/CorpusStats.pxd", "mekano/AtomVector.pxd"],
                        language="c++")

ext_LR = Extension("mekano.LogisticRegressionClassifier",
                             ["mekano/LogisticRegressionClassifier.pyx", "mekano/AtomVector.pxd", "support/LRHelper.cpp"],
                             language="c++")

ext_InvertedIndex = Extension("mekano.InvertedIndex",
                        ["mekano/InvertedIndex.pyx", "mekano/InvertedIndex.pxd", "mekano/AtomVector.pxd", "support/CUtils.cpp"],
                        language="c++")

ext_KNNClassifier = Extension("mekano.KNNClassifier",
                        ["mekano/KNNClassifier.pyx", "mekano/AtomVector.pxd", "mekano/InvertedIndex.pxd", "support/CUtils.cpp"],
                        language="c++")

ext_WV = Extension("mekano.WeightVectors",
                        ["mekano/WeightVectors.pyx"],
                        language="c++")

ext_Evaluator = Extension("mekano.Evaluator",
                        ["mekano/Evaluator.pyx"],
                        language="c++")


setup(
  name = "mekano",
  version = "1.4",
  description = "Mekano: Building blocks for IR & ML",
  author = "Abhimanyu Lad",
  author_email = "alad@cs.cmu.edu",
  packages = ["mekano"],
  ext_modules = [ext_AtomVector, ext_AtomVectorStore, ext_CS, ext_LR,
                 ext_KNNClassifier, ext_Evaluator, ext_InvertedIndex, ext_WV],
  cmdclass = {'build_ext': build_ext},
)

