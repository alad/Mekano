from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_AtomVector = Extension("mekano.atoms.atomvector",
                             ["mekano/atoms/atomvector.pyx", "mekano/atoms/atomvector.pxd", "support/CUtils.cpp"],
                             language="c++")

ext_AtomVectorStore = Extension("mekano.atoms.atomvectorstore",
                             ["mekano/atoms/atomvectorstore.pyx", "support/CUtils.cpp"],
                             language="c++")

ext_CS = Extension("mekano.atoms.corpusstats",
                        ["mekano/atoms/corpusstats.pyx", "mekano/atoms/corpusstats.pxd", "mekano/atoms/atomvector.pxd"],
                        language="c++")

ext_LR = Extension("mekano.ml.logreg",
                             ["mekano/ml/logreg.pyx", "mekano/atoms/atomvector.pxd", "support/LRHelper.cpp"],
                             language="c++")

ext_InvertedIndex = Extension("mekano.atoms.invidx",
                        ["mekano/atoms/invidx.pyx", "mekano/atoms/invidx.pxd", "mekano/atoms/atomvector.pxd", "support/CUtils.cpp"],
                        language="c++")

ext_KNNClassifier = Extension("mekano.ml.knn",
                        ["mekano/ml/knn.pyx", "mekano/atoms/atomvector.pxd", "mekano/atoms/invidx.pxd", "support/CUtils.cpp"],
                        language="c++")

ext_WV = Extension("mekano.atoms.weightvectors",
                        ["mekano/atoms/weightvectors.pyx"],
                        language="c++")

ext_Evaluator = Extension("mekano.evaluator",
                        ["mekano/evaluator.pyx"],
                        language="c++")


setup(
  name = "mekano",
  version = "2.0",
  description = "Mekano: Building blocks for IR & ML",
  author = "Abhimanyu Lad",
  author_email = "alad@cs.cmu.edu",
  packages = ["mekano", "mekano.ml", "mekano.atoms"],
  ext_modules = [ext_AtomVector, ext_AtomVectorStore, ext_CS, ext_LR,
                 ext_KNNClassifier, ext_Evaluator, ext_InvertedIndex, ext_WV],
  cmdclass = {'build_ext': build_ext},
)

