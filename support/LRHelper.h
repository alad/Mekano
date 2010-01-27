#include <iostream>
#include <math.h>
#include <set>
#include <vector>
#include <utility>
#include <CUtils.h>

using namespace std;

#if LRLONGDOUBLE
    typedef long double LRREAL;
#else
    typedef double LRREAL;
#endif

typedef set<unsigned int> UISet;
typedef pair<unsigned int, double> InvIndPair;
typedef vector<InvIndPair> InvIndVec;
typedef vector<InvIndVec> SparseData;
typedef vector<double> DVec;

//
// Calculate the LOSS, which is the negative version of likelihood.
// This function is applicable to both normal and sparse method.
//
LRREAL LogisticRegressionLoss1(DVec& SIGMA,
					DVec& S, 
					DVec& LAMBDA,
					DVec& MU,
					DVec& W,
                    const UISet& docList);

void CG_HS1(DVec& old_grad,
		     DVec& grad,
             DVec& u);
                             

void LogisticRegressionTrainSparse(SparseData& X,
				      DVec& Y,
				      DVec& S,
				      DVec& LAMBDA,
				      DVec& MU, 
				      int MAXITER,
				      LRREAL EPSILON,
				      int VERBOSE,
				      DVec& W,
				      bool UpdateW,
				      DVec& sigma,
                      const UISet& docList);

void copyToDVec(PyObject *src, DVec *dest);
PyObject* ctrain(PyObject *docs, PyObject *labels, PyObject *mu, double lambda, int maxiter, double epsilon);