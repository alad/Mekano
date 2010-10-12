#include <LRHelper.h>

PyObject* ctrain(PyObject *docs, PyObject *labels, PyObject *mu, double lambda, int maxiter, double epsilon, double c) {
    SparseData sparseData;
    UISet docset;
    DVec Y;
    DVec MU, LAMBDA, SIGMA, W;
    int pos = 0, neg = 0;
    int nf_found = 0;                             // no. of features found in the data

    Py_ssize_t ndocs = PyList_GET_SIZE(docs);
    for(Py_ssize_t i=0; i<ndocs; i++) {
        InvIndVec invind;
        docset.insert(i);

        if (PyObject_IsTrue(PyList_GET_ITEM(labels, i))) {
            Y.push_back(1.0);
            pos++;
        } else {
            Y.push_back(-1.0);
            neg++;
        }
            
        PyObject *doc = PyList_GET_ITEM(docs, i);
        PyObject *result = PyObject_CallMethod(doc, "iteritems", NULL);
        PyObject *iter = PyObject_GetIter(result);
        PyObject *item;
        //cout << iter->ob_type->tp_name << "\n";
        while(item = PyIter_Next(iter)) {
            int k = PyInt_AS_LONG(PyTuple_GET_ITEM(item, 0));
            double v = PyFloat_AS_DOUBLE(PyTuple_GET_ITEM(item, 1));
            if (k > nf_found) nf_found = k;
            InvIndPair pair(k-1, v);            // features start from 0, atoms start from 1.
            invind.push_back(pair);
            Py_DECREF(item);
        }
        sparseData.push_back(invind);
        Py_DECREF(iter);
        Py_DECREF(result);
    }
    
    // Instance Weights
    double posweight = c/(1+c)/pos;
    double negweight = 1.0/(1+c)/neg;
    DVec S;
    for (int i=0;i<ndocs;i++) {
        if (Y[i] > 0.0) {
            S.push_back(posweight);
        } else {
            S.push_back(negweight);
        }
    }

    int nf = nf_found;    
    // Mu
    if (PyList_Check(mu)) {
        // Number of features implied by the caller.
        // (Caller will send nf+1 to us if he knows what he is doing).
        nf = PyList_GET_SIZE(mu)-1; 
        // nf can be >= nf_found, but not smaller.
        if (nf < nf_found) {
            cout << "Wrong size for mu, should be at least" << nf_found+1 << "\n";
            PyErr_SetString(PyExc_RuntimeError, "Wrong mu");
            return NULL;
        } else {
            copyToDVec(mu, &MU);
        }
    } else {
        double mu_value = PyFloat_AS_DOUBLE(mu);
        for(Py_ssize_t i=0;i<nf_found+1;i++) {
            MU.push_back(mu_value);
        }
    }

    // Lambda
    // Now, nf would be max(len(MU), nf_found)
    for(Py_ssize_t i=0;i<nf+1;i++) {
        LAMBDA.push_back(lambda);
    }
    
    LogisticRegressionTrainSparse(sparseData, Y, S, LAMBDA, MU, maxiter, epsilon, 0, W, true, SIGMA, docset);
    
    PyObject *w = PyList_New(0);
    for(int i=1;i<nf+1;i++) {
        PyList_Append(w, PyFloat_FromDouble(W[i]));
    }
    PyObject *b = PyFloat_FromDouble(W[0]);
    return Py_BuildValue("(NN)", w, b);
    // OR:
    // PyObject *ret = PyTuple_Pack(2, w, b);
    // Py_DECREF(w);
    // Py_DECREF(b);
    // return ret;
}

void copyToDVec(PyObject *src, DVec *dest) {
    Py_ssize_t len = PyList_GET_SIZE(src);
    for (Py_ssize_t i=0;i<len;i++) {
        double v = PyFloat_AS_DOUBLE(PyList_GET_ITEM(src, i));
        dest->push_back(v);
    }
}

// Following code contributed by Jian Zhang
LRREAL LogisticRegressionLoss1(DVec& SIGMA,
						DVec& S, 
						DVec& LAMBDA,
						DVec& MU,
						DVec& W,
						const UISet& docList)
    {
      assert(SIGMA.size() == S.size());
      assert(LAMBDA.size() == MU.size());
      assert(LAMBDA.size() == W.size());

      // std::cout << "{";
      LRREAL loss = 0.0;
      //BK for (unsigned int i=0; i<SIGMA.size(); i++){
      for(UISet::const_iterator i= docList.begin(); i != docList.end(); i++) {
	loss += (LRREAL)S[*i]*(LRREAL)log((LRREAL)1.0/(LRREAL)SIGMA[*i]);
	// std::cout << SIGMA[*i] << "(" << S[*i] << "),";
      }
      for (unsigned int f=0; f<LAMBDA.size(); f++) {
	loss += (LRREAL)LAMBDA[f]*(W[f]-(LRREAL)MU[f])*(W[f]-(LRREAL)MU[f]);
	// std::cout << LAMBDA[f]*(W[f]-MU[f])*(W[f]-MU[f]) << ",";
      }
      // std::cout << "}\n";
      return loss;
    }

    // Obtain the new conjugate direction using Hestenes-Stiefel method
    //
    void CG_HS1(DVec& old_grad,
			     DVec& grad,
			     DVec& u)
    {
      // 1. calculate the beta
      LRREAL beta = 0.0, numerator = 0.0, denominator = 0.0;
      for (unsigned int f=0; f<u.size(); f++) {
	numerator += grad[f] * (grad[f] - old_grad[f]);
	denominator += u[f] * (grad[f] - old_grad[f]);
      }
      beta = numerator/denominator;

      // 2. update u (also make ||u|| = 1.0 
      LRREAL normsquare = 0.0, norm = 0.0;
      for (unsigned int f=0; f<u.size(); f++) {
	u[f] = grad[f] - beta*u[f];
	normsquare += u[f]*u[f];
      }
      norm = sqrt(normsquare);
      for (unsigned int f=0; f<u.size(); f++) {
	u[f] /= norm;
      }
    }


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
				      const UISet& docList)
{
    if (VERBOSE >= 1) {
        std::cout << "Initialize the logistic regression model ..." << std::endl;
    }

  assert(Y.size() == S.size());
  assert(LAMBDA.size() == MU.size());

  unsigned int datanum = Y.size();

  // featurenum = 1 (additional intercept) + actual feauture num
  unsigned int featurenum = LAMBDA.size();

  if (UpdateW) { // if no previous W is available
W.clear();  
W.resize(featurenum, 0.0);
  } else {
assert(W.size() == LAMBDA.size());
  }
  DVec grad, u, old_grad;
  old_grad.resize(featurenum, 0.0);

  // 2. start the loop
  LRREAL loss = 0.0, lastloss = 1e20;
  for (int iter=1; iter<= MAXITER; iter++){
// 2.1 compute the \sigma_i = (1+\exp(-y_i * (W^T x_i)))^{-1}
sigma.clear();
// first, \sigma_i = W^T x_i
sigma.resize(datanum, W[0]); // add the intercept

for(UISet::const_iterator dit= docList.begin();
    dit != docList.end();
    dit++)
{
  unsigned int dataid= *dit;
  const InvIndVec &fVec= X[dataid];
  for(InvIndVec::const_iterator fit= fVec.begin();
      fit != fVec.end();
      fit++)
  {
    // W[feature] * weight
    sigma[dataid] += W[fit->first+1]*(LRREAL)fit->second;
  }
}

/* Old II
   for (unsigned int f=1; f<featurenum; f++){ // skip the intercept
     for (unsigned int item=0; item<X[f-1].size(); item++){
       unsigned int dataid = X[f-1][item].first;
       if(docList.find(dataid) == docList.end()) continue;  //BK: TODO: merge-sort loop for speed & verify correctness
       double value = X[f-1][item].second;
       // std::cout << "[" << dataid << "," << f << "," << value << "]";
       sigma[dataid] += W[f]*value;
       // std::cout << "<" << sigma[dataid] << "," << W[f] << ">";
     }
   }
   // std::cout << std::endl;
*/

// then, calculate the true sigma
//BK for (unsigned int i=0; i<datanum; i++){
for (UISet::const_iterator i= docList.begin();
     i != docList.end();
     i++)
{
  sigma[*i] = (LRREAL)1.0/((LRREAL)1.0 + (LRREAL)exp((LRREAL)-Y[*i]*sigma[*i]));
}

// 2.2 calculate the loss
loss = LogisticRegressionLoss1(sigma, S, LAMBDA, MU, W, docList);
if (VERBOSE >= 1) {
  std::cout.precision(20);
  std::cout << "Iter " << iter << ": Loss = " << loss << std::endl;
}
if (!(fabs((loss-lastloss)/loss) > EPSILON)) break; // exit condition
lastloss = loss;

// 2.3 compute the gradient
grad.clear();
grad.resize(featurenum, 0.0);


for(UISet::const_iterator dit= docList.begin();
    dit != docList.end();
    dit++)
{
  unsigned int dataid= *dit;
  LRREAL meat=
    (LRREAL)S[dataid]*((LRREAL)1.0-sigma[dataid])*(LRREAL)Y[dataid];

  // Intercept
  grad[0] += meat;

  // Features in this document
  const InvIndVec &fVec= X[dataid];
  for(InvIndVec::const_iterator fit= fVec.begin();
      fit != fVec.end();
      fit++)
  {
    grad[fit->first+1] += meat*(LRREAL)fit->second;
  }
}

// add the prior/regularization part
for(unsigned int f=0; f<featurenum; f++) {
  grad[f] =
    grad[f] - (LRREAL)2.0*(LRREAL)LAMBDA[f]*(W[f]-(LRREAL)MU[f]);
}

// 2.4 compute the new conjugate direction (Hestenes-Stiefel)
if (iter == 1) {
  u = grad;
} else {
  CG_HS1(old_grad, grad, u);
}

// 2.5 based on the formula compute "g^T u"
LRREAL gu = 0.0;
for (unsigned int f=0; f<featurenum; f++) {
  gu += u[f]*grad[f];
  // std::cout << "[" << u[f] << "," << grad[f] << "]";
}
// std::cout << std::endl;

// SO FAR

// 2.6 compute "-u^T H u = \sum_i a_{ii}*(u^T x_i)^2 + 2 \sum_j (\lambda_j * u_j * u_j)"
// where H is the Hessian matrix, u is the direction, and a_{ii} = S_i * \sigma_i * (1-\sigma_i).
//
// first compute "u^T x_i"
LRREAL minus_uHu = 0.0;
DVec ux;
ux.clear();
ux.resize(datanum, u[0]); // the intercept part!

for(UISet::const_iterator dit= docList.begin();
    dit != docList.end();
    dit++)
{
  unsigned int dataid= *dit;
  const InvIndVec &fVec= X[dataid];
  for(InvIndVec::const_iterator fit= fVec.begin();
      fit != fVec.end();
      fit++)
  {
    ux[dataid] += u[fit->first+1]*(LRREAL)fit->second;
  }
}

/* Old II
   for (unsigned int f=1; f<featurenum; f++) {
     for (unsigned int i=0; i<X[f-1].size(); i++) {
       unsigned int dataid = X[f-1][i].first;
       if(docList.find(dataid) == docList.end()) continue;  //BK: TODO: merge-sort loop for speed & verify correctness
       double value = X[f-1][i].second;
       ux[dataid] += u[f]*value;
     }
   }
*/
// compute the data component of -u^T H u
//BK for (unsigned int i=0; i<datanum; i++){
for(UISet::const_iterator i= docList.begin();
    i != docList.end();
    i++)
    {
  minus_uHu +=
    (LRREAL)S[*i]*(LRREAL)sigma[*i]*((LRREAL)1.0-(LRREAL)sigma[*i])*ux[*i]*ux[*i];
    }
// add the regularization part
for (unsigned int f=0; f<featurenum; f++) {
  minus_uHu += (LRREAL)2.0*(LRREAL)LAMBDA[f]*u[f]*u[f];
}

// 2.7 update the new weight std::vector
    //
    // Shinjae addition: gu/minus_uHu can become NaN due to loss of
    // precision.  In this case, we stop iterating to avoid having W full
    // of NaNs.  TODO: Can this source of NaN be better isolated from
    // other possible sources?
    LRREAL tw= (gu/minus_uHu);
    if(isnan(tw)) break;
for (unsigned int f=0; f<featurenum; f++) {
  W[f] += tw*u[f];
}

// 2.8 update the old gradient

    // Incidently, this is decidedly faster than making grad and old_grad
    // pointers and just swapping them here.
old_grad = grad;      
  }

  // print out the weights W?
  if (VERBOSE >= 2) {
for (unsigned int f=0; f<featurenum; f++){
  if ((f % 3 == 0) && f > 0) std::cout << std::endl;
  std::cout << "W[" << f << "]=" << W[f] << " ";
}
std::cout << std::endl;
  }
}
