#ifndef __DEPEND_FUNCS__
#define __DEPEND_FUNCS__
#include <RcppArmadillo.h>

arma::mat construct_sigma1(const arma::vec &sig_all, const arma::vec &rho);
arma::mat fisher_z_trans(arma::mat z);
double mean_diff_nonzero(arma::mat x, unsigned int j);

#endif