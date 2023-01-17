
// [[Rcpp::depends(RcppArmadillo)]]
#include "depend_funcs.h"

//' @export
// [[Rcpp::export]]
arma::mat construct_sigma1(const arma::vec &sig_all, const arma::vec &rho){
  double sig_tp = sig_all(0);
  double sig_fp = sig_all(1);
  arma::mat Sigma(4,4);
  Sigma(0,0) = sig_tp * sig_tp;
  Sigma(0,1) = rho(0) * sig_tp * sig_fp;
  Sigma(0,2) = rho(1) * sig_tp;
  Sigma(0,3) = rho(2) * sig_tp;
  Sigma(1,1) = sig_fp * sig_fp;
  Sigma(1,2) = rho(3) * sig_fp;
  Sigma(1,3) = rho(4) * sig_fp;
  Sigma(2,2) = 1;
  Sigma(2,3) = rho(5);
  Sigma(3,3) = 1;
  Sigma(1,0) = Sigma(0,1);
  Sigma(2,0) = Sigma(0,2);
  Sigma(3,0) = Sigma(0,3);
  Sigma(2,1) = Sigma(1,2);
  Sigma(3,1) = Sigma(1,3);
  Sigma(3,2) = Sigma(2,3);
  return Sigma;
}

//' @export
// [[Rcpp::export]]
arma::mat fisher_z_trans(arma::mat z){
  // arma::mat tmp = arma::exp(2*z);
  // return (tmp-1)/(tmp+1);
  return z;
}

//' @export 
// [[Rcpp::export]]
double mean_diff_nonzero(arma::mat x, unsigned int j){
  arma::vec tmp = x.col(j);
  arma::vec tmp1 = arma::diff(tmp);
  tmp1 = arma::nonzeros(tmp1);
  return tmp1.n_rows / (x.n_rows - 1.0);
}