// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "lvmcorr_omp.h"
#include <RcppArmadilloExtensions/sample.h>

// [[Rcpp::export]]
arma::uvec sample_xi(arma::mat x_covariates, arma::mat g,
                     arma::mat beta_tp, arma::mat alpha_tp, arma::mat beta_fp, arma::mat alpha_fp,
                     arma::mat ytp, arma::vec b_tp, arma::vec e_tp,
                     arma::mat yfp, arma::vec b_fp, arma::vec e_fp,
                     arma::mat e12){
  
  double val_neg_20 = 1 / ( 1 + std::exp(20));
  int N = ytp.n_rows;
  
  arma::mat m = arma::exp(x_covariates * g);
  m.each_col() /= arma::sum(m, 1);
  
  arma::mat m_tp = arma::zeros(N, 4);
  arma::mat m_fp = arma::zeros(N, 4);
#pragma omp parallel for num_threads(getlvmcorr_threads())
  for(unsigned int i=0;i<N;++i){
    arma::vec ytp_i = ytp.row(i).t();
    arma::uvec ytp_i_nonmis_loc = arma::find_finite(ytp_i);
    arma::vec ytp_i_nonmis = ytp_i(ytp_i_nonmis_loc);
    m_tp(i, 0) = arma::prod(ytp_i_nonmis * val_neg_20 + (1-ytp_i_nonmis) * (1-val_neg_20));
    m_tp(i, 1) = m_tp(i, 0);
    
    arma::vec tmp = beta_tp.row(i).t() + alpha_tp.row(i).t() * (arma::as_scalar(x_covariates.row(i) * b_tp) + e_tp(i));
    m_tp(i, 2) = arma::prod(arma::exp(tmp(ytp_i_nonmis_loc) % ytp_i_nonmis) /
      (1 + arma::exp(tmp(ytp_i_nonmis_loc))));
    tmp = beta_tp.row(i).t() + alpha_tp.row(i).t() * (arma::as_scalar(x_covariates.row(i) * b_tp) + e12(i,0));
    m_tp(i, 3) = arma::prod(arma::exp(tmp(ytp_i_nonmis_loc) % ytp_i_nonmis) /
      (1 + arma::exp(tmp(ytp_i_nonmis_loc))));
    
    arma::vec yfp_i = yfp.row(i).t();
    arma::uvec yfp_i_nonmis_loc = arma::find_finite(yfp_i);
    arma::vec yfp_i_nonmis = yfp_i(yfp_i_nonmis_loc);
    m_fp(i, 0) = arma::prod(yfp_i_nonmis * val_neg_20 + (1-yfp_i_nonmis) * (1-val_neg_20));
    m_fp(i, 2) = m_fp(i, 0);
    
    tmp = beta_fp.row(i).t() + alpha_fp.row(i).t() * (arma::as_scalar(x_covariates.row(i) * b_fp) + e_fp(i));
    m_fp(i, 1) = arma::prod(arma::exp(tmp(yfp_i_nonmis_loc) % yfp_i_nonmis) /
      (1 + arma::exp(tmp(yfp_i_nonmis_loc))));
    tmp = beta_fp.row(i).t() + alpha_fp.row(i).t() * (arma::as_scalar(x_covariates.row(i) * b_fp) + e12(i,1));
    m_fp(i, 3) = arma::prod(arma::exp(tmp(yfp_i_nonmis_loc) % yfp_i_nonmis) /
      (1 + arma::exp(tmp(yfp_i_nonmis_loc))));
  }
  //arma::cout << "im here again" << arma::endl;
  arma::mat prob_m = m % m_tp % m_fp;
  // prob_m.each_col() /= arma::sum(prob_m,1);
  // arma::mat cum_prob_m = arma::cumsum(prob_m, 1);
  // arma::vec randu = arma::randu(N);
  // arma::mat ind_m = arma::conv_to<arma::mat>::from(cum_prob_m < arma::repmat(randu, 1, 4));
  arma::uvec ress(N);
  arma::uvec xx = {1,2,3,4};
  for(int i=0;i<N;++i){
    arma::vec tmp = prob_m.row(i).t();
    ress(i) = arma::as_scalar(Rcpp::RcppArmadillo::sample(xx, 1, true, tmp));
  }
  return ress;
}
arma::uvec sample_xi_new(const arma::mat &x_covariates, const arma::mat &g,
                     const arma::mat &beta_tp, const arma::mat &alpha_tp, 
                     const arma::mat &beta_fp, const arma::mat &alpha_fp,
                     const arma::mat &ytp, const arma::vec &b_tp,
                     const arma::mat &yfp, const arma::vec &b_fp,
                     const arma::vec &ytp_fin, const arma::vec &u_tp,
                     const arma::vec &yfp_fin, const arma::vec &u_fp,
                     const arma::mat &e1234){
  
  double val_neg_20 = 1 / ( 1 + std::exp(20));
  int N = ytp.n_rows;

  arma::mat m = arma::exp(x_covariates * g);
  m.each_col() /= arma::sum(m, 1);
  
  arma::mat m_tp = arma::zeros(N, 4);
  arma::mat m_fp = arma::zeros(N, 4);
  for(unsigned int i=0;i<N;++i){
    arma::vec ytp_i = ytp.row(i).t();
    arma::uvec ytp_i_nonmis_loc = arma::find_finite(ytp_i);
    arma::vec ytp_i_nonmis = ytp_i(ytp_i_nonmis_loc);
    
    m_tp(i, 0) = arma::prod(ytp_i_nonmis * val_neg_20 + (1-ytp_i_nonmis) * (1-val_neg_20)) * 
      (ytp_fin(i) * val_neg_20 + (1-ytp_fin(i)) * (1-val_neg_20));
    m_tp(i, 1) = m_tp(i, 0);
    
    arma::vec tmp = beta_tp.row(i).t() + alpha_tp.row(i).t() * (arma::as_scalar(x_covariates.row(i) * b_tp) + e1234(i,0));
    tmp = Rcpp::pnorm5(Rcpp::NumericVector(tmp.begin(), tmp.end()), 0.0, 1.0, 1, 0);
    arma::vec tmp1 = tmp(ytp_i_nonmis_loc) % ytp_i_nonmis + (1 - tmp(ytp_i_nonmis_loc)) % (1 - ytp_i_nonmis);
    
    // double p1 = R::pnorm5(arma::as_scalar(x_covariates.row(i) * u_tp) + e1234(i, 2), 0.0, 1.0, 1, 0);
    double p1 = R::pnorm5(arma::as_scalar(x_covariates.row(i) * u_tp), 0.0, 1.0, 1, 0);
    m_tp(i, 2) = arma::prod(tmp1) * (ytp_fin(i) * p1 + (1- ytp_fin(i)) * (1-p1));
    m_tp(i, 3) = m_tp(i, 2);
    
    arma::vec yfp_i = yfp.row(i).t();
    arma::uvec yfp_i_nonmis_loc = arma::find_finite(yfp_i);
    arma::vec yfp_i_nonmis = yfp_i(yfp_i_nonmis_loc);
    m_fp(i, 0) = arma::prod(yfp_i_nonmis * val_neg_20 + (1-yfp_i_nonmis) * (1-val_neg_20))* 
      (yfp_fin(i) * val_neg_20 + (1-yfp_fin(i)) * (1-val_neg_20));
    m_fp(i, 2) = m_fp(i, 0);
    
    tmp = beta_fp.row(i).t() + alpha_fp.row(i).t() * (arma::as_scalar(x_covariates.row(i) * b_fp) + e1234(i, 1));
    tmp = Rcpp::pnorm5(Rcpp::NumericVector(tmp.begin(), tmp.end()), 0.0, 1.0, 1, 0);
    tmp1 = tmp(yfp_i_nonmis_loc) % yfp_i_nonmis + (1 - tmp(yfp_i_nonmis_loc)) % (1-yfp_i_nonmis);
    
    // double p2 = R::pnorm5(arma::as_scalar(x_covariates.row(i) * u_fp) + e1234(i, 3), 0.0, 1.0, 1, 0);
    double p2 = R::pnorm5(arma::as_scalar(x_covariates.row(i) * u_fp), 0.0, 1.0, 1, 0);
    m_fp(i, 1) = arma::prod(tmp1) * (yfp_fin(i) * p2 + (1- yfp_fin(i)) * (1- p2));
    m_fp(i, 3) = m_fp(i, 1);
  }
  //arma::cout << "im here again" << arma::endl;
  arma::mat prob_m = m % m_tp % m_fp;
  // prob_m.each_col() /= arma::sum(prob_m,1);
  // arma::mat cum_prob_m = arma::cumsum(prob_m, 1);
  // arma::vec randu = arma::randu(N);
  // arma::mat ind_m = arma::conv_to<arma::mat>::from(cum_prob_m < arma::repmat(randu, 1, 4));
  arma::uvec ress(N);
  arma::uvec xx = {1,2,3,4};
  for(int i=0;i<N;++i){
    arma::vec tmp = prob_m.row(i).t();
    ress(i) = arma::as_scalar(Rcpp::RcppArmadillo::sample(xx, 1, true, tmp));
  }
  return ress;
}
arma::uvec sample_xi_new1(const arma::mat &x_covariates, const arma::mat &g,
                         const arma::mat &beta_tp, const arma::mat &alpha_tp, 
                         const arma::mat &beta_fp, const arma::mat &alpha_fp,
                         const arma::mat &ytp, const arma::vec &b_tp,
                         const arma::mat &yfp, const arma::vec &b_fp,
                         const arma::vec &ytp_fin, const arma::vec &u_tp,
                         const arma::vec &yfp_fin, const arma::vec &u_fp,
                         const arma::mat &e1234){
  
  double log_p1_neg20 = -20;
  double log_p0_neg20 = -std::exp(-20);
  int N = ytp.n_rows;
  
  arma::mat m = x_covariates * g;
  // m.each_col() /= arma::sum(m, 1);
  
  arma::mat m_tp = arma::zeros(N, 4);
  arma::mat m_fp = arma::zeros(N, 4);
  for(unsigned int i=0;i<N;++i){
    arma::vec ytp_i = ytp.row(i).t();
    arma::uvec ytp_i_nonmis_loc = arma::find_finite(ytp_i);
    arma::vec ytp_i_nonmis = ytp_i(ytp_i_nonmis_loc);
    
    m_tp(i, 0) = arma::sum(ytp_i_nonmis * log_p1_neg20 + (1-ytp_i_nonmis) * log_p0_neg20) + 
      ytp_fin(i) * log_p1_neg20 + (1-ytp_fin(i)) * log_p0_neg20;
    m_tp(i, 1) = m_tp(i, 0);
    
    arma::vec tmp = beta_tp.row(i).t() + alpha_tp.row(i).t() * (arma::as_scalar(x_covariates.row(i) * b_tp) + e1234(i,0));
    arma::vec log_p1 = Rcpp::pnorm5(Rcpp::NumericVector(tmp.begin(), tmp.end()), 0.0, 1.0, 1, 1);
    arma::vec log_p0 = Rcpp::pnorm5(Rcpp::NumericVector(tmp.begin(), tmp.end()), 0.0, 1.0, 0, 1);
    arma::vec tmp1 = log_p1(ytp_i_nonmis_loc) % ytp_i_nonmis + log_p0(ytp_i_nonmis_loc) % (1 - ytp_i_nonmis);
    
    // double p1 = R::pnorm5(arma::as_scalar(x_covariates.row(i) * u_tp) + e1234(i, 2), 0.0, 1.0, 1, 0);
    double log_p1f = R::pnorm5(arma::as_scalar(x_covariates.row(i) * u_tp), 0.0, 1.0, 1, 1);
    double log_p0f = R::pnorm5(arma::as_scalar(x_covariates.row(i) * u_tp), 0.0, 1.0, 0, 1);
    m_tp(i, 2) = arma::sum(tmp1) + ytp_fin(i) * log_p1f + (1- ytp_fin(i)) * log_p0f;
    m_tp(i, 3) = m_tp(i, 2);
    
    arma::vec yfp_i = yfp.row(i).t();
    arma::uvec yfp_i_nonmis_loc = arma::find_finite(yfp_i);
    arma::vec yfp_i_nonmis = yfp_i(yfp_i_nonmis_loc);
    m_fp(i, 0) = arma::sum(yfp_i_nonmis * log_p1_neg20 + (1-yfp_i_nonmis) * log_p0_neg20) +
      yfp_fin(i) * log_p1_neg20 + (1-yfp_fin(i)) * log_p0_neg20;
    m_fp(i, 2) = m_fp(i, 0);
    
    tmp = beta_fp.row(i).t() + alpha_fp.row(i).t() * (arma::as_scalar(x_covariates.row(i) * b_fp) + e1234(i, 1));
    log_p1 = Rcpp::pnorm5(Rcpp::NumericVector(tmp.begin(), tmp.end()), 0.0, 1.0, 1, 1);
    log_p0 = Rcpp::pnorm5(Rcpp::NumericVector(tmp.begin(), tmp.end()), 0.0, 1.0, 0, 1);
    tmp1 = log_p1(yfp_i_nonmis_loc) % yfp_i_nonmis + log_p0(yfp_i_nonmis_loc) % (1-yfp_i_nonmis);
    
    // double p2 = R::pnorm5(arma::as_scalar(x_covariates.row(i) * u_fp) + e1234(i, 3), 0.0, 1.0, 1, 0);
    log_p1f = R::pnorm5(arma::as_scalar(x_covariates.row(i) * u_fp), 0.0, 1.0, 1, 1);
    log_p0f = R::pnorm5(arma::as_scalar(x_covariates.row(i) * u_fp), 0.0, 1.0, 0, 1);
    m_fp(i, 1) = arma::sum(tmp1) + yfp_fin(i) * log_p1f + (1- yfp_fin(i)) * log_p0f;
    m_fp(i, 3) = m_fp(i, 1);
  }
  //arma::cout << "im here again" << arma::endl;
  arma::mat prob_m = m + m_tp + m_fp;
  prob_m.each_col() -= arma::max(prob_m,1);
  prob_m = arma::exp(prob_m);
  // prob_m.each_col() /= arma::sum(prob_m,1);
  // arma::mat cum_prob_m = arma::cumsum(prob_m, 1);
  // arma::vec randu = arma::randu(N);
  // arma::mat ind_m = arma::conv_to<arma::mat>::from(cum_prob_m < arma::repmat(randu, 1, 4));
  arma::uvec ress(N);
  Rcpp::NumericVector xx = {1,2,3,4};
  for(unsigned int i=0;i<N;++i){
    arma::vec temp = prob_m.row(i).t();
    Rcpp::NumericVector res = Rcpp::sample(xx, 1, 0, Rcpp::NumericVector(temp.begin(), temp.end()));
    ress(i) = res(0);
  }
  return ress;
}
//' @export
// [[Rcpp::export]]
arma::mat print_prob_m1(const arma::mat &x_covariates, const arma::mat &g,
                          const arma::mat &beta_tp, const arma::mat &alpha_tp, 
                          const arma::mat &beta_fp, const arma::mat &alpha_fp,
                          const arma::mat &ytp, const arma::vec &b_tp,
                          const arma::mat &yfp, const arma::vec &b_fp,
                          const arma::vec &ytp_fin, const arma::vec &u_tp,
                          const arma::vec &yfp_fin, const arma::vec &u_fp,
                          const arma::mat &e1234){
  
  double log_p1_neg20 = -20;
  double log_p0_neg20 = -std::exp(-20);
  int N = ytp.n_rows;
  
  arma::mat m = x_covariates * g;
  // m.each_col() /= arma::sum(m, 1);
  
  arma::mat m_tp = arma::zeros(N, 4);
  arma::mat m_fp = arma::zeros(N, 4);
  for(unsigned int i=0;i<N;++i){
    arma::vec ytp_i = ytp.row(i).t();
    arma::uvec ytp_i_nonmis_loc = arma::find_finite(ytp_i);
    arma::vec ytp_i_nonmis = ytp_i(ytp_i_nonmis_loc);
    
    m_tp(i, 0) = arma::sum(ytp_i_nonmis * log_p1_neg20 + (1-ytp_i_nonmis) * log_p0_neg20) + 
      ytp_fin(i) * log_p1_neg20 + (1-ytp_fin(i)) * log_p0_neg20;
    m_tp(i, 1) = m_tp(i, 0);
    
    arma::vec tmp = beta_tp.row(i).t() + alpha_tp.row(i).t() * (arma::as_scalar(x_covariates.row(i) * b_tp) + e1234(i,0));
    arma::vec log_p1 = Rcpp::pnorm5(Rcpp::NumericVector(tmp.begin(), tmp.end()), 0.0, 1.0, 1, 1);
    arma::vec log_p0 = Rcpp::pnorm5(Rcpp::NumericVector(tmp.begin(), tmp.end()), 0.0, 1.0, 0, 1);
    arma::vec tmp1 = log_p1(ytp_i_nonmis_loc) % ytp_i_nonmis + log_p0(ytp_i_nonmis_loc) % (1 - ytp_i_nonmis);
    
    // double p1 = R::pnorm5(arma::as_scalar(x_covariates.row(i) * u_tp) + e1234(i, 2), 0.0, 1.0, 1, 0);
    double log_p1f = R::pnorm5(arma::as_scalar(x_covariates.row(i) * u_tp), 0.0, 1.0, 1, 1);
    double log_p0f = R::pnorm5(arma::as_scalar(x_covariates.row(i) * u_tp), 0.0, 1.0, 0, 1);
    m_tp(i, 2) = arma::sum(tmp1) + ytp_fin(i) * log_p1f + (1- ytp_fin(i)) * log_p0f;
    m_tp(i, 3) = m_tp(i, 2);
    
    arma::vec yfp_i = yfp.row(i).t();
    arma::uvec yfp_i_nonmis_loc = arma::find_finite(yfp_i);
    arma::vec yfp_i_nonmis = yfp_i(yfp_i_nonmis_loc);
    m_fp(i, 0) = arma::sum(yfp_i_nonmis * log_p1_neg20 + (1-yfp_i_nonmis) * log_p0_neg20) +
      yfp_fin(i) * log_p1_neg20 + (1-yfp_fin(i)) * log_p0_neg20;
    m_fp(i, 2) = m_fp(i, 0);
    
    tmp = beta_fp.row(i).t() + alpha_fp.row(i).t() * (arma::as_scalar(x_covariates.row(i) * b_fp) + e1234(i, 1));
    log_p1 = Rcpp::pnorm5(Rcpp::NumericVector(tmp.begin(), tmp.end()), 0.0, 1.0, 1, 1);
    log_p0 = Rcpp::pnorm5(Rcpp::NumericVector(tmp.begin(), tmp.end()), 0.0, 1.0, 0, 1);
    tmp1 = log_p1(yfp_i_nonmis_loc) % yfp_i_nonmis + log_p0(yfp_i_nonmis_loc) % (1-yfp_i_nonmis);
    
    // double p2 = R::pnorm5(arma::as_scalar(x_covariates.row(i) * u_fp) + e1234(i, 3), 0.0, 1.0, 1, 0);
    log_p1f = R::pnorm5(arma::as_scalar(x_covariates.row(i) * u_fp), 0.0, 1.0, 1, 1);
    log_p0f = R::pnorm5(arma::as_scalar(x_covariates.row(i) * u_fp), 0.0, 1.0, 0, 1);
    m_fp(i, 1) = arma::sum(tmp1) + yfp_fin(i) * log_p1f + (1- yfp_fin(i)) * log_p0f;
    m_fp(i, 3) = m_fp(i, 1);
  }
  //arma::cout << "im here again" << arma::endl;
  arma::mat prob_m = m + m_tp + m_fp;
  Rcpp::Rcout << "m1: " << m.row(0) << "m_tp1: " << m_tp.row(0) <<
    "m_fp1: " << m_fp.row(0) << std::endl;
  prob_m.each_col() -= arma::max(prob_m,1);
  prob_m = arma::exp(prob_m);
  prob_m.each_col() /= arma::sum(prob_m,1);
  // arma::mat cum_prob_m = arma::cumsum(prob_m, 1);
  // arma::vec randu = arma::randu(N);
  // arma::mat ind_m = arma::conv_to<arma::mat>::from(cum_prob_m < arma::repmat(randu, 1, 4));
  // arma::uvec ress(N);
  // Rcpp::NumericVector xx = {1,2,3,4};
  // for(unsigned int i=0;i<N;++i){
  //   arma::vec temp = prob_m.row(i).t();
  //   Rcpp::NumericVector res = Rcpp::sample(xx, 1, 0, Rcpp::NumericVector(temp.begin(), temp.end()));
  //   ress(i) = res(0);
  // }
  return prob_m;
}
//' @export
// [[Rcpp::export]]
arma::mat print_prob_m(const arma::mat &x_covariates, const arma::mat &g,
                         const arma::mat &beta_tp, const arma::mat &alpha_tp, 
                         const arma::mat &beta_fp, const arma::mat &alpha_fp,
                         const arma::mat &ytp, const arma::vec &b_tp,
                         const arma::mat &yfp, const arma::vec &b_fp,
                         const arma::vec &ytp_fin, const arma::vec &u_tp,
                         const arma::vec &yfp_fin, const arma::vec &u_fp,
                         const arma::mat &e1234){
  
  double val_neg_20 = 1 / ( 1 + std::exp(20));
  int N = ytp.n_rows;
  
  arma::mat m = arma::exp(x_covariates * g);
  m.each_col() /= arma::sum(m, 1);
  arma::mat m_tp = arma::zeros(N, 4);
  arma::mat m_fp = arma::zeros(N, 4);
  for(unsigned int i=0;i<N;++i){
    arma::vec ytp_i = ytp.row(i).t();
    arma::uvec ytp_i_nonmis_loc = arma::find_finite(ytp_i);
    arma::vec ytp_i_nonmis = ytp_i(ytp_i_nonmis_loc);
    
    m_tp(i, 0) = arma::prod(ytp_i_nonmis * val_neg_20 + (1-ytp_i_nonmis) * (1-val_neg_20)) * 
      (ytp_fin(i) * val_neg_20 + (1-ytp_fin(i)) * (1-val_neg_20));
    m_tp(i, 1) = m_tp(i, 0);
    
    arma::vec tmp = beta_tp.row(i).t() + alpha_tp.row(i).t() * (arma::as_scalar(x_covariates.row(i) * b_tp) + e1234(i,0));
    tmp = Rcpp::pnorm5(Rcpp::NumericVector(tmp.begin(), tmp.end()), 0.0, 1.0, 1, 0);
    arma::vec tmp1 = tmp(ytp_i_nonmis_loc) % ytp_i_nonmis + (1 - tmp(ytp_i_nonmis_loc)) % (1 - ytp_i_nonmis);
    
    // double p1 = R::pnorm5(arma::as_scalar(x_covariates.row(i) * u_tp) + e1234(i, 2), 0.0, 1.0, 1, 0);
    double p1 = R::pnorm5(arma::as_scalar(x_covariates.row(i) * u_tp), 0.0, 1.0, 1, 0);
    m_tp(i, 2) = arma::prod(tmp1) * (ytp_fin(i) * p1 + (1- ytp_fin(i)) * (1-p1));
    m_tp(i, 3) = m_tp(i, 2);
    
    arma::vec yfp_i = yfp.row(i).t();
    arma::uvec yfp_i_nonmis_loc = arma::find_finite(yfp_i);
    arma::vec yfp_i_nonmis = yfp_i(yfp_i_nonmis_loc);
    m_fp(i, 0) = arma::prod(yfp_i_nonmis * val_neg_20 + (1-yfp_i_nonmis) * (1-val_neg_20))* 
      (yfp_fin(i) * val_neg_20 + (1-yfp_fin(i)) * (1-val_neg_20));
    m_fp(i, 2) = m_fp(i, 0);
    
    tmp = beta_fp.row(i).t() + alpha_fp.row(i).t() * (arma::as_scalar(x_covariates.row(i) * b_fp) + e1234(i, 1));
    tmp = Rcpp::pnorm5(Rcpp::NumericVector(tmp.begin(), tmp.end()), 0.0, 1.0, 1, 0);
    tmp1 = tmp(yfp_i_nonmis_loc) % yfp_i_nonmis + (1 - tmp(yfp_i_nonmis_loc)) % (1-yfp_i_nonmis);
    
    // double p2 = R::pnorm5(arma::as_scalar(x_covariates.row(i) * u_fp) + e1234(i, 3), 0.0, 1.0, 1, 0);
    double p2 = R::pnorm5(arma::as_scalar(x_covariates.row(i) * u_fp), 0.0, 1.0, 1, 0);
    m_fp(i, 1) = arma::prod(tmp1) * (yfp_fin(i) * p2 + (1- yfp_fin(i)) * (1- p2));
    m_fp(i, 3) = m_fp(i, 1);
  }
  //arma::cout << "im here again" << arma::endl;
  Rcpp::Rcout << "m1: " << m.row(0) << "m_tp1: " << m_tp.row(0) <<
    "m_fp1: " << m_fp.row(0) << std::endl;
  arma::mat prob_m = m % m_tp % m_fp;
  prob_m.each_col() /= arma::sum(prob_m,1);
  // arma::mat cum_prob_m = arma::cumsum(prob_m, 1);
  // arma::vec randu = arma::randu(N);
  // arma::mat ind_m = arma::conv_to<arma::mat>::from(cum_prob_m < arma::repmat(randu, 1, 4));
  // arma::uvec ress(N);
  // arma::uvec xx = {1,2,3,4};
  // for(int i=0;i<N;++i){
  //   arma::vec tmp = prob_m.row(i).t();
  //   ress(i) = arma::as_scalar(Rcpp::RcppArmadillo::sample(xx, 1, true, tmp));
  // }
  return prob_m;
}