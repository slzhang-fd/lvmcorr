// [[Rcpp::depends(RcppArmadillo, RcppDist, RcppTN)]]
#include <RcppDist.h>
#include "arms_ori.h"
#include "lvmcorr_omp.h"
#include <random>
#include <RcppTN.h>

struct log_ei_tp_fp_params{
  arma::vec ytp_fp_i;
  double sig2_tp_fp;
  arma::vec beta_tp_fp_i;
  arma::vec alpha_tp_fp_i;
  arma::rowvec x_covariates_i;
  arma::vec b_tp_fp;
};

// self-defined log_sum_exp to prevent possible underflow / overflow
arma::vec log_sum_exp1(arma::vec tmp){
  arma::vec tmp_zeros = arma::zeros(tmp.n_elem);
  arma::vec max_tmp_0 = arma::max(tmp_zeros, tmp);
  return max_tmp_0 + arma::log(arma::exp(- max_tmp_0) + arma::exp(tmp - max_tmp_0));
}

double log_ei_tp_fp_c(double x, void* params){
  struct log_ei_tp_fp_params *d;
  d = static_cast<struct log_ei_tp_fp_params *> (params);
  
  double res = - 0.5 * x * x / d->sig2_tp_fp;
  
  arma::vec tmp = d->beta_tp_fp_i + d->alpha_tp_fp_i * (arma::as_scalar((d->x_covariates_i)*(d->b_tp_fp)) + x);
  //tmp = 1 / (1+arma::exp(-tmp));
  //arma::vec tmp1 = arma::log(d->ytp_fp_i % tmp + (1-d->ytp_fp_i) % (1-tmp));
  // arma::vec tmp1 = tmp % d->ytp_fp_i - log_sum_exp1(tmp);
  arma::vec tmp1 = tmp % d->ytp_fp_i - arma::log(1+arma::exp(tmp));
  //res += arma::accu(tmp1(arma::find_finite(tmp1)));
  res += arma::accu(tmp1);
  return res;
}

// [[Rcpp::export]]
arma::vec sample_e_tp_c(arma::vec e_tp, arma::mat ytp, arma::uvec xi, arma::vec b_tp,
                         double sig2_tp, arma::mat beta_tp, arma::mat alpha_tp, arma::mat x_covariates){
  int err, ninit = 5, npoint = 100, nsamp = 1, ncent = 0 ;
  int neval;
  double xinit[10]={-4.0, -1.0, 0.0, 1.0, 4.0}, xl = -100.0, xr = 100.0;
  double xcent[10], qcent[10] = {5., 30., 70., 95.};
  double convex = 1.;
  int dometrop = 1;
  
  arma::uvec xi_124_loc = arma::find(xi==1 || xi==2 || xi==4);
  arma::uvec xi_3_loc = arma::find(xi==3);
  e_tp(xi_124_loc) = arma::randn(xi_124_loc.n_elem) * std::sqrt(sig2_tp);
#pragma omp parallel for num_threads(getlvmcorr_threads())
  for(unsigned int n=0;n<xi_3_loc.n_elem;++n){
    double xsamp[100];
    int ii = xi_3_loc(n);
    double xprev = e_tp(ii);
    log_ei_tp_fp_params log_ei_tp_data;
    log_ei_tp_data.alpha_tp_fp_i = alpha_tp.row(ii).t();
    log_ei_tp_data.beta_tp_fp_i = beta_tp.row(ii).t();
    log_ei_tp_data.b_tp_fp = b_tp;
    log_ei_tp_data.sig2_tp_fp = sig2_tp;
    log_ei_tp_data.x_covariates_i = x_covariates.row(ii);
    log_ei_tp_data.ytp_fp_i = ytp.row(ii).t();

    err = arms(xinit,ninit,&xl,&xr,log_ei_tp_fp_c,&log_ei_tp_data,&convex,
               npoint,dometrop,&xprev,xsamp,nsamp,qcent,xcent,ncent,&neval);
    if(err>0){
      Rprintf("error code: %d", err);
      Rcpp::stop("\n");
    }
    e_tp(ii) = xsamp[0];
  }
  return e_tp;
}

// [[Rcpp::export]]
arma::vec sample_e_fp_c(arma::vec e_fp, arma::mat yfp, arma::uvec xi, arma::vec b_fp,
                        double sig2_fp, arma::mat beta_fp, arma::mat alpha_fp, arma::mat x_covariates){
  int err, ninit = 5, npoint = 100, nsamp = 1, ncent = 0 ;
  int neval;
  double xinit[10]={-4.0, -1.0, 0.0, 1.0, 4.0}, xl = -100.0, xr = 100.0;
  double xcent[10], qcent[10] = {5., 30., 70., 95.};
  double convex = 1.;
  int dometrop = 1;
  
  arma::uvec xi_134_loc = arma::find(xi==1 || xi==3 || xi==4);
  arma::uvec xi_2_loc = arma::find(xi==2);
  e_fp(xi_134_loc) = arma::randn(xi_134_loc.n_elem) * std::sqrt(sig2_fp);
#pragma omp parallel for num_threads(getlvmcorr_threads())
  for(unsigned int n=0;n<xi_2_loc.n_elem;++n){
    double xsamp[100];
    int ii = xi_2_loc(n);
    double xprev = e_fp(ii);
    log_ei_tp_fp_params log_ei_fp_data;
    log_ei_fp_data.alpha_tp_fp_i = alpha_fp.row(ii).t();
    log_ei_fp_data.beta_tp_fp_i = beta_fp.row(ii).t();
    log_ei_fp_data.b_tp_fp = b_fp;
    log_ei_fp_data.sig2_tp_fp = sig2_fp;
    log_ei_fp_data.x_covariates_i = x_covariates.row(ii);
    log_ei_fp_data.ytp_fp_i = yfp.row(ii).t();
    
    err = arms(xinit,ninit,&xl,&xr,log_ei_tp_fp_c,&log_ei_fp_data,&convex,
               npoint,dometrop,&xprev,xsamp,nsamp,qcent,xcent,ncent,&neval);
    if(err>0){
      Rprintf("error code: %d", err);
      Rcpp::stop("\n");
    }
    e_fp(ii) = xsamp[0];
  }
  return e_fp;
}

struct log_e12_i_params{
  arma::vec ytp_fp_i;
  arma::vec beta_tp_fp_i;
  arma::vec alpha_tp_fp_i;
  double sig2_tp;
  double sig2_fp;
  arma::vec b_tp_fp;
  arma::rowvec x_covariates_i;
  double rho;
  double e21_i;
};
double log_e1_i_c(double x, void* params){
  struct log_e12_i_params *d;
  d = static_cast<struct log_e12_i_params *> (params);
  
  double res = - 0.5 / ( 1 - d->rho * d->rho )*
    ( x*x / d->sig2_tp - 2*d->rho * x * d->e21_i / std::sqrt(d->sig2_tp * d->sig2_fp) );
  
  arma::vec tmp = d->beta_tp_fp_i + d->alpha_tp_fp_i * (arma::as_scalar(d->x_covariates_i * d->b_tp_fp) + x);
  //tmp = 1 / (1+arma::exp(-tmp));
  //arma::vec tmp1 = arma::log(d->ytp_fp_i % tmp + (1-d->ytp_fp_i) % (1-tmp));
  //arma::vec tmp1 = tmp % d->ytp_fp_i - log_sum_exp1(tmp);
  arma::vec tmp1 = tmp % d->ytp_fp_i - arma::log(1+arma::exp(tmp));
  //res += arma::accu(tmp1(arma::find_finite(tmp1)));
  res += arma::accu(tmp1);
  return res;
}
double log_e2_i_c(double x, void* params){
  struct log_e12_i_params *d;
  d = static_cast<struct log_e12_i_params *> (params);
  
  double res = - 0.5 / ( 1 - d->rho * d->rho )*
    ( x*x / d->sig2_fp - 2*d->rho * x * d->e21_i / std::sqrt(d->sig2_tp * d->sig2_fp) );
  
  arma::vec tmp = d->beta_tp_fp_i + d->alpha_tp_fp_i * (arma::as_scalar(d->x_covariates_i * d->b_tp_fp) + x);
  // tmp = 1 / (1+arma::exp(-tmp));
  // arma::vec tmp1 = arma::log(d->ytp_fp_i % tmp + (1-d->ytp_fp_i) % (1-tmp));
  //arma::vec tmp1 = tmp % d->ytp_fp_i - log_sum_exp1(tmp);
  arma::vec tmp1 = tmp % d->ytp_fp_i - arma::log(1+arma::exp(tmp));
  
  //res += arma::accu(tmp1(arma::find_finite(tmp1)));
  res += arma::accu(tmp1);
  return res;
}
// [[Rcpp::export]]
arma::mat sample_e12_c(arma::mat e12, arma::mat ytp, arma::mat yfp, arma::uvec xi,
                       arma::mat beta_tp, arma::mat alpha_tp, arma::mat beta_fp, arma::mat alpha_fp,
                       double sig2_tp, double sig2_fp, double rho,
                       arma::vec b_tp, arma::vec b_fp, arma::mat x_covariates){
  int err, ninit = 5, npoint = 100, nsamp = 1, ncent = 0 ;
  int neval;
  double xinit[10]={-4.0, -1.0, 0.0, 1.0, 4.0}, xl = -100.0, xr = 100.0;
  double xcent[10], qcent[10] = {5., 30., 70., 95.};
  double convex = 1.;
  int dometrop = 1;
  
  arma::uvec xi_123_loc = arma::find(xi==1 || xi==2 || xi==3);
  arma::uvec xi_4_loc = arma::find(xi==4);
  arma::mat Sigma(2,2);
  Sigma(0,0) = sig2_tp;
  Sigma(1,1) = sig2_fp;
  Sigma(0,1) = rho * std::sqrt(sig2_tp * sig2_fp);
  Sigma(1,0) = Sigma(0,1);
  e12.rows(xi_123_loc) = arma::mvnrnd(arma::zeros(2), Sigma, xi_123_loc.n_elem).t();
#pragma omp parallel for num_threads(getlvmcorr_threads())
  for(unsigned int n=0;n<xi_4_loc.n_elem;++n){
    double xsamp[100];
    int ii = xi_4_loc(n);
    // sample e1_i
    double xprev = e12(ii, 0);
    log_e12_i_params log_e12_i_data;
    log_e12_i_data.ytp_fp_i = ytp.row(ii).t();
    log_e12_i_data.alpha_tp_fp_i = alpha_tp.row(ii).t();
    log_e12_i_data.beta_tp_fp_i = beta_tp.row(ii).t();
    log_e12_i_data.sig2_tp = sig2_tp;
    log_e12_i_data.sig2_fp = sig2_fp;
    log_e12_i_data.b_tp_fp = b_tp;
    log_e12_i_data.x_covariates_i = x_covariates.row(ii);
    log_e12_i_data.rho = rho;
    log_e12_i_data.e21_i = e12(ii, 1);
    err = arms(xinit,ninit,&xl,&xr,log_e1_i_c,&log_e12_i_data,&convex,
               npoint,dometrop,&xprev,xsamp,nsamp,qcent,xcent,ncent,&neval);
    if(err>0){
      Rprintf("error code: %d", err);
      Rcpp::stop("\n");
    }
    e12(ii, 0) = xsamp[0];
    // sample e2_i
    xprev = e12(ii, 1);
    log_e12_i_data.ytp_fp_i = yfp.row(ii).t();
    log_e12_i_data.alpha_tp_fp_i = alpha_fp.row(ii).t();
    log_e12_i_data.beta_tp_fp_i = beta_fp.row(ii).t();
    log_e12_i_data.b_tp_fp = b_fp;
    log_e12_i_data.e21_i = e12(ii, 0);
    err = arms(xinit,ninit,&xl,&xr,log_e2_i_c,&log_e12_i_data,&convex,
               npoint,dometrop,&xprev,xsamp,nsamp,qcent,xcent,ncent,&neval);
    if(err>0){
      Rprintf("error code: %d", err);
      Rcpp::stop("\n");
    }
    e12(ii, 1) = xsamp[0];
  }
  return e12;
}
//' @export 
// [[Rcpp::export]]
double test1(arma::vec tmp, arma::vec y_i){
  // return r_truncnorm(mu, sigma, a, b);
  tmp = Rcpp::pnorm(Rcpp::NumericVector(tmp.begin(),tmp.end()), 0.0, 1.0, 1, 0);
  arma::vec tmp1 = arma::log(y_i % tmp + (1 - y_i) % (1-tmp));
  return arma::accu(tmp1(arma::find_finite(tmp1)));
}

//' @export 
// [[Rcpp::export]]
double test2(arma::vec tmp, arma::vec y_i){
  // return RcppTN::rtn1(mu, sigma, a, b);
  arma::vec log_p1 = Rcpp::pnorm(Rcpp::NumericVector(tmp.begin(), tmp.end()), 0.0, 1.0, 1, 1);
  arma::vec log_p2 = Rcpp::pnorm(Rcpp::NumericVector(tmp.begin(), tmp.end()), 0.0, 1.0, 0, 1);
  arma::vec tmp1 = y_i % log_p1 + (1 - y_i) % log_p2;
  return arma::accu(tmp1);
}

//' @export 
// [[Rcpp::export]]
arma::vec test13(arma::vec tmp, int tail){
  return Rcpp::pnorm5(Rcpp::NumericVector(tmp.begin(), tmp.end()), 0.0, 1.0, tail, 1);
}
  