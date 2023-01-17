#include <RcppDist.h>
#include <RcppTN.h>
#include "arms_ori.h"
#include "lvmcorr_omp.h"
#include "progress.hpp"
#include "eta_progress_bar.hpp"
#include "depend_funcs.h"

// [[Rcpp::depends(RcppArmadillo, RcppDist, RcppTN)]]
struct log_uk_tp_param{
  int kk;
  arma::vec u_tp;
  arma::vec ytp_fin;
  double sig2_u_tp;
  arma::mat x_covariates;
};

double log_uk_tp(double x, void* params){
  struct log_uk_tp_param *d;
  d = static_cast<struct log_uk_tp_param *> (params);
  arma::vec u_tp = d->u_tp;
  u_tp(d->kk) = x;
  
  // u_tp_k prior log likelihood
  double res = -0.5 * x * x / d->sig2_u_tp;
  
  // log likelihood
  arma::vec tmp = d->x_covariates * u_tp;
  res += arma::accu( tmp % d->ytp_fin - arma::log( 1 + arma::exp(tmp) ) );
  return res;
}
//' @export
// [[Rcpp::export]]
arma::vec sample_u_c(arma::vec ytp_fin, arma::mat x_covariates, 
                        arma::vec u_tp, double sig2_u_tp){
  int err, ninit = 4, npoint = 100, nsamp = 1, ncent = 0;
  int neval;
  double xl = -100.0, xr = 100.0;
  double xinit[10]={-6.0, -2.0, 2.0, 6.0}, xsamp[100], xcent[10], qcent[10] = {5., 30., 70., 95.};
  double convex = 1.;
  int dometrop = 1;
  double xprev = 0.0;
  
  log_uk_tp_param log_uk_tp_data;
  log_uk_tp_data.sig2_u_tp = sig2_u_tp;
  log_uk_tp_data.u_tp = u_tp;
  log_uk_tp_data.x_covariates = x_covariates;
  log_uk_tp_data.ytp_fin = ytp_fin;
  
  int K = u_tp.n_rows;
  for(auto k=0;k<K;++k){
    log_uk_tp_data.kk = k;
    xprev = log_uk_tp_data.u_tp(k);
    err = arms(xinit,ninit,&xl,&xr,log_uk_tp,&log_uk_tp_data,&convex,
               npoint,dometrop,&xprev,xsamp,nsamp,qcent,xcent,ncent,&neval);
    if(err>0){
      Rprintf("error code: %d", err);
      Rcpp::stop("\n");
    }
    //Rprintf("jj=%d, log_b_tp_jj=%f",jj, log_bj_tp())
    log_uk_tp_data.u_tp(k) = xsamp[0];
  }
  return log_uk_tp_data.u_tp;
}

//' @export
// [[Rcpp::export]]
double sample_sig2_u(arma::vec u_tp){
  double alpha_0 = 1.5;
  double beta_0 = 0.01;
  return 1 / arma::randg(arma::distr_param(alpha_0 + 0.5*u_tp.n_elem, 1 / (beta_0 + 0.5 * arma::accu(arma::square(u_tp))))); 
}

//' @export
// [[Rcpp::export]]
arma::vec test_inv_gamma(double alpha, double beta, int nn){
  return 1 / arma::randg(nn, arma::distr_param(alpha, 1 / beta));
}

//' @export
// [[Rcpp::export]]
arma::vec my_trunc_norm(arma::vec mu, arma::vec direction){
  int n = mu.n_elem;
  arma::vec res(n);
  double tmp;
  for(auto i=0;i<n;++i){
    tmp = arma::randn() + mu(i);
    if(direction(i)){
      while(tmp < 0){
        tmp = arma::randn() + mu(i);
      }
      res(i) = tmp;
    }
    else{
      while(tmp >= 0){
        tmp = arma::randn() + mu(i);
      }
      res(i) = tmp;
    }
  }
  return res;
}

// [[Rcpp::export]]
arma::vec sample_u_tp_new(arma::mat Sigma, arma::mat e1234, arma::vec u_tp, const arma::mat &x_covariates){
  double sig2_0 = 100.0;
  arma::vec phi_tp = x_covariates * u_tp + e1234.col(2);
  arma::uvec ind_e3 = {0,1,3};
  arma::vec sigma_e3_12(3);
  sigma_e3_12(0) = Sigma(0,2);
  sigma_e3_12(1) = Sigma(1,2);
  sigma_e3_12(2) = Sigma(3,2);
  arma::mat sigma_e3_22_inv = arma::inv_sympd(Sigma(ind_e3,ind_e3));
  double sig2_e3 = Sigma(2,2) - arma::as_scalar(sigma_e3_12.t() * sigma_e3_22_inv * sigma_e3_12);
  arma::mat S_n = x_covariates.t() * x_covariates / sig2_e3;
  S_n.diag() += 1.0 / sig2_0;
  S_n = arma::inv_sympd(S_n);
  arma::vec mu_n = S_n * x_covariates.t() * (phi_tp - e1234.cols(ind_e3) * sigma_e3_22_inv * sigma_e3_12) / sig2_e3;
  return arma::mvnrnd(mu_n, S_n);
}
// [[Rcpp::export]]
arma::vec sample_u_fp_new(arma::mat Sigma, arma::mat e1234, arma::vec u_fp, const arma::mat &x_covariates){
  double sig2_0 = 100.0;
  arma::vec phi_fp = x_covariates * u_fp + e1234.col(3);
  arma::uvec ind_e4 = {0,1,2};
  arma::vec sigma_e4_12(3);
  sigma_e4_12(0) = Sigma(0,3);
  sigma_e4_12(1) = Sigma(1,3);
  sigma_e4_12(2) = Sigma(2,3);
  arma::mat sigma_e4_22_inv = arma::inv_sympd(Sigma(ind_e4,ind_e4));
  double sig2_e4 = Sigma(3,3) - arma::as_scalar(sigma_e4_12.t() * sigma_e4_22_inv * sigma_e4_12);
  arma::mat S_n = x_covariates.t() * x_covariates / sig2_e4;
  S_n.diag() += 1.0 / sig2_0;
  S_n = arma::inv_sympd(S_n);
  arma::vec mu_n = S_n * x_covariates.t() * (phi_fp - e1234.cols(ind_e4) * sigma_e4_22_inv * sigma_e4_12) / sig2_e4;
  return arma::mvnrnd(mu_n, S_n);
}
arma::vec sample_u_new1(arma::mat Sigma1, arma::mat Sigma2, arma::mat e1234, arma::vec u_tp, 
                        const arma::mat &x_covariates, arma::vec female, bool tp_flag){
  double sig2_0 = 100.0;
  arma::uvec female_loc = arma::find(female==1);
  arma::uvec male_loc = arma::find(female==0);
  arma::mat x_cov_female = x_covariates.rows(female_loc);
  arma::mat x_cov_male = x_covariates.rows(male_loc);
  arma::mat e1234_female = e1234.rows(female_loc);
  arma::mat e1234_male = e1234.rows(male_loc);
  
  arma::vec phi_tp_female, phi_tp_male;
  arma::uvec ind_e3(3);
  arma::vec sigma_e3_12_female(3);
  arma::mat sigma_e3_22_inv_female;
  double sig2_e3_female;
  arma::vec sigma_e3_12_male(3);
  arma::mat sigma_e3_22_inv_male;
  double sig2_e3_male;
  if(tp_flag){
    ind_e3 = {0,1,3};
    sigma_e3_12_female(0) = Sigma1(0,2);
    sigma_e3_12_female(1) = Sigma1(1,2);
    sigma_e3_12_female(2) = Sigma1(3,2);
    phi_tp_female = x_cov_female * u_tp + e1234_female.col(2);
    sigma_e3_22_inv_female = arma::inv_sympd(Sigma1(ind_e3,ind_e3));
    sig2_e3_female = Sigma1(2,2) - arma::as_scalar(sigma_e3_12_female.t() * sigma_e3_22_inv_female * sigma_e3_12_female);
    sigma_e3_12_male(0) = Sigma2(0,2);
    sigma_e3_12_male(1) = Sigma2(1,2);
    sigma_e3_12_male(2) = Sigma2(3,2);
    phi_tp_male = x_cov_male * u_tp + e1234_male.col(2);
    sigma_e3_22_inv_male = arma::inv_sympd(Sigma2(ind_e3,ind_e3));
    sig2_e3_male = Sigma2(2,2) - arma::as_scalar(sigma_e3_12_male.t() * sigma_e3_22_inv_male * sigma_e3_12_male);
  }
  else{
    ind_e3 = {0,1,2};
    sigma_e3_12_female(0) = Sigma1(0,3);
    sigma_e3_12_female(1) = Sigma1(1,3);
    sigma_e3_12_female(2) = Sigma1(2,3);
    phi_tp_female = x_cov_female * u_tp + e1234_female.col(3);
    sigma_e3_22_inv_female = arma::inv_sympd(Sigma1(ind_e3,ind_e3));
    sig2_e3_female = Sigma1(3,3) - arma::as_scalar(sigma_e3_12_female.t() * sigma_e3_22_inv_female * sigma_e3_12_female);
    sigma_e3_12_male(0) = Sigma2(0,3);
    sigma_e3_12_male(1) = Sigma2(1,3);
    sigma_e3_12_male(2) = Sigma2(2,3);
    phi_tp_male = x_cov_male * u_tp + e1234_male.col(3);
    sigma_e3_22_inv_male = arma::inv_sympd(Sigma2(ind_e3,ind_e3));
    sig2_e3_male = Sigma2(3,3) - arma::as_scalar(sigma_e3_12_male.t() * sigma_e3_22_inv_male * sigma_e3_12_male);
  }
  
  arma::mat S_n = x_cov_female.t() * x_cov_female / sig2_e3_female + x_cov_male.t() * x_cov_male / sig2_e3_male;
  S_n.diag() += 1.0 / sig2_0;
  S_n = arma::inv_sympd(S_n);
  arma::vec mu_n = S_n * (x_cov_female.t() * (phi_tp_female - e1234_female.cols(ind_e3) * sigma_e3_22_inv_female * sigma_e3_12_female) / sig2_e3_female+
                          x_cov_male.t() * (phi_tp_male - e1234_male.cols(ind_e3) * sigma_e3_22_inv_male * sigma_e3_12_male) / sig2_e3_male);
  return arma::mvnrnd(mu_n, S_n);
}
arma::vec sample_u_general(const arma::vec sig_all, const arma::mat &rho_mat, const arma::mat &e1234, 
                        const arma::vec &u_tp_fp, const arma::mat &x_covariates, bool tp_flag){
  int p_num = u_tp_fp.n_rows;
  int nn = rho_mat.n_cols;

  double sig2_0 = 100.0;
  arma::mat S_n = arma::zeros(p_num, p_num);
  arma::vec mu_n = arma::zeros(p_num);
  if(tp_flag){
    arma::uvec ind_e3 = {0,1,3};
    arma::vec phi_tp_fp = x_covariates * u_tp_fp + e1234.col(2);
    for(int i=0; i<nn;++i){
      arma::mat Sigma_ii = construct_sigma1(sig_all, rho_mat.col(i));
      arma::vec x_cov_i = x_covariates.row(i).t();
      arma::vec e1234_i = e1234.row(i).t();
      arma::vec sigma_e3_12 = arma::zeros(3);
      sigma_e3_12(0) = Sigma_ii(0,2);
      sigma_e3_12(1) = Sigma_ii(1,2);
      sigma_e3_12(2) = Sigma_ii(3,2);
      arma::mat sigma_e3_22_inv = arma::inv_sympd(Sigma_ii(ind_e3,ind_e3));
      double sig2_e3 = Sigma_ii(2,2) - arma::as_scalar(sigma_e3_12.t() * sigma_e3_22_inv * sigma_e3_12);
      S_n += x_cov_i * x_cov_i.t() / sig2_e3;
      mu_n += x_cov_i * (phi_tp_fp(i) - arma::as_scalar(e1234_i(ind_e3).t() * sigma_e3_22_inv * sigma_e3_12)) / sig2_e3;
    }
  }
  else{
    arma::uvec ind_e3 = {0,1,2};
    arma::vec phi_tp_fp = x_covariates * u_tp_fp + e1234.col(3);
    for(int i=0; i<nn;++i){
      arma::mat Sigma_ii = construct_sigma1(sig_all, rho_mat.col(i));
      arma::vec x_cov_i = x_covariates.row(i).t();
      arma::vec e1234_i = e1234.row(i).t();
      arma::vec sigma_e3_12 = arma::zeros(3);
      sigma_e3_12(0) = Sigma_ii(0,3);
      sigma_e3_12(1) = Sigma_ii(1,3);
      sigma_e3_12(2) = Sigma_ii(2,3);
      arma::mat sigma_e3_22_inv = arma::inv_sympd(Sigma_ii(ind_e3,ind_e3));
      double sig2_e3 = Sigma_ii(3,3) - arma::as_scalar(sigma_e3_12.t() * sigma_e3_22_inv * sigma_e3_12);
      S_n += x_cov_i * x_cov_i.t() / sig2_e3;
      mu_n += x_cov_i * (phi_tp_fp(i) - arma::as_scalar(e1234_i(ind_e3).t() * sigma_e3_22_inv * sigma_e3_12)) / sig2_e3;
    }
  }
  S_n.diag() += 1.0 / sig2_0;
  S_n = arma::inv_sympd(S_n);
  mu_n = S_n * mu_n;
  return arma::mvnrnd(mu_n, S_n);
}
arma::vec sample_u_new2(arma::mat Sigma1, arma::mat Sigma2, arma::mat Sigma3, arma::mat Sigma4,
                        arma::mat e1234, arma::vec u_tp, const arma::mat &x_covariates, 
                        arma::vec female, arma::vec distlong, bool tp_flag){
  arma::uvec female_loc = arma::find(female==1 && distlong == 1);
  arma::uvec male_loc = arma::find(female==0 && distlong == 1);
  arma::uvec part3_loc = arma::find(female==1 && distlong == 0);
  arma::uvec part4_loc = arma::find(female==0 && distlong == 0);
  double sig2_0 = 100.0;
  arma::mat x_cov_female = x_covariates.rows(female_loc);
  arma::mat x_cov_male = x_covariates.rows(male_loc);
  arma::mat x_cov_part3 = x_covariates.rows(part3_loc);
  arma::mat x_cov_part4 = x_covariates.rows(part4_loc);
  arma::mat e1234_female = e1234.rows(female_loc);
  arma::mat e1234_male = e1234.rows(male_loc);
  arma::mat e1234_part3 = e1234.rows(part3_loc);
  arma::mat e1234_part4 = e1234.rows(part4_loc);
  
  arma::vec phi_tp_female, phi_tp_male, phi_tp_part3, phi_tp_part4;
  arma::uvec ind_e3(3);
  
  arma::vec sigma_e3_12_female(3);
  arma::mat sigma_e3_22_inv_female;
  double sig2_e3_female;
  arma::vec sigma_e3_12_male(3);
  arma::mat sigma_e3_22_inv_male;
  double sig2_e3_male;
  arma::vec sigma_e3_12_part3(3);
  arma::mat sigma_e3_22_inv_part3;
  double sig2_e3_part3;
  arma::vec sigma_e3_12_part4(3);
  arma::mat sigma_e3_22_inv_part4;
  double sig2_e3_part4;
  if(tp_flag){
    ind_e3 = {0,1,3};
    sigma_e3_12_female(0) = Sigma1(0,2);
    sigma_e3_12_female(1) = Sigma1(1,2);
    sigma_e3_12_female(2) = Sigma1(3,2);
    phi_tp_female = x_cov_female * u_tp + e1234_female.col(2);
    sigma_e3_22_inv_female = arma::inv_sympd(Sigma1(ind_e3,ind_e3));
    sig2_e3_female = Sigma1(2,2) - arma::as_scalar(sigma_e3_12_female.t() * sigma_e3_22_inv_female * sigma_e3_12_female);
    
    sigma_e3_12_male(0) = Sigma2(0,2);
    sigma_e3_12_male(1) = Sigma2(1,2);
    sigma_e3_12_male(2) = Sigma2(3,2);
    phi_tp_male = x_cov_male * u_tp + e1234_male.col(2);
    sigma_e3_22_inv_male = arma::inv_sympd(Sigma2(ind_e3,ind_e3));
    sig2_e3_male = Sigma2(2,2) - arma::as_scalar(sigma_e3_12_male.t() * sigma_e3_22_inv_male * sigma_e3_12_male);
    
    sigma_e3_12_part3(0) = Sigma3(0,2);
    sigma_e3_12_part3(1) = Sigma3(1,2);
    sigma_e3_12_part3(2) = Sigma3(3,2);
    phi_tp_part3 = x_cov_part3 * u_tp + e1234_part3.col(2);
    sigma_e3_22_inv_part3 = arma::inv_sympd(Sigma3(ind_e3,ind_e3));
    sig2_e3_part3 = Sigma3(2,2) - arma::as_scalar(sigma_e3_12_part3.t() * sigma_e3_22_inv_part3 * sigma_e3_12_part3);
    
    sigma_e3_12_part4(0) = Sigma4(0,2);
    sigma_e3_12_part4(1) = Sigma4(1,2);
    sigma_e3_12_part4(2) = Sigma4(3,2);
    phi_tp_part4 = x_cov_part4 * u_tp + e1234_part4.col(2);
    sigma_e3_22_inv_part4 = arma::inv_sympd(Sigma4(ind_e3,ind_e3));
    sig2_e3_part4 = Sigma4(2,2) - arma::as_scalar(sigma_e3_12_part4.t() * sigma_e3_22_inv_part4 * sigma_e3_12_part4);
  }
  else{
    ind_e3 = {0,1,2};
    sigma_e3_12_female(0) = Sigma1(0,3);
    sigma_e3_12_female(1) = Sigma1(1,3);
    sigma_e3_12_female(2) = Sigma1(2,3);
    phi_tp_female = x_cov_female * u_tp + e1234_female.col(3);
    sigma_e3_22_inv_female = arma::inv_sympd(Sigma1(ind_e3,ind_e3));
    sig2_e3_female = Sigma1(3,3) - arma::as_scalar(sigma_e3_12_female.t() * sigma_e3_22_inv_female * sigma_e3_12_female);
    
    sigma_e3_12_male(0) = Sigma2(0,3);
    sigma_e3_12_male(1) = Sigma2(1,3);
    sigma_e3_12_male(2) = Sigma2(2,3);
    phi_tp_male = x_cov_male * u_tp + e1234_male.col(3);
    sigma_e3_22_inv_male = arma::inv_sympd(Sigma2(ind_e3,ind_e3));
    sig2_e3_male = Sigma2(3,3) - arma::as_scalar(sigma_e3_12_male.t() * sigma_e3_22_inv_male * sigma_e3_12_male);
    
    sigma_e3_12_part3(0) = Sigma3(0,3);
    sigma_e3_12_part3(1) = Sigma3(1,3);
    sigma_e3_12_part3(2) = Sigma3(2,3);
    phi_tp_part3 = x_cov_part3 * u_tp + e1234_part3.col(3);
    sigma_e3_22_inv_part3 = arma::inv_sympd(Sigma3(ind_e3,ind_e3));
    sig2_e3_part3 = Sigma3(3,3) - arma::as_scalar(sigma_e3_12_part3.t() * sigma_e3_22_inv_part3 * sigma_e3_12_part3);
    
    sigma_e3_12_part4(0) = Sigma4(0,3);
    sigma_e3_12_part4(1) = Sigma4(1,3);
    sigma_e3_12_part4(2) = Sigma4(2,3);
    phi_tp_part4 = x_cov_part4 * u_tp + e1234_part4.col(3);
    sigma_e3_22_inv_part4 = arma::inv_sympd(Sigma4(ind_e3,ind_e3));
    sig2_e3_part4 = Sigma4(3,3) - arma::as_scalar(sigma_e3_12_part4.t() * sigma_e3_22_inv_part4 * sigma_e3_12_part4);
  }
  
  arma::mat S_n = x_cov_female.t() * x_cov_female / sig2_e3_female +
                  x_cov_male.t() * x_cov_male / sig2_e3_male+
                  x_cov_part3.t() * x_cov_part3 / sig2_e3_part3+
                  x_cov_part4.t() * x_cov_part4 / sig2_e3_part4;
  S_n.diag() += 1.0 / sig2_0;
  S_n = arma::inv_sympd(S_n);
  arma::vec mu_n = S_n * (x_cov_female.t() * (phi_tp_female - e1234_female.cols(ind_e3) * sigma_e3_22_inv_female * sigma_e3_12_female) / sig2_e3_female+
                          x_cov_male.t() * (phi_tp_male - e1234_male.cols(ind_e3) * sigma_e3_22_inv_male * sigma_e3_12_male) / sig2_e3_male+
                          x_cov_part3.t() * (phi_tp_part3 - e1234_part3.cols(ind_e3) * sigma_e3_22_inv_part3 * sigma_e3_12_part3) / sig2_e3_part3+
                          x_cov_part4.t() * (phi_tp_part4 - e1234_part4.cols(ind_e3) * sigma_e3_22_inv_part4 * sigma_e3_12_part4) / sig2_e3_part4);
  return arma::mvnrnd(mu_n, S_n);
}
struct log_e12_i_new_params{
  arma::vec ytp_fp_i;
  arma::vec beta_tp_fp_i;
  arma::vec alpha_tp_fp_i;
  arma::vec b_tp_fp;
  arma::rowvec x_covariates_i;
  arma::mat Sigma_inv;
  arma::vec e1234_i;
};

double log_e1_i_new(double x, void* params){
  struct log_e12_i_new_params *d;
  d = static_cast<struct log_e12_i_new_params *> (params);
  
  arma::vec e1234_i = d->e1234_i;
  e1234_i(0) = x;
  
  double res = -0.5 * arma::as_scalar(e1234_i.t() * d->Sigma_inv * e1234_i);
  
  arma::vec tmp = d->beta_tp_fp_i + d->alpha_tp_fp_i * (arma::as_scalar(d->x_covariates_i * d->b_tp_fp) + x);
  // arma::vec tmp1 = tmp % d->ytp_fp_i - arma::log(1+arma::exp(tmp));
  // res += arma::accu(tmp1);
  arma::vec log_p1 = Rcpp::pnorm(Rcpp::NumericVector(tmp.begin(), tmp.end()), 0.0, 1.0, 1, 1);
  arma::vec log_p2 = Rcpp::pnorm(Rcpp::NumericVector(tmp.begin(), tmp.end()), 0.0, 1.0, 0, 1);
  arma::vec tmp1 = d->ytp_fp_i % log_p1 + (1 - d->ytp_fp_i) % log_p2; // nan in y_i
  // res += arma::accu(tmp1);
  // if(!tmp1.is_finite()){
  //   Rcpp::Rcout << "\t\n non finite tmp value: " << tmp << "\t\n y_i" << d->ytp_fp_i << 
  //     "\t\nlog_p1" << log_p1 <<"\t\nlog_p2" << log_p2<< "\t\n tmp1" << tmp1 << std::endl;
  //   Rcpp::stop("");
  // }
  // tmp = Rcpp::pnorm5(Rcpp::NumericVector(tmp.begin(),tmp.end()), 0.0, 1.0, 1, 0);
  // arma::vec tmp1 = arma::log(d->ytp_fp_i % tmp + (1 - d->ytp_fp_i) % (1-tmp));
  res += arma::accu(tmp1(arma::find_finite(tmp1)));

  return res;
}
double log_e2_i_new(double x, void* params){
  struct log_e12_i_new_params *d;
  d = static_cast<struct log_e12_i_new_params *> (params);
  
  arma::vec e1234_i = d->e1234_i;
  e1234_i(1) = x;
  
  double res = -0.5 * arma::as_scalar(e1234_i.t() * d->Sigma_inv * e1234_i);
  
  arma::vec tmp = d->beta_tp_fp_i + d->alpha_tp_fp_i * (arma::as_scalar(d->x_covariates_i * d->b_tp_fp) + x);
  // arma::vec tmp1 = tmp % d->ytp_fp_i - arma::log(1+arma::exp(tmp));
  // res += arma::accu(tmp1);
  arma::vec log_p1 = Rcpp::pnorm(Rcpp::NumericVector(tmp.begin(), tmp.end()), 0.0, 1.0, 1, 1);
  arma::vec log_p2 = Rcpp::pnorm(Rcpp::NumericVector(tmp.begin(), tmp.end()), 0.0, 1.0, 0, 1);
  arma::vec tmp1 = d->ytp_fp_i % log_p1 + (1 - d->ytp_fp_i) % log_p2;
  // res += arma::accu(tmp1);
  
  // tmp = Rcpp::pnorm5(Rcpp::NumericVector(tmp.begin(),tmp.end()), 0.0, 1.0, 1, 0);
  // arma::vec tmp1 = arma::log(d->ytp_fp_i % tmp + (1 - d->ytp_fp_i) % (1-tmp));
  res += arma::accu(tmp1(arma::find_finite(tmp1)));
  return res;
}
struct log_uni_norm_params{
  double mu;
  double sigma2;
};

double log_uni_norm(double x, void* params){
  struct log_uni_norm_params *d;
  d = static_cast<struct log_uni_norm_params *> (params);
  
  return -0.5 / d->sigma2 * std::pow(x-d->mu,2);
}
//' @export 
// [[Rcpp::export]]
arma::vec test3(double mu, double sig2, double cut_point, bool direct, int n){
  int err, ninit = 4, npoint = 100, nsamp = 1, ncent = 0 ;
  int neval;
  double xinit[10]={-4.0, -1.0, 0.0, 1.0, 4.0}, xl = -100.0, xr = 100.0;
  double xcent[10], qcent[10] = {5., 30., 70., 95.};
  double convex = 1.;
  double xprev = 0;
  double xsamp[100];
  int dometrop = 0;
  if(direct){
    xl = cut_point;
    xr = cut_point + 100.0;
    xinit[0] = cut_point +1;
    xinit[1] = cut_point +2;
    xinit[2] = cut_point +3;
    xinit[3] = cut_point +4;
  }
  else{
    xl = cut_point - 100.0;
    xr = cut_point;
    xinit[0] = cut_point -4;
    xinit[1] = cut_point -3;
    xinit[2] = cut_point -2;
    xinit[3] = cut_point -1;
  }
  log_uni_norm_params log_uni_norm_data;
  log_uni_norm_data.mu = mu;
  log_uni_norm_data.sigma2 = sig2;
  arma::vec res(n);
  for(auto i=0;i<n;++i){
    err = arms(xinit,ninit,&xl,&xr,log_uni_norm,&log_uni_norm_data,&convex,
               npoint,dometrop,&xprev,xsamp,nsamp,qcent,xcent,ncent,&neval);
    if(err>0){
      Rprintf("error code: %d", err);
      Rcpp::stop("\n");
    }
    res(i) = xsamp[0];
  }
  return res;
}
//' @export
// [[Rcpp::export]]
void sample_e1234(arma::mat &e1234, const arma::mat &ytp, const arma::mat &yfp, 
                       const arma::vec &ytp_fin, const arma::vec &yfp_fin, const arma::uvec &xi, 
                       const arma::mat &beta_tp, const arma::mat &alpha_tp, 
                       const arma::mat &beta_fp, const arma::mat &alpha_fp,
                       const arma::vec &b_tp, const arma::vec &b_fp, const arma::vec &u_tp, const arma::vec &u_fp, 
                       const arma::mat &x_covariates, const arma::mat &Sigma){
  // variables for armh sampler
  int err, ninit = 5, npoint = 100, nsamp = 1, ncent = 0 ;
  int neval;
  double xinit[10]={-4.0, -1.0, 0.0, 1.0, 4.0}, xl = -100.0, xr = 100.0;
  double xcent[10], qcent[10] = {5., 30., 70., 95.};
  double convex = 1.;
  int dometrop = 1;
  double xsamp[100];
  double xprev = 0;
  
  if ( Progress::check_abort() )
    return;
  arma::mat Sigma_inv = arma::inv_sympd(Sigma);
  
  // sample e12, conditional on e3,4
  // arma::uvec ind_e12 = {0,1};
  // arma::uvec ind_e34 = {2,3};
  // arma::mat e12 = e1234.cols(0,1);
  // arma::uvec xi_123_loc = arma::find(xi==1 || xi==2 || xi==3);
  // arma::uvec xi_4_loc = arma::find(xi==4);
  // arma::mat sigma_12(2,2);
  // sigma_12(0,0) = Sigma(0,2);
  // sigma_12(0,1) = Sigma(0,3);
  // sigma_12(1,0) = Sigma(1,2);
  // sigma_12(1,1) = Sigma(1,3);
  // arma::mat sigma_22_inv = arma::inv_sympd(Sigma(ind_e34,ind_e34));
  // arma::mat mu_tmp = e1234.cols(2,3) * sigma_22_inv * sigma_12.t();
  // arma::mat sigma_tmp = Sigma(ind_e12,ind_e12) - sigma_12 * sigma_22_inv * sigma_12.t();
  // e12.rows(xi_123_loc) = mu_tmp.rows(xi_123_loc) + arma::mvnrnd(arma::zeros(2), sigma_tmp, xi_123_loc.n_elem).t();
  
  // conditional mean and var for e3
  arma::vec mu3 = x_covariates * u_tp;
  arma::uvec ind_e3 = {0,1,3};
  arma::vec sigma_e3_12(3);
  sigma_e3_12(0) = Sigma(2,0);
  sigma_e3_12(1) = Sigma(2,1);
  sigma_e3_12(2) = Sigma(2,3);
  arma::mat sigma_e3_22_inv = arma::inv_sympd(Sigma(ind_e3,ind_e3));
  double sigma2_e3 = std::sqrt(Sigma(2,2) - arma::as_scalar(sigma_e3_12.t() * sigma_e3_22_inv * sigma_e3_12));
  // arma::vec mu_e3 = e1234.cols(ind_e3) * sigma_e3_22_inv * sigma_e3_12; 
  
  // conditional mean and var for e4
  arma::vec mu4 = x_covariates * u_fp;
  arma::uvec ind_e4 = {0,1,2};
  arma::vec sigma_e4_12(3);
  sigma_e4_12(0) = Sigma(3,0);
  sigma_e4_12(1) = Sigma(3,1);
  sigma_e4_12(2) = Sigma(3,2);
  arma::mat sigma_e4_22_inv = arma::inv_sympd(Sigma(ind_e4, ind_e4));
  double sigma2_e4 = std::sqrt(Sigma(3,3) - arma::as_scalar(sigma_e4_12.t() * sigma_e4_22_inv * sigma_e4_12));
  // arma::vec mu_e4 = e1234.cols(ind_e4) * sigma_e4_22_inv * sigma_e4_12;
  
  // for xi = 4: xi_G=1 & xi_R=1
  arma::uvec xi_4_loc = arma::find(xi == 4);
  for(auto &ii:xi_4_loc){
    arma::vec e1234_ii = e1234.row(ii).t();
    
    // sample e1
    xprev = e1234_ii(0);
    log_e12_i_new_params log_e12_i_new_data;
    log_e12_i_new_data.alpha_tp_fp_i = alpha_tp.row(ii).t();
    log_e12_i_new_data.b_tp_fp = b_tp;
    log_e12_i_new_data.beta_tp_fp_i = beta_tp.row(ii).t();
    log_e12_i_new_data.e1234_i = e1234_ii;
    log_e12_i_new_data.Sigma_inv = Sigma_inv;
    log_e12_i_new_data.x_covariates_i = x_covariates.row(ii);
    log_e12_i_new_data.ytp_fp_i = ytp.row(ii).t();
    
    err = arms(xinit,ninit,&xl,&xr,log_e1_i_new,&log_e12_i_new_data,&convex,
               npoint,dometrop,&xprev,xsamp,nsamp,qcent,xcent,ncent,&neval);
    if(err>0){
      Rprintf("error code: %d", err);
      Rcpp::stop("\n");
    }
    e1234_ii(0) = xsamp[0];
    if(arma::max(xsamp[0]) > 100.0) Rcpp::Rcout << "error in 4.1\n";
    // sample e2
    xprev = e1234_ii(1);
    log_e12_i_new_data.alpha_tp_fp_i = alpha_fp.row(ii).t();
    log_e12_i_new_data.b_tp_fp = b_fp;
    log_e12_i_new_data.beta_tp_fp_i = beta_fp.row(ii).t();
    log_e12_i_new_data.e1234_i = e1234_ii;
    log_e12_i_new_data.ytp_fp_i = yfp.row(ii).t();
    err = arms(xinit,ninit,&xl,&xr,log_e2_i_new,&log_e12_i_new_data,&convex,
               npoint,dometrop,&xprev,xsamp,nsamp,qcent,xcent,ncent,&neval);
    if(err>0){
      Rprintf("error code: %d", err);
      Rcpp::stop("\n");
    }
    e1234_ii(1) = xsamp[0];
    if(arma::max(xsamp[0]) > 100.0) Rcpp::Rcout << "error in 4.2\n";
    // sample e3
    //Rcpp::Rcout << e1234_ii(ind_e3) << "\n" << sigma_e3_22_inv << "\n";
    double mu_e3 = arma::as_scalar(e1234_ii(ind_e3).t() * sigma_e3_22_inv * sigma_e3_12);
    if(ytp_fin(ii)){
      e1234_ii(2) = RcppTN::rtn1(mu_e3, sigma2_e3, -mu3(ii), R_PosInf);
    }
    else{
      e1234_ii(2) = RcppTN::rtn1(mu_e3, sigma2_e3, R_NegInf, -mu3(ii));
    }
    if(arma::max(e1234_ii(2)) > 100.0) Rcpp::Rcout << "error in 4.3\n";
    // sample e4
    double mu_e4 = arma::as_scalar(e1234_ii(ind_e4).t() * sigma_e4_22_inv * sigma_e4_12);
    if(yfp_fin(ii)){
      e1234_ii(3) = RcppTN::rtn1(mu_e4, sigma2_e4, -mu4(ii), R_PosInf);
    }
    else{
      e1234_ii(3) = RcppTN::rtn1(mu_e4, sigma2_e4, R_NegInf, -mu4(ii));
    }
    if(arma::max(e1234_ii(3)) > 100.0) Rcpp::Rcout << "error in 4.4\n";
    e1234.row(ii) = e1234_ii.t();
  }
  //Rprintf("hello 2 ");
  // for xi_i = 3: xi_Gi = 1 and xi_Ri = 0
  // conditional mean and cov for sample e24, conditional on e13
  arma::uvec ind_e24 = {1,3};
  arma::uvec ind_e13 = {0,2};
  arma::mat sigma_12 = Sigma(ind_e24, ind_e13);
  arma::mat sigma_22_inv = arma::inv_sympd(Sigma(ind_e13,ind_e13));
  arma::mat sigma_tmp = Sigma(ind_e24,ind_e24) - sigma_12 * sigma_22_inv * sigma_12.t();

  arma::uvec xi_3_loc = arma::find(xi==3);
  for(auto &ii : xi_3_loc){
    arma::vec e1234_ii = e1234.row(ii).t();
    
    // sample e1 (e_GP)
    xprev = e1234_ii(0);
    log_e12_i_new_params log_e12_i_new_data;
    log_e12_i_new_data.alpha_tp_fp_i = alpha_tp.row(ii).t();
    log_e12_i_new_data.b_tp_fp = b_tp;
    log_e12_i_new_data.beta_tp_fp_i = beta_tp.row(ii).t();
    log_e12_i_new_data.e1234_i = e1234_ii;
    log_e12_i_new_data.Sigma_inv = Sigma_inv;
    log_e12_i_new_data.x_covariates_i = x_covariates.row(ii);
    log_e12_i_new_data.ytp_fp_i = ytp.row(ii).t();
    
    err = arms(xinit,ninit,&xl,&xr,log_e1_i_new,&log_e12_i_new_data,&convex,
               npoint,dometrop,&xprev,xsamp,nsamp,qcent,xcent,ncent,&neval);
    if(err>0){
      Rprintf("error code: %d", err);
      Rcpp::stop("\n");
    }
    e1234_ii(0) = xsamp[0];
    
    // sample e3 (e_GF)
    double mu_e3 = arma::as_scalar(e1234_ii(ind_e3).t() * sigma_e3_22_inv * sigma_e3_12);
    if(ytp_fin(ii)){
      e1234_ii(2) = RcppTN::rtn1(mu_e3, sigma2_e3, -mu3(ii), R_PosInf);
    }
    else{
      e1234_ii(2) = RcppTN::rtn1(mu_e3, sigma2_e3, R_NegInf, -mu3(ii));
    }
    
    // sample e2 (e_RP) and e4 (eRF) conditional on e13
    arma::vec mu_24 =  sigma_12 * sigma_22_inv * e1234_ii(ind_e13);
    e1234_ii(ind_e24) = arma::mvnrnd(mu_24, sigma_tmp);
    
    e1234.row(ii) = e1234_ii.t();
    if(arma::max(arma::abs(e1234_ii)) > 100.0) Rcpp::Rcout << "error in 3\n";
  }

  // for xi = 2: xi_G = 0 & xi_R = 1
  // conditional mean and cov for sample e13, conditional on e24
  sigma_12 = Sigma(ind_e13, ind_e24);
  sigma_22_inv = arma::inv_sympd(Sigma(ind_e24,ind_e24));
  // mu_tmp = e1234.cols(2,3) * sigma_22_inv * sigma_12.t();
  sigma_tmp = Sigma(ind_e13,ind_e13) - sigma_12 * sigma_22_inv * sigma_12.t();
  
  arma::uvec xi_2_loc = arma::find(xi==2);
  for(auto &ii : xi_2_loc){
    arma::vec e1234_ii = e1234.row(ii).t();
    
    // sample e2 (e_RP)
    xprev = e1234_ii(1);
    log_e12_i_new_params log_e12_i_new_data;
    log_e12_i_new_data.alpha_tp_fp_i = alpha_fp.row(ii).t();
    log_e12_i_new_data.b_tp_fp = b_fp;
    log_e12_i_new_data.beta_tp_fp_i = beta_fp.row(ii).t();
    log_e12_i_new_data.e1234_i = e1234_ii;
    log_e12_i_new_data.ytp_fp_i = yfp.row(ii).t();
    log_e12_i_new_data.Sigma_inv = Sigma_inv;
    log_e12_i_new_data.x_covariates_i = x_covariates.row(ii);
    
    err = arms(xinit,ninit,&xl,&xr,log_e2_i_new,&log_e12_i_new_data,&convex,
               npoint,dometrop,&xprev,xsamp,nsamp,qcent,xcent,ncent,&neval);
    if(err>0){
      Rprintf("error code: %d", err);
      Rcpp::stop("\n");
    }
    e1234_ii(1) = xsamp[0];
    
    // sample e4
    double mu_e4 = arma::as_scalar(e1234_ii(ind_e4).t() * sigma_e4_22_inv * sigma_e4_12);
    if(yfp_fin(ii)){
      e1234_ii(3) = RcppTN::rtn1(mu_e4, sigma2_e4, -mu4(ii), R_PosInf);
    }
    else{
      e1234_ii(3) = RcppTN::rtn1(mu_e4, sigma2_e4, R_NegInf, -mu4(ii));
    }
    
    // sample e13, conditional on e24
    arma::vec mu_13 =  sigma_12 * sigma_22_inv * e1234_ii(ind_e24);
    e1234_ii(ind_e13) = arma::mvnrnd(mu_13, sigma_tmp);
    
    e1234.row(ii) = e1234_ii.t();
    if(arma::max(arma::abs(e1234_ii)) > 100.0) Rcpp::Rcout << "error in 2\n";
  }
  
  // for xi_i = 1
  arma::uvec xi_1_loc = arma::find(xi==1);
  e1234.rows(xi_1_loc) = arma::mvnrnd(arma::zeros(4), Sigma, xi_1_loc.n_elem).t();
}

// [[Rcpp::export]]
void sample_e1234_general(arma::mat &e1234, const arma::mat &ytp, const arma::mat &yfp, 
                       const arma::vec &ytp_fin, const arma::vec &yfp_fin, const arma::uvec &xi, 
                       const arma::mat &beta_tp, const arma::mat &alpha_tp, 
                       const arma::mat &beta_fp, const arma::mat &alpha_fp,
                       const arma::vec &b_tp, const arma::vec &b_fp, const arma::vec &u_tp, const arma::vec &u_fp, 
                       const arma::mat &x_covariates, const arma::vec &sig_all, const arma::mat rho_mat){
  // variables for armh sampler
  int err, ninit = 5, npoint = 100, nsamp = 1, ncent = 0 ;
  int neval;
  double xinit[10]={-4.0, -1.0, 0.0, 1.0, 4.0}, xl = -100.0, xr = 100.0;
  double xcent[10], qcent[10] = {5., 30., 70., 95.};
  double convex = 1.;
  int dometrop = 1;
  double xsamp[100];
  double xprev = 0;
  
  if ( Progress::check_abort() )
    return;
  // arma::mat Sigma_inv = arma::inv_sympd(Sigma);
  
  // sample e12, conditional on e3,4
  // arma::uvec ind_e12 = {0,1};
  // arma::uvec ind_e34 = {2,3};
  // arma::mat e12 = e1234.cols(0,1);
  // arma::uvec xi_123_loc = arma::find(xi==1 || xi==2 || xi==3);
  // arma::uvec xi_4_loc = arma::find(xi==4);
  // arma::mat sigma_12(2,2);
  // sigma_12(0,0) = Sigma(0,2);
  // sigma_12(0,1) = Sigma(0,3);
  // sigma_12(1,0) = Sigma(1,2);
  // sigma_12(1,1) = Sigma(1,3);
  // arma::mat sigma_22_inv = arma::inv_sympd(Sigma(ind_e34,ind_e34));
  // arma::mat mu_tmp = e1234.cols(2,3) * sigma_22_inv * sigma_12.t();
  // arma::mat sigma_tmp = Sigma(ind_e12,ind_e12) - sigma_12 * sigma_22_inv * sigma_12.t();
  // e12.rows(xi_123_loc) = mu_tmp.rows(xi_123_loc) + arma::mvnrnd(arma::zeros(2), sigma_tmp, xi_123_loc.n_elem).t();
  
  arma::vec mu3 = x_covariates * u_tp;
  arma::uvec ind_e3 = {0,1,3};
  arma::vec sigma_e3_12(3);

  arma::vec mu4 = x_covariates * u_fp;
  arma::uvec ind_e4 = {0,1,2};
  arma::vec sigma_e4_12(3);

  // for xi = 4: xi_G=1 & xi_R=1
  arma::uvec xi_4_loc = arma::find(xi == 4);
  for(auto &ii:xi_4_loc){
    arma::vec e1234_ii = e1234.row(ii).t();
    arma::mat Sigma_ii = construct_sigma1(sig_all, rho_mat.col(ii));

    // sample e1
    xprev = e1234_ii(0);
    log_e12_i_new_params log_e12_i_new_data;
    log_e12_i_new_data.alpha_tp_fp_i = alpha_tp.row(ii).t();
    log_e12_i_new_data.b_tp_fp = b_tp;
    log_e12_i_new_data.beta_tp_fp_i = beta_tp.row(ii).t();
    log_e12_i_new_data.e1234_i = e1234_ii;
    log_e12_i_new_data.Sigma_inv = arma::inv_sympd(Sigma_ii);
    log_e12_i_new_data.x_covariates_i = x_covariates.row(ii);
    log_e12_i_new_data.ytp_fp_i = ytp.row(ii).t();
    
    err = arms(xinit,ninit,&xl,&xr,log_e1_i_new,&log_e12_i_new_data,&convex,
               npoint,dometrop,&xprev,xsamp,nsamp,qcent,xcent,ncent,&neval);
    if(err>0){
      Rprintf("error code: %d", err);
      Rcpp::stop("\n");
    }
    e1234_ii(0) = xsamp[0];
    if(arma::max(xsamp[0]) > 100.0) Rcpp::Rcout << "error in 4.1\n";
    // sample e2
    xprev = e1234_ii(1);
    log_e12_i_new_data.alpha_tp_fp_i = alpha_fp.row(ii).t();
    log_e12_i_new_data.b_tp_fp = b_fp;
    log_e12_i_new_data.beta_tp_fp_i = beta_fp.row(ii).t();
    log_e12_i_new_data.e1234_i = e1234_ii;
    log_e12_i_new_data.ytp_fp_i = yfp.row(ii).t();
    err = arms(xinit,ninit,&xl,&xr,log_e2_i_new,&log_e12_i_new_data,&convex,
               npoint,dometrop,&xprev,xsamp,nsamp,qcent,xcent,ncent,&neval);
    if(err>0){
      Rprintf("error code: %d", err);
      Rcpp::stop("\n");
    }
    e1234_ii(1) = xsamp[0];
    if(arma::max(xsamp[0]) > 100.0) Rcpp::Rcout << "error in 4.2\n";

    // sample e3
    //Rcpp::Rcout << e1234_ii(ind_e3) << "\n" << sigma_e3_22_inv << "\n";
    // conditional mean and var for e3

    sigma_e3_12(0) = Sigma_ii(2,0);
    sigma_e3_12(1) = Sigma_ii(2,1);
    sigma_e3_12(2) = Sigma_ii(2,3);
    arma::mat sigma_e3_22_inv = arma::inv_sympd(Sigma_ii(ind_e3,ind_e3));
    double sigma2_e3 = std::sqrt(Sigma_ii(2,2) - arma::as_scalar(sigma_e3_12.t() * sigma_e3_22_inv * sigma_e3_12));
    double mu_e3 = arma::as_scalar(e1234_ii(ind_e3).t() * sigma_e3_22_inv * sigma_e3_12);
    if(ytp_fin(ii)){
      e1234_ii(2) = RcppTN::rtn1(mu_e3, sigma2_e3, -mu3(ii), R_PosInf);
    }
    else{
      e1234_ii(2) = RcppTN::rtn1(mu_e3, sigma2_e3, R_NegInf, -mu3(ii));
    }
    if(arma::max(e1234_ii(2)) > 100.0) Rcpp::Rcout << "error in 4.3\n";

    // sample e4
    // conditional mean and var for e4
    sigma_e4_12(0) = Sigma_ii(3,0);
    sigma_e4_12(1) = Sigma_ii(3,1);
    sigma_e4_12(2) = Sigma_ii(3,2);
    arma::mat sigma_e4_22_inv = arma::inv_sympd(Sigma_ii(ind_e4, ind_e4));
    double sigma2_e4 = std::sqrt(Sigma_ii(3,3) - arma::as_scalar(sigma_e4_12.t() * sigma_e4_22_inv * sigma_e4_12));
    double mu_e4 = arma::as_scalar(e1234_ii(ind_e4).t() * sigma_e4_22_inv * sigma_e4_12);
    if(yfp_fin(ii)){
      e1234_ii(3) = RcppTN::rtn1(mu_e4, sigma2_e4, -mu4(ii), R_PosInf); // sigma2_e4 is standard deviation
    }
    else{
      e1234_ii(3) = RcppTN::rtn1(mu_e4, sigma2_e4, R_NegInf, -mu4(ii));
    }
    if(arma::max(e1234_ii(3)) > 100.0) Rcpp::Rcout << "error in 4.4\n";
    e1234.row(ii) = e1234_ii.t();
  }
  //Rprintf("hello 2 ");


  // for xi_i = 3: xi_Gi = 1 and xi_Ri = 0
  // conditional mean and cov for sample e24, conditional on e13
  arma::uvec ind_e24 = {1,3};
  arma::uvec ind_e13 = {0,2};

  arma::uvec xi_3_loc = arma::find(xi==3);
  for(auto &ii : xi_3_loc){
    arma::vec e1234_ii = e1234.row(ii).t();
    arma::mat Sigma_ii = construct_sigma1(sig_all, rho_mat.col(ii));

    // sample e1 (e_GP)
    xprev = e1234_ii(0);
    log_e12_i_new_params log_e12_i_new_data;
    log_e12_i_new_data.alpha_tp_fp_i = alpha_tp.row(ii).t();
    log_e12_i_new_data.b_tp_fp = b_tp;
    log_e12_i_new_data.beta_tp_fp_i = beta_tp.row(ii).t();
    log_e12_i_new_data.e1234_i = e1234_ii;
    log_e12_i_new_data.Sigma_inv = arma::inv_sympd(Sigma_ii);
    log_e12_i_new_data.x_covariates_i = x_covariates.row(ii);
    log_e12_i_new_data.ytp_fp_i = ytp.row(ii).t();
    
    err = arms(xinit,ninit,&xl,&xr,log_e1_i_new,&log_e12_i_new_data,&convex,
               npoint,dometrop,&xprev,xsamp,nsamp,qcent,xcent,ncent,&neval);
    if(err>0){
      Rprintf("error code: %d", err);
      Rcpp::stop("\n");
    }
    e1234_ii(0) = xsamp[0];
    
    // sample e3 (e_GF)
    sigma_e3_12(0) = Sigma_ii(2,0);
    sigma_e3_12(1) = Sigma_ii(2,1);
    sigma_e3_12(2) = Sigma_ii(2,3);
    arma::mat sigma_e3_22_inv = arma::inv_sympd(Sigma_ii(ind_e3,ind_e3));
    double sigma2_e3 = std::sqrt(Sigma_ii(2,2) - arma::as_scalar(sigma_e3_12.t() * sigma_e3_22_inv * sigma_e3_12));
    double mu_e3 = arma::as_scalar(e1234_ii(ind_e3).t() * sigma_e3_22_inv * sigma_e3_12);
    if(ytp_fin(ii)){
      e1234_ii(2) = RcppTN::rtn1(mu_e3, sigma2_e3, -mu3(ii), R_PosInf);
    }
    else{
      e1234_ii(2) = RcppTN::rtn1(mu_e3, sigma2_e3, R_NegInf, -mu3(ii));
    }
    
    // sample e2 (e_RP) and e4 (eRF) conditional on e13
    arma::mat sigma_12 = Sigma_ii(ind_e24, ind_e13);
    arma::mat sigma_22_inv = arma::inv_sympd(Sigma_ii(ind_e13,ind_e13));
    arma::mat sigma_tmp = Sigma_ii(ind_e24,ind_e24) - sigma_12 * sigma_22_inv * sigma_12.t();
    arma::vec mu_24 =  sigma_12 * sigma_22_inv * e1234_ii(ind_e13);
    e1234_ii(ind_e24) = arma::mvnrnd(mu_24, sigma_tmp);
    
    e1234.row(ii) = e1234_ii.t();
    if(arma::max(arma::abs(e1234_ii)) > 100.0) Rcpp::Rcout << "error in 3\n";
  }

  // for xi = 2: xi_G = 0 & xi_R = 1
  arma::uvec xi_2_loc = arma::find(xi==2);
  for(auto &ii : xi_2_loc){
    arma::vec e1234_ii = e1234.row(ii).t();
    arma::mat Sigma_ii = construct_sigma1(sig_all, rho_mat.col(ii));

    // sample e2 (e_RP)
    xprev = e1234_ii(1);
    log_e12_i_new_params log_e12_i_new_data;
    log_e12_i_new_data.alpha_tp_fp_i = alpha_fp.row(ii).t();
    log_e12_i_new_data.b_tp_fp = b_fp;
    log_e12_i_new_data.beta_tp_fp_i = beta_fp.row(ii).t();
    log_e12_i_new_data.e1234_i = e1234_ii;
    log_e12_i_new_data.ytp_fp_i = yfp.row(ii).t();
    log_e12_i_new_data.Sigma_inv = arma::inv_sympd(Sigma_ii);
    log_e12_i_new_data.x_covariates_i = x_covariates.row(ii);
    
    err = arms(xinit,ninit,&xl,&xr,log_e2_i_new,&log_e12_i_new_data,&convex,
               npoint,dometrop,&xprev,xsamp,nsamp,qcent,xcent,ncent,&neval);
    if(err>0){
      Rprintf("error code: %d", err);
      Rcpp::stop("\n");
    }
    e1234_ii(1) = xsamp[0];
    
    // sample e4
    // conditional mean and var for e4
    sigma_e4_12(0) = Sigma_ii(3,0);
    sigma_e4_12(1) = Sigma_ii(3,1);
    sigma_e4_12(2) = Sigma_ii(3,2);
    arma::mat sigma_e4_22_inv = arma::inv_sympd(Sigma_ii(ind_e4, ind_e4));
    double sigma2_e4 = std::sqrt(Sigma_ii(3,3) - arma::as_scalar(sigma_e4_12.t() * sigma_e4_22_inv * sigma_e4_12));
    double mu_e4 = arma::as_scalar(e1234_ii(ind_e4).t() * sigma_e4_22_inv * sigma_e4_12);
    if(yfp_fin(ii)){
      e1234_ii(3) = RcppTN::rtn1(mu_e4, sigma2_e4, -mu4(ii), R_PosInf);
    }
    else{
      e1234_ii(3) = RcppTN::rtn1(mu_e4, sigma2_e4, R_NegInf, -mu4(ii));
    }
    
    // sample e13, conditional on e24
      // conditional mean and cov for sample e13, conditional on e24
    arma::mat sigma_12 = Sigma_ii(ind_e13, ind_e24);
    arma::mat sigma_22_inv = arma::inv_sympd(Sigma_ii(ind_e24,ind_e24));
    arma::mat sigma_tmp = Sigma_ii(ind_e13,ind_e13) - sigma_12 * sigma_22_inv * sigma_12.t();
    arma::vec mu_13 =  sigma_12 * sigma_22_inv * e1234_ii(ind_e24);
    e1234_ii(ind_e13) = arma::mvnrnd(mu_13, sigma_tmp);
    
    e1234.row(ii) = e1234_ii.t();
    if(arma::max(arma::abs(e1234_ii)) > 100.0) Rcpp::Rcout << "error in 2\n";
  }
  
  // for xi_i = 1
  arma::uvec xi_1_loc = arma::find(xi==1);
  for(auto &ii : xi_1_loc){
    arma::mat Sigma_ii = construct_sigma1(sig_all, rho_mat.col(ii));
    e1234.row(ii) = arma::mvnrnd(arma::zeros(4), Sigma_ii).t();
  }

}




