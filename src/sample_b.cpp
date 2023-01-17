// [[Rcpp::depends(RcppArmadillo)]]
#include "arms_ori.h"
#include "lvmcorr_omp.h"
#include "depend_funcs.h"

struct log_bj_tp_param{
  arma::vec b_tp;
  unsigned int jj;
  arma::mat ytp;
  arma::uvec xi_3_loc;
  arma::uvec xi_4_loc;
  arma::mat x_covariates;
  arma::mat beta_tp;
  arma::mat alpha_tp;
  arma::vec e_tp;
  arma::vec e1;
};
struct log_bj_fp_param{
  arma::vec b_fp;
  unsigned int jj;
  arma::mat yfp;
  arma::uvec xi_2_loc;
  arma::uvec xi_4_loc;
  arma::mat x_covariates;
  arma::mat beta_fp;
  arma::mat alpha_fp;
  arma::vec e_fp;
  arma::vec e2;
};

// self-defined log_sum_exp to prevent possible underflow / overflow
arma::mat log_sum_exp1_mat(arma::mat tmp){
  arma::mat tmp_zeros = arma::zeros(arma::size(tmp));
  arma::mat max_tmp_0 = arma::max(tmp_zeros, tmp);
  return max_tmp_0 + arma::log(arma::exp(- max_tmp_0) + arma::exp(tmp - max_tmp_0));
}

double log_bj_tp(double x, void* params){
  struct log_bj_tp_param *d;
  d = static_cast<struct log_bj_tp_param *> (params);
  arma::uvec xi_3_loc = d->xi_3_loc;
  arma::uvec xi_4_loc = d->xi_4_loc;
  arma::vec b_tp = d->b_tp;
  b_tp(d->jj) = x;
  
  double res = - x * x / 200;
  
  arma::mat tmp = (d->alpha_tp).rows(xi_3_loc);
  tmp.each_col() %= (d->x_covariates).rows(xi_3_loc) * b_tp +(d->e_tp)(xi_3_loc);
  tmp += (d->beta_tp).rows(xi_3_loc);
  //arma::mat tmp1 = tmp % (d->ytp).rows(xi_3_loc) - log_sum_exp1_mat(tmp);
  arma::mat tmp1 = tmp % (d->ytp).rows(xi_3_loc) - arma::log(1+arma::exp(tmp));
  //res += arma::accu( tmp1.elem(arma::find_finite(tmp1)) );
  res += arma::accu(tmp1);
  
  arma::mat tmp2 = (d->alpha_tp).rows(xi_4_loc);
  tmp2.each_col() %= (d->x_covariates).rows(xi_4_loc) * b_tp + (d->e1)(xi_4_loc);
  tmp2 += (d->beta_tp).rows(xi_4_loc);
  //arma::mat tmp3 = tmp2 % (d->ytp).rows(xi_4_loc) - log_sum_exp1_mat(tmp2);
  arma::mat tmp3 = tmp2 % (d->ytp).rows(xi_4_loc) - arma::log(1+arma::exp(tmp2));
  //res += arma::accu( tmp3.elem(arma::find_finite(tmp3)) );
  res += arma::accu( tmp3 );
  return res;
}
double log_bj_fp(double x, void* params){
  struct log_bj_fp_param *d;
  d = static_cast<struct log_bj_fp_param *> (params);
  arma::uvec xi_2_loc = d->xi_2_loc;
  arma::uvec xi_4_loc = d->xi_4_loc;
  arma::vec b_fp = d->b_fp;
  b_fp(d->jj) = x;
  
  double res = - x * x / 200;
  
  arma::mat tmp = (d->alpha_fp).rows(xi_2_loc);
  tmp.each_col() %= (d->x_covariates).rows(xi_2_loc) * b_fp +(d->e_fp)(xi_2_loc);
  tmp += (d->beta_fp).rows(xi_2_loc);
  //arma::mat tmp1 = tmp % (d->yfp).rows(xi_2_loc) - log_sum_exp1_mat(tmp);
  arma::mat tmp1 = tmp % (d->yfp).rows(xi_2_loc) - arma::log(1+arma::exp(tmp));
  //res += arma::accu( tmp1.elem(arma::find_finite(tmp1)) );
  res += arma::accu( tmp1 );
  
  arma::mat tmp2 = (d->alpha_fp).rows(xi_4_loc);
  tmp2.each_col() %= (d->x_covariates).rows(xi_4_loc) * b_fp + (d->e2)(xi_4_loc);
  tmp2 += (d->beta_fp).rows(xi_4_loc);
  //arma::mat tmp3 = tmp2 % (d->yfp).rows(xi_4_loc) - log_sum_exp1_mat(tmp2);
  arma::mat tmp3 = tmp2 % (d->yfp).rows(xi_4_loc) - arma::log(1+arma::exp(tmp2));
  //res += arma::accu( tmp3.elem(arma::find_finite(tmp3)) );
  res += arma::accu( tmp3 );
  return res;
}
// [[Rcpp::export]]
arma::vec sample_b_tp_c(arma::vec b_tp, const arma::mat &ytp, const arma::uvec &xi, 
                        const arma::mat &x_covariates, const arma::mat &beta_tp, 
                        const arma::mat &alpha_tp, const arma::vec &e_tp, const arma::vec &e1){
  int err, ninit = 4, npoint = 100, nsamp = 1, ncent = 0;
  int neval;
  double xl = -100.0, xr = 100.0;
  double xinit[10]={-6.0, -2.0, 2.0, 6.0}, xsamp[100], xcent[10], qcent[10] = {5., 30., 70., 95.};
  double convex = 1.;
  int dometrop = 1;
  double xprev = 0.0;
  
  log_bj_tp_param bj_tp_data;
  bj_tp_data.b_tp = b_tp;
  bj_tp_data.ytp = ytp;
  bj_tp_data.xi_3_loc = arma::find(xi==3);
  bj_tp_data.xi_4_loc = arma::find(xi==4);
  bj_tp_data.x_covariates = x_covariates;
  bj_tp_data.beta_tp = beta_tp;
  bj_tp_data.alpha_tp = alpha_tp;
  bj_tp_data.e_tp = e_tp;
  bj_tp_data.e1 = e1;
  for(unsigned int jj = 0; jj < b_tp.n_elem; ++jj){
    bj_tp_data.jj = jj;
    err = arms(xinit,ninit,&xl,&xr,log_bj_tp,&bj_tp_data,&convex,
               npoint,dometrop,&xprev,xsamp,nsamp,qcent,xcent,ncent,&neval);
    if(err>0){
      Rprintf("error code: %d", err);
      Rcpp::stop("\n");
    }
    //Rprintf("jj=%d, log_b_tp_jj=%f",jj, log_bj_tp())
    bj_tp_data.b_tp(jj) = xsamp[0];
  }
  return bj_tp_data.b_tp;
}
// [[Rcpp::export]]
arma::vec sample_b_fp_c(arma::vec b_fp, const arma::mat &yfp, const arma::uvec &xi, 
                        const arma::mat &x_covariates, const arma::mat &beta_fp, 
                        const arma::mat &alpha_fp, const arma::vec &e_fp, const arma::vec &e2){
  int err, ninit = 4, npoint = 100, nsamp = 1, ncent = 0 ;
  int neval;
  double xl = -100.0, xr = 100.0;
  double xinit[10]={-6.0, -2.0, 2.0, 6.0}, xsamp[100], xcent[10], qcent[10] = {5., 30., 70., 95.};
  double convex = 1.;
  int dometrop = 1;
  double xprev = 0.0;
  
  log_bj_fp_param bj_fp_data;
  bj_fp_data.b_fp = b_fp;
  bj_fp_data.yfp = yfp;
  bj_fp_data.xi_2_loc = arma::find(xi==2);
  bj_fp_data.xi_4_loc = arma::find(xi==4);
  bj_fp_data.x_covariates = x_covariates;
  bj_fp_data.beta_fp = beta_fp;
  bj_fp_data.alpha_fp = alpha_fp;
  bj_fp_data.e_fp = e_fp;
  bj_fp_data.e2 = e2;
  
  for(unsigned int jj = 0; jj < b_fp.n_elem; ++jj){
    bj_fp_data.jj = jj;
    err = arms(xinit,ninit,&xl,&xr,log_bj_fp,&bj_fp_data,&convex,
               npoint,dometrop,&xprev,xsamp,nsamp,qcent,xcent,ncent,&neval);
    if(err>0){
      Rprintf("error code: %d", err);
      Rcpp::stop("\n");
    }
    bj_fp_data.b_fp(jj) = xsamp[0];
  }
  return bj_fp_data.b_fp;
}
// [[Rcpp::export]]
arma::vec sample_b_linear_reg(arma::vec e_tp1, arma::vec e_tp2, arma::vec e_fp2,
                              arma::mat x_covariates1, arma::mat x_covariates2,
                              arma::vec b_tp, arma::vec b_fp, 
                              double sig2_tp, double sig2_fp, double rho){
  double sig2_0 = 100;
  arma::vec eta_tp1 = x_covariates1 * b_tp + e_tp1;
  arma::vec eta_tp2 = x_covariates2 * b_tp + e_tp2;
  arma::vec eta_fp2 = x_covariates2 * b_fp + e_fp2;
  arma::mat S_n = x_covariates1.t() * x_covariates1 / sig2_tp +
    x_covariates2.t() * x_covariates2 / sig2_tp / (1 - rho * rho);
  S_n.diag() += 1.0 / sig2_0;
  S_n = arma::inv_sympd(S_n);
  arma::vec mu_n = S_n * (x_covariates1.t() * eta_tp1 / sig2_tp +
    x_covariates2.t() * eta_tp2 / sig2_tp / (1-rho*rho) - 
    rho/(1-rho*rho)/std::sqrt(sig2_tp * sig2_fp) * x_covariates2.t() * (eta_fp2 - x_covariates2 * b_fp));
  return arma::mvnrnd(mu_n, S_n);
}

// [[Rcpp::export]]
arma::vec sample_b_tp_new(arma::mat Sigma, arma::mat e1234, arma::mat x_covariates, arma::vec b_tp){
  double sig2_0 = 100;
  arma::vec sigma_e1_12(3);
  sigma_e1_12(0) = Sigma(1,0);
  sigma_e1_12(1) = Sigma(2,0);
  sigma_e1_12(2) = Sigma(3,0);
  arma::uvec ind_e1 = {1,2,3};
  arma::mat sigma_e1_22_inv = arma::inv_sympd(Sigma(ind_e1, ind_e1));
  // arma::vec eta_tp = e_tp1 + x_covariates1 * b_tp;
  arma::vec eta_tp = e1234.col(0) + x_covariates * b_tp;
  double sig2_e1 = Sigma(0,0) - arma::as_scalar(sigma_e1_12.t() * sigma_e1_22_inv * sigma_e1_12);
  arma::mat S_n = x_covariates.t() * x_covariates / sig2_e1;
  S_n.diag() += 1.0 / sig2_0;
  S_n = arma::inv_sympd(S_n);
  arma::vec mu_n = S_n * (x_covariates.t() * (eta_tp - e1234.cols(ind_e1) * sigma_e1_22_inv * sigma_e1_12) / sig2_e1);
  
  return arma::mvnrnd(mu_n, S_n);
}
// [[Rcpp::export]]
arma::vec sample_b_fp_new(arma::mat Sigma, arma::mat e1234, arma::mat x_covariates, arma::vec b_fp){
  double sig2_0 = 100;
  arma::vec sigma_e2_12(3);
  sigma_e2_12(0) = Sigma(0,1);
  sigma_e2_12(1) = Sigma(2,1);
  sigma_e2_12(2) = Sigma(3,1);
  arma::uvec ind_e2 = {0,2,3};
  arma::mat sigma_e2_22_inv = arma::inv_sympd(Sigma(ind_e2, ind_e2));
  //arma::vec eta_fp1 = e_fp1 + x_covariates1 * b_fp;
  arma::vec eta_fp = e1234.col(1) + x_covariates * b_fp;
  double sig2_e2 = Sigma(1,1) - arma::as_scalar(sigma_e2_12.t() * sigma_e2_22_inv * sigma_e2_12);
  arma::mat S_n = x_covariates.t() * x_covariates / sig2_e2;
  S_n.diag() += 1.0 / sig2_0;
  S_n = arma::inv_sympd(S_n);
  arma::vec mu_n = S_n * (x_covariates.t() * (eta_fp - e1234.cols(ind_e2) * sigma_e2_22_inv * sigma_e2_12) / sig2_e2);
  
  return arma::mvnrnd(mu_n, S_n);
}

arma::vec sample_b_new1(arma::mat Sigma1, arma::mat Sigma2, arma::mat e1234, arma::mat x_covariates, 
                        arma::vec b_tp, arma::vec female, bool tp_flag){
  arma::uvec female_loc = arma::find(female == 1); 
  arma::uvec male_loc = arma::find(female == 0);
  arma::mat x_cov_female = x_covariates.rows(female_loc);
  arma::mat x_cov_male = x_covariates.rows(male_loc);
  arma::mat e1234_female = e1234.rows(female_loc);
  arma::mat e1234_male = e1234.rows(male_loc);
  
  double sig2_0 = 100;
  
  arma::uvec ind_e1(3);
  arma::vec sigma_e1_12_female(3);
  arma::vec sigma_e1_12_male(3);
  arma::vec eta_tp_female;
  arma::mat sigma_e1_22_inv_female;
  double sig2_e1_female;
  arma::vec eta_tp_male;
  arma::mat sigma_e1_22_inv_male;
  double sig2_e1_male;
  if(tp_flag){
    ind_e1 = {1,2,3};
    sigma_e1_12_female(0) = Sigma1(1,0);
    sigma_e1_12_female(1) = Sigma1(2,0);
    sigma_e1_12_female(2) = Sigma1(3,0);
    sigma_e1_12_male(0) = Sigma2(1,0);
    sigma_e1_12_male(1) = Sigma2(2,0);
    sigma_e1_12_male(2) = Sigma2(3,0);
    eta_tp_female = e1234_female.col(0) + x_cov_female * b_tp;
    sigma_e1_22_inv_female = arma::inv_sympd(Sigma1(ind_e1, ind_e1));
    sig2_e1_female = Sigma1(0,0) - arma::as_scalar(sigma_e1_12_female.t() * sigma_e1_22_inv_female * sigma_e1_12_female);
    eta_tp_male = e1234_male.col(0) + x_cov_male * b_tp;
    sigma_e1_22_inv_male = arma::inv_sympd(Sigma2(ind_e1, ind_e1));
    sig2_e1_male = Sigma2(0,0) - arma::as_scalar(sigma_e1_12_male.t() * sigma_e1_22_inv_male * sigma_e1_12_male);
  }
  else{
    ind_e1 = {0,2,3};
    sigma_e1_12_female(0) = Sigma1(0,1);
    sigma_e1_12_female(1) = Sigma1(2,1);
    sigma_e1_12_female(2) = Sigma1(3,1);
    sigma_e1_12_male(0) = Sigma2(0,1);
    sigma_e1_12_male(1) = Sigma2(2,1);
    sigma_e1_12_male(2) = Sigma2(3,1);
    eta_tp_female = e1234_female.col(1) + x_cov_female * b_tp;
    sigma_e1_22_inv_female = arma::inv_sympd(Sigma1(ind_e1, ind_e1));
    sig2_e1_female = Sigma1(1,1) - arma::as_scalar(sigma_e1_12_female.t() * sigma_e1_22_inv_female * sigma_e1_12_female);

    eta_tp_male = e1234_male.col(1) + x_cov_male * b_tp;
    sigma_e1_22_inv_male = arma::inv_sympd(Sigma2(ind_e1, ind_e1));
    sig2_e1_male = Sigma2(1,1) - arma::as_scalar(sigma_e1_12_male.t() * sigma_e1_22_inv_male * sigma_e1_12_male);
  }

  arma::mat S_n = x_cov_female.t() * x_cov_female / sig2_e1_female + x_cov_male.t() * x_cov_male / sig2_e1_male;
  S_n.diag() += 1.0 / sig2_0;
  S_n = arma::inv_sympd(S_n);
  
  arma::vec mu_n = S_n * (x_cov_female.t() * (eta_tp_female - e1234_female.cols(ind_e1) * sigma_e1_22_inv_female * sigma_e1_12_female) / sig2_e1_female+
                          x_cov_male.t() * (eta_tp_male - e1234_male.cols(ind_e1) * sigma_e1_22_inv_male * sigma_e1_12_male) / sig2_e1_male);
  
  return arma::mvnrnd(mu_n, S_n);
}
arma::vec sample_b_general(const arma::vec &sig_all, const arma::mat &rho_mat,
                          const arma::mat &e1234, const arma::mat &x_covariates, 
                        const arma::vec &b_tp_fp, bool tp_flag){
  int p_num = b_tp_fp.n_rows;
  int nn = rho_mat.n_cols;

  double sig2_0 = 100;
  arma::mat S_n = arma::zeros(p_num, p_num);
  arma::vec mu_n = arma::zeros(p_num);
  if(tp_flag){
    arma::vec eta_tp_fp = e1234.col(0) + x_covariates * b_tp_fp;
    arma::uvec ind_e1 = {1,2,3};
    for(int i=0;i < nn; ++i){
      arma::mat Sigma_ii = construct_sigma1(sig_all, rho_mat.col(i));
      arma::vec x_cov_i = x_covariates.row(i).t();
      arma::vec e1234_i = e1234.row(i).t();
      arma::vec sigma_e1_12 = arma::zeros(3);
      sigma_e1_12(0) = Sigma_ii(1,0);
      sigma_e1_12(1) = Sigma_ii(2,0);
      sigma_e1_12(2) = Sigma_ii(3,0);
      arma::mat sigma_e1_22_inv = arma::inv_sympd(Sigma_ii(ind_e1, ind_e1));
      double sig2_e1 = Sigma_ii(0,0) - arma::as_scalar(sigma_e1_12.t() * sigma_e1_22_inv * sigma_e1_12);
      S_n += x_cov_i * x_cov_i.t() / sig2_e1;
      mu_n += x_cov_i * (eta_tp_fp(i) - arma::as_scalar(e1234_i(ind_e1).t() * sigma_e1_22_inv * sigma_e1_12)) / sig2_e1;
    }
  }
  else{
    arma::vec eta_tp_fp = e1234.col(1) + x_covariates * b_tp_fp;
    arma::uvec ind_e1 = {0,2,3};
    for(int i=0;i < nn; ++i){
      arma::mat Sigma_ii = construct_sigma1(sig_all, rho_mat.col(i));
      arma::vec x_cov_i = x_covariates.row(i).t();
      arma::vec e1234_i = e1234.row(i).t();
      arma::vec sigma_e1_12 = arma::zeros(3);
      sigma_e1_12(0) = Sigma_ii(0,1);
      sigma_e1_12(1) = Sigma_ii(2,1);
      sigma_e1_12(2) = Sigma_ii(3,1);
      arma::mat sigma_e1_22_inv = arma::inv_sympd(Sigma_ii(ind_e1, ind_e1));
      double sig2_e1 = Sigma_ii(1,1) - arma::as_scalar(sigma_e1_12.t() * sigma_e1_22_inv * sigma_e1_12);
      S_n += x_cov_i * x_cov_i.t() / sig2_e1;
      mu_n += x_cov_i * (eta_tp_fp(i) - arma::as_scalar(e1234_i(ind_e1).t() * sigma_e1_22_inv * sigma_e1_12)) / sig2_e1;
    }
  }
  S_n.diag() += 1.0 / sig2_0;
  S_n = arma::inv_sympd(S_n);
  mu_n = S_n * mu_n;
  
  return arma::mvnrnd(mu_n, S_n);
}
arma::vec sample_b_new2(arma::mat Sigma1, arma::mat Sigma2, arma::mat Sigma3, arma::mat Sigma4, 
                        arma::mat e1234, arma::mat x_covariates, arma::vec b_tp, 
                        arma::vec female, arma::vec distlong, bool tp_flag){
  arma::uvec female_loc = arma::find(female==1 && distlong == 1);
  arma::uvec male_loc = arma::find(female==0 && distlong == 1);
  arma::uvec part3_loc = arma::find(female==1 && distlong == 0);
  arma::uvec part4_loc = arma::find(female==0 && distlong == 0);
  
  arma::mat x_cov_female = x_covariates.rows(female_loc);
  arma::mat x_cov_male = x_covariates.rows(male_loc);
  arma::mat x_cov_part3 = x_covariates.rows(part3_loc);
  arma::mat x_cov_part4 = x_covariates.rows(part4_loc);
  
  arma::mat e1234_female = e1234.rows(female_loc);
  arma::mat e1234_male = e1234.rows(male_loc);
  arma::mat e1234_part3 = e1234.rows(part3_loc);
  arma::mat e1234_part4 = e1234.rows(part4_loc);
  
  double sig2_0 = 100;
  
  arma::uvec ind_e1(3);
  arma::vec sigma_e1_12_female(3);
  arma::vec sigma_e1_12_male(3);
  arma::vec sigma_e1_12_part3(3);
  arma::vec sigma_e1_12_part4(3);
  
  arma::vec eta_tp_female;
  arma::mat sigma_e1_22_inv_female;
  double sig2_e1_female;
  arma::vec eta_tp_male;
  arma::mat sigma_e1_22_inv_male;
  double sig2_e1_male;
  arma::vec eta_tp_part3;
  arma::mat sigma_e1_22_inv_part3;
  double sig2_e1_part3;
  arma::vec eta_tp_part4;
  arma::mat sigma_e1_22_inv_part4;
  double sig2_e1_part4;
  if(tp_flag){
    ind_e1 = {1,2,3};
    sigma_e1_12_female(0) = Sigma1(1,0);
    sigma_e1_12_female(1) = Sigma1(2,0);
    sigma_e1_12_female(2) = Sigma1(3,0);
    sigma_e1_12_male(0) = Sigma2(1,0);
    sigma_e1_12_male(1) = Sigma2(2,0);
    sigma_e1_12_male(2) = Sigma2(3,0);
    sigma_e1_12_part3(0) = Sigma3(1,0);
    sigma_e1_12_part3(1) = Sigma3(2,0);
    sigma_e1_12_part3(2) = Sigma3(3,0);
    sigma_e1_12_part4(0) = Sigma4(1,0);
    sigma_e1_12_part4(1) = Sigma4(2,0);
    sigma_e1_12_part4(2) = Sigma4(3,0);
    
    eta_tp_female = e1234_female.col(0) + x_cov_female * b_tp;
    sigma_e1_22_inv_female = arma::inv_sympd(Sigma1(ind_e1, ind_e1));
    sig2_e1_female = Sigma1(0,0) - arma::as_scalar(sigma_e1_12_female.t() * sigma_e1_22_inv_female * sigma_e1_12_female);
    eta_tp_male = e1234_male.col(0) + x_cov_male * b_tp;
    sigma_e1_22_inv_male = arma::inv_sympd(Sigma2(ind_e1, ind_e1));
    sig2_e1_male = Sigma2(0,0) - arma::as_scalar(sigma_e1_12_male.t() * sigma_e1_22_inv_male * sigma_e1_12_male);
    eta_tp_part3 = e1234_part3.col(0) + x_cov_part3 * b_tp;
    sigma_e1_22_inv_part3 = arma::inv_sympd(Sigma3(ind_e1, ind_e1));
    sig2_e1_part3 = Sigma3(0,0) - arma::as_scalar(sigma_e1_12_part3.t() * sigma_e1_22_inv_part3 * sigma_e1_12_part3);
    eta_tp_part4 = e1234_part4.col(0) + x_cov_part4 * b_tp;
    sigma_e1_22_inv_part4 = arma::inv_sympd(Sigma4(ind_e1, ind_e1));
    sig2_e1_part4 = Sigma4(0,0) - arma::as_scalar(sigma_e1_12_part4.t() * sigma_e1_22_inv_part4 * sigma_e1_12_part4);
  }
  else{
    ind_e1 = {0,2,3};
    sigma_e1_12_female(0) = Sigma1(0,1);
    sigma_e1_12_female(1) = Sigma1(2,1);
    sigma_e1_12_female(2) = Sigma1(3,1);
    sigma_e1_12_male(0) = Sigma2(0,1);
    sigma_e1_12_male(1) = Sigma2(2,1);
    sigma_e1_12_male(2) = Sigma2(3,1);
    sigma_e1_12_part3(0) = Sigma3(0,1);
    sigma_e1_12_part3(1) = Sigma3(2,1);
    sigma_e1_12_part3(2) = Sigma3(3,1);
    sigma_e1_12_part4(0) = Sigma4(0,1);
    sigma_e1_12_part4(1) = Sigma4(2,1);
    sigma_e1_12_part4(2) = Sigma4(3,1);
    eta_tp_female = e1234_female.col(1) + x_cov_female * b_tp;
    sigma_e1_22_inv_female = arma::inv_sympd(Sigma1(ind_e1, ind_e1));
    sig2_e1_female = Sigma1(1,1) - arma::as_scalar(sigma_e1_12_female.t() * sigma_e1_22_inv_female * sigma_e1_12_female);
    
    eta_tp_male = e1234_male.col(1) + x_cov_male * b_tp;
    sigma_e1_22_inv_male = arma::inv_sympd(Sigma2(ind_e1, ind_e1));
    sig2_e1_male = Sigma2(1,1) - arma::as_scalar(sigma_e1_12_male.t() * sigma_e1_22_inv_male * sigma_e1_12_male);
    
    eta_tp_part3 = e1234_part3.col(1) + x_cov_part3 * b_tp;
    sigma_e1_22_inv_part3 = arma::inv_sympd(Sigma3(ind_e1, ind_e1));
    sig2_e1_part3 = Sigma3(1,1) - arma::as_scalar(sigma_e1_12_part3.t() * sigma_e1_22_inv_part3 * sigma_e1_12_part3);
    
    eta_tp_part4 = e1234_part4.col(1) + x_cov_part4 * b_tp;
    sigma_e1_22_inv_part4 = arma::inv_sympd(Sigma4(ind_e1, ind_e1));
    sig2_e1_part4 = Sigma4(1,1) - arma::as_scalar(sigma_e1_12_part4.t() * sigma_e1_22_inv_part4 * sigma_e1_12_part4);
  }
  
  arma::mat S_n = x_cov_female.t() * x_cov_female / sig2_e1_female + 
                  x_cov_male.t() * x_cov_male / sig2_e1_male +
                  x_cov_part3.t() * x_cov_part3 / sig2_e1_part3 +
                  x_cov_part4.t() * x_cov_part4 / sig2_e1_part4;
  S_n.diag() += 1.0 / sig2_0;
  S_n = arma::inv_sympd(S_n);
  
  arma::vec mu_n = S_n * (x_cov_female.t() * (eta_tp_female - e1234_female.cols(ind_e1) * sigma_e1_22_inv_female * sigma_e1_12_female) / sig2_e1_female+
                          x_cov_male.t() * (eta_tp_male - e1234_male.cols(ind_e1) * sigma_e1_22_inv_male * sigma_e1_12_male) / sig2_e1_male,
                          x_cov_part3.t() * (eta_tp_part3 - e1234_part3.cols(ind_e1) * sigma_e1_22_inv_part3 * sigma_e1_12_part3) / sig2_e1_part3,
                          x_cov_part4.t() * (eta_tp_part4 - e1234_part4.cols(ind_e1) * sigma_e1_22_inv_part4 * sigma_e1_12_part4) / sig2_e1_part4);
  
  return arma::mvnrnd(mu_n, S_n);
}
