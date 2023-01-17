// [[Rcpp::depends(RcppArmadillo, RcppDist)]]
#include <RcppDist.h>
#include "arms_ori.h"
#include "depend_funcs.h"
struct log_sig2_param{
  double sig2_fp_tp;
  double rho;
  arma::vec e_tp_fp;
  arma::mat e12;
};
struct log_rho_params{
  arma::mat e12;
  double sig2_tp;
  double sig2_fp;
};

double log_sig2_tp_c(double x, void* params){
  struct log_sig2_param *d;
  d = static_cast<struct log_sig2_param *> (params);
  double sig2_fp = (d->sig2_fp_tp);
  arma::vec e_tp = (d->e_tp_fp);
  double rho = d->rho;
  double inv_gamma_alpha = 0.01;
  double inv_gamma_beta = 0.01;

  arma::vec e1 = (d->e12).col(0);
  arma::vec e2 = (d->e12).col(1);

  double res = -(inv_gamma_alpha + 1 + 0.5 * 2 * e_tp.n_elem) * std::log(x);
  res += - ( 0.5 * arma::accu(arma::square(e_tp)) + inv_gamma_beta) / x;
  res += - 0.5 / (1-rho * rho) * (arma::accu(arma::square(e1)) / x -
    2*rho*arma::accu(e1 % e2) / std::sqrt(x * sig2_fp));
  return res;
}
double log_sig2_fp_c(double x, void* params){
  struct log_sig2_param *d;
  d = static_cast<struct log_sig2_param *> (params);
  double sig2_tp = (d->sig2_fp_tp);
  arma::vec e_fp = (d->e_tp_fp);
  double rho = d->rho;
  double inv_gamma_alpha = 0.01;
  double inv_gamma_beta = 0.01;
  
  arma::vec e1 = (d->e12).col(0);
  arma::vec e2 = (d->e12).col(1);
  
  double res = -(inv_gamma_alpha + 1 + 0.5 * 2 * e_fp.n_elem) * std::log(x);
  res += - ( 0.5 * arma::accu(arma::square(e_fp)) + inv_gamma_beta) / x;
  res += - 0.5 / (1-rho * rho) * (arma::accu(arma::square(e2)) / x -
    2*rho*arma::accu(e1 % e2) / std::sqrt(x * sig2_tp));
  return res;
}
// [[Rcpp::export]]
double sample_sig2_tp_c(double sig2_tp, double sig2_fp, double rho,
                      arma::vec e_tp, arma::mat e12, arma::uvec xi){
  
  int err, ninit = 4, npoint = 100, nsamp = 1, ncent = 0 ;
  int neval;
  double xinit[10]={0.5, 2.0, 4.0, 8.0}, xl = 0.0, xr = 100.0;
  double xsamp[100], xcent[10], qcent[10] = {5., 30., 70., 95.};
  double convex = 1.;
  int dometrop = 1;
  double xprev = sig2_tp;
  
  log_sig2_param log_sig2_data;
  log_sig2_data.sig2_fp_tp = sig2_fp;
  log_sig2_data.rho = rho;
  log_sig2_data.e_tp_fp = e_tp;
  log_sig2_data.e12 = e12;
  
  err = arms(xinit,ninit,&xl,&xr,log_sig2_tp_c,&log_sig2_data,&convex,
             npoint,dometrop,&xprev,xsamp,nsamp,qcent,xcent,ncent,&neval);
  if(err>0){
    Rprintf("error code: %d", err);
    Rcpp::stop("\n");
  }
  return xsamp[0];
}
// [[Rcpp::export]]
double sample_sig2_fp_c(double sig2_tp, double sig2_fp, double rho,
                        arma::vec e_fp, arma::mat e12, arma::uvec xi){
  
  int err, ninit = 4, npoint = 100, nsamp = 1, ncent = 0 ;
  int neval;
  double xinit[10]={0.5, 2.0, 4.0, 8.0}, xl = 0.0, xr = 100.0;
  double xsamp[100], xcent[10], qcent[10] = {5., 30., 70., 95.};
  double convex = 1.;
  int dometrop = 1;
  double xprev = sig2_fp;
  
  log_sig2_param log_sig2_data;
  log_sig2_data.sig2_fp_tp = sig2_tp;
  log_sig2_data.rho = rho;
  log_sig2_data.e_tp_fp = e_fp;
  log_sig2_data.e12 = e12;
  
  err = arms(xinit,ninit,&xl,&xr,log_sig2_fp_c,&log_sig2_data,&convex,
             npoint,dometrop,&xprev,xsamp,nsamp,qcent,xcent,ncent,&neval);
  if(err>0){
    Rprintf("error code: %d", err);
    Rcpp::stop("\n");
  }
  return xsamp[0];
}
 
double log_rho_c(double x, void* params){
  struct log_rho_params *d;
  d = static_cast<struct log_rho_params *> (params);
  arma::vec e1 = (d->e12).col(0);
  arma::vec e2 = (d->e12).col(1);
  double sig2_tp = d->sig2_tp;
  double sig2_fp = d->sig2_fp;
  double res = -0.5 * e1.n_elem * std::log( 1 - x * x) -
    0.5 / (1 - x * x) * 
    (arma::accu(arma::square(e1)) / sig2_tp + arma::accu(arma::square(e2)) / sig2_fp -
    2* x * arma::accu(e1 % e2) / std::sqrt(sig2_tp * sig2_fp));
  
  return res;
}
// [[Rcpp::export]]
double sample_rho_c(double rho, const arma::uvec &xi, double sig2_tp, double sig2_fp, const arma::mat &e12){
  int err, ninit = 4, npoint = 100, nsamp = 1, ncent = 0 ;
  int neval;
  double xinit[10] = {-0.4, 0.2, 0.4, 0.6}, xl = -1.0, xr = 1.0;
  double xsamp[100], xcent[10], qcent[10] = {5., 30., 70., 95.};
  double convex = 1.;
  int dometrop = 1; // this is a log-concave function
  double xprev = rho;
  
  log_rho_params log_rho_data;
  log_rho_data.sig2_tp = sig2_tp;
  log_rho_data.sig2_fp = sig2_fp;
  log_rho_data.e12 = e12;
  
  err = arms(xinit, ninit,&xl,&xr,log_rho_c,&log_rho_data,&convex,
             npoint,dometrop,&xprev,xsamp,nsamp,qcent,xcent,ncent,&neval);
  if(err>0){
    Rprintf("error code: %d", err);
    Rcpp::stop("\n");
  }
  return xsamp[0];
}

struct log_sig2_new_param{
  double sig2_fp_tp;
  arma::vec rho;
  arma::mat e1234;
};

arma::mat construct_sigma(double sig2_tp, double sig2_fp, arma::vec rho){
  arma::mat Sigma(4,4);
  Sigma(0,0) = sig2_tp;
  Sigma(0,1) = rho(0) * std::sqrt(sig2_tp*sig2_fp);
  Sigma(0,2) = rho(1) * std::sqrt(sig2_tp);
  Sigma(0,3) = rho(2) * std::sqrt(sig2_tp);
  Sigma(1,1) = sig2_fp;
  Sigma(1,2) = rho(3) * std::sqrt(sig2_fp);
  Sigma(1,3) = rho(4) * std::sqrt(sig2_fp);
  Sigma(2,2) = 1;
  Sigma(2,3) = rho(5);
  Sigma(3,3) = 1;
  Sigma(1,0) = Sigma(0,1);
  Sigma(2,0) = Sigma(0,2);
  Sigma(3,0) = Sigma(0,3);
  Sigma(2,1) = Sigma(1,2);
  Sigma(3,1) = Sigma(1,3);
  Sigma(3,2) = Sigma(2,3);
  if(!Sigma.is_symmetric()){
    Rprintf("Sigma not symmetric!");
    Rcpp::Rcout << Sigma << sig2_tp << sig2_fp << rho << std::endl;
    Rcpp::stop("\n");
  }
  return Sigma;
}
double log_sig2_tp_new(double x, void* params){
  struct log_sig2_new_param *d;
  d = static_cast<struct log_sig2_new_param *> (params);
  double inv_gamma_alpha = 0.01;
  double inv_gamma_beta = 0.01;
  
  int N = d->e1234.n_rows;
  arma::mat Sigma = construct_sigma(x, d->sig2_fp_tp, d->rho);
  arma::mat Sigma_inv = arma::inv_sympd(Sigma);
  
  double res = -(inv_gamma_alpha + 1) * std::log(x) - inv_gamma_beta / x;
  // res += - ( 0.5 * arma::accu(arma::square(d->e_tp_fp)) + inv_gamma_beta) / x;
  res += -0.5 * N * std::log(arma::det(Sigma));
  for(auto i=0;i<N;++i){
    arma::rowvec e1234_row_i = d->e1234.row(i);
    res += -0.5 * arma::as_scalar(e1234_row_i * Sigma_inv * e1234_row_i.t());
  }
  return res;
}
double log_sig2_fp_new(double x, void* params){
  struct log_sig2_new_param *d;
  d = static_cast<struct log_sig2_new_param *> (params);
  double inv_gamma_alpha = 0.01;
  double inv_gamma_beta = 0.01;
  
  int N = d->e1234.n_rows;
  arma::mat Sigma = construct_sigma(d->sig2_fp_tp, x, d->rho);
  arma::mat Sigma_inv = arma::inv_sympd(Sigma);
  
  double res = -(inv_gamma_alpha + 1) * std::log(x) - inv_gamma_beta / x;
  // res += - ( 0.5 * arma::accu(arma::square(d->e_tp_fp)) + inv_gamma_beta) / x;
  res += -0.5 * N * std::log(arma::det(Sigma));
  for(auto i=0;i<N;++i){
    arma::rowvec e1234_row_i = d->e1234.row(i);
    res += -0.5 * arma::as_scalar(e1234_row_i * Sigma_inv * e1234_row_i.t());
  }
  
  return res;
}
double sample_sig2_tp_new(arma::mat e1234, double sig2_tp, double sig2_fp, arma::vec rho){
  int err, ninit = 4, npoint = 100, nsamp = 1, ncent = 0 ;
  int neval;
  double xinit[10]={0.5, 2.0, 4.0, 8.0}, xl = 0.0, xr = 100.0;
  double xsamp[100], xcent[10], qcent[10] = {5., 30., 70., 95.};
  double convex = 1.;
  int dometrop = 1;
  double xprev = sig2_tp;
  
  log_sig2_new_param log_sig2_new_data;
  log_sig2_new_data.e1234 = e1234;
  log_sig2_new_data.rho = rho;
  log_sig2_new_data.sig2_fp_tp = sig2_fp;
  
  err = arms(xinit,ninit,&xl,&xr,log_sig2_tp_new,&log_sig2_new_data,&convex,
             npoint,dometrop,&xprev,xsamp,nsamp,qcent,xcent,ncent,&neval);
  if(err>0){
    Rprintf("error code: %d", err);
    Rcpp::stop("\n");
  }
  return xsamp[0];
}

double sample_sig2_fp_new(arma::mat e1234, double sig2_tp, double sig2_fp, arma::vec rho){
  int err, ninit = 4, npoint = 100, nsamp = 1, ncent = 0 ;
  int neval;
  double xinit[10]={0.5, 2.0, 4.0, 8.0}, xl = 0.0, xr = 100.0;
  double xsamp[100], xcent[10], qcent[10] = {5., 30., 70., 95.};
  double convex = 1.;
  int dometrop = 1;
  double xprev = sig2_fp;
  
  log_sig2_new_param log_sig2_new_data;
  log_sig2_new_data.e1234 = e1234;
  log_sig2_new_data.rho = rho;
  log_sig2_new_data.sig2_fp_tp = sig2_tp;
  
  err = arms(xinit,ninit,&xl,&xr,log_sig2_fp_new,&log_sig2_new_data,&convex,
             npoint,dometrop,&xprev,xsamp,nsamp,qcent,xcent,ncent,&neval);
  if(err>0){
    Rprintf("error code: %d", err);
    Rcpp::stop("\n");
  }
  return xsamp[0];
}

struct log_rho_new_param{
  arma::mat e1234;
  double sig2_tp;
  double sig2_fp;
  arma::vec rho;
  unsigned int rho_loc;
};

double log_rho_new(double x, void* params){
  struct log_rho_new_param *d;
  d = static_cast<struct log_rho_new_param *> (params);
  
  int N = d->e1234.n_rows;
  arma::vec rho = d->rho;
  if(!std::isnan(x)){
    rho(d->rho_loc) = x;
  }
  else{
    Rprintf("Warning! x is nan");
    Rcpp::Rcout << "i = " << d->rho_loc << std::endl;
  }

  arma::mat Sigma = construct_sigma(d->sig2_tp, d->sig2_fp, rho);
  arma::mat Sigma_inv = arma::inv_sympd(Sigma);
  
  double res = -0.5 * N * std::log(arma::det(Sigma)) - std::log(1-x*x);

  for(auto i=0;i<N;++i){
    arma::rowvec e1234_row_i = d->e1234.row(i);
    res += -0.5 * arma::as_scalar(e1234_row_i * Sigma_inv * e1234_row_i.t());
  }
  return res;
}
//' @export
// [[Rcpp::export]]
double log_rho_export(double x, unsigned int rho_loc, arma::mat e1234, double sig2_tp, double sig2_fp, arma::vec rho){
  log_rho_new_param log_rho_new_data;
  log_rho_new_data.e1234 = e1234;
  log_rho_new_data.rho = rho;
  log_rho_new_data.rho_loc = rho_loc;
  log_rho_new_data.sig2_fp = sig2_fp;
  log_rho_new_data.sig2_tp = sig2_tp;
  
  return log_rho_new(x, &log_rho_new_data);
}
arma::vec sample_rho_new(arma::mat e1234, double sig2_tp, double sig2_fp, arma::vec rho, arma::vec xinit_center){
  int err, ninit = 5, npoint = 100, nsamp = 1, ncent = 0 ;
  int neval;
  double xinit[10] = {0.1, 0.2, 0.4, 0.6, 0.7}, xl = -1.0, xr = 1.0;
  double xsamp[100], xcent[10], qcent[10] = {5., 30., 70., 95.};
  double convex = 50.;
  int dometrop = 1;
  double xprev = 0.0;
  
  log_rho_new_param log_rho_new_data;
  log_rho_new_data.e1234 = e1234;
  log_rho_new_data.sig2_fp = sig2_fp;
  log_rho_new_data.sig2_tp = sig2_tp;
  log_rho_new_data.rho = rho;
  
  for(unsigned int i=0;i<6;++i){
    xprev = log_rho_new_data.rho(i);
    xinit[0] = xinit_center(i)-0.1;
    xinit[1] = xinit_center(i)-0.05;
    xinit[2] = xinit_center(i);
    xinit[3] = xinit_center(i)+0.05;
    xinit[4] = xinit_center(i)+0.1; 
    log_rho_new_data.rho_loc = i;

    err = arms(xinit, ninit,&xl,&xr,log_rho_new,&log_rho_new_data,&convex,
               npoint,dometrop,&xprev,xsamp,nsamp,qcent,xcent,ncent,&neval);
    if(err>0){
      Rprintf("error code: %d", err);
      Rcpp::stop("\n");
    }
    log_rho_new_data.rho(i) = xsamp[0];
  }
  return log_rho_new_data.rho;
}

// [[Rcpp::export]]
arma::mat sample_Sigma_new1(const arma::mat &e1234){
  arma::mat S = arma::eye(4,4) + e1234.t() * e1234;
  return riwish(4 + e1234.n_rows, S);
}

//' @export
// [[Rcpp::export]]
void sample_rho(const arma::mat &e1234, const arma::vec &sig_all, arma::vec &rho, double step_rho){
  int N = e1234.n_rows;
  arma::mat Sigma = construct_sigma1(sig_all, rho);

  arma::vec perturb(Rcpp::as<arma::vec>(Rcpp::rnorm(6)));
  arma::vec rho_pro = rho + step_rho * perturb;
  arma::mat Sigma_propose = construct_sigma1(sig_all, rho_pro);

  if(Sigma_propose.is_sympd()){
    arma::mat Sigma_inv = arma::inv_sympd(Sigma);
    arma::mat Sigma_pro_inv = arma::inv_sympd(Sigma_propose);
    arma::mat eet = e1234.t() * e1234;
    double log_det1, log_det2;
    double sign_det1, sign_det2;
    arma::log_det(log_det1, sign_det1, Sigma_propose);
    arma::log_det(log_det2, sign_det2, Sigma);
    double odds = std::exp(-0.5 * N * (log_det1 - log_det2) -
                           0.5 * (arma::trace(eet * Sigma_pro_inv) - arma::trace(eet * Sigma_inv)));
    if(R::runif(0,1) < odds){
      rho = rho_pro;
    }
  }
}
void sample_beta_rho(const arma::mat &e1234, const arma::vec &sig_all, const arma::mat &rho_covs,
                      arma::mat &beta_rho, double step_rho, unsigned int loc1, unsigned int loc2){
  int N = e1234.n_rows;

  arma::mat beta_rho_pro = beta_rho;
  // arma::vec perturb(Rcpp::as<arma::vec>(Rcpp::rnorm(6)));
  double perturb(R::rnorm(0, 1));
  beta_rho_pro(loc1, loc2) = beta_rho_pro(loc1, loc2) + step_rho * perturb;

  arma::mat rho_mat = fisher_z_trans(beta_rho * rho_covs.t());
  arma::mat rho_mat_pro = fisher_z_trans(beta_rho_pro * rho_covs.t());

  double log_odds = 0;
  for(int i=0;i<N;++i){
    arma::vec e1234_i = e1234.row(i).t();
    arma::mat Sigma_ii = construct_sigma1(sig_all, rho_mat.col(i));
    arma::mat Sigma_ii_pro = construct_sigma1(sig_all, rho_mat_pro.col(i));
    if(!Sigma_ii_pro.is_sympd()){
      Rcpp::Rcout << " rho reject ";
      return;
    };
    arma::mat Sigma_ii_inv = arma::inv_sympd(Sigma_ii);
    arma::mat Sigma_ii_pro_inv = arma::inv_sympd(Sigma_ii_pro);
    double log_det=0, log_det_pro=0, sign_det=0, sign_det_pro=0;
    arma::log_det(log_det_pro, sign_det_pro, Sigma_ii_pro);
    arma::log_det(log_det, sign_det, Sigma_ii);

    log_odds += -0.5 * (log_det_pro - log_det) - 
                0.5 * (arma::as_scalar(e1234_i.t() * Sigma_ii_pro_inv * e1234_i) - 
                        arma::as_scalar(e1234_i.t() * Sigma_ii_inv * e1234_i));
  }
  if(R::runif(0,1) < std::exp(log_odds)){
      beta_rho = beta_rho_pro;
      // Rcpp::Rcout << " beta_rho= " << beta_rho;
    }
}
//' @export
// [[Rcpp::export]]
void sample_sig2(const arma::mat &e1234, arma::vec &sig_all, const arma::vec &rho, double step_sigma){
  int N = e1234.n_rows;
  arma::mat Sigma = construct_sigma1(sig_all, rho);

  arma::vec perturb(Rcpp::as<arma::vec>(Rcpp::rnorm(2)));
  arma::vec sig_all_pro = sig_all + step_sigma * perturb;
  arma::mat Sigma_propose = construct_sigma1(sig_all_pro, rho);

  if(Sigma_propose.is_sympd()){
    arma::mat Sigma_inv = arma::inv_sympd(Sigma);
    arma::mat Sigma_pro_inv = arma::inv_sympd(Sigma_propose);
    arma::mat eet = e1234.t() * e1234;
    double log_det1, log_det2;
    double sign_det1, sign_det2;
    arma::log_det(log_det1, sign_det1, Sigma_propose);
    arma::log_det(log_det2, sign_det2, Sigma);
    double odds = std::exp(-0.5 * N * (log_det1 - log_det2) -
                           0.5 * (arma::trace(eet * Sigma_pro_inv) - arma::trace(eet * Sigma_inv)));
    if(R::runif(0,1) < odds){
      sig_all = sig_all_pro;
    }
  }
}
//' @export
// [[Rcpp::export]]
void sample_sig2_sep(const arma::mat &e1234_1, const arma::mat &e1234_2,
                     arma::vec &sig_all, const arma::vec &rho1, const arma::vec &rho2,
                     double step_sigma){
  int N1 = e1234_1.n_rows;
  int N2 = e1234_2.n_rows;
  arma::mat Sigma1 = construct_sigma1(sig_all, rho1);
  arma::mat Sigma2 = construct_sigma1(sig_all, rho2);

  arma::vec perturb(Rcpp::as<arma::vec>(Rcpp::rnorm(2)));
  arma::vec sig_all_pro = sig_all + step_sigma * perturb;
  arma::mat Sigma_propose1 = construct_sigma1(sig_all_pro, rho1);
  arma::mat Sigma_propose2 = construct_sigma1(sig_all_pro, rho2);

  if(Sigma_propose1.is_sympd() && Sigma_propose2.is_sympd()){
    arma::mat Sigma_inv1 = arma::inv_sympd(Sigma1);
    arma::mat Sigma_inv2 = arma::inv_sympd(Sigma2);
    arma::mat Sigma_pro_inv1 = arma::inv_sympd(Sigma_propose1);
    arma::mat Sigma_pro_inv2 = arma::inv_sympd(Sigma_propose2);
    arma::mat eet1 = e1234_1.t() * e1234_1;
    arma::mat eet2 = e1234_2.t() * e1234_2;

    double log_det1=0, log_det_pro1=0, log_det2=0, log_det_pro2=0;
    double sign_det1=0, sign_det_pro1=0, sign_det2=0, sign_det_pro2=0;
    arma::log_det(log_det_pro1, sign_det_pro1, Sigma_propose1);
    arma::log_det(log_det1, sign_det1, Sigma1);
    arma::log_det(log_det_pro2, sign_det_pro2, Sigma_propose2);
    arma::log_det(log_det2, sign_det2, Sigma2);
    double odds = std::exp(-0.5 * N1 * (log_det_pro1 - log_det1) - 0.5 * N2 * (log_det_pro2 - log_det2) -
                           0.5 * (arma::trace(eet1 * Sigma_pro_inv1) - arma::trace(eet1 * Sigma_inv1)) -
                           0.5 * (arma::trace(eet2 * Sigma_pro_inv2) - arma::trace(eet2 * Sigma_inv2)));
    if(R::runif(0,1) < odds){
      sig_all = sig_all_pro;
    }
  }

}
void sample_sig2_general(const arma::mat &e1234, arma::vec &sig_all, 
                        const arma::mat &rho_mat, double step_sigma){
  arma::vec perturb(Rcpp::as<arma::vec>(Rcpp::rnorm(2)));
  arma::vec sig_all_pro = sig_all + step_sigma * perturb;

  int N = e1234.n_rows;
  double log_odds = 0;
  for(int i=0; i< N; ++i){
    arma::mat Sigma_ii = construct_sigma1(sig_all, rho_mat.col(i));
    arma::mat Sigma_ii_pro = construct_sigma1(sig_all_pro, rho_mat.col(i));
    if(!Sigma_ii_pro.is_sympd()){
      return;
    }
    arma::mat Sigma_ii_inv = arma::inv_sympd(Sigma_ii);
    arma::mat Sigma_ii_pro_inv = arma::inv_sympd(Sigma_ii_pro);
    arma::vec e1234_i = e1234.row(i).t();
    double log_det = 0, log_det_pro = 0;
    double sign_det = 0, sign_det_pro = 0;
    arma::log_det(log_det, sign_det, Sigma_ii);
    arma::log_det(log_det_pro, sign_det_pro, Sigma_ii_pro);
    log_odds += -0.5 * (log_det_pro - log_det) -
                0.5 * (arma::as_scalar(e1234_i.t() * Sigma_ii_pro_inv * e1234_i) - 
                       arma::as_scalar(e1234_i.t() * Sigma_ii_inv * e1234_i));
  }
  if(R::runif(0,1) < std::exp(log_odds)){
      sig_all = sig_all_pro;
    }
  return;
}

