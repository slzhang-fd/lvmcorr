#include <RcppArmadillo.h>
#include "arms_ori.h"
#include "lvmcorr_omp.h"
#include "depend_funcs.h"

// [[Rcpp::depends(RcppArmadillo)]]
struct log_meaj_param{
  arma::vec ytp_fp_j;
  double alpha_tp_fp_j;
  double beta_tp_fp_j;
  arma::vec eta_tpfp;
};

double log_mea_betaj(double x, void* params){
  struct log_meaj_param *d;
  d = static_cast<struct log_meaj_param *> (params);
  
  double res = -1.0/200 * x * x; // normal prior with sd=10
  
  arma::vec tmp = x + d->alpha_tp_fp_j * d->eta_tpfp;
  arma::vec log_p1 = Rcpp::pnorm(Rcpp::NumericVector(tmp.begin(), tmp.end()), 0.0, 1.0, 1, 1);
  arma::vec log_p2 = Rcpp::pnorm(Rcpp::NumericVector(tmp.begin(), tmp.end()), 0.0, 1.0, 0, 1);
  arma::vec tmp1 = d->ytp_fp_j % log_p1 + (1 - d->ytp_fp_j) % log_p2; // nan in y_i
  res += arma::accu(tmp1(arma::find_finite(tmp1)));
  return res;
}

double log_mea_alphaj(double x, void* params){
  struct log_meaj_param *d;
  d = static_cast<struct log_meaj_param *> (params);
  
  double res = -1.0/200 * x * x; // normal prior with sd =10
  
  arma::vec tmp = d->beta_tp_fp_j + x * d->eta_tpfp;
  arma::vec log_p1 = Rcpp::pnorm(Rcpp::NumericVector(tmp.begin(), tmp.end()), 0.0, 1.0, 1, 1);
  arma::vec log_p2 = Rcpp::pnorm(Rcpp::NumericVector(tmp.begin(), tmp.end()), 0.0, 1.0, 0, 1);
  arma::vec tmp1 = d->ytp_fp_j % log_p1 + (1 - d->ytp_fp_j) % log_p2; // nan in y_i
  res += arma::accu(tmp1(arma::find_finite(tmp1)));
  return res;
}
//' @export
// [[Rcpp::export]]
void sample_alpha_beta(arma::vec &alpha_tpfp, arma::vec &beta_tpfp, const arma::mat &ytpfp,
                       arma::vec eta_tpfp){
  int err, ninit = 4, npoint = 100, nsamp = 1, ncent = 0;
  int neval;
  double xinit[10]={-3.0, -1.0, 1.0, 3.0}, xl = -5.0, xr = 5.0;
  double xsamp[100], xcent[10], qcent[10] = {5., 30., 70., 95.};
  double convex = 1.;
  int dometrop = 0;
  double xprev = 0.0;
  
  int J = ytpfp.n_cols;
  log_meaj_param log_meaj_data;
  log_meaj_data.eta_tpfp = eta_tpfp; // eta_tpfp = x_covariates * b_tpfp + e_tpfp
  for(int j=0;j<J;j++){
    log_meaj_data.ytp_fp_j = ytpfp.col(j);
    // sample beta_j_tpfp
    log_meaj_data.alpha_tp_fp_j = alpha_tpfp(j);
    log_meaj_data.beta_tp_fp_j = beta_tpfp(j);
    err = arms(xinit,ninit,&xl,&xr,log_mea_betaj,&log_meaj_data,&convex,
               npoint,dometrop,&xprev,xsamp,nsamp,qcent,xcent,ncent,&neval);
    if(err>0){
      Rprintf("error code: %d", err);
      Rcpp::stop("\n");
    }
    beta_tpfp(j) = xsamp[0];
    // sample alpha_j_tpfp
    log_meaj_data.alpha_tp_fp_j = alpha_tpfp(j);
    log_meaj_data.beta_tp_fp_j = beta_tpfp(j);
    err = arms(xinit,ninit,&xl,&xr,log_mea_alphaj,&log_meaj_data,&convex,
               npoint,dometrop,&xprev,xsamp,nsamp,qcent,xcent,ncent,&neval);
    if(err>0){
      Rprintf("error code: %d", err);
      Rcpp::stop("\n");
    }
    alpha_tpfp(j) = xsamp[0];
  }
  // return(Rcpp::List::create(Rcpp::Named("alpha_tpfp")= alpha_tpfp,
  //                           Rcpp::Named("beta_tpfp") = beta_tpfp));
}

struct log_etai_param{
  arma::vec ytp_fp_i;
  arma::vec alpha_tpfp;
  arma::vec beta_tpfp;
};

double log_eta_i(double x, void* params){
  struct log_etai_param *d;
  d = static_cast<struct log_etai_param *> (params);
  
  double res = -1.0/2 * x * x; // normal prior with sd=1
  
  arma::vec tmp = d->beta_tpfp + d->alpha_tpfp * x;
  arma::vec log_p1 = Rcpp::pnorm(Rcpp::NumericVector(tmp.begin(), tmp.end()), 0.0, 1.0, 1, 1);
  arma::vec log_p2 = Rcpp::pnorm(Rcpp::NumericVector(tmp.begin(), tmp.end()), 0.0, 1.0, 0, 1);
  arma::vec tmp1 = d->ytp_fp_i % log_p1 + (1 - d->ytp_fp_i) % log_p2; // nan in y_i
  res += arma::accu(tmp1(arma::find_finite(tmp1)));
  return res;
}
//' @export
// [[Rcpp::export]]
void sample_eta(arma::vec &eta_tpfp, const arma::mat &ytpfp, 
                arma::vec beta_tpfp, arma::vec alpha_tpfp){
  int err, ninit = 4, npoint = 100, nsamp = 1, ncent = 0;
  int neval;
  double xinit[10]={-3.0, -1.0, 1.0, 3.0}, xl = -5.0, xr = 5.0;
  double xsamp[100], xcent[10], qcent[10] = {5., 30., 70., 95.};
  double convex = 1.;
  int dometrop = 0;
  double xprev = 0.0;
  
  int N = ytpfp.n_rows;
  log_etai_param log_etai_data;
  log_etai_data.alpha_tpfp = alpha_tpfp;
  log_etai_data.beta_tpfp = beta_tpfp;
  for(int i=0;i<N;++i){
    log_etai_data.ytp_fp_i = ytpfp.row(i).t();
    err = arms(xinit,ninit,&xl,&xr,log_eta_i,&log_etai_data,&convex,
               npoint,dometrop,&xprev,xsamp,nsamp,qcent,xcent,ncent,&neval);
    if(err>0){
      Rprintf("error code: %d", err);
      Rcpp::stop("\n");
    }
    eta_tpfp(i) = xsamp[0];
  }
}

