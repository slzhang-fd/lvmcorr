// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "arms_ori.h"
#include "lvmcorr_omp.h"
#include "progress.hpp"
#include "eta_progress_bar.hpp"
#include "depend_funcs.h"

arma::mat sample_g_c(arma::mat g, const arma::mat &g_free_ind, const arma::uvec &xi, 
                     const arma::mat &x_covariates);
arma::vec sample_e_tp_c(arma::vec e_tp, arma::mat ytp, arma::uvec xi, arma::vec b_tp,
                        double sig2_tp, arma::mat beta_tp, arma::mat alpha_tp, arma::mat x_covariates);
arma::vec sample_e_fp_c(arma::vec e_fp, arma::mat yfp, arma::uvec xi, arma::vec b_fp,
                        double sig2_fp, arma::mat beta_fp, arma::mat alpha_fp, arma::mat x_covariates);
arma::mat sample_e12_c(arma::mat e12, arma::mat ytp, arma::mat yfp, arma::uvec xi,
                       arma::mat beta_tp, arma::mat alpha_tp, arma::mat beta_fp, arma::mat alpha_fp,
                       double sig2_tp, double sig2_fp, double rho,
                       arma::vec b_tp, arma::vec b_fp, arma::mat x_covariates);
arma::vec sample_b_tp_c(arma::vec b_tp, const arma::mat &ytp, const arma::uvec &xi, 
                        const arma::mat &x_covariates, const arma::mat &beta_tp, 
                        const arma::mat &alpha_tp, const arma::vec &e_tp, const arma::vec &e1);
arma::vec sample_b_fp_c(arma::vec b_fp, const arma::mat &yfp, const arma::uvec &xi, 
                        const arma::mat &x_covariates, const arma::mat &beta_fp, 
                        const arma::mat &alpha_fp, const arma::vec &e_fp, const arma::vec &e2);
arma::vec sample_b_linear_reg(arma::vec e_tp1, arma::vec e_tp2, arma::vec e_fp2,
                              arma::mat x_covariates1, arma::mat x_covariates2,
                              arma::vec b_tp, arma::vec b_fp, 
                              double sig2_tp, double sig2_fp, double rho);
arma::vec sample_b_tp_new(arma::mat Sigma, arma::mat e1234, arma::mat x_covariates, arma::vec b_tp);
arma::vec sample_b_fp_new(arma::mat Sigma, arma::mat e1234, arma::mat x_covariates, arma::vec b_fp);

arma::vec sample_b_new1(arma::mat Sigma1, arma::mat Sigma2, arma::mat e1234, arma::mat x_covariates, arma::vec b_tp, arma::vec female, bool tp_flag);
arma::vec sample_b_general(const arma::vec &sig_all, const arma::mat &rho_mat,
                          const arma::mat &e1234, const arma::mat &x_covariates, 
                        const arma::vec &b_tp_fp, bool tp_flag);

arma::vec sample_b_new2(arma::mat Sigma1, arma::mat Sigma2, arma::mat Sigma3, arma::mat Sigma4, 
                        arma::mat e1234, arma::mat x_covariates, arma::vec b_tp, 
                        arma::vec female, arma::vec distlong, bool tp_flag);

void sample_e1234(arma::mat &e1234, const arma::mat &ytp, const arma::mat &yfp, 
                  const arma::vec &ytp_fin, const arma::vec &yfp_fin, const arma::uvec &xi, 
                  const arma::mat &beta_tp, const arma::mat &alpha_tp, 
                  const arma::mat &beta_fp, const arma::mat &alpha_fp,
                  const arma::vec &b_tp, const arma::vec &b_fp, const arma::vec &u_tp, const arma::vec &u_fp, 
                  const arma::mat &x_covariates, const arma::mat &Sigma);
void sample_e1234_general(arma::mat &e1234, const arma::mat &ytp, const arma::mat &yfp, 
                       const arma::vec &ytp_fin, const arma::vec &yfp_fin, const arma::uvec &xi, 
                       const arma::mat &beta_tp, const arma::mat &alpha_tp, 
                       const arma::mat &beta_fp, const arma::mat &alpha_fp,
                       const arma::vec &b_tp, const arma::vec &b_fp, const arma::vec &u_tp, const arma::vec &u_fp, 
                       const arma::mat &x_covariates, const arma::vec &sig_all, const arma::mat rho_mat);

arma::mat sample_e12_c(arma::mat e12, arma::mat ytp, arma::mat yfp, arma::vec xi,
                       arma::mat beta_tp, arma::mat alpha_tp, arma::mat beta_fp, arma::mat alpha_fp,
                       double sig2_tp, double sig2_fp, double rho,
                       arma::vec b_tp, arma::vec b_fp, arma::mat x_covariates);
double sample_sig2_tp_c(double sig2_tp, double sig2_fp, double rho,
                        arma::vec e_tp, arma::mat e12, arma::uvec xi);
double sample_sig2_fp_c(double sig2_tp, double sig2_fp, double rho,
                        arma::vec e_fp, arma::mat e12, arma::uvec xi);
double sample_rho_c(double rho, const arma::uvec &xi, double sig2_tp, double sig2_fp, const arma::mat &e12);
void sample_beta_rho(const arma::mat &e1234, const arma::vec &sig_all, const arma::mat &rho_covs,
                      arma::mat &beta_rho, double step_rho, unsigned int loc1, unsigned int loc2);
// arma::uvec sample_xi(arma::mat x_covariates, arma::mat g,
//                     arma::mat beta_tp, arma::mat alpha_tp, arma::mat beta_fp, arma::mat alpha_fp,
//                     arma::mat ytp, arma::vec b_tp, arma::vec e_tp,
//                     arma::mat yfp, arma::vec b_fp, arma::vec e_fp,
//                     arma::mat e12);
double sample_sig2_tp_new(arma::mat e1234, double sig2_tp, double sig2_fp, arma::vec rho);
double sample_sig2_fp_new(arma::mat e1234, double sig2_tp, double sig2_fp, arma::vec rho);

arma::vec sample_rho_new(arma::mat e1234, double sig2_tp, double sig2_fp, arma::vec rho, arma::vec xinit_center);
arma::mat sample_Sigma_new1(const arma::mat &e1234);

// arma::mat construct_sigma1(arma::vec sig2_all, arma::vec rho);
void sample_sig2_sep(const arma::mat &e1234_1, const arma::mat &e1234_2,
                     arma::vec &sig_all, const arma::vec &rho1, const arma::vec &rho2,
                     double step_sigma);
void sample_sig2_general(const arma::mat &e1234, arma::vec &sig_all, 
                        const arma::mat &rho_mat, double step_sigma);

void sample_rho(const arma::mat &e1234, const arma::vec &sig_all, arma::vec &rho, double step_rho);
void sample_sig2(const arma::mat &e1234, arma::vec &sig_all, const arma::vec &rho, double step_sigma);

arma::vec sample_y_fin_star(arma::vec ytp_fin, arma::mat x_covariates, arma::vec u_tp);
arma::vec sample_u_new(arma::vec epsilon_tp_fp, arma::vec u_old, arma::mat x_covariates, double sig2);
arma::uvec sample_xi(arma::mat x_covariates, arma::mat g,
                     arma::mat beta_tp, arma::mat alpha_tp, arma::mat beta_fp, arma::mat alpha_fp,
                     arma::mat ytp, arma::vec b_tp, arma::vec e_tp,
                     arma::mat yfp, arma::vec b_fp, arma::vec e_fp,
                     arma::mat e12);
arma::uvec sample_xi_new(const arma::mat &x_covariates, const arma::mat &g,
                         const arma::mat &beta_tp, const arma::mat &alpha_tp, 
                         const arma::mat &beta_fp, const arma::mat &alpha_fp,
                         const arma::mat &ytp, const arma::vec &b_tp,
                         const arma::mat &yfp, const arma::vec &b_fp,
                         const arma::vec &ytp_fin, const arma::vec &u_tp,
                         const arma::vec &yfp_fin, const arma::vec &u_fp,
                         const arma::mat &e1234);
arma::uvec sample_xi_new1(const arma::mat &x_covariates, const arma::mat &g,
                          const arma::mat &beta_tp, const arma::mat &alpha_tp, 
                          const arma::mat &beta_fp, const arma::mat &alpha_fp,
                          const arma::mat &ytp, const arma::vec &b_tp,
                          const arma::mat &yfp, const arma::vec &b_fp,
                          const arma::vec &ytp_fin, const arma::vec &u_tp,
                          const arma::vec &yfp_fin, const arma::vec &u_fp,
                          const arma::mat &e1234);
arma::vec sample_u_tp_new(arma::mat , arma::mat , arma::vec , const arma::mat &);
arma::vec sample_u_fp_new(arma::mat , arma::mat , arma::vec , const arma::mat &);
arma::vec sample_u_new1(arma::mat Sigma1, arma::mat Sigma2, arma::mat e1234, arma::vec u_tp, 
                        const arma::mat &x_covariates, arma::vec female, bool tp_flag);
arma::vec sample_u_general(const arma::vec sig_all, const arma::mat &rho_mat, const arma::mat &e1234, 
                        const arma::vec &u_tp_fp, const arma::mat &x_covariates, bool tp_flag);
                        
arma::vec sample_u_new2(arma::mat Sigma1, arma::mat Sigma2, arma::mat Sigma3, arma::mat Sigma4,
                        arma::mat e1234, arma::vec u_tp, const arma::mat &x_covariates, 
                        arma::vec female, arma::vec distlong, bool tp_flag);

void impute_ytp_mis(arma::mat &ytp, arma::uvec ytp_mis_ind, arma::mat beta_tp, arma::mat alpha_tp,
                    arma::mat x_covariates, arma::vec b_tp, arma::vec e_tp, arma::vec e1, arma::uvec xi){
  arma::mat mu_res = -20 * arma::ones(arma::size(beta_tp));
  arma::uvec xi_3_loc = arma::find(xi==3);
  arma::uvec xi_4_loc = arma::find(xi==4);
  arma::uvec xi_34_loc = arma::join_cols(xi_3_loc, xi_4_loc);
  arma::vec eta_tp = x_covariates.rows(xi_34_loc) * b_tp + 
    arma::join_cols(e_tp(xi_3_loc), e1(xi_4_loc));
  arma::mat tmp = alpha_tp.rows(xi_34_loc);
  tmp.each_col() %= eta_tp;
  mu_res.rows(xi_34_loc) = beta_tp.rows(xi_34_loc) + tmp;
  arma::mat prob = 1.0 / (1.0 + arma::exp(-mu_res));
  arma::mat temp(arma::size(prob),arma::fill::randu);
  arma::mat draws = arma::conv_to<arma::mat>::from(temp<prob);
  ytp(ytp_mis_ind) = draws(ytp_mis_ind);
  if ( Progress::check_abort() )
    return;
}
void impute_ytp_mis_core(arma::mat &ytp, arma::uvec ytp_mis_ind, arma::mat beta_tp, arma::mat alpha_tp,
                    arma::mat x_covariates, arma::vec b_tp, arma::vec e_tp, arma::vec e1, arma::uvec xi){
  arma::mat mu_res = -20 * arma::ones(arma::size(beta_tp));
  arma::uvec xi_3_loc = arma::find(xi==3);
  arma::uvec xi_4_loc = arma::find(xi==4);
  arma::uvec xi_34_loc = arma::join_cols(xi_3_loc, xi_4_loc);
  arma::vec eta_tp = x_covariates.rows(xi_34_loc) * b_tp + 
    arma::join_cols(e_tp(xi_3_loc), e1(xi_4_loc));
  arma::mat tmp = alpha_tp.rows(xi_34_loc);
  tmp.each_col() %= eta_tp;
  mu_res.rows(xi_34_loc) = beta_tp.rows(xi_34_loc) + tmp;
  arma::mat prob = 1.0 / (1.0 + arma::exp(-mu_res));
  arma::mat temp(arma::size(prob),arma::fill::randu);
  arma::mat draws = arma::conv_to<arma::mat>::from(temp<prob);
  ytp(ytp_mis_ind) = draws(ytp_mis_ind);
}
void impute_yfp_mis(arma::mat &yfp, arma::uvec yfp_mis_ind, arma::mat beta_fp, arma::mat alpha_fp,
                    arma::mat x_covariates, arma::vec b_fp, arma::vec e_fp, arma::vec e2, arma::uvec xi){
  arma::mat mu_res = -20 * arma::ones(arma::size(beta_fp));
  arma::uvec xi_2_loc = arma::find(xi==2);
  arma::uvec xi_4_loc = arma::find(xi==4);
  arma::uvec xi_24_loc = arma::join_cols(xi_2_loc, xi_4_loc);
  arma::vec eta_fp = x_covariates.rows(xi_24_loc) * b_fp + 
    arma::join_cols(e_fp(xi_2_loc), e2(xi_4_loc));
  arma::mat tmp = alpha_fp.rows(xi_24_loc);
  tmp.each_col() %= eta_fp;
  mu_res.rows(xi_24_loc) = beta_fp.rows(xi_24_loc) + tmp;
  arma::mat prob = 1.0 / (1.0 + arma::exp(-mu_res));
  arma::mat temp(arma::size(prob),arma::fill::randu);
  arma::mat draws = arma::conv_to<arma::mat>::from(temp<prob);
  yfp(yfp_mis_ind) = draws(yfp_mis_ind);
}
// [[Rcpp::export]]
void sample_g_b(arma::mat &g, const arma::mat &g_free_ind, const arma::uvec &xi, 
                const arma::mat &x_covariates, const arma::mat &ytp, const arma::mat &yfp,
                const arma::mat &beta_tp, const arma::mat &alpha_tp,
                const arma::mat &beta_fp, const arma::mat &alpha_fp,
                arma::vec &b_tp, arma::vec &b_fp,
                const arma::vec &e_tp, const arma::vec &e_fp,
                const arma::mat &e12, const double sig2_tp, const double sig2_fp, const double rho){
  arma::mat g_res;
  arma::vec b_tp_res,b_fp_res;
  arma::vec e1 = e12.col(0);
  arma::vec e2 = e12.col(1);
  #pragma omp parallel sections num_threads(getlvmcorr_threads())
  {
    #pragma omp section
    {
      g_res = sample_g_c(g, g_free_ind, xi, x_covariates);
    }
    #pragma omp section
    {
      // b_tp_res = sample_b_tp_c(b_tp, ytp, xi, x_covariates, beta_tp, alpha_tp, e_tp, e1);
      arma::uvec xi_3_loc = arma::find(xi==3);
      arma::uvec xi_4_loc = arma::find(xi==4);
      b_tp_res = sample_b_linear_reg(e_tp(xi_3_loc), e1(xi_4_loc), e2(xi_4_loc),
                                    x_covariates.rows(xi_3_loc), x_covariates.rows(xi_4_loc), 
                                    b_tp, b_fp, sig2_tp, sig2_fp, rho);
    }
    #pragma omp section
    {
      // b_fp_res = sample_b_fp_c(b_fp, yfp, xi, x_covariates, beta_fp, alpha_fp, e_fp, e2);
      arma::uvec xi_2_loc = arma::find(xi==2);
      arma::uvec xi_4_loc = arma::find(xi==4);
      b_fp_res = sample_b_linear_reg(e_fp(xi_2_loc), e2(xi_4_loc), e1(xi_4_loc),
                                     x_covariates.rows(xi_2_loc), x_covariates.rows(xi_4_loc),
                                     b_fp, b_tp, sig2_fp, sig2_tp, rho);
    }
  }
  g = g_res;
  b_tp = b_tp_res;
  b_fp = b_fp_res;
}
//' @export
// [[Rcpp::export]]
Rcpp::List lvmcorr_cpp(arma::mat ytp, arma::mat yfp, arma::mat x_covariates, arma::vec b_tp, arma::vec b_fp,
                    arma::mat g, arma::mat g_free_ind, double sig2_tp, double sig2_fp, double rho, 
                    arma::vec e_tp, arma::vec e_fp, arma::mat e12, arma::uvec xi, 
                    arma::mat beta_tp, arma::mat alpha_tp,arma::mat beta_fp, arma::mat alpha_fp,
                    int mcmc_len, bool verbose=false){
  arma::mat g_draws = arma::zeros(g.n_elem,mcmc_len);
  arma::mat b_tp_draws = arma::zeros(b_tp.n_elem,mcmc_len);
  arma::mat b_fp_draws = arma::zeros(b_fp.n_elem,mcmc_len);
  arma::vec sig2_tp_draws = arma::zeros(mcmc_len);
  arma::vec sig2_fp_draws = arma::zeros(mcmc_len);
  arma::vec rho_draws = arma::zeros(mcmc_len);
  
  arma::uvec ytp_mis_ind = arma::find_nonfinite(ytp);
  arma::uvec yfp_mis_ind = arma::find_nonfinite(yfp);
  ETAProgressBar pb;
  Progress p(mcmc_len, !verbose, pb);
  for(int i=0;i<mcmc_len;++i){
    if ( p.increment() ) {
      // sample e_tp, e_fp, e12
      e_tp = sample_e_tp_c(e_tp, ytp, xi, b_tp, sig2_tp, beta_tp, alpha_tp, x_covariates);
      e_fp = sample_e_fp_c(e_fp, yfp, xi, b_fp, sig2_fp, beta_fp, alpha_fp, x_covariates);
      e12 = sample_e12_c(e12, ytp, yfp, xi, beta_tp, alpha_tp, beta_fp, alpha_fp, sig2_tp, sig2_fp,
                         rho, b_tp, b_fp, x_covariates);
      // sample missing ytp and yfp
      impute_ytp_mis(ytp, ytp_mis_ind, beta_tp, alpha_tp, x_covariates, b_tp, e_tp, e12.col(0), xi);
      impute_yfp_mis(yfp, yfp_mis_ind, beta_fp, alpha_fp, x_covariates, b_fp, e_fp, e12.col(1), xi);
      // sample g, b_tp, b_fp
      sample_g_b(g, g_free_ind, xi, x_covariates, ytp, yfp, beta_tp, alpha_tp, beta_fp, alpha_fp,
                 b_tp, b_fp, e_tp, e_fp, e12, sig2_tp, sig2_fp, rho);
      // sample sig2_tp, sig2_fp, rho
      sig2_tp = sample_sig2_tp_c(sig2_tp, sig2_fp, rho, e_tp, e12, xi);
      sig2_fp = sample_sig2_fp_c(sig2_tp, sig2_fp, rho, e_fp, e12, xi);
      rho = sample_rho_c(rho, xi, sig2_tp, sig2_fp, e12);
      // sample xi
      xi = sample_xi(x_covariates, g, beta_tp, alpha_tp, beta_fp, alpha_fp, 
                     ytp, b_tp, e_tp, yfp, b_fp, e_fp, e12);
      g_draws.col(i) = g.as_col();
      b_tp_draws.col(i) = b_tp;
      b_fp_draws.col(i) = b_fp;
      sig2_tp_draws(i) = sig2_tp;
      sig2_fp_draws(i) = sig2_fp;
      rho_draws(i) = rho;
      if(verbose)
        Rprintf("step: %d\t b_tp1 %f\t b_fp1 %f\t g[1,2] %f\t sig2_tp %f\t sig2_fp %f\t rho %f\n",
                i, b_tp(0), b_fp(0), g(0,1), sig2_tp, sig2_fp, rho);
    }
  }
  return Rcpp::List::create(Rcpp::Named("g_draws") = g_draws.t(),
                             Rcpp::Named("b_tp_draws") = b_tp_draws.t(),
                             Rcpp::Named("b_fp_draws") = b_fp_draws.t(),
                             Rcpp::Named("sig2_tp_draws") = sig2_tp_draws,
                             Rcpp::Named("sig2_fp_draws") = sig2_fp_draws,
                             Rcpp::Named("rho_draws") = rho_draws,
                             Rcpp::Named("e_tp_last") = e_tp,
                             Rcpp::Named("e_fp_last") = e_fp,
                             Rcpp::Named("e12_last") = e12,
                             Rcpp::Named("xi_last") = xi,
                             Rcpp::Named("ytp") = ytp,
                             Rcpp::Named("yfp") = yfp,
                             Rcpp::Named("x_covariates") = x_covariates,
                             Rcpp::Named("g_free_ind") = g_free_ind,
                             Rcpp::Named("beta_tp") = beta_tp,
                             Rcpp::Named("alpha_tp") = alpha_tp,
                             Rcpp::Named("beta_fp") = beta_fp,
                             Rcpp::Named("alpha_fp") = alpha_fp);
}
Rcpp::List lvmcorr_core(arma::mat ytp, arma::mat yfp, arma::mat x_covariates, arma::vec b_tp, arma::vec b_fp,
                    arma::mat g, arma::mat g_free_ind, double sig2_tp, double sig2_fp, double rho, 
                    arma::vec e_tp, arma::vec e_fp, arma::mat e12, arma::uvec xi, 
                    arma::mat beta_tp, arma::mat alpha_tp,arma::mat beta_fp, arma::mat alpha_fp,
                    int mcmc_len, bool verbose=false){
  arma::mat g_draws = arma::zeros(g.n_elem,mcmc_len);
  arma::mat b_tp_draws = arma::zeros(b_tp.n_elem,mcmc_len);
  arma::mat b_fp_draws = arma::zeros(b_fp.n_elem,mcmc_len);
  arma::vec sig2_tp_draws = arma::zeros(mcmc_len);
  arma::vec sig2_fp_draws = arma::zeros(mcmc_len);
  arma::vec rho_draws = arma::zeros(mcmc_len);
  
  arma::uvec ytp_mis_ind = arma::find_nonfinite(ytp);
  arma::uvec yfp_mis_ind = arma::find_nonfinite(yfp);
  for(int i=0;i<mcmc_len;++i){
    // sample e_tp, e_fp, e12
      e_tp = sample_e_tp_c(e_tp, ytp, xi, b_tp, sig2_tp, beta_tp, alpha_tp, x_covariates);
      e_fp = sample_e_fp_c(e_fp, yfp, xi, b_fp, sig2_fp, beta_fp, alpha_fp, x_covariates);
      e12 = sample_e12_c(e12, ytp, yfp, xi, beta_tp, alpha_tp, beta_fp, alpha_fp, sig2_tp, sig2_fp,
                         rho, b_tp, b_fp, x_covariates);
      // sample missing ytp and yfp
      impute_ytp_mis_core(ytp, ytp_mis_ind, beta_tp, alpha_tp, x_covariates, b_tp, e_tp, e12.col(0), xi);
      impute_yfp_mis(yfp, yfp_mis_ind, beta_fp, alpha_fp, x_covariates, b_fp, e_fp, e12.col(1), xi);
      // sample g, b_tp, b_fp
      sample_g_b(g, g_free_ind, xi, x_covariates, ytp, yfp, beta_tp, alpha_tp, beta_fp, alpha_fp,
                 b_tp, b_fp, e_tp, e_fp, e12, sig2_tp, sig2_fp, rho);
      // sample sig2_tp, sig2_fp, rho
      sig2_tp = sample_sig2_tp_c(sig2_tp, sig2_fp, rho, e_tp, e12, xi);
      sig2_fp = sample_sig2_fp_c(sig2_tp, sig2_fp, rho, e_fp, e12, xi);
      rho = sample_rho_c(rho, xi, sig2_tp, sig2_fp, e12);
      // sample xi
      xi = sample_xi(x_covariates, g, beta_tp, alpha_tp, beta_fp, alpha_fp, 
                     ytp, b_tp, e_tp, yfp, b_fp, e_fp, e12);
      g_draws.col(i) = g.as_col();
      b_tp_draws.col(i) = b_tp;
      b_fp_draws.col(i) = b_fp;
      sig2_tp_draws(i) = sig2_tp;
      sig2_fp_draws(i) = sig2_fp;
      rho_draws(i) = rho;
      if(verbose)
        Rprintf("step: %d\t b_tp1 %f\t b_fp1 %f\t g[1,2] %f\t sig2_tp %f\t sig2_fp %f\t rho %f\n",
                i, b_tp(0), b_fp(0), g(0,1), sig2_tp, sig2_fp, rho);
  }
  return Rcpp::List::create(Rcpp::Named("g_draws") = g_draws.t(),
                            Rcpp::Named("b_tp_draws") = b_tp_draws.t(),
                            Rcpp::Named("b_fp_draws") = b_fp_draws.t(),
                            Rcpp::Named("sig2_tp_draws") = sig2_tp_draws,
                            Rcpp::Named("sig2_fp_draws") = sig2_fp_draws,
                            Rcpp::Named("rho_draws") = rho_draws,
                            Rcpp::Named("e_tp_last") = e_tp,
                            Rcpp::Named("e_fp_last") = e_fp,
                            Rcpp::Named("e12_last") = e12,
                            Rcpp::Named("xi_last") = xi);
}
// [[Rcpp::export]]
Rcpp::List lvmcorr_multi_chains(arma::mat ytp, arma::mat yfp, arma::mat x_covariates, arma::vec b_tp, arma::vec b_fp,
                    arma::mat g, arma::mat g_free_ind, double sig2_tp, double sig2_fp, double rho, arma::uvec xi, 
                    arma::mat beta_tp, arma::mat alpha_tp,arma::mat beta_fp, arma::mat alpha_fp,
                    int mcmc_len, int chains = 1){
  Rcpp::List res(chains);
  // int old_thread_num = setlvmcorr_threads(1);
  int N = ytp.n_rows;
#pragma omp parallel num_threads(std::min(chains, omp_get_num_procs()))
{
  if(omp_get_thread_num() == 0){
    // N <- nrow(data$ytp)
    // g_init <- matrix(0, 14, 4)
    // g_init[,1] <- 0
    // g_init[3,2] <- g_init[6,3] <- g_init[7,3] <- -5
    // initvals <- list(b_tp_init = matrix(0, nparam, 1), b_fp_init = matrix(0, nparam, 1), 
    //                  g_init = g_init, sig2_tp_init = 1/0.35,
    //                  sig2_fp_init = 1/0.2, rho_init = 0,
    //                  e_tp = rnorm(N), e_fp = rnorm(N), e12 = matrix(rnorm(2*N), N, 2), xi = data$xi)
    res[0] = lvmcorr_cpp(ytp, yfp, x_covariates, b_tp, b_fp, g, g_free_ind, sig2_tp, sig2_fp, rho,
                       arma::randn(N), arma::randn(N), arma::mat(N, 2, arma::fill::randn), 
                       xi, beta_tp, alpha_tp, beta_fp, alpha_fp, mcmc_len);
  }
  else{
    res[omp_get_thread_num()] = lvmcorr_core(ytp, yfp, x_covariates, b_tp, b_fp, g, g_free_ind, sig2_tp, sig2_fp, rho,
                           arma::randn(N), arma::randn(N), arma::mat(N, 2, arma::fill::randn), 
                           xi, beta_tp, alpha_tp, beta_fp, alpha_fp, mcmc_len);
  }
}
  // setlvmcorr_threads(old_thread_num);
  return res;
}

//' @export
// [[Rcpp::export]]
Rcpp::List lvmcorr_seperate_fin_cpp(arma::mat ytp_all, arma::mat yfp_all, arma::mat x_covariates,
                                 arma::vec b_tp, arma::vec b_fp,
                                 arma::mat g, arma::mat g_free_ind, arma::mat e1234, arma::uvec xi,
                                 arma::mat beta_tp_all, arma::mat alpha_tp_all,
                                 arma::mat beta_fp_all, arma::mat alpha_fp_all,
                                 arma::vec u_tp, arma::vec u_fp,
                                 int mcmc_len,bool verbose=false){
  arma::mat g_draws = arma::zeros(g.n_elem,mcmc_len);
  arma::mat b_tp_draws = arma::zeros(b_tp.n_elem,mcmc_len);
  arma::mat b_fp_draws = arma::zeros(b_fp.n_elem,mcmc_len);
  arma::mat u_tp_draws = arma::zeros(u_tp.n_elem, mcmc_len);
  arma::mat u_fp_draws = arma::zeros(u_fp.n_elem, mcmc_len);
  arma::mat sig_draws = arma::zeros(2, mcmc_len);
  arma::mat rho_draws = arma::zeros(6, mcmc_len);

  arma::mat ytp = ytp_all.cols(0, 6);
  arma::mat yfp = yfp_all.cols(0, 6);
  arma::vec ytp_fin = ytp_all.col(7);
  arma::vec yfp_fin = yfp_all.col(7);
  arma::mat beta_tp = beta_tp_all.cols(0, 6);
  arma::mat beta_fp = beta_fp_all.cols(0, 6);
  arma::mat alpha_tp = alpha_tp_all.cols(0, 6);
  arma::mat alpha_fp = alpha_fp_all.cols(0, 6);

  // arma::vec xinit_center = rho;
  ETAProgressBar pb;
  Progress p(mcmc_len, !verbose, pb);

  arma::mat Sigma = arma::eye(4,4);
  arma::vec sig_all = arma::ones(2);
  arma::vec rho_all = arma::zeros(6);

  for(int i=0;i<mcmc_len;++i){
    // sample e_tp, e_fp, e12
    if ( p.increment() ) {
      Sigma = construct_sigma1(sig_all, rho_all);
      // // sample missing ytp and yfp
      // impute_ytp_mis(ytp, ytp_mis_ind, beta_tp, alpha_tp, x_covariates, b_tp, e_tp, e1234.col(0), xi);
      // impute_yfp_mis(yfp, yfp_mis_ind, beta_fp, alpha_fp, x_covariates, b_fp, e_fp, e1234.col(1), xi);
      // note: sampling of missing responses are depracated due to that the missing responses are all from two items
      //       they are actually inapplicable rather than missing values.

      // sample e1234
      Rcpp::Rcout << "part 1, ";
      // e_tp = sample_e_tp_c(e_tp, ytp, xi, b_tp, sig2_tp, beta_tp, alpha_tp, x_covariates);
      // e_fp = sample_e_fp_c(e_fp, yfp, xi, b_fp, sig2_fp, beta_fp, alpha_fp, x_covariates);
      sample_e1234(e1234, ytp, yfp, ytp_fin, yfp_fin, xi, beta_tp, alpha_tp, beta_fp, alpha_fp,
                           b_tp, b_fp, u_tp, u_fp, x_covariates, Sigma);

      // sample g, b_tp, b_fp
      Rcpp::Rcout << "part 2, ";
      g = sample_g_c(g, g_free_ind, xi, x_covariates);

      // sample coefficient for eta
      arma::uvec xi_34_loc = arma::find(xi==3 || xi==4);
      b_tp = sample_b_tp_new(Sigma, e1234.rows(xi_34_loc), x_covariates.rows(xi_34_loc), b_tp);
      arma::uvec xi_24_loc = arma::find(xi==2 || xi==4);
      b_fp = sample_b_fp_new(Sigma, e1234.rows(xi_24_loc), x_covariates.rows(xi_24_loc), b_fp);


      // sample coefficients for financial help latent variables
      Rcpp::Rcout << "part 3, ";
      u_tp = sample_u_tp_new(Sigma, e1234.rows(xi_34_loc), u_tp, x_covariates.rows(xi_34_loc));
      u_fp = sample_u_fp_new(Sigma, e1234.rows(xi_24_loc), u_fp, x_covariates.rows(xi_24_loc));
      // u_tp /= std::abs(u_tp(0));
      // u_fp /= std::abs(u_fp(0));
      // sample sig2_tp, sig2_fp, rho
      // Rcpp::Rcout << "part 4, ";
      // sig2_tp = sample_sig2_tp_new(e1234, sig2_tp, sig2_fp, rho);
      // sig2_fp = sample_sig2_fp_new(e1234, sig2_tp, sig2_fp, rho);
      //
      // Rcpp::Rcout << "part 5" << std::endl;
      //
      // // adaptively choose sample interval centers
      // if(i < adaptive_steps){
      //   xinit_center = rho;
      // }
      // else if(i == adaptive_steps){
      //   xinit_center = arma::mean(rho_draws.cols(adaptive_steps-10,adaptive_steps-1),1);
      //   xinit_center = arma::min(xinit_center, 0.85 * arma::ones(xinit_center.n_rows));
      // }
      // rho = sample_rho_new(e1234, sig2_tp, sig2_fp, rho, xinit_center);

      sample_sig2(e1234, sig_all, rho_all, 0.01);
      sample_rho(e1234, sig_all, rho_all, 0.007);
      // Sigma = sample_Sigma_new1(e1234);

      Rcpp::Rcout << "part 6" << std::endl;
      // sample xi
      xi = sample_xi_new(x_covariates, g, beta_tp, alpha_tp, beta_fp, alpha_fp,
                     ytp, b_tp, yfp, b_fp,
                     ytp_fin, u_tp,
                     yfp_fin, u_fp,
                     e1234);

      g_draws.col(i) = g.as_col();
      b_tp_draws.col(i) = b_tp;
      b_fp_draws.col(i) = b_fp;
      u_tp_draws.col(i) = u_tp;
      u_fp_draws.col(i) = u_fp;

      sig_draws.col(i) = sig_all;
      rho_draws.col(i) = rho_all;

      // if(verbose)
      //   Rprintf("step: %d\t b_tp1 %f\t b_fp1 %f\t u_tp1 %f\t u_fp1 %f\t g[1,2] %f\t sig2_tp %f\t sig2_fp %f\t rho_0 %f\t rho_1 %f rho_2 %f rho_3 %f rho_4 %f rho_5 %f\n",
      //           i, b_tp(0), b_fp(0), u_tp(0), u_fp(0), g(0,1), sig2_tp, sig2_fp, rho(0), rho(1), rho(2), rho(3), rho(4), rho(5));
      if(verbose)
        Rcpp::Rcout << "step: " << i <<
          "\t b_tp0:" << b_tp(0) <<
          "\t b_fp0:" << b_fp(0) <<
          "\t u_tp0:" << u_tp(0) <<
          "\t u_fp0:" << u_fp(0) <<
          "\t sig_tp_fp:" << sig_all.t() <<
          "\t rho:" << rho_all.t() << std::endl;

    }
  }
  return Rcpp::List::create(Rcpp::Named("g_draws") = g_draws.t(),
                            Rcpp::Named("b_tp_draws") = b_tp_draws.t(),
                            Rcpp::Named("b_fp_draws") = b_fp_draws.t(),
                            Rcpp::Named("u_tp_draws") = u_tp_draws.t(),
                            Rcpp::Named("u_fp_draws") = u_fp_draws.t(),
                            Rcpp::Named("sig_draws") = sig_draws.t(),
                            Rcpp::Named("rho_draws") = rho_draws.t(),
                            Rcpp::Named("e1234_last") = e1234,
                            Rcpp::Named("xi_last") = xi,
                            Rcpp::Named("ytp") = ytp,
                            Rcpp::Named("yfp") = yfp,
                            Rcpp::Named("x_covariates") = x_covariates,
                            Rcpp::Named("g_free_ind") = g_free_ind,
                            Rcpp::Named("beta_tp") = beta_tp,
                            Rcpp::Named("alpha_tp") = alpha_tp,
                            Rcpp::Named("beta_fp") = beta_fp,
                            Rcpp::Named("alpha_fp") = alpha_fp,
                            Rcpp::Named("Sigma") = Sigma);
}

//' @export
// [[Rcpp::export]]
Rcpp::List lvmcorr_seperate_fin_cpp1(arma::mat ytp_all, arma::mat yfp_all, arma::mat x_covariates,
                                 arma::vec b_tp, arma::vec b_fp,
                                 arma::mat g, arma::mat g_free_ind,
                                 arma::mat e1234,
                                 arma::uvec xi,
                                 arma::mat beta_tp_all, arma::mat alpha_tp_all,arma::mat beta_fp_all, arma::mat alpha_fp_all,
                                 arma::vec u_tp, arma::vec u_fp, int mcmc_len, bool verbose=false){
  arma::mat g_draws = arma::zeros(g.n_elem, mcmc_len);
  arma::mat b_tp_draws = arma::zeros(b_tp.n_elem,mcmc_len);
  arma::mat b_fp_draws = arma::zeros(b_fp.n_elem,mcmc_len);
  arma::mat u_tp_draws = arma::zeros(u_tp.n_elem, mcmc_len);
  arma::mat u_fp_draws = arma::zeros(u_fp.n_elem, mcmc_len);
  arma::mat sig_draws = arma::zeros(2, mcmc_len);
  arma::mat rho_draws1 = arma::zeros(6, mcmc_len);
  arma::mat rho_draws2 = arma::zeros(6, mcmc_len);

  arma::mat ytp = ytp_all.cols(0, 6);
  arma::mat yfp = yfp_all.cols(0, 6);
  arma::vec ytp_fin = ytp_all.col(7);
  arma::vec yfp_fin = yfp_all.col(7);
  arma::mat beta_tp = beta_tp_all.cols(0, 6);
  arma::mat beta_fp = beta_fp_all.cols(0, 6);
  arma::mat alpha_tp = alpha_tp_all.cols(0, 6);
  arma::mat alpha_fp = alpha_fp_all.cols(0, 6);

  // arma::uvec ytp_mis_ind = arma::find_nonfinite(ytp);
  // arma::uvec yfp_mis_ind = arma::find_nonfinite(yfp);

  ETAProgressBar pb;
  Progress p(mcmc_len, !verbose, pb);
  arma::mat Sigma1 = arma::eye(4,4);
  arma::mat Sigma2 = arma::eye(4,4);
  arma::vec sig_all = arma::ones(2);
  arma::vec rho_all1 = arma::zeros(6);
  arma::vec rho_all2 = arma::zeros(6);

  arma::vec female = x_covariates.col(3);
  arma::uvec female_loc = arma::find(female==1);
  arma::uvec male_loc = arma::find(female==0);
  
  for(int i=0;i<mcmc_len;++i){
    // sample e_tp, e_fp, e12
    if ( p.increment() ) {
      Sigma1 = construct_sigma1(sig_all, rho_all1);
      Sigma2 = construct_sigma1(sig_all, rho_all2);
      // // sample missing ytp and yfp
      // impute_ytp_mis(ytp, ytp_mis_ind, beta_tp, alpha_tp, x_covariates, b_tp, e_tp, e1234.col(0), xi);
      // impute_yfp_mis(yfp, yfp_mis_ind, beta_fp, alpha_fp, x_covariates, b_fp, e_fp, e1234.col(1), xi);
      // note: sampling of missing responses are depracated due to that the missing responses are all from two items
      //       they are actually inapplicable rather than missing values.
      // sample e1234
      // e_tp = sample_e_tp_c(e_tp, ytp, xi, b_tp, sig2_tp, beta_tp, alpha_tp, x_covariates);
      // e_fp = sample_e_fp_c(e_fp, yfp, xi, b_fp, sig2_fp, beta_fp, alpha_fp, x_covariates);
      arma::mat e1234_1 = e1234.rows(female_loc);
      sample_e1234(e1234_1, ytp.rows(female_loc), yfp.rows(female_loc),
                   ytp_fin.rows(female_loc), yfp_fin.rows(female_loc), xi.rows(female_loc),
                   beta_tp, alpha_tp, beta_fp, alpha_fp,
                   b_tp, b_fp, u_tp, u_fp, x_covariates.rows(female_loc), Sigma1);
      arma::mat e1234_2 = e1234.rows(male_loc);
      sample_e1234(e1234_2, ytp.rows(male_loc), yfp.rows(male_loc),
                   ytp_fin.rows(male_loc), yfp_fin.rows(male_loc), xi.rows(male_loc),
                   beta_tp, alpha_tp, beta_fp, alpha_fp,
                   b_tp, b_fp, u_tp, u_fp, x_covariates.rows(male_loc), Sigma2); // previously Sigma1 here!!
      e1234.rows(female_loc) = e1234_1;
      e1234.rows(male_loc) = e1234_2;
      // sample g, b_tp, b_fp
      Rcpp::Rcout << "part 2, ";
      g = sample_g_c(g, g_free_ind, xi, x_covariates);

      // sample coefficient for eta
      arma::uvec xi_34_loc = arma::find(xi==3 || xi==4);
      b_tp = sample_b_new1(Sigma1, Sigma2, e1234.rows(xi_34_loc), x_covariates.rows(xi_34_loc), b_tp, female.rows(xi_34_loc), true);
      arma::uvec xi_24_loc = arma::find(xi==2 || xi==4);
      b_fp = sample_b_new1(Sigma1, Sigma2, e1234.rows(xi_24_loc), x_covariates.rows(xi_24_loc), b_fp, female.rows(xi_24_loc), false);


      // sample coefficients for financial help latent variables
      Rcpp::Rcout << "part 3, ";
      u_tp = sample_u_new1(Sigma1, Sigma2, e1234.rows(xi_34_loc), u_tp, x_covariates.rows(xi_34_loc), female.rows(xi_34_loc), true);
      u_fp = sample_u_new1(Sigma1, Sigma2, e1234.rows(xi_24_loc), u_fp, x_covariates.rows(xi_24_loc), female.rows(xi_24_loc), false);
      // u_tp /= std::abs(u_tp(0));
      // u_fp /= std::abs(u_fp(0));

      // // sample sig2_tp, sig2_fp, rho
      // Rcpp::Rcout << "part 4, ";
      // sig2_tp = sample_sig2_tp_new(e1234, sig2_tp, sig2_fp, rho);
      // sig2_fp = sample_sig2_fp_new(e1234, sig2_tp, sig2_fp, rho);
      //
      // Rcpp::Rcout << "part 5" << std::endl;
      //
      // // adaptively choose sample interval centers
      // if(i < adaptive_steps){
      //   xinit_center = rho;
      // }
      // else if(i == adaptive_steps){
      //   xinit_center = arma::mean(rho_draws.cols(adaptive_steps-10,adaptive_steps-1),1);
      //   xinit_center = arma::min(xinit_center, 0.85 * arma::ones(xinit_center.n_rows));
      // }
      //
      // rho = sample_rho_new(e1234, sig2_tp, sig2_fp, rho, xinit_center);

      // Sigma1 = sample_Sigma_new1(e1234.rows(female_loc));
      // Sigma2 = sample_Sigma_new1(e1234.rows(male_loc));
      sample_sig2_sep(e1234.rows(female_loc), e1234.rows(male_loc), sig_all, rho_all1, rho_all2, 0.01);
      // sample_sig2(e1234.rows(female_loc), sig_all1, rho_all1, 0.01);
      // sample_sig2(e1234.rows(male_loc), sig_all2, rho_all2, 0.01);
      sample_rho(e1234.rows(female_loc), sig_all, rho_all1, 0.01);
      sample_rho(e1234.rows(male_loc), sig_all, rho_all2, 0.01);

      Rcpp::Rcout << "part 6" << std::endl;
      // sample xi
      xi = sample_xi_new(x_covariates, g, beta_tp, alpha_tp, beta_fp, alpha_fp,
                         ytp, b_tp, yfp, b_fp,
                         ytp_fin, u_tp,
                         yfp_fin, u_fp,
                         e1234);


      g_draws.col(i) = g.as_col();
      b_tp_draws.col(i) = b_tp;
      b_fp_draws.col(i) = b_fp;
      u_tp_draws.col(i) = u_tp;
      u_fp_draws.col(i) = u_fp;
      
      sig_draws.col(i) = sig_all;
      rho_draws1.col(i) = rho_all1;
      rho_draws2.col(i) = rho_all2;

      if(verbose)
        Rcpp::Rcout << "step: " << i << "b_tp1" << b_tp(0) << "\t b_fp1 " <<  b_fp(0) <<
            "\t u_tp1" << u_tp(0) << "\t u_fp1" << u_fp(0) << "\t g[1,2] " << g(0,1) <<
            "\t sig_all" <<sig_all.t() <<
            "\t rho_0" << rho_all1.t() <<
            "\t rho_1" << rho_all2.t() << std::endl;

    }
  }
  return Rcpp::List::create(Rcpp::Named("g_draws") = g_draws.t(),
                            Rcpp::Named("b_tp_draws") = b_tp_draws.t(),
                            Rcpp::Named("b_fp_draws") = b_fp_draws.t(),
                            Rcpp::Named("u_tp_draws") = u_tp_draws.t(),
                            Rcpp::Named("u_fp_draws") = u_fp_draws.t(),
                            Rcpp::Named("sig_draws") = sig_draws.t(),
                            Rcpp::Named("rho_draws1") = rho_draws1.t(),
                            Rcpp::Named("rho_draws2") = rho_draws2.t(),
                            Rcpp::Named("e1234_last") = e1234,
                            Rcpp::Named("xi_last") = xi,
                            Rcpp::Named("Sigma1") = Sigma1,
                            Rcpp::Named("Sigma2") = Sigma2);
}
//' @export
// [[Rcpp::export]]
Rcpp::List lvmcorr_seperate_fin_general(arma::mat ytp_all, arma::mat yfp_all, arma::mat x_covariates,
                                  arma::mat rho_covs,
                                  arma::vec b_tp, arma::vec b_fp,
                                  arma::mat g, arma::mat g_free_ind,
                                  arma::mat e1234, arma::vec sig_init, arma::mat beta_rho_init,
                                  arma::uvec xi,
                                  arma::mat beta_tp_all, arma::mat alpha_tp_all,
                                  arma::mat beta_fp_all, arma::mat alpha_fp_all,
                                  arma::vec u_tp, arma::vec u_fp, arma::vec rho_stepsize,
                                  int mcmc_len, bool verbose=false){

  int N = ytp_all.n_rows;
  unsigned int categories = beta_rho_init.n_cols;

  arma::mat g_draws = arma::zeros(g.n_elem, mcmc_len);
  arma::mat b_tp_draws = arma::zeros(b_tp.n_elem,mcmc_len);
  arma::mat b_fp_draws = arma::zeros(b_fp.n_elem,mcmc_len);
  arma::mat u_tp_draws = arma::zeros(u_tp.n_elem, mcmc_len);
  arma::mat u_fp_draws = arma::zeros(u_fp.n_elem, mcmc_len);
  arma::mat sig_draws = arma::zeros(2, mcmc_len);
  arma::mat beta_rho_draws = arma::zeros(6 * categories, mcmc_len);

  arma::mat ytp = ytp_all.cols(0, 6);
  arma::mat yfp = yfp_all.cols(0, 6);
  arma::vec ytp_fin = ytp_all.col(7);
  arma::vec yfp_fin = yfp_all.col(7);
  arma::mat beta_tp = beta_tp_all.cols(0, 6);
  arma::mat beta_fp = beta_fp_all.cols(0, 6);
  arma::mat alpha_tp = alpha_tp_all.cols(0, 6);
  arma::mat alpha_fp = alpha_fp_all.cols(0, 6);

  ETAProgressBar pb;
  Progress p(mcmc_len, !verbose, pb);
  arma::vec sig_all = sig_init; // temp vec for sds
  arma::mat beta_rho = beta_rho_init; // temp mat for rho coefficients
  arma::mat rho_mat = arma::zeros(6, N); // temp mat 6 * N for all correlations

  for(int i=0;i<mcmc_len;++i){
    if ( p.increment() ) {
      // sample e1234
      rho_mat = fisher_z_trans(beta_rho * rho_covs.t());// fisher z transformation
      sample_e1234_general(e1234, ytp, yfp, ytp_fin, yfp_fin, xi,
                           beta_tp, alpha_tp, beta_fp, alpha_fp,
                           b_tp, b_fp, u_tp, u_fp, x_covariates, sig_all, rho_mat);

      // sample g, b_tp, b_fp
      // Rcpp::Rcout << "part 2, ";
      g = sample_g_c(g, g_free_ind, xi, x_covariates);

      // sample coefficient for eta
      // arma::uvec xi_34_loc = arma::find(xi==3 || xi==4);
      // arma::uvec xi_24_loc = arma::find(xi==2 || xi==4);
      // b_tp = sample_b_new1(Sigma1, Sigma2, e1234.rows(xi_34_loc), x_covariates.rows(xi_34_loc), b_tp, female.rows(xi_34_loc), true);
      // b_fp = sample_b_new1(Sigma1, Sigma2, e1234.rows(xi_24_loc), x_covariates.rows(xi_24_loc), b_fp, female.rows(xi_24_loc), false);
      // Rcpp::Rcout << "part 2.5, ";
      // b_tp = sample_b_general(sig_all, rho_mat.cols(xi_34_loc), e1234.rows(xi_34_loc), x_covariates.rows(xi_34_loc), b_tp, true);
      // b_fp = sample_b_general(sig_all, rho_mat.cols(xi_24_loc), e1234.rows(xi_24_loc), x_covariates.rows(xi_24_loc), b_fp, false);
      b_tp = sample_b_general(sig_all, rho_mat, e1234, x_covariates, b_tp, true);
      b_fp = sample_b_general(sig_all, rho_mat, e1234, x_covariates, b_fp, false);
      
      // sample coefficients for financial help latent variables
      // Rcpp::Rcout << "part 3, ";
      // u_tp = sample_u_new1(Sigma1, Sigma2, e1234.rows(xi_34_loc), u_tp, x_covariates.rows(xi_34_loc), female.rows(xi_34_loc), true);
      // u_fp = sample_u_new1(Sigma1, Sigma2, e1234.rows(xi_24_loc), u_fp, x_covariates.rows(xi_24_loc), female.rows(xi_24_loc), false);
      // u_tp = sample_u_general(sig_all, rho_mat.cols(xi_34_loc), e1234.rows(xi_34_loc), u_tp, x_covariates.rows(xi_34_loc), true);
      // u_fp = sample_u_general(sig_all, rho_mat.cols(xi_24_loc), e1234.rows(xi_24_loc), u_fp, x_covariates.rows(xi_24_loc), false);
      u_tp = sample_u_general(sig_all, rho_mat, e1234, u_tp, x_covariates, true);
      u_fp = sample_u_general(sig_all, rho_mat, e1234, u_fp, x_covariates, false);
      
      // Rcpp::Rcout << "part 4, ";
      // sample sig2_tp, sig2_fp, rho
      sample_sig2_general(e1234, sig_all, rho_mat, 0.01);

      // Rcpp::Rcout << "part 5, ";
      for(unsigned int cate=0;cate < categories; ++cate){
        for(unsigned int loc1=0; loc1 < 6; ++loc1){
          sample_beta_rho(e1234, sig_all, rho_covs, beta_rho, rho_stepsize(cate), loc1, cate);
        }
      }

      // Rcpp::Rcout << "part 6" << std::endl;
      // sample xi
      xi = sample_xi_new1(x_covariates, g, beta_tp, alpha_tp, beta_fp, alpha_fp,
                         ytp, b_tp, yfp, b_fp,
                         ytp_fin, u_tp,
                         yfp_fin, u_fp,
                         e1234);

      g_draws.col(i) = g.as_col();
      b_tp_draws.col(i) = b_tp;
      b_fp_draws.col(i) = b_fp;
      u_tp_draws.col(i) = u_tp;
      u_fp_draws.col(i) = u_fp;

      sig_draws.col(i) = sig_all;
      beta_rho_draws.col(i) = beta_rho.as_col();

      if(verbose)
        Rcpp::Rcout << "step: " << i << "b_tp1" << b_tp(0) << "\t b_fp1 " <<  b_fp(0) <<
          "\t u_tp1" << u_tp(0) << "\t u_fp1" << u_fp(0) << "\t g[1,2] " << g(0,1) <<
            "\t sig_all" <<sig_all.t() <<
              "\t beta_rho_0" << beta_rho.col(0).t() <<
                "\t beta_rho_1" << beta_rho.col(1).t() <<
                "\t rho_mat1" << rho_mat.col(0).t();
      // if(i % 100 == 0){
      //   Rcpp::Rcout << "\t acception:";
      //   for(int kk=0;kk<categories;++kk){
      //     arma::vec tmp = arma::diff(beta_rho_draws.row(6*kk).t());
      //     tmp = arma::find(tmp);
      //     Rcpp::Rcout << tmp.n_elem / (i-1) << ", ";
      //   }
      // } 

    }
  }
  return Rcpp::List::create(Rcpp::Named("g_draws") = g_draws.t(),
                            Rcpp::Named("b_tp_draws") = b_tp_draws.t(),
                            Rcpp::Named("b_fp_draws") = b_fp_draws.t(),
                            Rcpp::Named("u_tp_draws") = u_tp_draws.t(),
                            Rcpp::Named("u_fp_draws") = u_fp_draws.t(),
                            Rcpp::Named("sig_draws") = sig_draws.t(),
                            Rcpp::Named("beta_rho_draws") = beta_rho_draws.t(),
                            Rcpp::Named("e1234_last") = e1234,
                            Rcpp::Named("xi_last") = xi);
}

// Rcpp::List lvmcorr_seperate_fin_rho_only(arma::mat ytp_all, arma::mat yfp_all, arma::mat x_covariates,
//                                      arma::mat rho_covs,
//                                      arma::vec b_tp, arma::vec b_fp,
//                                      arma::mat g, arma::mat g_free_ind,
//                                      arma::mat e1234, arma::vec sig_init, arma::mat beta_rho_init,
//                                      arma::uvec xi,
//                                      arma::mat beta_tp_all, arma::mat alpha_tp_all,
//                                      arma::mat beta_fp_all, arma::mat alpha_fp_all,
//                                      arma::vec u_tp, arma::vec u_fp, arma::vec rho_stepsize,
//                                      int mcmc_len, bool verbose=false){
//   int N = ytp_all.n_rows;
//   unsigned int categories = beta_rho_init.n_cols;
//   arma::mat beta_rho_draws = arma::zeros(6 * categories, mcmc_len);
//   arma::mat ytp = ytp_all.cols(0, 6);
//   arma::mat yfp = yfp_all.cols(0, 6);
//   arma::vec ytp_fin = ytp_all.col(7);
//   arma::vec yfp_fin = yfp_all.col(7);
//   arma::mat beta_tp = beta_tp_all.cols(0, 6);
//   arma::mat beta_fp = beta_fp_all.cols(0, 6);
//   arma::mat alpha_tp = alpha_tp_all.cols(0, 6);
//   arma::mat alpha_fp = alpha_fp_all.cols(0, 6);
//   
//   ETAProgressBar pb;
//   Progress p(mcmc_len, !verbose, pb);
//   arma::mat beta_rho = beta_rho_init; // temp mat for rho coefficients
//   arma::mat rho_mat = arma::zeros(6, N); // temp mat 6 * N for all correlations
//   for(int i=0;i<mcmc_len;++i){
//     // sample e_tp, e_fp, e12
//     if ( p.increment() ) {
//       rho_mat = fisher_z_trans(beta_rho * rho_covs.t());// fisher z transformation
//       sample_e1234_general(e1234, ytp, yfp, ytp_fin, yfp_fin, xi,
//                            beta_tp, alpha_tp, beta_fp, alpha_fp,
//                            b_tp, b_fp, u_tp, u_fp, x_covariates, sig_all, rho_mat);
//       
//       Rcpp::Rcout << "part 5, ";
//       for(unsigned int cate=0;cate < categories; ++cate){
//         sample_beta_rho(e1234, sig_all, rho_covs, beta_rho, rho_stepsize(cate), cate);
//       }
//       xi = sample_xi_new(x_covariates, g, beta_tp, alpha_tp, beta_fp, alpha_fp,
//                          ytp, b_tp, yfp, b_fp,
//                          ytp_fin, u_tp,
//                          yfp_fin, u_fp,
//                          e1234);
//       
//       beta_rho_draws.col(i) = beta_rho.as_col();
//     }
//   }
//   return Rcpp::List::create(Rcpp::Named("beta_rho_draws") = beta_rho_draws.t(),
//                             Rcpp::Named("e1234_last") = e1234,
//                             Rcpp::Named("xi_last") = xi);
// }
// Rcpp::List lvmcorr_seperate_fin_cpp2(arma::mat ytp_all, arma::mat yfp_all, arma::mat x_covariates, 
//                                   arma::vec b_tp, arma::vec b_fp,
//                                   arma::mat g, arma::mat g_free_ind, //double sig2_tp, double sig2_fp,
//                                   arma::mat e1234,
//                                   arma::uvec xi, 
//                                   arma::mat beta_tp_all, arma::mat alpha_tp_all,arma::mat beta_fp_all, arma::mat alpha_fp_all,
//                                   arma::vec u_tp, arma::vec u_fp, // the first elements of u_tp and u_fp are set to be 1 for identifiability
//                                   int mcmc_len, int adaptive_steps=50, bool verbose=false){
//   arma::mat g_draws = arma::zeros(g.n_elem,mcmc_len);
//   arma::mat b_tp_draws = arma::zeros(b_tp.n_elem,mcmc_len);
//   arma::mat b_fp_draws = arma::zeros(b_fp.n_elem,mcmc_len);
//   arma::mat u_tp_draws = arma::zeros(u_tp.n_elem, mcmc_len);
//   arma::mat u_fp_draws = arma::zeros(u_fp.n_elem, mcmc_len);
//   arma::mat sig2_draws1 = arma::zeros(4, mcmc_len);
//   arma::mat sig2_draws2 = arma::zeros(4, mcmc_len);
//   arma::mat sig2_draws3 = arma::zeros(4, mcmc_len);
//   arma::mat sig2_draws4 = arma::zeros(4, mcmc_len);
//   arma::mat rho_draws1 = arma::zeros(16, mcmc_len);
//   arma::mat rho_draws2 = arma::zeros(16, mcmc_len);
//   arma::mat rho_draws3 = arma::zeros(16, mcmc_len);
//   arma::mat rho_draws4 = arma::zeros(16, mcmc_len);
//   
//   arma::mat ytp = ytp_all.cols(0, 6);
//   arma::mat yfp = yfp_all.cols(0, 6);
//   arma::vec ytp_fin = ytp_all.col(7);
//   arma::vec yfp_fin = yfp_all.col(7);
//   arma::mat beta_tp = beta_tp_all.cols(0, 6);
//   arma::mat beta_fp = beta_fp_all.cols(0, 6);
//   arma::mat alpha_tp = alpha_tp_all.cols(0, 6);
//   arma::mat alpha_fp = alpha_fp_all.cols(0, 6);
//   
//   ETAProgressBar pb;
//   Progress p(mcmc_len, !verbose, pb);
//   arma::mat Sigma1 = arma::eye(4,4);
//   arma::mat Sigma2 = arma::eye(4,4);
//   arma::mat Sigma3 = arma::eye(4,4);
//   arma::mat Sigma4 = arma::eye(4,4);
//   
//   arma::vec female = x_covariates.col(1);
//   arma::vec distlong = x_covariates.col(2);
//   arma::uvec female_loc = arma::find(female==1 && distlong == 1);
//   arma::uvec male_loc = arma::find(female==0 && distlong == 1);
//   arma::uvec part3_loc = arma::find(female==1 && distlong == 0);
//   arma::uvec part4_loc = arma::find(female==0 && distlong == 0);
//   
//   for(int i=0;i<mcmc_len;++i){
//     // sample e_tp, e_fp, e12
//     if ( p.increment() ) {
//       // Sigma = construct_sigma(sig2_all, rho);
//       // // sample missing ytp and yfp
//       // impute_ytp_mis(ytp, ytp_mis_ind, beta_tp, alpha_tp, x_covariates, b_tp, e_tp, e1234.col(0), xi);
//       // impute_yfp_mis(yfp, yfp_mis_ind, beta_fp, alpha_fp, x_covariates, b_fp, e_fp, e1234.col(1), xi);
//       // note: sampling of missing responses are depracated due to that the missing responses are all from two items
//       //       they are actually inapplicable rather than missing values.
//       // sample e1234
//       Rcpp::Rcout << "part 1, ";
//       // e_tp = sample_e_tp_c(e_tp, ytp, xi, b_tp, sig2_tp, beta_tp, alpha_tp, x_covariates);
//       // e_fp = sample_e_fp_c(e_fp, yfp, xi, b_fp, sig2_fp, beta_fp, alpha_fp, x_covariates);
//       arma::mat e1234_1 = e1234.rows(female_loc);
//       sample_e1234(e1234_1, ytp.rows(female_loc), yfp.rows(female_loc), 
//                    ytp_fin.rows(female_loc), yfp_fin.rows(female_loc), xi.rows(female_loc), 
//                    beta_tp, alpha_tp, beta_fp, alpha_fp, 
//                    b_tp, b_fp, u_tp, u_fp, x_covariates.rows(female_loc), Sigma1);
//       
//       arma::mat e1234_2 = e1234.rows(male_loc);
//       sample_e1234(e1234_2, ytp.rows(male_loc), yfp.rows(male_loc), 
//                    ytp_fin.rows(male_loc), yfp_fin.rows(male_loc), xi.rows(male_loc), 
//                    beta_tp, alpha_tp, beta_fp, alpha_fp, 
//                    b_tp, b_fp, u_tp, u_fp, x_covariates.rows(male_loc), Sigma2);
//       
//       arma::mat e1234_3 = e1234.rows(part3_loc);
//       sample_e1234(e1234_3, ytp.rows(part3_loc), yfp.rows(part3_loc), 
//                    ytp_fin.rows(part3_loc), yfp_fin.rows(part3_loc), xi.rows(part3_loc), 
//                    beta_tp, alpha_tp, beta_fp, alpha_fp, 
//                    b_tp, b_fp, u_tp, u_fp, x_covariates.rows(part3_loc), Sigma3);
//       
//       arma::mat e1234_4 = e1234.rows(part4_loc);
//       sample_e1234(e1234_4, ytp.rows(part4_loc), yfp.rows(part4_loc), 
//                    ytp_fin.rows(part4_loc), yfp_fin.rows(part4_loc), xi.rows(part4_loc), 
//                    beta_tp, alpha_tp, beta_fp, alpha_fp, 
//                    b_tp, b_fp, u_tp, u_fp, x_covariates.rows(part4_loc), Sigma4);
//       e1234.rows(female_loc) = e1234_1;
//       e1234.rows(male_loc)   = e1234_2;
//       e1234.rows(part3_loc)  = e1234_3;
//       e1234.rows(part4_loc)  = e1234_4;
//       // sample g, b_tp, b_fp
//       Rcpp::Rcout << "part 2, ";
//       g = sample_g_c(g, g_free_ind, xi, x_covariates);
//       
//       // sample coefficient for eta
//       arma::uvec xi_34_loc = arma::find(xi==3 || xi==4);
//       b_tp = sample_b_new2(Sigma1, Sigma2, Sigma3, Sigma4, e1234.rows(xi_34_loc), x_covariates.rows(xi_34_loc), b_tp, 
//                            female.rows(xi_34_loc), distlong.rows(xi_34_loc), true);
//       arma::uvec xi_24_loc = arma::find(xi==2 || xi==4);
//       b_fp = sample_b_new2(Sigma1, Sigma2, Sigma3, Sigma4, e1234.rows(xi_24_loc), x_covariates.rows(xi_24_loc), b_fp, 
//                            female.rows(xi_24_loc), distlong.rows(xi_24_loc), false);
//       
//       
//       // sample coefficients for financial help latent variables
//       Rcpp::Rcout << "part 3, ";
//       u_tp = sample_u_new2(Sigma1, Sigma2, Sigma3, Sigma4, e1234.rows(xi_34_loc), u_tp, x_covariates.rows(xi_34_loc), 
//                            female.rows(xi_34_loc), distlong.rows(xi_34_loc), true);
//       u_fp = sample_u_new2(Sigma1, Sigma2, Sigma3, Sigma4, e1234.rows(xi_24_loc), u_fp, x_covariates.rows(xi_24_loc),
//                            female.rows(xi_24_loc), distlong.rows(xi_24_loc), false);
//       u_tp /= std::abs(u_tp(0));
//       u_fp /= std::abs(u_fp(0));
//       
//       Sigma1 = sample_Sigma_new1(e1234.rows(female_loc));
//       Sigma2 = sample_Sigma_new1(e1234.rows(male_loc));
//       Sigma3 = sample_Sigma_new1(e1234.rows(part3_loc));
//       Sigma4 = sample_Sigma_new1(e1234.rows(part4_loc));
//       
//       Rcpp::Rcout << "part 6" << std::endl;
//       // sample xi
//       xi = sample_xi_new(x_covariates, g, beta_tp, alpha_tp, beta_fp, alpha_fp,
//                          ytp, b_tp, yfp, b_fp,
//                          ytp_fin, u_tp,
//                          yfp_fin, u_fp,
//                          e1234);
//       
//       
//       g_draws.col(i) = g.as_col();
//       b_tp_draws.col(i) = b_tp;
//       b_fp_draws.col(i) = b_fp;
//       u_tp_draws.col(i) = u_tp;
//       u_fp_draws.col(i) = u_fp;
//       
//       
//       arma::vec sig2_all1 = Sigma1.diag();
//       arma::vec sig2_all2 = Sigma2.diag();
//       arma::vec sig2_all3 = Sigma3.diag();
//       arma::vec sig2_all4 = Sigma4.diag();
//       
//       sig2_draws1.col(i) = sig2_all1;
//       sig2_draws2.col(i) = sig2_all2;
//       sig2_draws3.col(i) = sig2_all3;
//       sig2_draws4.col(i) = sig2_all4;
//       rho_draws1.col(i) = (arma::diagmat(1.0 / arma::sqrt(sig2_all1)) * Sigma1 * arma::diagmat(1.0 / arma::sqrt(sig2_all1))).as_col();
//       rho_draws2.col(i) = (arma::diagmat(1.0 / arma::sqrt(sig2_all2)) * Sigma2 * arma::diagmat(1.0 / arma::sqrt(sig2_all2))).as_col();
//       rho_draws3.col(i) = (arma::diagmat(1.0 / arma::sqrt(sig2_all3)) * Sigma3 * arma::diagmat(1.0 / arma::sqrt(sig2_all3))).as_col();
//       rho_draws4.col(i) = (arma::diagmat(1.0 / arma::sqrt(sig2_all4)) * Sigma4 * arma::diagmat(1.0 / arma::sqrt(sig2_all4))).as_col();
//       
//       if(verbose)
//         Rcpp::Rcout << "step: " << i << "b_tp1" << b_tp(0) << "\t b_fp1 " <<  b_fp(0) << 
//           "\t u_tp1" << u_tp(0) << "\t u_fp1" << u_fp(0) << "\t g[1,2] " << g(0,1) << 
//             "\t sig2_all1" <<sig2_all1.t() <<  "\t sig2_all2" << sig2_all2.t() <<
//               "\t rho_0" << rho_draws1(1,i) << rho_draws1(2,i) << rho_draws1(3,i) << rho_draws1(6,i)<< rho_draws1(7,i)<< rho_draws1(11,i)<<
//                 "\t rho_1" << rho_draws2(1,i) << rho_draws2(2,i) << rho_draws2(3,i) << rho_draws2(6,i)<< rho_draws2(7,i)<< rho_draws2(11,i)<< std::endl;
//       
//     }
//   }
//   return Rcpp::List::create(Rcpp::Named("g_draws") = g_draws.t(),
//                             Rcpp::Named("b_tp_draws") = b_tp_draws.t(),
//                             Rcpp::Named("b_fp_draws") = b_fp_draws.t(),
//                             Rcpp::Named("rho_draws1") = rho_draws1.t(),
//                             Rcpp::Named("rho_draws2") = rho_draws2.t(),
//                             Rcpp::Named("rho_draws3") = rho_draws3.t(),
//                             Rcpp::Named("rho_draws4") = rho_draws4.t(),
//                             Rcpp::Named("u_tp_draws") = u_tp_draws.t(),
//                             Rcpp::Named("u_fp_draws") = u_fp_draws.t(),
//                             Rcpp::Named("sig2_draws1") = sig2_draws1.t(),
//                             Rcpp::Named("sig2_draws2") = sig2_draws2.t(),
//                             Rcpp::Named("sig2_draws3") = sig2_draws3.t(),
//                             Rcpp::Named("sig2_draws4") = sig2_draws4.t(),
//                             Rcpp::Named("e1234_last") = e1234,
//                             Rcpp::Named("xi_last") = xi,
//                             Rcpp::Named("Sigma1") = Sigma1,
//                             Rcpp::Named("Sigma2") = Sigma2,
//                             Rcpp::Named("Sigma3") = Sigma3,
//                             Rcpp::Named("Sigma4") = Sigma4);
// }