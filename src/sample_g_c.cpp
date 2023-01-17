// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "arms_ori.h"
#include "lvmcorr_omp.h"

struct log_g_param{
  arma::uvec xi;
  arma::mat x_covariates;
  arma::mat g;
  arma::uword ii;
  arma::uword jj;
};

double log_g(double x, void* params){
  struct log_g_param *d;
  d = static_cast<struct log_g_param *> (params);
  arma::mat g = d->g;
  g(d->ii,d->jj) = x;
  arma::mat xg_mat = d->x_covariates * g;
  arma::vec row_max = arma::max(xg_mat, 1);
  xg_mat.each_col() -= row_max;
  double log_sum_exp_val = arma::accu(row_max + arma::log(arma::sum(arma::exp(xg_mat), 1)));
  return(arma::accu(g.cols((d->xi)-1) % (d->x_covariates).t())  - log_sum_exp_val - x * x / 200.0 );
}

// [[Rcpp::export]]
arma::mat sample_g_c(arma::mat g, const arma::mat &g_free_ind, const arma::uvec &xi, 
                     const arma::mat &x_covariates){
  int err, ninit = 4, npoint = 100, nsamp = 1, ncent = 0;
  int neval;
  double xinit[10]={-3.0, -1.0, 1.0, 3.0}, xl = -5.0, xr = 5.0;
  double xsamp[100], xcent[10], qcent[10] = {5., 30., 70., 95.};
  double convex = 1.;
  int dometrop = 1;
  double xprev = 0.0;
  
  log_g_param g_data;
  g_data.g = g;
  g_data.xi = xi;
  g_data.x_covariates = x_covariates;
  for(unsigned int jj=1;jj<g.n_cols;++jj){
    for(unsigned int ii=0;ii<g.n_rows;++ii){
      if(g_free_ind(ii,jj)){
        g_data.ii = ii;
        g_data.jj = jj;
        xprev = g_data.g(ii,jj);
        err = arms(xinit,ninit,&xl,&xr,log_g,&g_data,&convex,
                   npoint,dometrop,&xprev,xsamp,nsamp,qcent,xcent,ncent,&neval);
        if(err>0){
          Rprintf("error code: %d", err);
          Rcpp::stop("\n");
        }
        g_data.g(ii,jj) = xsamp[0];
      }
    }
  }
  return g_data.g;
}
