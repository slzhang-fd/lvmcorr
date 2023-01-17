#' An R package for joint structural equation modeling
#' 
#' @useDynLib lvmcorr
#' @importFrom Rcpp evalCpp
#' @importFrom stats rbinom
#' 
#' @export lvmcorr
lvmcorr <- function(ytp, yfp, x_covariates, b_tp_init, b_fp_init, g_init, g_free_ind, sig2_tp_init, sig2_fp_init,
                 rho_init, e_tp, e_fp, e12, xi, beta_tp, alpha_tp, beta_fp, alpha_fp, mcmc_len){
  N <- nrow(ytp)
  mcmc_params_draw <- matrix(0, mcmc_len, length(g_init) + length(b_tp_init) + length(b_fp_init) + 3)
  colnames(mcmc_params_draw) <- c(paste0("g[",c(outer(1:nrow(g_init), 1:ncol(g_init), paste, sep=",")), "]"),
                                  paste0("b_tp[", 1:length(b_tp_init), "]"),
                                  paste0("b_fp[", 1:length(b_fp_init), "]"),
                                  "sig2_tp", "sig2_fp", "rho")
  mcmc_params_draw[1,] <- c(g_init, b_tp_init, b_fp_init, sig2_tp_init, sig2_fp_init, rho_init)
  time_elap <- matrix(0, mcmc_len-1, 4)
  ytp_mis_ind <- which(is.na(ytp))
  yfp_mis_ind <- which(is.na(yfp))
  for(mcmc_step in 2:mcmc_len){
    ## sample e_tp, e_fp, e12
    time1 <- Sys.time()
    e_tp <- sample_e_tp_c(e_tp, ytp, xi, b_tp_init, sig2_tp_init, beta_tp, alpha_tp, x_covariates)
    e_fp <- sample_e_fp_c(e_fp, yfp, xi, b_fp_init, sig2_fp_init, beta_fp, alpha_fp, x_covariates)
    e12 <- sample_e12_c(e12, ytp, yfp, xi, beta_tp, alpha_tp, beta_fp, alpha_fp, sig2_tp_init, sig2_fp_init,
                        rho_init, b_tp_init, b_fp_init, x_covariates)
    ## optional: sample missing ytp and yfp
    prob_tp <- calcu_mu_tp(beta_tp, alpha_tp, x_covariates, b_tp_init, e_tp, e12[,1], xi)
    ytp[ytp_mis_ind] <- rbinom(length(ytp_mis_ind), 1, prob_tp[ytp_mis_ind])
    prob_fp <- calcu_mu_fp(beta_fp, alpha_fp, x_covariates, b_fp_init, e_fp, e12[,2], xi)
    yfp[yfp_mis_ind] <- rbinom(length(yfp_mis_ind), 1, prob_fp[yfp_mis_ind])
    ## sample g, b_tp, b_fp
    time2 <- Sys.time()
    sample_g_b(g_init, g_free_ind, xi, x_covariates, ytp, yfp, beta_tp, alpha_tp, beta_fp, alpha_fp,
               b_tp_init, b_fp_init, e_tp, e_fp, e12, sig2_tp_init, sig2_fp_init, rho_init)
    #g_init <- tmp$g; b_tp_init <- tmp$b_tp; b_fp_init <- tmp$b_fp;
    ## sample sig2_tp, sig2_fp, rho
    time3 <- Sys.time()
    sig2_tp_init <- sample_sig2_tp_c(sig2_tp_init, sig2_fp_init, rho_init, e_tp, e12, xi)
    sig2_fp_init <- sample_sig2_fp_c(sig2_tp_init, sig2_fp_init, rho_init, e_fp, e12, xi)
    rho_init <- sample_rho_c(rho_init, xi, sig2_tp_init, sig2_fp_init, e12)
    ## sample xi
    time4 <- Sys.time()
    xi <- sample_xi(x_covariates, g_init, beta_tp, alpha_tp, beta_fp, alpha_fp, 
                    ytp, b_tp_init, e_tp, yfp, b_fp_init, e_fp, e12)
    time5 <- Sys.time()
    time_elap[mcmc_step-1,] <- c(time2-time1, time3-time2, time4-time3, time5-time4)
    mcmc_params_draw[mcmc_step,] <- c(g_init, b_tp_init, b_fp_init, sig2_tp_init, sig2_fp_init, rho_init)
    cat("step: ", mcmc_step, "\t b_tp1 ", b_tp_init[1], "\t b_fp1 ", b_fp_init[1], "\t g[1,2]", g_init[1,2],
        "\t sig2_tp ", sig2_tp_init, "\t sig2_fp ", sig2_fp_init, "\t rho ", rho_init, "\n")
  }
  return(structure(list("mcmc_params_draw" = mcmc_params_draw,
              "mcmc_lv_draw_last" = list('xi'=xi, 'e_tp'=e_tp, 'e_fp'=e_fp, 'e12'=e12),
              "time_elap" = time_elap), class="lvmcorr"))
}
#' @export lvmcorr_update
lvmcorr_update <- function(lvmcorr_res, ytp, yfp, x_covariates, g_free_ind, beta_tp, alpha_tp, beta_fp, alpha_fp, mcmc_len){
  mcmc_len_old <- nrow(lvmcorr_res$mcmc_params_draw)
  b_tp_old <- lvmcorr_res$mcmc_params_draw[mcmc_len_old,grep("b_tp",colnames(lvmcorr_res$mcmc_params_draw))]
  b_fp_old <- lvmcorr_res$mcmc_params_draw[mcmc_len_old,grep("b_fp",colnames(lvmcorr_res$mcmc_params_draw))]
  g_old <- matrix(lvmcorr_res$mcmc_params_draw[mcmc_len_old,grep("g[",colnames(lvmcorr_res$mcmc_params_draw), fixed=T)], nrow(g_free_ind))
  sig2_tp_old <- lvmcorr_res$mcmc_params_draw[mcmc_len_old,grep("sig2_tp",colnames(lvmcorr_res$mcmc_params_draw))]
  sig2_fp_old <- lvmcorr_res$mcmc_params_draw[mcmc_len_old,grep("sig2_fp",colnames(lvmcorr_res$mcmc_params_draw))]
  rho_old <- lvmcorr_res$mcmc_params_draw[mcmc_len_old,grep("rho",colnames(lvmcorr_res$mcmc_params_draw))]
  lv_draws_old <- lvmcorr_res$mcmc_lv_draw_last
  lvmcorr_res_add <- lvmcorr(ytp, yfp, x_covariates, b_tp_old, b_fp_old, g_old, g_free_ind, sig2_tp_old, sig2_fp_old, rho_old,
                       lv_draws_old$e_tp, lv_draws_old$e_fp, lv_draws_old$e12, lv_draws_old$xi, 
                       beta_tp, alpha_tp, beta_fp, alpha_fp, mcmc_len)
  # ytp, yfp, x_covariates, b_tp_init, b_fp_init, g_init, g_free_ind, sig2_tp_init, sig2_fp_init,
  # rho_init, e_tp, e_fp, e12, xi, beta_tp, alpha_tp, beta_fp, alpha_fp, mcmc_len
  return(list("mcmc_params_draw" = rbind(lvmcorr_res$mcmc_params_draw, lvmcorr_res_add$mcmc_params_draw),
              "mcmc_lv_draw_last" = lvmcorr_res_add$mcmc_lv_draw_last,
              "time_elap" = lvmcorr_res_add$time_elap))
}
#' @export dylanie_model
dylanie_model <- function(formula, data, initvals=NULL, mcmc_len, verbose=F){
  term_label <- unlist(attributes(terms(formula))['term.labels'])
  nparam <- length(term_label) + 1
  if(is.null(initvals)){
    N <- nrow(data$ytp)
    g_init <- matrix(0, 14, 4)
    g_init[,1] <- 0
    g_init[3,2] <- g_init[6,3] <- g_init[7,3] <- -5
    initvals <- list(b_tp_init = matrix(0, nparam, 1), b_fp_init = matrix(0, nparam, 1), 
                     g_init = g_init, sig2_tp_init = 1/0.35,
                     sig2_fp_init = 1/0.2, rho_init = 0,
                     e_tp = rnorm(N), e_fp = rnorm(N), e12 = matrix(rnorm(2*N), N, 2), xi = data$xi)
  }
  res_cpp <- lvmcorr_cpp(data$ytp, data$yfp, as.matrix(data$x_covariates[c('intercept',term_label)]), 
                      matrix(initvals$b_tp_init[1:nparam],ncol=1), matrix(initvals$b_fp_init[1:nparam],ncol=1), 
                      matrix(initvals$g_init[1:nparam,],nrow=nparam), matrix(data$g_free_ind[1:nparam,],nrow=nparam), 
                      initvals$sig2_tp_init, initvals$sig2_fp_init, initvals$rho_init,
                      initvals$e_tp, initvals$e_fp, initvals$e12, initvals$xi, 
                      data$beta_tp, data$alpha_tp, data$beta_fp, data$alpha_fp, mcmc_len, verbose)
  return(res_cpp)
}
#' @export dylanie_model_simple
dylanie_model_simple <- function(formula, data, mcmc_len, verbose = F){
  N <- nrow(data$ytp)
  J <- ncol(data$ytp)
  
  ## set the fixed parameters for the measurement model
  # loadings
  lambda_tp <- c(1.159, 1.873, 1.073, 1.560, 1.640, 1, 0.625, 0.588)
  lambda_fp <- c(0.833, 1.293, 0.760, 0.453, 0.770, 1, 0.604, 0.578)
  
  # coefficients
  a_tp <- matrix(0, 5, 8)
  a_fp <- matrix(0, 5, 8)
  
  # intercepts
  a_tp[1,] <- c(2.012, 1.694, -0.329, -2.335, -0.965, 0, 0.907, -1.291)
  a_fp[1,] <- c(1.855, 2.621, 2.107, 2.473, 0.488, 0, 0.514, 0.991)
  
  # female
  a_tp[2,] <- c(-0.635, 0, 0, 0, 0, 0, -1.529, -0.642)
  a_fp[2,] <- c(0, 0, -0.222, 0, 0, 0, 0, -0.322)
  
  # distlong
  # a_tp[3,] <- c(-0.831, -0.128, 0.415, 0, 0.466, 0, 0.038, 0)
  a_tp[3,] <- c(-0.831, -0.128, 0.415, 0, 0.466, 0, 0.056, 0)
  a_fp[3,] <- c(1.100, 0.603, 1.997, 0.689, 2.536, 0, -0.685, 0)
  
  # etaXfemale
  a_tp[4,] <- c(-0.461, 0, 0, 0, 0, 0, 0.090, -0.030)
  a_fp[4,] <- c(0, 0, 0.054, 0, 0, 0, 0, -0.057)
  
  # etaXdistlong
  a_tp[5,] <- c(0.905, 1.225, 0.952, 0, 0.626, 0, 0.535, 0)
  a_fp[5,] <- c(0.898, 0.740, 1.039, 0.618, 1.213, 0, 0.049, 0)
  
  beta_tp <- cbind(rep(1, N), jags.data$female, jags.data$distlong) %*% a_tp[1:3,]
  beta_fp <- cbind(rep(1, N), jags.data$female, jags.data$distlong) %*% a_fp[1:3,]
  
  alpha_tp <- cbind(rep(1, N), jags.data$female, jags.data$distlong) %*% rbind(lambda_tp, a_tp[4:5,])
  alpha_fp <- cbind(rep(1, N), jags.data$female, jags.data$distlong) %*% rbind(lambda_fp, a_fp[4:5,])
  
  g_free_ind <- matrix(1, 14, 4)
  g_free_ind[,1] <- g_free_ind[3,2] <- g_free_ind[6,3] <- g_free_ind[7,3] <- 0
  
  term_label <- unlist(attributes(terms(formula))['term.labels'])
  nparam <- length(term_label) + 1
  
  g_init <- matrix(0, 14, 4)
  g_init[,1] <- 0
  g_init[3,2] <- g_init[6,3] <- g_init[7,3] <- -5
  initvals <- list(b_tp_init = matrix(0, nparam, 1), b_fp_init = matrix(0, nparam, 1), 
                   g_init = g_init, sig2_tp_init = 1/0.35,
                   sig2_fp_init = 1/0.2, rho_init = 0,
                   e_tp = rnorm(N), e_fp = rnorm(N), e12 = matrix(rnorm(2*N), N, 2))
  x_covariates <- matrix(rep(1, N), ncol = 1)
  if(length(term_label)) x_covariates <- cbind(x_covariates, as.matrix(data.frame(data[term_label])))
  res_cpp <- lvmcorr_cpp(data$ytp, data$yfp, x_covariates, 
                      initvals$b_tp_init, initvals$b_fp_init, 
                      matrix(g_init[1:nparam,],nrow=nparam), matrix(g_free_ind[1:nparam,],nrow=nparam), 
                      initvals$sig2_tp_init, initvals$sig2_fp_init, initvals$rho_init,
                      initvals$e_tp, initvals$e_fp, initvals$e12, data$xi, 
                      beta_tp, alpha_tp, beta_fp, alpha_fp, mcmc_len, verbose)
  return(res_cpp)
}
#' @export dylanie_model_update
dylanie_model_update <- function(dylanie_res, mcmc_len, verbose = F){
  mcmc_len_old <- nrow(dylanie_res$g_draws)
  b_tp_old <- dylanie_res$b_tp_draws[mcmc_len_old,]
  b_fp_old <- dylanie_res$b_fp_draws[mcmc_len_old,]
  g_old <- matrix(dylanie_res$g_draws[mcmc_len_old,], ncol = 4)
  updated_res <- lvmcorr_cpp(dylanie_res$ytp, dylanie_res$yfp, dylanie_res$x_covariates, 
                      matrix(b_tp_old, ncol = 1), matrix(b_fp_old, ncol = 1), 
                      g_old, dylanie_res$g_free_ind, 
                      dylanie_res$sig2_tp_draws[mcmc_len_old], dylanie_res$sig2_fp_draws[mcmc_len_old], 
                      dylanie_res$rho_draws[mcmc_len_old],
                      dylanie_res$e_tp_last, dylanie_res$e_fp_last, dylanie_res$e12_last, dylanie_res$xi_last, 
                      dylanie_res$beta_tp, dylanie_res$alpha_tp, dylanie_res$beta_fp, dylanie_res$alpha_fp, 
                      mcmc_len, verbose)
  return(list("g_draws" = rbind(dylanie_res$g_draws, updated_res$g_draws),
              "b_tp_draws" = rbind(dylanie_res$b_tp_draws, updated_res$b_tp_draws),
              "b_fp_draws" = rbind(dylanie_res$b_fp_draws, updated_res$b_fp_draws),
              "sig2_tp_draws" = c(dylanie_res$sig2_tp_draws, updated_res$sig2_tp_draws),
              "sig2_fp_draws" = c(dylanie_res$sig2_fp_draws, updated_res$sig2_fp_draws),
              "rho_draws" = c(dylanie_res$rho_draws, updated_res$rho_draws),
              "e_tp_last" = updated_res$e_tp_last,
              "e_fp_last" = updated_res$e_fp_last,
              "e12_last" = updated_res$e12_last,
              "xi_last" = updated_res$xi_last,
              "ytp" = updated_res$ytp,
              "yfp" = updated_res$yfp,
              "x_covariates" = updated_res$x_covariates,
              "g_free_ind" = updated_res$g_free_ind,
              "beta_tp" = updated_res$beta_tp,
              "alpha_tp" = updated_res$alpha_tp,
              "beta_fp" = updated_res$beta_fp,
              "alpha_fp" = updated_res$alpha_fp))
}
#' @export dylanie_model_mchains
dylanie_model_mchains <- function(formula, data, mcmc_len, chains=1,verbose=F){
  term_label <- unlist(attributes(terms(formula))['term.labels'])
  nparam <- length(term_label) + 1
  N <- nrow(data$ytp)
  g_init <- matrix(0, 14, 4)
  g_init[,1] <- 0
  g_init[3,2] <- g_init[6,3] <- g_init[7,3] <- -5
  b_tp_init <- rep(0, 14)
  b_fp_init <- rep(0, 14)
  res_cpp <- lvmcorr_multi_chains(data$ytp, data$yfp, as.matrix(data$x_covariates[c('intercept',term_label)]), 
                               matrix(b_tp_init[1:nparam],ncol=1), matrix(b_fp_init[1:nparam],ncol=1), 
                               matrix(g_init[1:nparam,],nrow=nparam), matrix(data$g_free_ind[1:nparam,],nrow=nparam), 
                               1/0.35, 1/0.2, 0, data$xi,
                               data$beta_tp, data$alpha_tp, data$beta_fp, data$alpha_fp, mcmc_len, chains)
  return(res_cpp)
}
#' @export jcorr_model_simple
jcorr_model_simple <- function(formula_mean, formula_corr, data, mcmc_len, verbose = F){
  N <- nrow(data$ytp)
  J <- ncol(data$ytp)
  ## intercept parameters
  beta_tp <- outer(
    rep(1, N),
    c(0.832, 1.016, -0.284, -1.324, -0.766, 0.000, -0.215)
  )
  beta_fp <- outer(
    rep(1, N),
    c(1.536, 2.081, 1.574, 2.247, 0.819, 0.000, 0.372)
  )
  ## slope parameters
  alpha_tp <- outer(
    rep(1, N),
    c(1.121, 2.375, 1.237, 1.321, 1.319, 1.000, 0.574)
  )
  alpha_fp <- outer(
    rep(1, N),
    c(1.138, 1.698, 1.150, 0.890, 1.146, 1.000, 0.742)
  )
  ## define covariates for mean
  term_label <- unlist(attributes(terms(formula_mean))['term.labels'])
  nparam <- length(term_label) + 1
  x_covariates <- data.frame("intercept"= rep(1, N),
                              data[term_label])
  ## define covariates for correlation
  term_label_corr <- unlist(attributes(terms(formula_corr))['term.labels'])
  cor_vars <- as.matrix(data.frame("intercept" = rep(1,N),
                          data[term_label_corr]))
  
  #### SET UP initial values ############################
  anytp <- rowSums(data$ytp[, 1:8], na.rm = TRUE)
  anytp[anytp > 0] <- 1
  anyfp <- rowSums(data$yfp[, 1:8], na.rm = TRUE)
  anyfp[anyfp > 0] <- 1
  prop.table(table(anytp))
  prop.table(table(anyfp))
  prop.table(table(anytp, anyfp))
  xi.start <- anytp
  xi.start[anytp == 0 & anyfp == 0] <- 1
  xi.start[anytp == 0 & anyfp == 1] <- 2
  xi.start[anytp == 1 & anyfp == 0] <- 3
  xi.start[anytp == 1 & anyfp == 1] <- 4
  b_tp_init <- rep(0, nparam)
  b_fp_init <- rep(0, nparam)
  u_tp_init <- rep(0, nparam)
  u_fp_init <- rep(0, nparam)
  
  g_init <- matrix(0, nparam, 4)
  g_init[, 1] <- 0
  g_free_ind <- matrix(1, nparam, 4)
  g_free_ind[, 1] <- 0
  
  e1234 <- matrix(rnorm(4 * N), N, 4)
  
  sig_init <- c(1.054, 1.539)
  beta_rho_init <- matrix(0, 6, ncol(cor_vars))
  rho_stepsize <- c(0.02, 2e-3, 2e-5, 0.02, 0.02, 2e-3)
  
  ## feed data to the MCMC main program
  res <-
    lvmcorr_seperate_fin_general(data$ytp, data$yfp,
                                 matrix(unlist(x_covariates[, 1:nparam]), ncol = nparam),
                                 cor_vars,
                                 b_tp_init[1:nparam], b_fp_init[1:nparam],
                                 matrix(g_init[1:nparam, ], nrow = nparam),
                                 matrix(g_free_ind[1:nparam, ], nrow = nparam),
                                 e1234, sig_init, beta_rho_init,
                                 xi.start, beta_tp, alpha_tp, beta_fp, alpha_fp,
                                 u_tp_init[1:nparam], u_fp_init[1:nparam],
                                 rho_stepsize, mcmc_len, verbose)
  return(res)
}
# dylanie_model_fin_sep_simple <- function(formula, data, mcmc_len, verbose = F){
#   N <- nrow(data$ytp)
#   J <- ncol(data$ytp)
# 
#   ## set the fixed parameters for the measurement model
#   # loadings
#   lambda_tp <- c(1.159, 1.873, 1.073, 1.560, 1.640, 1, 0.625, 0.588)
#   lambda_fp <- c(0.833, 1.293, 0.760, 0.453, 0.770, 1, 0.604, 0.578)
# 
#   # coefficients
#   a_tp <- matrix(0, 5, 8)
#   a_fp <- matrix(0, 5, 8)
# 
#   # intercepts
#   a_tp[1,] <- c(2.012, 1.694, -0.329, -2.335, -0.965, 0, 0.907, -1.291)
#   a_fp[1,] <- c(1.855, 2.621, 2.107, 2.473, 0.488, 0, 0.514, 0.991)
# 
#   # female
#   a_tp[2,] <- c(-0.635, 0, 0, 0, 0, 0, -1.529, -0.642)
#   a_fp[2,] <- c(0, 0, -0.222, 0, 0, 0, 0, -0.322)
# 
#   # distlong
#   # a_tp[3,] <- c(-0.831, -0.128, 0.415, 0, 0.466, 0, 0.038, 0)
#   a_tp[3,] <- c(-0.831, -0.128, 0.415, 0, 0.466, 0, 0.056, 0)
#   a_fp[3,] <- c(1.100, 0.603, 1.997, 0.689, 2.536, 0, -0.685, 0)
# 
#   # etaXfemale
#   a_tp[4,] <- c(-0.461, 0, 0, 0, 0, 0, 0.090, -0.030)
#   a_fp[4,] <- c(0, 0, 0.054, 0, 0, 0, 0, -0.057)
# 
#   # etaXdistlong
#   a_tp[5,] <- c(0.905, 1.225, 0.952, 0, 0.626, 0, 0.535, 0)
#   a_fp[5,] <- c(0.898, 0.740, 1.039, 0.618, 1.213, 0, 0.049, 0)
# 
#   beta_tp <- cbind(rep(1, N), jags.data$female, jags.data$distlong) %*% a_tp[1:3,]
#   beta_fp <- cbind(rep(1, N), jags.data$female, jags.data$distlong) %*% a_fp[1:3,]
# 
#   alpha_tp <- cbind(rep(1, N), jags.data$female, jags.data$distlong) %*% rbind(lambda_tp, a_tp[4:5,])
#   alpha_fp <- cbind(rep(1, N), jags.data$female, jags.data$distlong) %*% rbind(lambda_fp, a_fp[4:5,])
# 
#   g_free_ind <- matrix(1, 14, 4)
#   g_free_ind[,1] <- g_free_ind[3,2] <- g_free_ind[6,3] <- g_free_ind[7,3] <- 0
# 
#   term_label <- unlist(attributes(terms(formula))['term.labels'])
#   nparam <- length(term_label) + 1
# 
#   g_init <- matrix(0, 14, 4)
#   g_init[,1] <- 0
#   g_init[3,2] <- g_init[6,3] <- g_init[7,3] <- -5
# 
#   #Starting values for latent class variables
#   anytp <- rowSums(data$ytp[,1:7], na.rm=TRUE)
#   anytp[anytp>0] <- 1
#   anyfp <- rowSums(data$yfp[,1:7], na.rm=TRUE)
#   anyfp[anyfp>0] <- 1
#   xi.start <- anytp
#   xi.start[anytp==0 & anyfp==0] <- 1
#   xi.start[anytp==0 & anyfp==1] <- 2
#   xi.start[anytp==1 & anyfp==0] <- 3
#   xi.start[anytp==1 & anyfp==1] <- 4
# 
#   initvals <- list(b_tp_init = matrix(0, nparam, 1), b_fp_init = matrix(0, nparam, 1),
#                    u_tp_init = rep(0, nparam), u_fp_init = rep(0, nparam),
#                    sig2_u_tp_init = 1, sig2_u_fp_init = 1,
#                    g_init = g_init, sig2_tp_init = 1/0.35,
#                    sig2_fp_init = 1/0.2, rho_init = 0,
#                    e_tp = rnorm(N), e_fp = rnorm(N), e1234 = matrix(rnorm(4*N), N, 4))
#   x_covariates <- matrix(rep(1, N), ncol = 1)
#   if(length(term_label)) x_covariates <- cbind(x_covariates, as.matrix(data.frame(data[term_label])))
#   res_cpp <- lvmcorr_seperate_fin_cpp(data$ytp, data$yfp, x_covariates,
#                       initvals$b_tp_init, initvals$b_fp_init,
#                       matrix(g_init[1:nparam,],nrow=nparam), matrix(g_free_ind[1:nparam,],nrow=nparam),
#                       initvals$sig2_tp_init, initvals$sig2_fp_init,
#                       initvals$rho_init,
#                       initvals$e_tp, initvals$e_fp, initvals$e1234,
#                       xi.start,
#                       beta_tp, alpha_tp, beta_fp, alpha_fp,
#                       initvals$u_tp_init, initvals$u_fp_init,
#                       mcmc_len, verbose)
#   return(res_cpp)
# }