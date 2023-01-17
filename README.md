# lvmcorr

### An R package for Multivariate Data Analysis
Paralleled full Bayesian estimation for multivariate dyads data analysis using 
joint latent variable framework with linear correlation model. 

### Description
The `lvmcorr` R package we provide contains code associated with the article 
> Zhang, S., Kuha, J., & Steele, F. (2023). [Modelling correlation matrices in multivariate data, with application to reciprocity and complementarity of child-parent exchanges of support.](https://arxiv.org/abs/2210.14751)

It contains tailored estimation code for the proposed joint latent variable framework using full Bayesian methods. R and C++ compiled code (Rcpp, RcppArmadillo) is used with OpenMP API for parallel computing to boost the estimation. Direct sampling (when applicable by using proper conjugate prior) and adaptive rejection Metropolis sampling are both employed in the program. See Appendix D and E in the manuscript for more details. Furthermore, practical features for MCMC sampling program such as time consumption estimation with progress bar and support for interruptions of estimation (intermediate result will be saved, even for multi-thread function) are also enabled.

### An example
[Fitting-synthetic-data](https://github.com/slzhang-fd/jsem-ukhls/wiki/Fitting-synthetic-data)

### Supporting software requirements
R, gcc with OpenMP enabled

#### Version of primary software used

<!--
(e.g., R version 3.6.0)
-->

- R version 4.0.4
- gcc version 9.3.0 (with OpenMP 5 support)

#### Libraries and dependencies used by the code

<!--
Include version numbers (e.g., version numbers for any R or Python packages used)
-->
- Rcpp 1.0.6
- RcppArmadillo 0.10.2.1.0
- RcppProgress 0.4.2
- RcppDist 0.1.1
- RcppTN
- coda

### License

GPL v3.0

### Additional information

#### Configuration of the C++ Toolchain 
This step is optional, but it can result in compiled jsem programs that execute much faster than they otherwise would. Simply paste the following into R before installation
```{r, eval=F}
dotR <- file.path(Sys.getenv("HOME"), ".R")
if (!file.exists(dotR)) dir.create(dotR)
M <- file.path(dotR, ifelse(.Platform$OS.type == "windows", "Makevars.win", "Makevars"))
if (!file.exists(M)) file.create(M)
cat("\nCXX11FLAGS=-O3 -march=native -mtune=native",
    if( grepl("^darwin", R.version$os)) "CXX11FLAGS += -arch x86_64 -ftemplate-depth-256" else 
    if (.Platform$OS.type == "windows") "CXX11FLAGS=-O3 -march=native -mtune=native" else
    "CXX11FLAGS += -fPIC",
    file = M, sep = "\n", append = TRUE)
```
However, be advised that setting the optimization level to O3 may cause problems for some packages. If you ever need to change anything with your C++ toolchain configuration, you can execute
```{r, eval=F}
M <- file.path(Sys.getenv("HOME"), ".R", ifelse(.Platform$OS.type == "windows",
                                                "Makevars.win", "Makevars"))
file.edit(M)
```

#### For Windows users
Please make sure [Rtools](https://cran.r-project.org/bin/windows/Rtools/) software is installed before installing the `jsem` package.

#### For MacOS users
If there is the "math.h not found" error during installation, please install the xcode command line tool first (run the following in the terminal)
```{r, eval=F}
xcode-select --install
```

and try to set the following variables in ~/.R/Makevars:
```{r, eval=F}
CFLAGS=-isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk
CCFLAGS=-isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk
CXXFLAGS=-isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk
```
