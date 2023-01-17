.onLoad <- function(libname, pkgname) {
  setlvmcorr_threads(3)
}

.onAttach <- function(libname, pkgname){
  v = packageVersion("lvmcorr")
  packageStartupMessage("lvmcorr ", v, " using ", getlvmcorr_threads(), " threads (see ?getlvmcorr_threads())")
}