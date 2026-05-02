fn main() {
    #[cfg(feature = "openblas")]
    {
        pkg_config::Config::new()
            .atleast_version("0.3")
            .probe("blas")
            .expect("openblas feature enabled but OpenBLAS was not found");
    }
}
