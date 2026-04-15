use cxx_build::CFG;
fn main() {
    CFG.include_prefix = "";

    cxx_build::bridge("src/lib.rs")
        .include("cpp/include")
        .file("cpp/src/gaussian.cpp")
        .flag_if_supported("-std=c++23")
        .flag("-Wno-system-headers")
        .flag_if_supported("-Wno-unused-parameter")
        .flag("-O3")
        .flag("-march=native")
        .compile("matmod_cpp");

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=cpp/src/gaussian.cpp");
    println!("cargo:rerun-if-changed=cpp/include/gaussian.hpp");
}
