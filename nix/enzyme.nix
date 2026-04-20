{
  pkgs,
  llvmPkgs,
}:
pkgs.clangStdenv.mkDerivation rec {
  name = "enzyme";
  version = "dev";
  nativeBuildInputs = [
    pkgs.cmake
    pkgs.ninja
    pkgs.python3
    pkgs.ccache
  ];
  buildInputs = [
    llvmPkgs.llvm
    llvmPkgs.libcxx
    llvmPkgs.libunwind
    llvmPkgs.lld
    llvmPkgs.clang-unwrapped
    llvmPkgs.clang
    llvmPkgs.mlir
    pkgs.libffi
    pkgs.libxml2
    pkgs.mpfr
    pkgs.zlib
    pkgs.zstd
    pkgs.ncurses
  ];
  src = pkgs.fetchFromGitHub {
    owner = "EnzymeAD";
    repo = "Enzyme";
    rev = "main";
    hash = "sha256-vKEtvt4Rqsrl005JdEHluZHAlr/XW0+ZsToaW+ThF/E=";
  };
  sourceRoot = "${src.name}/enzyme";
  cmakeFlags = [
    "-G Ninja"
    "-DLLVM_DIR=${llvmPkgs.llvm.dev}/lib/cmake/llvm"
    "-DCLANG_DIR=${llvmPkgs.clang-unwrapped.dev}/lib/cmake/clang"
    "-DLLVM_EXTERNAL_LIT=${pkgs.lit.dist}"
    "-DMLIR_DIR=${llvmPkgs.mlir.dev}/lib/cmake/mlir"
    "-DLLVM_ENABLE_PLUGINS=ON"
    "-DCMAKE_BUILD_TYPE=Release"
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=YES"
    "-DBUILD_SHARED_LIBS=ON"
  ];
}
