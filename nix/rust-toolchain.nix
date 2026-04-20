{
  pkgs,
  llvmPkgs,
  version,
  bootstrap,
}:
pkgs.clangStdenv.mkDerivation {
  pname = "rust-base";
  inherit version;

  src = pkgs.fetchurl {
    url = "https://static.rust-lang.org/dist/rustc-${version}-src.tar.xz";
    sha256 = "sha256-DXxdooM54tbSxtDzL83Lng3uwo1Ww62rCqCCWo/T95c=";
  };

  dontUpdateAutotoolsGnuConfigScripts = true;
  stripDebugList = ["bin"];

  nativeBuildInputs = with pkgs; [
    libffi
    cmake
    ninja
    rustc
    cargo
    perl
    curl
    libiconv
    python3
    file
    which
    xz
    pkg-config
    git
    sccache
  ];

  buildInputs = [
    llvmPkgs.llvm
    llvmPkgs.clang
    llvmPkgs.libcxx
    llvmPkgs.libunwind
    llvmPkgs.lld
    pkgs.zlib
    pkgs.openssl
    pkgs.xz
    bootstrap
  ];

  env = {
    LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
      llvmPkgs.llvm
      llvmPkgs.lld
      llvmPkgs.libcxx
      pkgs.xz
      pkgs.zlib
      pkgs.openssl
    ];
  };

  postPatch = ''
    patchShebangs src/etc
  '';

  configurePhase = ''
    mkdir -p $out

      cat > bootstrap.toml <<EOF
      change-id = 154587

    [build]
    build = "${pkgs.stdenv.buildPlatform.config}"
    host = ["${pkgs.stdenv.buildPlatform.config}"]
    target = ["${pkgs.stdenv.buildPlatform.config}"]
    build-dir = "build"
    rustc = "${bootstrap}/bin/rustc"
    cargo = "${bootstrap}/bin/cargo"
    rustfmt = "${pkgs.rustfmt}/bin/rustfmt"
    docs = false
    extended = true
    tools = ["cargo", "clippy", "miri", "rust-analyzer-proc-macro-srv", "rustdoc", "rustfmt"]

    [rust]
    channel = "nightly"
    download-rustc = false
    llvm-libunwind = "system"
    rpath = true
    lld = false

    [llvm]
    download-ci-llvm = false
    link-shared = true
    use-libcxx = true
    assertions = true
    ninja = true

    [target.${pkgs.stdenv.hostPlatform.config}]
    cc = "${llvmPkgs.clang}/bin/clang"
    cxx = "${llvmPkgs.clang}/bin/clang++"
    linker = "${llvmPkgs.clang}/bin/clang"
    llvm-config = "${llvmPkgs.llvm.dev}/bin/llvm-config"

    [install]
    prefix = "$out"
    sysconfdir = "$out/etc"
    EOF
  '';

  buildPhase = ''
    python3 x.py build --stage 1 \
      library/std \
      src/tools/cargo \
      src/tools/clippy \
      src/tools/rustfmt \
      src/tools/miri
  '';

  doCheck = false;
  checkPhase = ''
    python3 x.py test --stage 1 tests/codegen-llvm/autodiff
    python3 x.py test --stage 1 tests/pretty/autodiff
    python3 x.py test --stage 1 tests/ui/autodiff
    python3 x.py test --stage 1 tests/run-make/autodiff
    python3 x.py test --stage 1 tests/ui/feature-gates/feature-gate-autodiff.rs
  '';

  installPhase = ''
    python3 x.py install --stage 1
  '';

  postInstall = ''
    rm $out/lib/rustlib/install.log
    for m in $out/lib/rustlib/manifest-rust*
    do
      sort --output=$m < $m
    done

    # remove uninstall script that doesn't really make sense for Nix.
    rm $out/lib/rustlib/uninstall.sh
  '';

  passthru = {
    tests.autodiff = pkgs.runCommand "rust-autodiff-tests" {} ''
      export PATH=${placeholder "out"}/bin:$PATH
      cd $src

      python3 x.py test --stage 1 tests/codegen-llvm/autodiff
      touch $out
    '';
    isRustToolchain = true;
  };
}
