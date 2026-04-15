{
  description = "CSE Modeling Environment";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    naersk.url = "github:nix-community/naersk";
    naersk.inputs.nixpkgs.follows = "nixpkgs";
    fenix.url = "github:nix-community/fenix";
    fenix.inputs.nixpkgs.follows = "nixpkgs";
    treefmt-nix.url = "github:numtide/treefmt-nix";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs = inputs @ {
    self,
    nixpkgs,
    flake-utils,
    naersk,
    fenix,
    treefmt-nix,
    flake-parts,
    ...
  }:
    flake-parts.lib.mkFlake {inherit inputs;} {
      systems = flake-utils.lib.defaultSystems;
      imports = [];
      perSystem = {system, ...}: let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [fenix.overlays.default];
        };
        stdenv = pkgs.clangStdenv;

        rustToolchain = fenix.packages.${system}.latest.withComponents [
          "rustc"
          "rust-src"
          "cargo"
          "clippy"
          "rustfmt"
        ];

        coreBuild = with pkgs; [gcc cmake gnumake pkg-config clang clang-tools cppcheck clang-analyzer llvmPackages.libcxx llvmPackages.libunwind openssl openblas openmpi opencl-headers ocl-icd];
        cppPkgs = with pkgs; [boost eigen fmt];
        zigPkgs = with pkgs; [zig zls];
        futharkPkgs = with pkgs; [futhark];

        utilities = with pkgs; [
          valgrind
          lcov
          ccache
          bear
          doxygen
          perf
          hotspot
          gdb
          lldb
          cppcheck
          massif-visualizer
          hyperfine
          rust-analyzer-nightly
        ];

        pythonEnv = pkgs.python3.withPackages (ps:
          with ps; [
            numpy
            pandas
            matplotlib
            seaborn
            pyarrow
            torch
          ]);

        rEnv = pkgs.rWrapper.override {
          packages = with pkgs.rPackages; [
            tidyverse
            lme4
            lmerTest
            brms
            rmarkdown
          ];
        };

        cmdStan = stdenv.mkDerivation rec {
          pname = "cmdStan";
          version = "2.38.0";
          src = pkgs.fetchFromGitHub {
            owner = "stan-dev";
            repo = "cmdstan";
            tag = "v${version}";
            fetchSubmodules = true;
            hash = "sha256-4Mx4LvXW2lYOSSOgNT0f+unry6mBobgGTDLwtiypHBU=";
          };
          postPatch = ''
            substituteInPlace stan/lib/stan_math/make/libraries \
              --replace "/usr/bin/env/bash" "bash"
          '';

          nativeBuildInputs = with pkgs; [
            python3
            stanc
            gcc
            gnumake
            pkg-config
            openssl
          ];

          preConfigure =
            ''
              patchShebangs test-all.sh runCmdStanTests.py stan/
            ''
            + ''
              mkdir -p $out/opt
              cp -R . $out/opt/cmdstan
              cd $out/opt/cmdstan
              mkdir -p bin
              ln -s ${pkgs.stanc}/bin/stanc bin/stanc
            '';

          makeFlags =
            [
              "build"
            ]
            ++ pkgs.lib.optionals stdenv.hostPlatform.isDarwin [
              "arch=${stdenv.hostPlatform.darwinArch}"
            ];
          env.CXXFLAGS = pkgs.lib.optionalString stdenv.cc.isClang "-Xclang -fno-pch-timestamp";
          enableParallelBuilding = true;
          installPhase = ''
            runHook preInstall

            mkdir -p $out/bin
            ln -s $out/opt/cmdstan/bin/stanc $out/bin/stanc
            ln -s $out/opt/cmdstan/bin/stansummary $out/bin/stansummary
            cat > $out/bin/stan <<EOF
            #! ${pkgs.runtimeShell}
            make -C $out/opt/cmdstan "\$(realpath "\$1")"
            EOF
            chmod a+x $out/bin/stan

            runHook postInstall
          '';

          passthru.tests = {
            test = stdenv.runCommandCC "cmdstan-test" {inherit (self) cmdStan;} ''
              cp -R ${cmdStan}/opt/cmdstan cmdstan
              chmod -R +w cmdstan
              cd cmdstan
              ./runCmdStanTests.py -j$NIX_BUILD_CORES src/test/interface
              touch $out
            '';
          };

          meta = {
            description = "Command-line interface to Stan";
            longDescription = ''
              Stan is a probabilistic programming language implementing full Bayesian
              statistical inference with MCMC sampling (NUTS, HMC), approximate Bayesian
              inference with Variational inference (ADVI) and penalized maximum
              likelihood estimation with Optimization (L-BFGS).
            '';
            homepage = "https://mc-stan.org/interfaces/cmdstan.html";
            license = pkgs.lib.licenses.bsd3;
          };
        };

        naersk-lib = pkgs.callPackage naersk {
          rustc = rustToolchain;
          cargo = rustToolchain;
        };

        treefmtEval = treefmt-nix.lib.evalModule pkgs ./treefmt.nix;

        rustPackage = naersk-lib.buildPackage {
          src = ./crates;
          cargoBuildOptions = opts:
            opts ++ ["--features" "ffi-backend"];
          nativeBuildInputs = [
            pkgs.pkg-config
          ];
          buildInputs = [
            pkgs.openblas
          ];
          cargoEnv = {
            "CXXFLAGS" = "-I${pkgs.eigen}/include/eigen3";
            "OPENBLAS_NUM_THREADS" = 1;
          };
        };

        matmod = stdenv.mkDerivation {
          pname = "matmod";
          version = "0.1";
          dontUnpack = true;

          nativeBuildInputs = [pkgs.makeWrapper];

          installPhase = ''
            mkdir -p $out/bin

            makeWrapper ${rustPackage}/bin/app $out/bin/matmod \
              --set-default MATMOD_BACKEND dynamic
          '';
        };
      in {
        packages = {
          rustPackage = rustPackage;
          default = matmod;
        };
        apps.default = flake-utils.lib.mkApp {drv = self.packages.${system}.default;};

        formatter = treefmtEval.config.build.wrapper;

        checks = {
          cmdStan-test = cmdStan.passthru.tests.test;
          formatting = treefmtEval.config.build.check self;
        };

        devShells.default = pkgs.mkShell rec {
          hardeningDisable = ["all"];
          packages =
            [pythonEnv rEnv rustToolchain cmdStan]
            ++ coreBuild
            ++ zigPkgs
            ++ cppPkgs
            ++ futharkPkgs
            ++ utilities;

          shellHook = ''
            echo "---------------------------------------"
            echo "you are in the flake's default devshell"
            echo "---------------------------------------"

            export DCMAKE_EXPORT_COMPILE_COMMANDS=1

            exec zsh
          '';

          LD_LIBRARY_PATH =
            pkgs.lib.makeLibraryPath
            packages;
          LLDB_DEBUGSERVER_PATH = "${pkgs.lldb}/bin/lldb-server";
          CPLUS_INCLUDE_PATH = "crates/target/cxxbridge/";
          RUSTFLAGS = "-L${pkgs.openblas}/lib -lopenblas -C target-cpu=native -C target-feature=+avx2,+fma -C link-arg=-Wl,-rpath,${pkgs.gcc.cc.lib}/lib";
        };
      };
    };
}
