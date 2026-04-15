{pkgs, ...}: {
  projectRootFile = "flake.nix";

  programs = {
    alejandra.enable = true;
    rustfmt.enable = true;
    clang-format.enable = true;
    zig.enable = true;
  };

  settings.formatter = {
    futhark = {
      command = "${pkgs.futhark}/bin/futhark";
      options = ["fmt"];
      includes = ["*.fut"];
    };
  };
}
