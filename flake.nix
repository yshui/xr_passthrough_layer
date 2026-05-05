{
  inputs = {
    rustup.url = "github:yshui/rustup.nix";
    rust-manifest = {
      flake = false;
      url = "https://static.rust-lang.org/dist/2026-05-05/channel-rust-nightly.toml";
    };
  };
  description = "xr_passthrough_layer";

  outputs =
    {
      self,
      nixpkgs,
      rustup,
      rust-manifest,
    }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        overlays = [ rustup.overlays.default ];
      };
      rust-toolchain = (pkgs.rustToolchainFromManifestFile rust-manifest).minimal.override {
        extensions = [
          "rustfmt"
          "rust-src"
          "clippy"
          "rustc"
          "cargo"
          "miri-preview"
        ];
      };
    in
    with pkgs;
    {
      devShells.${system}.default = mkShell {
        nativeBuildInputs = [
          pkg-config
          cmake
          rust-toolchain
          rust-analyzer
          cargo-bloat
          shaderc
        ];
        buildInputs = [
          systemdLibs
          linuxHeaders
          openvr
          xorg.libxcb
        ];
        shellHook = ''
          export LD_LIBRARY_PATH="${
            lib.makeLibraryPath [
              systemdLibs
              libglvnd
              vulkan-loader
              util-linux
              shaderc
              libuvc
              SDL2
              opencv
              openvr
              libx11
              libxcursor
              libxi
              libxkbcommon
              openxr-loader
              gcc.cc.lib
            ]
          }:$LD_LIBRARY_PATH"
        '';
        LIBCLANG_PATH = lib.makeLibraryPath [ llvmPackages_21.libclang.lib ];
        SHADERC_LIB_DIR = "${lib.getLib shaderc}/lib";

        BINDGEN_EXTRA_CLANG_ARGS =
          # Includes with normal include path
          (builtins.map (a: ''-I"${a}/include"'') [
            # add dev libraries here (e.g. pkgs.libvmi.dev)
            linuxHeaders
          ])
          ++ [
            ''-isystem "${pkgs.llvmPackages_latest.libclang.lib}/lib/clang/${lib.versions.major pkgs.llvmPackages_latest.libclang.version}/include"''
            ''-isystem "${stdenv.cc.cc}/include/c++/${lib.getVersion stdenv.cc.cc}"''
            ''-isystem "${stdenv.cc.cc}/include/c++/${lib.getVersion stdenv.cc.cc}/${stdenv.hostPlatform.config}"''
            ''-isystem "${glibc.dev}/include"''
          ];
      };
    };
}
