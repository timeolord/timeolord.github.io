{
  description = "Hakyll blog - The Wei Zone";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        haskellPackages = pkgs.haskellPackages;

        site = haskellPackages.callCabal2nix "timeolord-github-io" ./. {};
      in
      {
        packages.default = site;

        devShells.default = pkgs.mkShell {
          buildInputs = [
            haskellPackages.ghc
            haskellPackages.cabal-install
            haskellPackages.hakyll
            haskellPackages.pandoc
            haskellPackages.haskell-language-server
            pkgs.zlib
          ];

          shellHook = ''
            echo "Hakyll development environment loaded"
            echo "Run './watch.sh' to start the dev server"
          '';
        };
      });
}
