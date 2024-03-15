let
  pkgs = import <nixpkgs> {};
in pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (python-pkgs: [
      python-pkgs.networkx
      python-pkgs.pygraphviz
      python-pkgs.matplotlib
      python-pkgs.tqdm
    ]))
  ];
}
