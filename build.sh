cabal clean
cabal build
cabal exec site -- clean
cabal exec site -- build
cp CNAME docs/CNAME