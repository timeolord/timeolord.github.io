rm -r ./docs
rm -r ./_site
site rebuild
cp -r ./_site ./docs
