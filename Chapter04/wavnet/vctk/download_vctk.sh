#!/usr/bin/env bash
echo "Available diskspace in $(pwd):"
df -h .
read -p "This will download the VCTK corputs 11GB and extract it 14.9GB are you sure (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    wget http://homepages.inf.ed.ac.uk/jyamagis/release/VCTK-Corpus.tar.gz
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open VCTK.Corpus.tar.gz
    else
        tar -xvfz VCTK.Corpus.tar.gz
    fi
fi