#!/usr/bin/env bash
echo "Available diskspace in $(pwd):"
df -h .
read -p "This will download the Glove corputs 822 MB and extract it  MB are you sure (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    wget wget http://nlp.stanford.edu/data/glove.6B.zip
    unzip glove.6B.zip
fi
