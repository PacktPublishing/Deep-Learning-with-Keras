#!/usr/bin/env bash
echo "Available diskspace in $(pwd):"
df -h .
read -p "This will download the Text8 corputs 31.3 MB and extract it 100 MB are you sure (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    wget http://mattmahoney.net/dc/text8.zip
    unzip text8.zip
fi
