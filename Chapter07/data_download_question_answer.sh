#!/usr/bin/env bash
echo "Available diskspace in $(pwd):"
df -h .
read -p "This will download the Text8 corputs 15 MB and extract it 149 MB are you sure (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz
    tar -xvzf tasks_1-20_v1-2.tar.gz
fi