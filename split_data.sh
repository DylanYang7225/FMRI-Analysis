#!/bin/bash

subdir_size=41
subdir_name="stack"
n=$((`find . -maxdepth 1 -type f | wc -l`/$subdir_size+1))
for i in `seq 1 $n`;
do
    mkdir -p "$subdir_name$i";
    find . -maxdepth 1 -type f | head -n $subdir_size | xargs -i mv "{}" "$subdir_name$i"
done

##INSTRUCTIONS
# 1. update subdir_size, subdir_name with the desired size and name of the subdirectories
# 2. cd to the target directories where the files are to be split into subdirectories
# 3. execute 'bash path_to_this_script/split_data.sh'
##END INSTRUCTIONS
