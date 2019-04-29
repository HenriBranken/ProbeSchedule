#!/bin/bash

# In this shell script we:
# 1. Find the index number at which a series of images stop.
#    The series of images always starts at index 0.
#    The final integer index number is assigned to the shell variable idx.
# 2. Next we use the convert command to create a simple .gif file that loops
#    over the series of images.
################################################################################
# We create .gif files for both the:
# 1. progression_iter_{}.png, and
# 2. healthy_iter_{}.png
# series.
################################################################################

idx=$(< progression_idx.txt)
if [[ $idx -ne 0 ]]
then
    eval convert -delay 100 -loop 0 progression_iter_{0..$idx}.png progression.gif
else
    echo "There is not enough iterations to create progression.gif."
fi

idx=$(< healthy_idx.txt)
if [[ $idx -ne 0 ]]
then
    eval convert -delay 100 -loop 0 healthy_iter_{0..$idx}.png healthy.gif
else
    echo "There is not enough iterations to create healthy.gif."
fi

