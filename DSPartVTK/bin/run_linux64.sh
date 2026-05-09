#!/bin/bash

# "dirout" is created to store results or it is cleaned if it already exists
dirout=out
if [ -e $dirout ]; then
  rm -f -r $dirout
fi
mkdir $dirout


# CODE is executed according the selected parameters of execution
errcode=0

if [ $errcode -eq 0 ]; then
  ./ToVTK_linux64 -dirin Piston2D/data -filexml Piston2D/Piston2D.xml -savevtk $dirout/parts -savecsv $dirout/data
  errcode=$?
fi



if [ $errcode -eq 0 ]; then
  echo All done
else
  echo Execution aborted
fi
read -n1 -r -p "Press any key to continue..." key
echo
