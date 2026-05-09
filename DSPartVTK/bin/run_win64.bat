@echo off
cls

@set dirout=out
if exist %dirout% del /Q %dirout%\*.*
if not exist %dirout% mkdir %dirout%

ToVTK_win64.exe -dirin Piston2D/data -filexml Piston2D/Piston2D.xml -savevtk %dirout%/parts -savecsv %dirout%/data


