@echo off
setlocal EnableDelayedExpansion

rem Don't remove the two jump lines after the next line [set NL=^]
set NL=^


rem "name" is named according to the testcase
set name=Solid_Case

rem "executables" are renamed and called from their directory
set dirbin=..\..\bin\windows
set dspartvtkdirbin=..\..\DSPartVTK\bin

set gencase="%dirbin%\GenCase_win64.exe"
set dualsphysicscpu="%dirbin%\SoliDualSPHysicsCPU_win64.exe"
set dualsphysicsgpu="%dirbin%\SoliDualSPHysics_win64.exe"
set boundaryvtk="%dirbin%\BoundaryVTK_win64.exe"
set partvtk="%dirbin%\PartVTK_win64.exe"
set dspartvtk="%dspartvtkdirbin%\DSPartVTK_win64.exe"
set partvtkout="%dirbin%\PartVTKOut_win64.exe"
set measuretool="%dirbin%\MeasureTool_win64.exe"
set computeforces="%dirbin%\ComputeForces_win64.exe"
set isosurface="%dirbin%\IsoSurface_win64.exe"
set flowtool="%dirbin%\FlowTool_win64.exe"
set floatinginfo="%dirbin%\FloatingInfo_win64.exe"
set tracerparts="%dirbin%\TracerParts_win64.exe"

rem ---------------------------------------------------------------------------
rem Choose execution mode FIRST
rem ---------------------------------------------------------------------------
:choosemode
echo.
echo Select execution mode:
echo   [1] CPU
echo   [2] GPU
set /p mode=Enter option (1/2): 

if "%mode%"=="1" (
    set runmode=CPU
    set solver=%dualsphysicscpu%
    set solverargs=
) else if "%mode%"=="2" (
    set runmode=GPU
    set solver=%dualsphysicsgpu%
    set solverargs=-gpu
) else (
    echo Invalid option. Please choose 1 or 2.
    goto choosemode
)

rem Output folders depend on chosen mode
set dirout=%runmode%_%name%_out
set diroutdata=%dirout%\bindata

:menu
if exist "%dirout%" (
    set /p option=The folder "!dirout!" already exists. Choose an option.!NL!  [1]- Delete it and continue.!NL!  [2]- Execute post-processing.!NL!  [3]- Abort and exit.!NL! 
    if "!option!"=="1" (
        goto run
    ) else if "!option!"=="2" (
        goto postprocessing
    ) else if "!option!"=="3" (
        goto fail
    ) else (
        goto menu
    )
)

:run
rem "dirout" to store results is removed if it already exists
if exist "%dirout%" rd /s /q "%dirout%"

rem CODES are executed according the selected parameters of execution in this testcase

%gencase% %name%_Def "%dirout%\%name%" -save:all
if not "%ERRORLEVEL%"=="0" goto fail

%solver% %solverargs% "%dirout%\%name%" "%dirout%" -dirdataout bindata -svres
if not "%ERRORLEVEL%"=="0" goto fail

:postprocessing
set dirout2=%dirout%\particles
if not exist "%dirout2%" mkdir "%dirout2%"

del /f /q "%dirout2%\DefStrucMapped*" 2>nul

set casexml=%dirout%\%name%.xml
%dspartvtk% -dirin "%diroutdata%" -filexml "%casexml%" ^
          -savevtk "%dirout2%\DefStrucBody_%%mk%%"
if not "%ERRORLEVEL%"=="0" goto fail

:success
echo All done
goto end

:fail
echo Execution aborted.

:end
pause
