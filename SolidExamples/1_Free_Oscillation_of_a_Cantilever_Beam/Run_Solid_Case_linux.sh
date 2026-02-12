#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Interactive CPU/GPU selection (no arguments), similar to Windows .bat
# ---------------------------------------------------------------------------

name="Solid_Case"

echo
echo "Select execution mode:"
echo "  [1] CPU"
echo "  [2] GPU"
read -r -p "Enter option (1/2): " mode

if [[ "$mode" == "1" ]]; then
  runmode="CPU"
elif [[ "$mode" == "2" ]]; then
  runmode="GPU"
else
  echo "Invalid option. Please choose 1 or 2."
  exit 1
fi

dirout="${runmode}_${name}_out"
diroutdata="${dirout}/bindata"
diroutvtk="${dirout}/particles"

# ---- binaries (relative to example directory) ----
dirbin="../../bin/linux"
dspartvtkdirbin="../../DSPartVTK/bin"

# Ensure the dynamic linker can find shared libs (e.g. libdsphchrono.so)
export LD_LIBRARY_PATH="$(cd "${dirbin}" && pwd)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

gencase="${dirbin}/GenCase_linux64"
solver_cpu="${dirbin}/SoliDualSPHysics_linux64"   # CPU mode invoked with -cpu
solver_gpu="${dirbin}/SoliDualSPHysics_linux64"   # GPU mode is default (no -cpu)
dspartvtk="${dspartvtkdirbin}/DSPartVTK_linux64"

# ---------------------------------------------------------------------------
# If output folder exists, prompt like the .bat script:
#   [1] delete and continue
#   [2] post-processing only
#   [3] abort
# ---------------------------------------------------------------------------
if [[ -d "${dirout}" ]]; then
  echo
  echo "The folder '${dirout}' already exists. Choose an option:"
  echo "  [1] Delete it and continue"
  echo "  [2] Execute post-processing"
  echo "  [3] Abort and exit"
  read -r -p "Enter option (1/2/3): " opt

  if [[ "$opt" == "1" ]]; then
    rm -rf "${dirout}"
  elif [[ "$opt" == "2" ]]; then
    goto_post="1"
  elif [[ "$opt" == "3" ]]; then
    echo "Execution aborted."
    exit 1
  else
    echo "Invalid option."
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# Run (GenCase + solver) unless we chose "post-processing only"
# ---------------------------------------------------------------------------
if [[ "${goto_post:-0}" != "1" ]]; then
  # Ensure vtk directory exists after creating output folder
  mkdir -p "${diroutvtk}"

  # Step 1: GenCase
  "${gencase}" "${name}_Def" "${dirout}/${name}" -save:all

  # Step 2: Run solver
  if [[ "${runmode}" == "CPU" ]]; then
    "${solver_cpu}" -cpu "${dirout}/${name}" "${dirout}" -dirdataout bindata -svres
  else
    "${solver_gpu}" "${dirout}/${name}" "${dirout}" -dirdataout bindata -svres
  fi
fi

# ---------------------------------------------------------------------------
# Post-processing (VTK)
# ---------------------------------------------------------------------------
mkdir -p "${diroutvtk}"
rm -f "${diroutvtk}/DefStrucMapped"* 2>/dev/null || true

casexml="${dirout}/${name}.xml"
"${dspartvtk}" -dirin "${diroutdata}" -filexml "${casexml}" \
  -savevtk "${diroutvtk}/DefStrucBody_%mk%"

echo "All done."
