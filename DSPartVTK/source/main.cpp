/*
 <TOVTK4_DEFSTRUCT>  Copyright (c) 2022 by Dr Jose M. Dominguez et al. (see http://dual.sphysics.org/index.php/developers/).

 EPHYSLAB Environmental Physics Laboratory, Universidade de Vigo, Ourense, Spain.
 School of Mechanical, Aerospace and Civil Engineering, University of Manchester, Manchester, U.K.

 This file is part of DualSPHysics.

 DualSPHysics is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License
 as published by the Free Software Foundation; either version 2.1 of the License, or (at your option) any later version.

 DualSPHysics is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

 You should have received a copy of the GNU Lesser General Public License along with DualSPHysics. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Functions.h"
#include "JException.h"
#include "JCfgRun.h"
#include "TypesDef.h"
#include "JPartDataBi4.h"
#include "JVtkLib.h"
#include "JDataArrays.h"
#include "JOutputCsv.h"
#include "JRangeFilter.h"
#include "FunctionsMath.h"

#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <iomanip>
#include <map>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <atomic>
#include <memory>
#ifdef _OPENMP
#include <omp.h>
#endif
//using namespace std;
using std::string;
using std::exception;

const char* APP_NAME = "ToVtk_DefStruct v5.2.029 (09-10-2022)";

//==============================================================================
// Invoca una excepcion referente a gestion de ficheros
//==============================================================================
void ExceptionFile(string msg, string file) {
    msg = msg + "\nFile:[" + file + "]\n";
    throw msg;
}
namespace {

    //==============================================================================
    // Converts data arrays into VTK-friendly buffers while reducing precision to
    // float where possible to minimize file size and memory use.
    //==============================================================================
    void BuildVtkCompatibleArrays(const JDataArrays& source, JDataArrays& target,
        std::vector<std::vector<float>>& scalarBuffers, std::vector<std::vector<tfloat3>>& vec3Buffers)
    {
        target.Reset();
        scalarBuffers.clear();
        vec3Buffers.clear();

        const unsigned narrays = source.Count();
        for (unsigned idx = 0; idx < narrays; ++idx) {
            const JDataArrays::StDataArray& ar = source.GetArrayCte(idx);
            if (ar.type == TypeSyMatrix3f) {
                const std::string fmt = source.GetArrayFmt(idx);
                const std::string units = source.GetArrayUnits(idx);
                const tsymatrix3f* data = static_cast<const tsymatrix3f*>(ar.ptr);
                static const char* suffix[6] = { "xx", "xy", "xz", "yy", "yz", "zz" };

                for (int comp = 0; comp < 6; ++comp) {
                    scalarBuffers.emplace_back(ar.count);
                    std::vector<float>& buffer = scalarBuffers.back();
                    for (unsigned c = 0; c < ar.count; ++c) {
                        const tsymatrix3f& m = data[c];
                        float value = 0.f;
                        switch (comp) {
                        case 0: value = m.xx; break;
                        case 1: value = m.xy; break;
                        case 2: value = m.xz; break;
                        case 3: value = m.yy; break;
                        case 4: value = m.yz; break;
                        case 5: value = m.zz; break;
                        }
                        buffer[c] = value;
                    }

                    std::string fullname = ar.keyname + "." + suffix[comp];
                    if (!fmt.empty()) fullname += ":" + fmt;
                    if (!units.empty()) fullname += ":" + units;
                    unsigned newIdx = target.AddArray(fullname, ar.count, buffer.data(), false);
                    target.GetArray(newIdx).tag = ar.tag;
                }
                continue;
            }
            if (ar.type == TypeDouble) {
                scalarBuffers.emplace_back(ar.count);
                std::vector<float>& buffer = scalarBuffers.back();
                const double* data = static_cast<const double*>(ar.ptr);
                for (unsigned c = 0; c < ar.count; ++c) buffer[c] = float(data[c]);
                unsigned newIdx = target.AddArray(ar.fullname, ar.count, buffer.data(), false);
                target.GetArray(newIdx).tag = ar.tag;
                continue;
            }

            if (ar.type == TypeDouble3) {
                vec3Buffers.emplace_back(ar.count);
                std::vector<tfloat3>& buffer = vec3Buffers.back();
                const tdouble3* data = static_cast<const tdouble3*>(ar.ptr);
                for (unsigned c = 0; c < ar.count; ++c) buffer[c] = TFloat3(float(data[c].x), float(data[c].y), float(data[c].z));
                unsigned newIdx = target.AddArray(ar.fullname, ar.count, buffer.data(), false);
                target.GetArray(newIdx).tag = ar.tag;
                continue;
            }
            unsigned newIdx = target.AddArray(ar.fullname, ar.type, ar.count, ar.ptr, false);
            target.GetArray(newIdx).tag = ar.tag;
        }
    }

    void SaveVtkDataCompatible(const std::string& fname, const JDataArrays& arrays,
        const std::string& posfield)
    {
        std::vector<std::vector<float>> floatBuffers;
        std::vector<std::vector<tfloat3>> vec3Buffers;
        JDataArrays vtkArrays;
        BuildVtkCompatibleArrays(arrays, vtkArrays, floatBuffers, vec3Buffers);
        JVtkLib::SaveVtkData(fname, vtkArrays, posfield);
    }

} // namespace
struct DsBodyInfo {
    unsigned mkbound = 0;
    unsigned npbody = 0;
    unsigned npstart = 0;
    bool fracture = false;
    double pflim = 0;
    double lambda = 0;
    double bulk = 0;
    double mu = 0;
    double vol0 = 0;
    double rho0 = 0;
    double gc = 0;
    double lc = 0;
    int mapfact = 1;
    unsigned nmeasplane = 0;
    double dp = 0;
    double czero = 0;
    TpConstitModel constitutive = CONSTITMODEL_SVK;
};

//==============================================================================
// Invoca una excepcion generica
//==============================================================================
void ExceptionText(string msg) {
    throw msg;
}
static std::string EnsureTrailingSlash(const std::string& dir) {
    if (dir.empty()) return dir;
    const char back = dir.back();
    if (back == '/' || back == '\\') return dir;
    return dir + "/";
}

static std::string ExtractDirectory(const std::string& path) {
    if (path.empty()) return path;
    const size_t pos = path.find_last_of("/\\");
    if (pos == string::npos) return string();
    return path.substr(0, pos + 1);
}

static std::string ExtractUpperDirectory(const std::string& path) {
    if (path.empty()) return path;
    const size_t pos = path.find_last_of("/\\");
    if (pos == string::npos) return string();
    std::string upperpath = path.substr(0, pos);
    const size_t pos2 = upperpath.find_last_of("/\\");
    return upperpath.substr(0, pos2 + 1);
}

static std::string EnsureCsvExtension(const std::string& path) {
    if (path.empty()) return path;
    std::string result = path;
    const std::string ext = fun::GetExtension(result);
    if (ext.empty()) return fun::AddExtension(result, "csv");
    if (fun::StrLower(ext) == "csv") return result;
    result.erase(result.size() - ext.size() - 1);
    return fun::AddExtension(result, "csv");
}
static void WriteEnergiesCsv(const std::string& file, bool csvComma, double timestep,
    const std::vector<DsBodyInfo>& bodies, const std::vector<tdouble3>& energies) {

    (void)csvComma;
    if (bodies.empty() || bodies.size() != energies.size() || file.empty()) return;

    const std::string outFile = EnsureCsvExtension(file);
    fun::MkdirPath(ExtractDirectory(outFile));

    struct EnergyTable {
        std::vector<std::string> header;
        std::vector<bool> fracture;
        std::map<double, std::vector<std::vector<std::string>>> rows;
    };

    static std::unordered_map<std::string, EnergyTable> sTables;
    static std::mutex tableMutex;
    std::lock_guard<std::mutex> lock(tableMutex);

    EnergyTable& table = sTables[outFile];

    if (table.header.empty()) {
        table.header.push_back("Time");
        table.fracture.resize(bodies.size());
        for (size_t i = 0; i < bodies.size(); ++i) {
            const std::string prefix = "mkbound_" + std::to_string(bodies[i].mkbound);
            table.header.push_back(prefix + "_STRAIN_ENERGY");
            table.header.push_back(prefix + "_KINETIC_ENERGY");
            if (bodies[i].fracture) {
                table.header.push_back(prefix + "_FRACTURE_ENERGY");
            }
            table.fracture[i] = bodies[i].fracture;
        }
    }
    std::vector<std::string> row;
    row.reserve(table.header.size());
    row.push_back(fun::PrintStr("%.15g", timestep));
    for (size_t i = 0; i < bodies.size(); ++i) {
        row.push_back(fun::PrintStr("%.15g", energies[i].x));
        row.push_back(fun::PrintStr("%.15g", energies[i].y));
        if (table.fracture[i]) row.push_back(fun::PrintStr("%.15g", energies[i].z));
    }
    table.rows[timestep].push_back(std::move(row));

    size_t totalRows = 0;
    for (const auto& kv : table.rows) totalRows += kv.second.size();
    std::vector<std::vector<std::string>> orderedRows;
    orderedRows.reserve(totalRows);
    for (const auto& kv : table.rows) {
        for (const auto& values : kv.second) orderedRows.push_back(values);
    }
    JOutputCsv::SaveCsvTable(outFile, table.header, orderedRows);
}

static void WritePlaneCsv(const std::string& file, bool csvComma, bool simulate2d, double timestep,
    const tdouble3& avgDisp, const tdouble3& avgForce) {

    (void)csvComma;
    if (file.empty()) return;

    const std::string outFile = EnsureCsvExtension(file);
    fun::MkdirPath(ExtractDirectory(outFile));

    struct PlaneTable {
        bool initialized = false;
        bool simulate2d = false;
        std::vector<std::string> header;
        std::map<double, std::vector<std::vector<std::string>>> rows;
    };

    static std::unordered_map<std::string, PlaneTable> sTables;
    static std::mutex tableMutex;
    std::lock_guard<std::mutex> lock(tableMutex);

    PlaneTable& table = sTables[outFile];
    if (!table.initialized) {
        table.simulate2d = simulate2d;
        table.header.push_back("Time");
        table.header.push_back("Av.Disp.x");
        if (!simulate2d) table.header.push_back("Av.Disp.y");
        table.header.push_back("Av.Disp.z");
        table.header.push_back("Av.Force.x");
        if (!simulate2d) table.header.push_back("Av.Force.y");
        table.header.push_back("Av.Force.z");
        table.initialized = true;
    }
    std::vector<std::string> row;
    row.reserve(table.header.size());
    row.push_back(fun::PrintStr("%.15g", timestep));
    row.push_back(fun::PrintStr("%.15g", avgDisp.x));
    if (!table.simulate2d) row.push_back(fun::PrintStr("%.15g", avgDisp.y));
    row.push_back(fun::PrintStr("%.15g", avgDisp.z));
    row.push_back(fun::PrintStr("%.15g", avgForce.x));
    if (!table.simulate2d) row.push_back(fun::PrintStr("%.15g", avgForce.y));
    row.push_back(fun::PrintStr("%.15g", avgForce.z));

    table.rows[timestep].push_back(std::move(row));

    size_t totalRows = 0;
    for (const auto& kv : table.rows) totalRows += kv.second.size();
    std::vector<std::vector<std::string>> orderedRows;
    orderedRows.reserve(totalRows);
    for (const auto& kv : table.rows) {
        for (const auto& values : kv.second) orderedRows.push_back(values);
    }
    JOutputCsv::SaveCsvTable(outFile, table.header, orderedRows);
}

static void ProcessDeformableStructures(const JCfgRun* cfg, const JPartDataBi4& pd, double timestep, unsigned part,
    const std::string& baseOutDir, bool csvComma) {

    JBinaryData* partdata = pd.GetPart();
    if (!partdata || !partdata->GetvBool("HasDeformStruc", true, false)) return;
    const unsigned bodyCount = partdata->GetvUint("DSBodyCount", true, 0);
    unsigned particleCount = partdata->GetvUint("DSParticleCount", true, 0);
    if (!bodyCount || !particleCount) return;
    const bool floatArrays = partdata->GetvBool("DSFloatArrays", true, false);
    const bool simulate2D = partdata->GetvBool("DSSimulate2D", true, false);
    const unsigned totalPlaneCount = partdata->GetvUint("DSNMeasPlanes", true, 0);
    const unsigned totalPlanePartCount = partdata->GetvUint("DSNPartMeasPlanes", true, 0);

    JBinaryData* dsitem = partdata->GetItem("DefStruc");
    if (!dsitem) return;

    // Store timestep for later use in VTK output
    const float timestepFloat = float(timestep);

    const std::string defaultDir = EnsureTrailingSlash(baseOutDir);
    const std::string dsVtkTemplate = (cfg ? cfg->DsSaveVtk : std::string());
    const std::string dsCsvTemplate = (cfg ? cfg->DsSaveCsv : std::string());
    const std::string fallbackVtkTemplate = defaultDir + "DeformStruc/mkbound_%mk%";

    enum {
        FieldDisp = 1u << 0, FieldFluidForce = 1u << 1, FieldCauchy = 1u << 2,
        FieldPhasefield = 1u << 3, FieldPlastic = 1u << 4, FieldEnergy = 1u << 5
    };
    const unsigned defaultMask = FieldDisp | FieldCauchy | FieldFluidForce | FieldPhasefield | FieldPlastic | FieldEnergy;

    auto parseFieldMask = [&](const std::string& text) -> unsigned {
        unsigned mask = 0;
        std::stringstream ss(text);
        std::string token;
        while (std::getline(ss, token, ',')) {
            token = fun::StrTrim(token);
            if (token.empty()) continue;
            std::string upper = fun::StrUpper(token);
            if (upper == "DISP" || upper == "DISPLACEMENT") mask |= FieldDisp;
            else if (upper == "CAUCHY" || upper == "CAUCHYSTRESS" || upper == "STRESS") mask |= FieldCauchy;
            else if (upper == "CAUCHYXYZ" || upper == "CAUCHYDIAG" || upper == "CAUCHYSHEAR" || upper == "CAUCHYXY_XZ_YZ") mask |= FieldCauchy;
            else if (upper == "FLUIDFORCE" || upper == "FORCE") mask |= FieldFluidForce;
            else if (upper == "PHASEFIELD" || upper == "PHASE") mask |= FieldPhasefield;
            else if (upper == "ENERGY" || upper == "ENERGIES") mask |= FieldEnergy;
            else if (upper == "PLASTICE" || upper == "PLASTIC" || upper == "PLASTICSTRAIN") mask |= FieldPlastic;
            else if (upper == "ALL") mask |= defaultMask;
        }
        return mask;
        };

    std::map<unsigned, unsigned> perBodyMask;
    unsigned wildcardMask = defaultMask;
    if (cfg) {
        for (const std::string& specRaw : cfg->DsFieldSpecs) {
            std::string spec = fun::StrTrim(specRaw);
            if (spec.empty()) continue;
            const size_t eq = spec.find('=');
            if (eq == std::string::npos) continue;
            std::string lhs = fun::StrTrim(spec.substr(0, eq));
            std::string rhs = spec.substr(eq + 1);
            unsigned mask = parseFieldMask(rhs);
            if (lhs == "*" || fun::StrUpper(lhs) == "ALL") {
                wildcardMask = (mask ? mask : 0u);
            }
            else {
                unsigned mk = (unsigned)atoi(lhs.c_str());
                perBodyMask[mk] = mask;
            }
        }
    }

    JRangeFilter bodyFilter;
    const bool filterBodies = (cfg && !cfg->DsOnlyMk.empty());
    if (filterBodies) bodyFilter.Config(cfg->DsOnlyMk);

    auto getMaskForBody = [&](unsigned mkbound, const DsBodyInfo& body) -> unsigned {
        unsigned mask = 0;
        if (!perBodyMask.empty()) {
            auto it = perBodyMask.find(mkbound);
            if (it != perBodyMask.end()) mask = it->second;
            else mask = wildcardMask;
        }
        else {
            mask = wildcardMask;
        }

        // Smart defaults: auto-add fields based on body properties
        if (body.fracture) mask |= FieldPhasefield;
        if (body.constitutive == CONSTITMODEL_J2) mask |= FieldPlastic;

        // ContactForce only makes sense for multi-body simulations
        if (bodyCount == 1) mask &= ~FieldFluidForce;

        return mask;
        };

    auto formatPattern = [](std::string pattern, unsigned mk)->std::string {
        bool replaced = false;
        size_t pos = 0;
        while ((pos = pattern.find("%mk%", pos)) != std::string::npos) {
            pattern.replace(pos, 4, std::to_string(mk));
            replaced = true;
        }
        pos = 0;
        while ((pos = pattern.find("%MK%", pos)) != std::string::npos) {
            pattern.replace(pos, 4, std::to_string(mk));
            replaced = true;
        }
        if (!replaced) pattern += "_" + std::to_string(mk);
        return pattern;
        };

    auto buildOutputFile = [&](const std::string& pattern, const std::string& fallback, unsigned mk, const char* ext) -> std::string {
        std::string basePattern = pattern.empty() ? fallback : pattern;
        if (basePattern.empty()) return std::string();
        std::string resolved = formatPattern(basePattern, mk);
        std::string withIndex = fun::FileNameSec(resolved, part);
        if (ext && *ext) {
            const std::string currentExt = fun::GetExtension(withIndex);
            const std::string desiredExt(ext);
            if (currentExt.empty()) withIndex = fun::AddExtension(withIndex, desiredExt);
            else if (fun::StrLower(currentExt) != fun::StrLower(desiredExt)) {
                withIndex.erase(withIndex.size() - currentExt.size() - 1);
                withIndex = fun::AddExtension(withIndex, desiredExt);
            }
        }
        return withIndex;
        };

    auto hasArray = [&](const std::string& name) -> bool { return(dsitem->GetArrayIndex(name) >= 0); };
    auto getElementCount = [](JBinaryDataArray* arr) -> unsigned {
        if (!arr) return 0u;
        unsigned count = arr->GetCount();
        if (!count) count = arr->GetFileDataCount();
        return count;
        };
    auto getArray = [&](const std::string& name, JBinaryDataDef::TpData type) -> JBinaryDataArray* {
        if (!hasArray(name)) return nullptr;
        JBinaryDataArray* arr = dsitem->GetArray(name);
        return (arr && arr->GetType() == type) ? arr : nullptr;
        };
    auto copyUintArray = [&](const std::string& name, std::vector<unsigned>& dest)->bool {
        JBinaryDataArray* arr = getArray(name, JBinaryDataDef::DatUint);
        if (!arr) return false;
        const unsigned count = getElementCount(arr);
        dest.resize(count);
        if (count) arr->GetDataCopy(count, dest.data()); return true;
        };

    std::vector<unsigned> mkbounds, bodynp, bodystart, bodymeas;
    std::vector<byte> bodyfracture;
    std::vector<float> bodypflim, bodylambda, bodybulk, bodymu, bodyvol0, bodyrho0, bodygc, bodylc, bodydp, bodyczero;
    std::vector<int> bodymapfact, bodyconstit;
    if (!copyUintArray("MkBound", mkbounds) || mkbounds.size() != bodyCount) return;
    if (!copyUintArray("NpBody", bodynp) || bodynp.size() != bodyCount) return;
    if (!copyUintArray("NpStart", bodystart) || bodystart.size() != bodyCount) return;
    if (hasArray("MeasurePlanes")) copyUintArray("MeasurePlanes", bodymeas);
    if (hasArray("FractureBody")) {
        if (JBinaryDataArray* arr = getArray("FractureBody", JBinaryDataDef::DatUchar)) {
            const unsigned count = getElementCount(arr);
            bodyfracture.resize(count);
            if (count) arr->GetDataCopy(count, bodyfracture.data());

        }
    }
    auto copyFloatArray = [&](const std::string& name, std::vector<float>& dest)->bool {
        JBinaryDataArray* arr = getArray(name, JBinaryDataDef::DatFloat);
        if (!arr) return false;
        const unsigned count = getElementCount(arr);
        dest.resize(count);
        if (count) arr->GetDataCopy(count, dest.data());
        return true;
        };
    auto copyIntArray = [&](const std::string& name, std::vector<int>& dest)->bool {
        JBinaryDataArray* arr = getArray(name, JBinaryDataDef::DatInt);
        if (!arr) return false;
        const unsigned count = getElementCount(arr);
        dest.resize(count);
        if (count) arr->GetDataCopy(count, dest.data());
        return true;
        };
    copyFloatArray("PfLimit", bodypflim);
    copyFloatArray("Lambda", bodylambda);
    copyFloatArray("Bulk", bodybulk);
    copyFloatArray("Mu", bodymu);
    copyFloatArray("Vol0", bodyvol0);
    copyFloatArray("Rho0", bodyrho0);
    copyFloatArray("Gc", bodygc);
    copyFloatArray("Lc", bodylc);
    copyFloatArray("BodyDp", bodydp);
    copyFloatArray("Czero", bodyczero);
    copyIntArray("MapFactor", bodymapfact);
    copyIntArray("ConstitutiveModel", bodyconstit);

    std::vector<DsBodyInfo> bodies(bodyCount);
    for (unsigned i = 0; i < bodyCount; ++i) {
        bodies[i].mkbound = mkbounds[i];
        bodies[i].npbody = bodynp[i];
        bodies[i].npstart = bodystart[i];
        bodies[i].fracture = (!bodyfracture.empty() ? bodyfracture[i] != 0 : false);
        bodies[i].pflim = (!bodypflim.empty() ? bodypflim[i] : 0.0f);
        bodies[i].lambda = (!bodylambda.empty() ? bodylambda[i] : 0.0f);
        bodies[i].bulk = (!bodybulk.empty() ? bodybulk[i] : 0.0f);
        bodies[i].mu = (!bodymu.empty() ? bodymu[i] : 0.0f);
        bodies[i].vol0 = (!bodyvol0.empty() ? bodyvol0[i] : 0.0f);
        bodies[i].rho0 = (!bodyrho0.empty() ? bodyrho0[i] : 0.0f);
        bodies[i].gc = (!bodygc.empty() ? bodygc[i] : 0.0f);
        bodies[i].lc = (!bodylc.empty() ? bodylc[i] : 0.0f);
        bodies[i].mapfact = (!bodymapfact.empty() ? bodymapfact[i] : 1);
        bodies[i].nmeasplane = (!bodymeas.empty() && i < bodymeas.size() ? bodymeas[i] : 0u);
        bodies[i].dp = (!bodydp.empty() ? bodydp[i] : 0.0f);
        bodies[i].czero = (!bodyczero.empty() ? bodyczero[i] : 0.0f);
        bodies[i].constitutive = (!bodyconstit.empty() ? TpConstitModel(bodyconstit[i]) : CONSTITMODEL_SVK);
        if (bodies[i].mapfact <= 0) bodies[i].mapfact = 1;
    }

    std::vector<unsigned> dsibodyridp;
    if (!copyUintArray("DSiBodyRidp", dsibodyridp) || dsibodyridp.empty()) return;
    if (particleCount && particleCount != dsibodyridp.size()) {
        printf("Warning: DSParticleCount metadata (%u) differs from deformable structure arrays (%zu). Using %zu entries.\n",
            particleCount, dsibodyridp.size(), dsibodyridp.size());
    }
    particleCount = (unsigned)dsibodyridp.size(); std::vector<unsigned> measPlaneCounts, measPlanePart;
    if (totalPlaneCount && hasArray("DSMeasPlnCnt")) copyUintArray("DSMeasPlnCnt", measPlaneCounts);
    if (totalPlanePartCount && hasArray("DSMeasPlnPart")) copyUintArray("DSMeasPlnPart", measPlanePart);

    std::string energyDir = ExtractUpperDirectory(defaultDir);
    if (cfg) {
        std::string templateForDir;
        if (!dsCsvTemplate.empty()) templateForDir = dsCsvTemplate;
        else if (!dsVtkTemplate.empty()) templateForDir = dsVtkTemplate;
        if (!templateForDir.empty() && !bodies.empty()) {
            std::string resolved = formatPattern(templateForDir, bodies.front().mkbound);
            energyDir = EnsureTrailingSlash(ExtractUpperDirectory(resolved));
        }
    }
    if (energyDir.empty()) energyDir = ExtractUpperDirectory(defaultDir);
    energyDir = EnsureTrailingSlash(energyDir);

    const std::string energyFile = energyDir + "DeformStruc_Energies.csv";

    if (floatArrays) {
        JBinaryDataArray* arrPos0w = getArray("DSPos0w", JBinaryDataDef::DatFloat);
        JBinaryDataArray* arrDispPhi = getArray("DSDispPhi", JBinaryDataDef::DatFloat);
        JBinaryDataArray* arrCauchy = getArray("cauchy", JBinaryDataDef::DatFloat);
        if (!arrPos0w || !arrDispPhi || !arrCauchy) return;
        const unsigned pos0wCount = getElementCount(arrPos0w);
        const unsigned dispPhiCount = getElementCount(arrDispPhi);
        const unsigned cauchyCount = getElementCount(arrCauchy);
        if (pos0wCount % 4 || dispPhiCount % 4 || cauchyCount % 6) {
            ExceptionText("Error: Deformable structure arrays have unexpected element counts.");
        }
        std::vector<tfloat4> dspos0w(pos0wCount / 4), dsdispphi(dispPhiCount / 4);
        std::vector<tsymatrix3f> cauchy(cauchyCount / 6);
        arrPos0w->GetDataCopy(pos0wCount, reinterpret_cast<float*>(dspos0w.data()));
        arrDispPhi->GetDataCopy(dispPhiCount, reinterpret_cast<float*>(dsdispphi.data()));
        arrCauchy->GetDataCopy(cauchyCount, reinterpret_cast<float*>(cauchy.data()));
        if (particleCount > dspos0w.size() || particleCount > dsdispphi.size() || particleCount > cauchy.size()) {
            ExceptionText("Error: Deformable structure particle metadata exceeds available array data.");
        }

        std::vector<tfloat3> energiesFloat;
        std::vector<tdouble3> energiesDouble;
        if (hasArray("Energies")) {
            if (JBinaryDataArray* arr = getArray("Energies", JBinaryDataDef::DatFloat3)) {
                const unsigned count = getElementCount(arr);
                energiesFloat.resize(count);
                if (count) arr->GetDataCopy(count, energiesFloat.data());
            }
            else if (JBinaryDataArray* arr = getArray("Energies", JBinaryDataDef::DatDouble3)) {
                const unsigned count = getElementCount(arr);
                energiesDouble.resize(count);
                if (count) arr->GetDataCopy(count, energiesDouble.data());
            }
        }
        std::vector<tfloat3> dsenergyParticle;
        std::vector<tdouble3> dsenergyParticleDouble;
        if (hasArray("DSEnergy")) {
            if (JBinaryDataArray* arr = getArray("DSEnergy", JBinaryDataDef::DatFloat3)) {
                const unsigned count = getElementCount(arr);
                dsenergyParticle.resize(count);
                if (count) arr->GetDataCopy(count, dsenergyParticle.data());
            }
            else if (JBinaryDataArray* arr = getArray("DSEnergy", JBinaryDataDef::DatDouble3)) {
                const unsigned count = getElementCount(arr);
                dsenergyParticleDouble.resize(count);
                if (count) arr->GetDataCopy(count, dsenergyParticleDouble.data());
            }
        }
        std::vector<tfloat3> dsfforceFloat;
        std::vector<tdouble3> dsfforceDouble;
        if (hasArray("FluidForce")) {
            if (JBinaryDataArray* arr = getArray("FluidForce", JBinaryDataDef::DatFloat3)) {
                const unsigned count = getElementCount(arr);
                dsfforceFloat.resize(count);
                arr->GetDataCopy((unsigned)dsfforceFloat.size(), dsfforceFloat.data());
            }
            else if (JBinaryDataArray* arr = getArray("FluidForce", JBinaryDataDef::DatDouble3)) {
                const unsigned count = getElementCount(arr);
                dsfforceDouble.resize(count);
                arr->GetDataCopy((unsigned)dsfforceDouble.size(), dsfforceDouble.data());
            }
        }

        std::vector<float> dsplastic;
        if (hasArray("DSEqPlastic")) {
            if (JBinaryDataArray* arr = getArray("DSEqPlastic", JBinaryDataDef::DatFloat)) {
                const unsigned count = getElementCount(arr);
                dsplastic.resize(count);
                if (count) arr->GetDataCopy(count, dsplastic.data());
                if (particleCount > dsplastic.size()) {
                    printf("Warning: DSEqPlastic array only stores %zu values; plastic strain output will be skipped.\n",
                        dsplastic.size());
                    dsplastic.clear();
                }
            }
        }

        std::vector<tdouble3> energyOut(bodies.size(), TDouble3(0));
        for (size_t i = 0; i < bodies.size(); ++i) {
            if (i < energiesFloat.size()) energyOut[i] = TDouble3(energiesFloat[i].x, energiesFloat[i].y, energiesFloat[i].z);
            else if (i < energiesDouble.size()) energyOut[i] = energiesDouble[i];
        }


        for (unsigned bodyid = 0; bodyid < bodyCount; ++bodyid) {
            const DsBodyInfo& body = bodies[bodyid];
            if (!body.npbody) continue;
            if (filterBodies && !bodyFilter.CheckValue(body.mkbound)) continue;
            const unsigned fieldMask = getMaskForBody(body.mkbound, body);
            const unsigned dfnp = body.npbody;
            const unsigned npstart = body.npstart;
            std::vector<tfloat3> pos0(dfnp), disp(dfnp);
            std::vector<tsymatrix3f> stress(dfnp);
            std::vector<float> strainEnergy, kineticEnergy, fractureOrPlasticEnergy;
            std::vector<tfloat3> fforceFloat;
            std::vector<tdouble3> fforceDouble;


            if (!dsenergyParticle.empty()) {
                strainEnergy.resize(dfnp);
                kineticEnergy.resize(dfnp);
                if (body.fracture || body.constitutive == CONSTITMODEL_J2) {
                    fractureOrPlasticEnergy.resize(dfnp);
                }
            }
            else if (!dsenergyParticleDouble.empty()) {
                strainEnergy.resize(dfnp);
                kineticEnergy.resize(dfnp);
                if (body.fracture || body.constitutive == CONSTITMODEL_J2) {
                    fractureOrPlasticEnergy.resize(dfnp);
                }
            }

            if (!dsfforceFloat.empty()) fforceFloat.resize(dfnp);
            else if (!dsfforceDouble.empty()) fforceDouble.resize(dfnp);

            std::vector<float> phit(dfnp), dsplasticbody(dfnp);
            // Parallelize particle loop for large bodies (biggest performance gain)
#pragma omp parallel for if(dfnp > 5000)
            for (int p = 0; p < (int)dfnp; ++p) {
                const unsigned idx = dsibodyridp[npstart + p];
                const tfloat4 pos = dspos0w[idx];
                const tfloat4 dispp = dsdispphi[idx];
                const tsymatrix3f stressfull = cauchy[idx];
                pos0[p] = TFloat3(pos.x, pos.y, pos.z);
                disp[p] = TFloat3(dispp.x, dispp.y, dispp.z);
                stress[p] = stressfull;
                if (body.fracture) phit[p] = dispp.w;
                if (!dsplastic.empty()) dsplasticbody[p] = dsplastic[idx];

                // Split energy components
                if (!dsenergyParticle.empty()) {
                    const tfloat3 e = dsenergyParticle[idx];
                    strainEnergy[p] = e.x;
                    kineticEnergy[p] = e.y;
                    if (body.fracture || body.constitutive == CONSTITMODEL_J2) {
                        fractureOrPlasticEnergy[p] = e.z;
                    }
                }
                else if (!dsenergyParticleDouble.empty()) {
                    const tdouble3 e = dsenergyParticleDouble[idx];
                    strainEnergy[p] = float(e.x);
                    kineticEnergy[p] = float(e.y);
                    if (body.fracture || body.constitutive == CONSTITMODEL_J2) {
                        fractureOrPlasticEnergy[p] = float(e.z);
                    }
                }

                if (!fforceFloat.empty()) fforceFloat[p] = dsfforceFloat[idx];
                else if (!fforceDouble.empty()) fforceDouble[p] = dsfforceDouble[idx];
            }
            JDataArrays arraysdef;
            arraysdef.AddArray("Pos0", dfnp, pos0.data());

            // Add timestep as a scalar field (constant for all particles)
            std::vector<float> timestepField(dfnp, timestepFloat);
            arraysdef.AddArray("TimeStep", dfnp, timestepField.data());

            if (fieldMask & FieldDisp) arraysdef.AddArray("Displacement", dfnp, disp.data());
            if (fieldMask & FieldCauchy) arraysdef.AddArray("Cauchy", dfnp, stress.data());
            if ((fieldMask & FieldPhasefield) && body.fracture) arraysdef.AddArray("Phasefield", dfnp, phit.data());
            if ((fieldMask & FieldPlastic) && !dsplastic.empty()) arraysdef.AddArray("Eqv_Plastic_Strain", dfnp, dsplasticbody.data());

            // ContactForce (only for multi-body)
            if ((fieldMask & FieldFluidForce) && bodyCount > 1) {
                if (!fforceDouble.empty()) arraysdef.AddArray("ContactForce", dfnp, fforceDouble.data());
                else if (!fforceFloat.empty()) arraysdef.AddArray("ContactForce", dfnp, fforceFloat.data());
            }

            // Energy components as separate scalar fields
            if ((fieldMask & FieldEnergy) && !strainEnergy.empty()) {
                arraysdef.AddArray("Strain_Energy", dfnp, strainEnergy.data());
                arraysdef.AddArray("Kinetic_Energy", dfnp, kineticEnergy.data());
                if (!fractureOrPlasticEnergy.empty()) {
                    if (body.fracture) {
                        arraysdef.AddArray("Fracture_Energy", dfnp, fractureOrPlasticEnergy.data());
                    }
                    else if (body.constitutive == CONSTITMODEL_J2) {
                        arraysdef.AddArray("Plastic_Energy", dfnp, fractureOrPlasticEnergy.data());
                    }
                }
            }
            const std::string vtkfile = buildOutputFile(dsVtkTemplate, fallbackVtkTemplate, body.mkbound, "vtk");
            if (!vtkfile.empty()) {
                fun::MkdirPath(ExtractDirectory(vtkfile));
                SaveVtkDataCompatible(vtkfile, arraysdef, "Pos0");
            }
            const std::string csvfile = buildOutputFile(dsCsvTemplate, std::string(), body.mkbound, "csv");
            if (!csvfile.empty()) {
                fun::MkdirPath(ExtractDirectory(csvfile));
                JOutputCsv ocsv(false, csvComma);
                ocsv.SaveCsv(csvfile, arraysdef);
            }
        }

        if (!energiesFloat.empty() || !energiesDouble.empty()) WriteEnergiesCsv(energyFile, csvComma, timestep, bodies, energyOut);

        if (totalPlaneCount && !measPlaneCounts.empty() && !measPlanePart.empty()) {
            size_t planeIndex = 0;
            size_t partOffset = 0;
            for (unsigned bodyid = 0; bodyid < bodyCount; ++bodyid) {
                const DsBodyInfo& body = bodies[bodyid];
                if (!body.nmeasplane) continue;
                double area = (body.mapfact ? (body.dp / body.mapfact) : body.dp);
                if (!simulate2D) area *= area;
                for (unsigned pl = 0; pl < body.nmeasplane && planeIndex < measPlaneCounts.size(); ++pl, ++planeIndex) {
                    const unsigned count = measPlaneCounts[planeIndex];
                    if (!count || partOffset + count > measPlanePart.size()) { partOffset += count; continue; }
                    tdouble3 avgDisp = TDouble3(0);
                    tdouble3 avgForce = TDouble3(0);
                    for (unsigned ip = 0; ip < count; ++ip) {
                        const unsigned idx = measPlanePart[partOffset + ip];
                        const tfloat4 dispp = dsdispphi[idx];
                        const tsymatrix3f stressfull = cauchy[idx];
                        avgDisp.x += dispp.x;
                        avgDisp.y += dispp.y;
                        avgDisp.z += dispp.z;
                        avgForce.x += (stressfull.xx + stressfull.xy + stressfull.xz) * area;
                        avgForce.y += (stressfull.yy + stressfull.xy + stressfull.yz) * area;
                        avgForce.z += (stressfull.zz + stressfull.xz + stressfull.yz) * area;
                    }
                    partOffset += count;
                    avgDisp.x /= count; avgDisp.y /= count; avgDisp.z /= count;
                    avgForce.x /= count; avgForce.y /= count; avgForce.z /= count;
                    const std::string planeFile = EnsureCsvExtension(energyDir + "MeasuringPlData_MK" + std::to_string(body.mkbound)
                        + "_PL" + std::to_string(pl + 1));
                    WritePlaneCsv(planeFile, csvComma, simulate2D, timestep, avgDisp, avgForce);
                }
            }
        }
    }
    else {
        JBinaryDataArray* arrPos0 = getArray("DSPos0", JBinaryDataDef::DatDouble3);
        JBinaryDataArray* arrDisp = getArray("DSDisp", JBinaryDataDef::DatDouble3);
        JBinaryDataArray* arrVel = getArray("DSVel", JBinaryDataDef::DatDouble3);
        //JBinaryDataArray* arrkervol = getArray("DSKerSumVol", JBinaryDataDef::DatDouble);
        JBinaryDataArray* arrPiol = getArray("DSPiolKir", JBinaryDataDef::DatDouble);
        JBinaryDataArray* arrDefGrad = getArray("DSDefGrad", JBinaryDataDef::DatDouble);
        if (!arrPos0 || !arrDisp || !arrVel || !arrPiol || !arrDefGrad) return;
        const unsigned posCount = getElementCount(arrPos0);
        const unsigned dispCount = getElementCount(arrDisp);
        //const unsigned volCount = getElementCount(arrkervol);
        const unsigned velCount = getElementCount(arrVel);
        const unsigned piolCount = getElementCount(arrPiol);
        const unsigned defGradCount = getElementCount(arrDefGrad);
        if (piolCount % 9 || defGradCount % 9) {
            ExceptionText("Error: Deformable structure tensors have unexpected element counts.");
        }
        std::vector<tdouble3> dspos(posCount), dsdisp(dispCount), dsvel(velCount), dskerderv0;
        //std::vector<double> dskervol(volCount);
        std::vector<tmatrix3d> dspiolkir(piolCount / 9), dsdefgrad(defGradCount / 9), dsplasticdev;
        arrPos0->GetDataCopy(posCount, dspos.data());
        arrDisp->GetDataCopy(dispCount, dsdisp.data());
        arrVel->GetDataCopy(velCount, dsvel.data());
        //arrkervol->GetDataCopy(volCount, dskervol.data());
        arrPiol->GetDataCopy(piolCount, reinterpret_cast<double*>(dspiolkir.data()));
        arrDefGrad->GetDataCopy(defGradCount, reinterpret_cast<double*>(dsdefgrad.data()));
        if (particleCount > dspos.size() || particleCount > dsdisp.size() || particleCount > dsvel.size()
            || particleCount > dspiolkir.size() || particleCount > dsdefgrad.size()) {
            ExceptionText("Error: Deformable structure particle metadata exceeds available array data.");
        }
        if (hasArray("DSKerDerV0")) {
            if (JBinaryDataArray* arr = getArray("DSKerDerV0", JBinaryDataDef::DatDouble3)) {
                const unsigned count = getElementCount(arr);
                dskerderv0.resize(count);
                if (count) arr->GetDataCopy(count, dskerderv0.data());
            }
        }
        if (hasArray("DSPlasticDev")) {
            if (JBinaryDataArray* arr = getArray("DSPlasticDev", JBinaryDataDef::DatDouble)) {
                const unsigned plasticDevCount = getElementCount(arr);
                if (plasticDevCount % 9) ExceptionText("Error: Plastic deformation tensors have unexpected element counts.");
                dsplasticdev.resize(plasticDevCount / 9);
                if (plasticDevCount) arr->GetDataCopy(plasticDevCount, reinterpret_cast<double*>(dsplasticdev.data()));
                if (particleCount > dsplasticdev.size()) ExceptionText("Error: Plastic deformation data is incomplete for deformable structures.");

            }
        }
        std::vector<double> dsphi, dseqplastic;
        if (hasArray("DSPhi")) {
            if (JBinaryDataArray* arr = getArray("DSPhi", JBinaryDataDef::DatDouble)) {
                const unsigned count = getElementCount(arr);
                dsphi.resize(count);
                if (count) arr->GetDataCopy(count, dsphi.data());
                if (particleCount > dsphi.size()) {
                    printf("Warning: DSPhi array only stores %zu values; fracture phase output will be skipped.\n",
                        dsphi.size());
                    dsphi.clear();
                }
            }
        }
        if (hasArray("DSEqPlastic")) {
            if (JBinaryDataArray* arr = getArray("DSEqPlastic", JBinaryDataDef::DatDouble)) {
                const unsigned count = getElementCount(arr);
                dseqplastic.resize(count);
                if (count) arr->GetDataCopy(count, dseqplastic.data());
                if (particleCount > dseqplastic.size()) {
                    printf("Warning: DSEqPlastic array only stores %zu values; plastic strain output will be skipped.\n",
                        dseqplastic.size());
                    dseqplastic.clear();
                }
            }
        }
        std::vector<unsigned> dspairn, dspairstart, dspairj;
        copyUintArray("DSPairN", dspairn);
        copyUintArray("DSPairStart", dspairstart);
        if (hasArray("DSPairJ")) {
            if (JBinaryDataArray* arr = getArray("DSPairJ", JBinaryDataDef::DatUint)) {
                const unsigned count = getElementCount(arr);
                dspairj.resize(count);
                if (count) arr->GetDataCopy(count, dspairj.data());

            }
        }

        std::vector<tfloat3> energiesFloat;
        std::vector<tdouble3> energiesDouble;
        if (hasArray("Energies")) {
            if (JBinaryDataArray* arr = getArray("Energies", JBinaryDataDef::DatFloat3)) {
                const unsigned count = getElementCount(arr);
                energiesFloat.resize(count);
                if (count) arr->GetDataCopy(count, energiesFloat.data());
            }
            else if (JBinaryDataArray* arr = getArray("Energies", JBinaryDataDef::DatDouble3)) {
                const unsigned count = getElementCount(arr);
                energiesDouble.resize(count);
                if (count) arr->GetDataCopy(count, energiesDouble.data());
            }
        }
        std::vector<tfloat3> dsenergyParticle;
        std::vector<tdouble3> dsenergyParticleDouble;
        bool energyd = false, energyf = false;
        if (hasArray("DSEnergy")) {
            if (JBinaryDataArray* arr = getArray("DSEnergy", JBinaryDataDef::DatFloat3)) {
                const unsigned count = getElementCount(arr);
                dsenergyParticle.resize(count);
                if (count) arr->GetDataCopy(count, dsenergyParticle.data());
                energyf = true;
            }
            else if (JBinaryDataArray* arr = getArray("DSEnergy", JBinaryDataDef::DatDouble3)) {
                const unsigned count = getElementCount(arr);
                dsenergyParticleDouble.resize(count);
                if (count) arr->GetDataCopy(count, dsenergyParticleDouble.data());
                energyd = true;
            }
        }
        std::vector<tfloat3> dsfforceFloat;
        std::vector<tdouble3> dsfforceDouble;
        if (hasArray("FluidForce")) {
            if (JBinaryDataArray* arr = getArray("FluidForce", JBinaryDataDef::DatFloat3)) {
                const unsigned count = getElementCount(arr);
                dsfforceFloat.resize(count);
                arr->GetDataCopy((unsigned)dsfforceFloat.size(), dsfforceFloat.data());
            }
            else if (JBinaryDataArray* arr = getArray("FluidForce", JBinaryDataDef::DatDouble3)) {
                const unsigned count = getElementCount(arr);
                dsfforceDouble.resize(count);
                arr->GetDataCopy((unsigned)dsfforceDouble.size(), dsfforceDouble.data());
            }
        }
        std::vector<tdouble3> totEnergy(bodyCount, TDouble3(0));
        // Parallel energy accumulation with reduction
        for (unsigned bodyid = 0; bodyid < bodyCount; ++bodyid) {
            const DsBodyInfo& body = bodies[bodyid];
            double bodyEnergyx = 0.0, bodyEnergyy = 0.0, bodyEnergyz = 0.0;
#pragma omp parallel for reduction(+:bodyEnergyx,bodyEnergyy,bodyEnergyz) if(body.npbody > 1000)
            for (int p = 0; p < (int)body.npbody; ++p) {
                const unsigned idx = dsibodyridp[body.npstart + p];
                tdouble3 energies = { 0,0,0 };
                if (energyd) {
                    energies = dsenergyParticleDouble[idx];
                }
                else if (energyf) {
                    const tfloat3 e = dsenergyParticle[idx];
                    energies.x = double(e.x);
                    energies.y = double(e.y);
                    energies.z = double(e.z);
                }
                bodyEnergyx += energies.x;
                bodyEnergyy += energies.y;
                bodyEnergyz += energies.z;
            }
            totEnergy[bodyid] = { bodyEnergyx,bodyEnergyy,bodyEnergyz, };
        }

        if (!energiesFloat.empty() || !energiesDouble.empty()) {
            for (unsigned bodyid = 0; bodyid < bodyCount; ++bodyid) {
                if (bodyid < energiesFloat.size()) {
                    const tfloat3 v = energiesFloat[bodyid];
                    totEnergy[bodyid] = TDouble3(v.x, v.y, v.z);
                }
                else if (bodyid < energiesDouble.size()) {
                    totEnergy[bodyid] = energiesDouble[bodyid];
                }

            }
        }

        for (unsigned bodyid = 0; bodyid < bodyCount; ++bodyid) {
            const DsBodyInfo& body = bodies[bodyid];
            if (!body.npbody) continue;
            if (filterBodies && !bodyFilter.CheckValue(body.mkbound)) continue;
            const unsigned fieldMask = getMaskForBody(body.mkbound, body);
            const unsigned dfnp = body.npbody;
            const unsigned npstart = body.npstart;
            std::vector<tdouble3> pos0(dfnp), disp(dfnp);
            std::vector<tsymatrix3f> stress(dfnp);
            std::vector<float> strainEnergy, kineticEnergy, fractureOrPlasticEnergy;
            std::vector<tfloat3> fforceFloat;
            std::vector<tdouble3> fforceDouble;

            if (!dsenergyParticle.empty() || !dsenergyParticleDouble.empty()) {
                strainEnergy.resize(dfnp);
                kineticEnergy.resize(dfnp);
                if (body.fracture || body.constitutive == CONSTITMODEL_J2) {
                    fractureOrPlasticEnergy.resize(dfnp);
                }
            }

            if (!dsfforceFloat.empty()) fforceFloat.resize(dfnp);
            else if (!dsfforceDouble.empty()) fforceDouble.resize(dfnp);

            std::vector<double> phit(dfnp), plasticbody(dfnp);
            // Parallelize particle loop for large bodies (biggest performance gain)
#pragma omp parallel for if(dfnp > 5000)
            for (int p = 0; p < (int)dfnp; ++p) {
                const unsigned idx = dsibodyridp[npstart + p];
                pos0[p] = dspos[idx];
                disp[p] = dsdisp[idx];

                // Compute Cauchy stress on-demand with safety check
                const tmatrix3d defg = dsdefgrad[idx];
                const double det = fmath::Determinant3x3(defg);
                if (fabs(det) > 1e-12) {
                    const tmatrix3d cauchyFull = (1.0 / det) * fmath::MulMatrix3x3(defg, dspiolkir[idx]);
                    stress[p].xx = float(cauchyFull.a11);
                    stress[p].xy = float(cauchyFull.a12);
                    stress[p].xz = float(cauchyFull.a13);
                    stress[p].yy = float(cauchyFull.a22);
                    stress[p].yz = float(cauchyFull.a23);
                    stress[p].zz = float(cauchyFull.a33);
                }
                else {
                    // Degenerate deformation gradient - set stress to zero
                    stress[p] = tsymatrix3f();
                }
                if (body.fracture && !dsphi.empty()) phit[p] = dsphi[idx];
                if (!dseqplastic.empty()) plasticbody[p] = dseqplastic[idx];

                // Split energy components
                if (!dsenergyParticle.empty()) {
                    const tfloat3 e = dsenergyParticle[idx];
                    strainEnergy[p] = e.x;
                    kineticEnergy[p] = e.y;
                    if (body.fracture || body.constitutive == CONSTITMODEL_J2) {
                        fractureOrPlasticEnergy[p] = e.z;
                    }
                }
                else if (!dsenergyParticleDouble.empty()) {
                    const tdouble3 e = dsenergyParticleDouble[idx];
                    strainEnergy[p] = float(e.x);
                    kineticEnergy[p] = float(e.y);
                    if (body.fracture || body.constitutive == CONSTITMODEL_J2) {
                        fractureOrPlasticEnergy[p] = float(e.z);
                    }
                }

                if (!fforceFloat.empty()) fforceFloat[p] = dsfforceFloat[idx];
                else if (!fforceDouble.empty()) fforceDouble[p] = dsfforceDouble[idx];

            }
            JDataArrays arraysdef;
            arraysdef.AddArray("Pos0", dfnp, pos0.data());

            // Add timestep as a scalar field (constant for all particles)
            std::vector<float> timestepField(dfnp, timestepFloat);
            arraysdef.AddArray("TimeStep", dfnp, timestepField.data());

            if (fieldMask & FieldDisp) arraysdef.AddArray("Displacement", dfnp, disp.data());
            if (fieldMask & FieldCauchy) arraysdef.AddArray("Cauchy", dfnp, stress.data());
            if ((fieldMask & FieldPhasefield) && body.fracture && !dsphi.empty()) arraysdef.AddArray("Phasefield", dfnp, phit.data());
            if ((fieldMask & FieldPlastic) && !dseqplastic.empty()) arraysdef.AddArray("Eqv_Plastic_Strain", dfnp, plasticbody.data());

            // ContactForce (only for multi-body)
            if ((fieldMask & FieldFluidForce) && bodyCount > 1) {
                if (!fforceDouble.empty()) arraysdef.AddArray("ContactForce", dfnp, fforceDouble.data());
                else if (!fforceFloat.empty()) arraysdef.AddArray("ContactForce", dfnp, fforceFloat.data());
            }

            // Energy components as separate scalar fields
            if ((fieldMask & FieldEnergy) && !strainEnergy.empty()) {
                arraysdef.AddArray("Strain_Energy", dfnp, strainEnergy.data());
                arraysdef.AddArray("Kinetic_Energy", dfnp, kineticEnergy.data());
                if (!fractureOrPlasticEnergy.empty()) {
                    if (body.fracture) {
                        arraysdef.AddArray("Fracture_Energy", dfnp, fractureOrPlasticEnergy.data());
                    }
                    else if (body.constitutive == CONSTITMODEL_J2) {
                        arraysdef.AddArray("Plastic_Energy", dfnp, fractureOrPlasticEnergy.data());
                    }
                }
            }
            const std::string vtkfile = buildOutputFile(dsVtkTemplate, fallbackVtkTemplate, body.mkbound, "vtk");
            if (!vtkfile.empty()) {
                fun::MkdirPath(ExtractDirectory(vtkfile));

                SaveVtkDataCompatible(vtkfile, arraysdef, "Pos0");
            }
            const std::string csvfile = buildOutputFile(dsCsvTemplate, std::string(), body.mkbound, "csv");
            if (!csvfile.empty()) {
                fun::MkdirPath(ExtractDirectory(csvfile));
                JOutputCsv ocsv(false, csvComma);
                ocsv.SaveCsv(csvfile, arraysdef);
            }
        }

        WriteEnergiesCsv(energyFile, csvComma, timestep, bodies, totEnergy);

        if (totalPlaneCount && !measPlaneCounts.empty() && !measPlanePart.empty()) {
            size_t planeIndex = 0;
            size_t partOffset = 0;
            for (unsigned bodyid = 0; bodyid < bodyCount; ++bodyid) {
                const DsBodyInfo& body = bodies[bodyid];
                if (!body.nmeasplane) continue;
                double area = (body.mapfact ? (body.dp / body.mapfact) : body.dp);
                if (!simulate2D) area *= area;
                for (unsigned pl = 0; pl < body.nmeasplane && planeIndex < measPlaneCounts.size(); ++pl, ++planeIndex) {
                    const unsigned count = measPlaneCounts[planeIndex];
                    if (!count || partOffset + count > measPlanePart.size()) { partOffset += count; continue; }
                    tdouble3 avgDisp = TDouble3(0);
                    tdouble3 avgForce = TDouble3(0);
                    for (unsigned ip = 0; ip < count; ++ip) {
                        const unsigned idx = measPlanePart[partOffset + ip];
                        const tdouble3 dispp = dsdisp[idx];

                        // Compute Cauchy stress on-demand with safety check
                        tsymatrix3f stress;
                        const tmatrix3d defg = dsdefgrad[idx];
                        const double det = fmath::Determinant3x3(defg);
                        if (fabs(det) > 1e-12) {
                            const tmatrix3d cauchyFull = (1.0 / det) * fmath::MulMatrix3x3(defg, dspiolkir[idx]);
                            stress.xx = float(cauchyFull.a11);
                            stress.xy = float(cauchyFull.a12);
                            stress.xz = float(cauchyFull.a13);
                            stress.yy = float(cauchyFull.a22);
                            stress.yz = float(cauchyFull.a23);
                            stress.zz = float(cauchyFull.a33);
                        }
                        else {
                            stress = tsymatrix3f();
                        }

                        avgDisp.x += dispp.x;
                        avgDisp.y += dispp.y;
                        avgDisp.z += dispp.z;
                        avgForce.x += (stress.xx + stress.xy + stress.xz) * area;
                        avgForce.y += (stress.yy + stress.xy + stress.yz) * area;
                        avgForce.z += (stress.zz + stress.xz + stress.yz) * area;
                    }
                    partOffset += count;
                    avgDisp.x /= count; avgDisp.y /= count; avgDisp.z /= count;
                    avgForce.x /= count; avgForce.y /= count; avgForce.z /= count;
                    const std::string planeFile = EnsureCsvExtension(energyDir + "MeasuringPlData_MK" + std::to_string(body.mkbound)
                        + "_PL" + std::to_string(pl + 1));
                    WritePlaneCsv(planeFile, csvComma, simulate2D, timestep, avgDisp, avgForce);
                }
            }
        }
    }
}

void RunFiles(const JCfgRun* cfg) {
    const bool onefile = !cfg->FileIn.empty();
    const string casein = cfg->FileIn;
    const string dirin = fun::GetDirWithSlash(cfg->DirIn);
    const int last = cfg->Last;
    const int firstPart = (onefile || cfg->First < 0 ? 0 : cfg->First);

    byte npie = 0;
    string firstFile = JPartDataBi4::GetFileData(casein, dirin, firstPart, npie);
    if (firstFile.empty())ExceptionText("Error: Data files not found.");

    unsigned npiece = 0;
    {
        JPartDataBi4 pd;
        if (onefile)pd.LoadFileCase("", casein, 0, npie);
        else pd.LoadFilePart(dirin, firstPart, 0, npie);
        npiece = pd.GetNpiece();
        if (npiece > 1)ExceptionText("Error: The number of pieces is higher than 1.");
    }

    std::vector<int> parts;
    if (onefile) {
        parts.push_back(0);
    }
    else {
        if (!firstFile.empty()) parts.push_back(firstPart);
        int current = firstPart + 1;
        while (last < 0 || current <= last) {
            std::string filename = dirin + JPartDataBi4::GetFileNamePart(current, 0, npiece);
            if (!fun::FileExists(filename)) break;
            parts.push_back(current);
            ++current;
        }
    }

    if (parts.empty())ExceptionText("Error: Data files not found.");

    const bool csvComma = true;
    std::atomic<bool> abortProcessing(false);
    std::atomic<size_t> processedCount(0);
    std::mutex errorMutex;
    std::mutex progressMutex;
    std::string firstError;

    // Use input directory as base output directory
    std::string baseOutDir = dirin;

    printf("\n*** Processing %zu part files ***\n", parts.size());

#pragma omp parallel for schedule(dynamic)
    for (int idx = 0; idx < parts.size(); ++idx) {
        if (abortProcessing.load(std::memory_order_acquire)) continue;
        const int part = parts[idx];
        try {
            // Load particle data
            JPartDataBi4 pd;
            if (onefile) pd.LoadFileCase("", casein, 0, npiece);
            else pd.LoadFilePart(dirin, part, 0, npiece);

            const double timestep = pd.Get_TimeStep();

            ProcessDeformableStructures(cfg, pd, timestep, part, baseOutDir, csvComma);

            // Update progress
            size_t completed = processedCount.fetch_add(1, std::memory_order_relaxed) + 1;
            {
                std::lock_guard<std::mutex> lock(progressMutex);
                printf("  [%zu/%zu] Completed Part_%04d (timestep=%.6f)\n",
                    completed, parts.size(), part, timestep);
            }
        }
        catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(errorMutex);
            if (firstError.empty()) firstError = e.what();
            abortProcessing.store(true, std::memory_order_release);
        }
        catch (const std::string& e) {
            std::lock_guard<std::mutex> lock(errorMutex);
            if (firstError.empty()) firstError = e;
            abortProcessing.store(true, std::memory_order_release);
        }
        catch (const char* e) {
            std::lock_guard<std::mutex> lock(errorMutex);
            if (firstError.empty()) firstError = (e ? std::string(e) : std::string());
            abortProcessing.store(true, std::memory_order_release);
        }
        catch (...) {
            std::lock_guard<std::mutex> lock(errorMutex);
            if (firstError.empty()) firstError = "Unknown exception";
            abortProcessing.store(true, std::memory_order_release);
        }
    }

    printf("\n*** Processing complete: %zu/%zu files processed successfully ***\n",
        processedCount.load(), parts.size());

    if (!firstError.empty())ExceptionText(firstError);
}
//==============================================================================
/// GPL License.
//==============================================================================
std::string getlicense_lgpl(const std::string& name) {
    std::string tx = "";
    tx = tx + "\n\n <" + fun::StrUpper(name) + ">  Copyright (c) 2020 by Dr Jose M. Dominguez";
    tx = tx + "\n (see http://dual.sphysics.org/index.php/developers/)\n";
    tx = tx + "\n EPHYSLAB Environmental Physics Laboratory, Universidade de Vigo";
    tx = tx + "\n School of Mechanical, Aerospace and Civil Engineering, University of Manchester\n";
    tx = tx + "\n DualSPHysics is free software: you can redistribute it and/or";
    tx = tx + "\n modify it under the terms of the GNU Lesser General Public License";
    tx = tx + "\n as published by the Free Software Foundation, either version 2.1 of";
    tx = tx + "\n the License, or (at your option) any later version.\n";
    tx = tx + "\n DualSPHysics is distributed in the hope that it will be useful,";
    tx = tx + "\n but WITHOUT ANY WARRANTY; without even the implied warranty of";
    tx = tx + "\n MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the";
    tx = tx + "\n GNU Lesser General Public License for more details.\n";
    tx = tx + "\n You should have received a copy of the GNU Lesser General Public License";
    tx = tx + "\n along with DualSPHysics. If not, see <http://www.gnu.org/licenses/>.\n\n";
    return(tx);
}

//==============================================================================
//==============================================================================
int main(int argc, char** argv) {
    int errcode = 1;
    printf("%s", getlicense_lgpl("TOVTK4_DEFSTRUCT").c_str());
    printf("\n%s\n", APP_NAME);
    for (unsigned c = 0; c <= strlen(APP_NAME); c++)printf("="); printf("\n");

    // OpenMP status check
#ifdef _OPENMP
    int maxThreads = omp_get_max_threads();
    if (maxThreads > 30) {
        omp_set_num_threads(30);
        maxThreads = 30;
    }
    printf("OpenMP: ENABLED (%d threads available)\n", maxThreads);
#else
    printf("WARNING: OpenMP DISABLED - running single-threaded!\n");
#endif
    printf("\n");

    JCfgRun cfg;
    try {
        cfg.LoadArgv(argc, argv);
        if (!cfg.PrintInfo) {
            cfg.ValidaCfg();
            RunFiles(&cfg);
        }
        errcode = 0;
    }
    catch (const char* cad) {
        printf("\n*** Exception: %s\n", cad);
    }
    catch (const string& e) {
        printf("\n*** Exception: %s\n", e.c_str());
    }
    catch (const exception& e) {
        printf("\n*** %s\n", e.what());
    }
    catch (...) {
        printf("\n*** Attention: Unknown exception...\n");
    }
    return(errcode);
}


