//HEAD_DSPH
/*
 <DUALSPHYSICS>  Copyright (c) 2020 by Dr Jose M. Dominguez et al. (see http://dual.sphysics.org/index.php/developers/).

 EPHYSLAB Environmental Physics Laboratory, Universidade de Vigo, Ourense, Spain.
 School of Mechanical, Aerospace and Civil Engineering, University of Manchester, Manchester, U.K.

 This file is part of DualSPHysics.

 DualSPHysics is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

 DualSPHysics is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

 You should have received a copy of the GNU General Public License, along with DualSPHysics. If not, see <http://www.gnu.org/licenses/>.

 ---

 <SOLIDUALSPHYSICS> SoliDualSPHysics Extensions:
 Copyright (c) 2024 by Dr. Naqib Rahimi and Dr. Georgios Moutsanidis.
 
 This file contains extensions for solid mechanics, including:
 - Deformable structure modeling (hyperelasticity, plasticity, fracture)
 - Phase-field fracture methods
 - Advanced boundary condition handling
 
 Developed as part of the PhD thesis:
 "Computational Mechanics of Extreme Events: Advanced Multi-physics Modeling and 
 Simulations with Smoothed Particle Hydrodynamics, Isogeometric Analysis, and Phase Field"
 by Dr. Naqib Rahimi, supervised by Dr. Georgios Moutsanidis.
 
 Related publication:
 Rahimi, N., & Moutsanidis, G. (2026). "SoliDualSPHysics: An extension of DualSPHysics 
 to solid mechanics with hyperelasticity, plasticity, and fracture." [In preparation]
 
 These extensions are distributed under the same GNU GPL v3 license as DualSPHysics.
*/

/// \file JSphDeformStruc.h \brief Declares the class \ref JSphDeformStruc.
/// 
/// This file implements the deformable structure framework for SoliDualSPHysics,
/// enabling solid mechanics simulations with hyperelasticity, J2 plasticity, and
/// phase-field fracture modeling within the SPH framework.
/// 
/// \authors Dr. Naqib Rahimi, Dr. Georgios Moutsanidis

#ifndef _JSphDeformStruc_
#define _JSphDeformStruc_
#include "JObject.h"
#include "DualSphDef.h"
#include "JDataArrays.h"
#include "JDsTimersCpu.h"
#include "JCellDivDataCpu.h"
#include "FunctionsMath.h"
#include "JCellSearch_inline.h"
#include "JSphCpu_ExpressionParser.h"
#include "JCellDivCpuSingle.h"
#include "JAppInfo.h"
#include "JVtkLib.h"
#include "JSaveCsv2.h"
#include "Functions.h"
#include <numeric>
#include <climits>
#include <unordered_map>
#include <array>
#include <functional>

using namespace std;
class JLog2;
class JXml;
class TiXmlElement;
class JSphMk;
class JCellDivCpuSingle;
class JCellDivCpu;
class JUserExpression;
//##############################################################################
//# JSphDeformStrucBody
//##############################################################################
/// \brief Manages the info of a single deformable structure.
class JSphDeformStrucBody : protected JObject
{
private:
	JLog2* Log;

	//-Selection of particles
	typecode BoundCode;							///<Code to select boundary particles.

	//-Body parameters
	float PartVol;								///<Initial particle volume.
	float Density;								///<Initial particle density.
	float YoungMod;								///<Young's modulus.
	float PoissRatio;							///<Poisson ratio.
	float RestCoeff;							///<Coefficient of restitution
	float KFric;								///<Coefficient of kinetic friction
	float LameLmbda;							///<Lame parameter, Lambda
	float LameMu;								///<Lame parameter, Mu
	float LameBulk;								///<Bulk modulus
	float cZero;								///<Initial speed of sound.
    float YieldStress;                                                       ///<Initial yield stress for J2 plasticity.
    float Hardening;                                                         ///<Isotropic hardening modulus for J2 plasticity.
	float kernelh;
	float kernelsize;
	TpConstitModel ConstitModel;				///<Constitutive model.
	float AvFactor1;								///<Artificial viscousity factor1.
    float AvFactor2;								///<Artificial viscousity factor2.

	void Reset();

	//-Data included by Naqib
	bool Fracture;								///<Switch to model fracture in the deformable structure
	float Gc;									///<Critical energy release rate.
	unsigned MapFact;							///<Map factor for particle discretization.
	unsigned NBsrange;							///<Restrict nieghbour search to nbsrange number of particle
	float PfLim;								///<Phase field limit, used for clearing the surface of the crack.
	double LenScale;							///<Phase field Length Scale, LenScale = PfLenFact * Dp
	double Dp;									///<Particle distancing for mapped domain, different for each defstruc body
	
	unsigned NvBC;									///<Number of velocity boundary conditions.
	unsigned NfBC;									///<Number of force boundary conditions.
	unsigned NphiBC;								///<Number of phi boundary conditions.
	unsigned Nnotch;								///<Number of preexisting notches.
	unsigned Nmeaspl;								///<Number of measuring planes.
	tbcstrucbody BodyVBClist[MAX_VELBC_DEFSTRUC];	///<List of velocity boundary conditions of the body [m/s].
	tbcstrucbody BodyFBClist[MAX_FORCEBC_DEFSTRUC];	///<List of force boundary conditions [surface: N/m^2(3D)|N/m(2D), body: N/m^3(3D)|N/m^2(2D), point: N].
	tbcstrucbody BodyPhiBClist[MAX_PHIBC_DEFSTRUC];	///<List of phi boundary conditions.
	plane4Nstruc BodyNList[MAX_NOTCH_DEFSTRUC];		///<List of preexisting notches in the body
	plane4Nstruc BodyMPList[MAX_MEASUREPLANE_DEFSTRUC];		///<List of measuring planes in the body

public:
	const unsigned IdBody;							///<Deformable structure ID.
	const word MkBound;								///<MkBound of deformable structure.

	JSphDeformStrucBody(const double dpb, const bool simulate2d, unsigned idbody, word mkbound, float density, double youngmod, double poissratio, TpConstitModel constitmodel, float zesfactor1, float zesfactor2,
		bool fracturei, float Gc, unsigned dxmfact, float pflimit, float pflfact, float restcoef, float kfric,
		double usrlambda, double usrMu, double usrBulk, double yieldstress, double hardening,
		tbcstrucbody* bcvel, unsigned nmbcvel, tbcstrucbody* bcforce, unsigned nmbcforce, tbcstrucbody* bcphi, unsigned nmbcphi,
		plane4Nstruc* notchlist, unsigned nmnotch, plane4Nstruc* measplist, unsigned nmmeasp, unsigned nbsrangei);
	~JSphDeformStrucBody();
	void calc_elastic_const(bool simulate2d, double usrlambda, double usrMu, double usrBulk, double youngmod, double poissratio);
	void ConfigBoundCode(typecode boundcode);

	void GetConfig(std::vector<std::string>& lines)const;
	void ConfigBCCodeBody(const JSphMk* mkinfo);

	typecode GetBoundCode()const { return(BoundCode); }
	TpConstitModel GetConstModel() const { return(ConstitModel); }

	inline float GetPartVol()const { return PartVol; };
	inline float GetDensity()const { return Density; };
	inline float GetYoungMod()const { return YoungMod; };
    inline float GetPoissRatio()const { return PoissRatio; };
    inline float GetLameMu()const { return LameMu; };
    inline float GetLameLmbda()const { return LameLmbda; };
    inline float GetLameBulk()const { return LameBulk; };
    inline float GetSoundSpeed()const { return cZero; };
    inline float GetYieldStress()const { return YieldStress; };
    inline float GetHardening()const { return Hardening; };
    inline float GetAvFactor1()const { return AvFactor1; };
    inline float GetAvFactor2()const { return AvFactor2; };
    inline float GetGc()const { return Gc; };
	inline float GetPfLim()const { return PfLim; };
	inline double GetDp()const { return Dp; };
	inline float GetKernelh()const { return kernelh; };
	inline void SetKernelh(float v) { kernelh = v; };
	inline float GetKernelsize()const { return kernelsize; };
	inline void SetKernelsize(float v) { kernelsize = v; };
	inline double GetLenScale()const { return LenScale; };
	inline bool GetFracture()const { return Fracture; };
	
	inline unsigned GetMapfact()const { return MapFact; };
	inline unsigned GetNBsfact()const { return NBsrange; };

	inline float GetKFric()const { return KFric; };
	inline float GetRestCoeff()const { return RestCoeff; };
	inline unsigned GetNvBC()const { return NvBC; };
	inline unsigned GetNfBC()const { return NfBC; };
	inline unsigned GetNphiBC()const { return NphiBC; };
	inline unsigned GetNnotch()const { return Nnotch; };
	inline unsigned GetNMeasPlane()const { return Nmeaspl; };
	inline const tbcstrucbody* GetVBClist()const { return BodyVBClist; };
	inline const tbcstrucbody* GetFBClist()const { return BodyFBClist; };
	inline const tbcstrucbody* GetPhiBClist()const { return BodyPhiBClist; };
	inline const plane4Nstruc* GetNotchList()const { return BodyNList; };
	inline const plane4Nstruc* GetMeasPlaneList()const { return BodyMPList; };
	inline float GetPartMass()const { return GetPartVol() * GetDensity(); };

};

//##############################################################################
//# JSphDeformStruc
//##############################################################################
/// \brief Manages the info of deformable structures.
class JSphDeformStruc : protected JObject
{
private:
	JLog2* Log;

	void Reset();
	
	bool ExistMk(word mkbound)const;
	void LoadXml(const JXml* sxml, const std::string& place, const bool simulate2d, const double dporg);
	void ReadXml(const JXml* sxml, TiXmlElement* lis, const bool simulate2d, const double dporg);
	void ConfigBoundCode(const JSphMk* mkinfo);

public:
	double UseUsrTimeStep;						///<Time step value from input file
	float ContPowerCoeff;						///<Coefficient to multiply contact power with ge 1.0
	std::vector<JSphDeformStrucBody*> List;     ///<List of deformable structure bodies.

	void DSConfigCode(const unsigned Npb, typecode* Codec, const JSphMk* MkInfo);
	JSphDeformStruc(bool simulate2d, double dp, JXml* sxml, const std::string& place, const JSphMk* mkinfo);
	~JSphDeformStruc();

	void VisuConfig(std::string txhead, std::string txfoot);
	unsigned GetCount()const { return(unsigned(List.size())); }
	const JSphDeformStrucBody* GetBody(unsigned idx)const { return(idx < GetCount() ? List[idx] : NULL); }
	
	/// Returns true if any deformable structure body has phi boundary conditions defined.
	bool HasPhiBC() const {
		for (unsigned c = 0; c < GetCount(); c++)
			if (List[c]->GetNphiBC() > 0) return true;
		return false;
	}

	void CheckUserExpForBC(JUserExpressionList* UserExpressions);
	double GetInitialDtMin();
	void DSSaveInitDomainInfo(const bool simulate2d, const int Np, const int dsnpsurf, const int casendeformstruc,
		const unsigned* dsparent, const tbcstruc* dspartvbc, const tbcstruc* dspartfbc,
		const typecode* dscodec, const tdouble3* dspos0, const tdouble3* dsvel,
		const unsigned* dspairn, const unsigned* dspairstart, const unsigned* dspairj, const double* dsker,
		const tdouble3* dskerderv0, const double* dskerlapc, const float* dskersumvol, const unsigned* dssurfpartlist,
		const unsigned* deformstrucridp, const unsigned* dsbestchild, const unsigned* dsibodyridp,
		const unsigned* dsmeaspartlist, const unsigned* dsmeasplpartnum, const unsigned dsmeasplnum,
		const StDeformStrucIntData* defstrucintdata, const StDeformStrucData* deformstrucdata, const std::string dirout,
		JUserExpressionList* UserExpressions)const;
};


namespace measplane
{
    // Your project already has tdouble3, TDouble3 and fmath::* utilities.

    struct PlaneEq {
        tdouble3 n; // unit normal
        double   d; // plane offset s.t. dot(n,x)+d = 0
    };

    struct QuadProjector {
        tdouble3 p0, n, u, v;
        double d;

        struct d2 { double x, y; };
        d2 c2[4];                               // 2D corners in projector space
        double minx, maxx, miny, maxy;          // AABB for quick reject
        bool ccw;                                // winding of projected corners
    };

    // --- Plane from 4 corners (robust against slightly non-coplanar inputs) ---
     PlaneEq plane_from_4(const tdouble3* corner);

    // Signed distance: >0 is in front of plane along +n
     double point_plane_signed_distance(const PlaneEq& pl, const tdouble3& x);

    // --- Order 4 corners into a proper loop (CCW in a temporary projection) ---
     void order_corners_loop(const tdouble3& n_hint,
        const tdouble3* in4,
        tdouble3* out4);

    // --- Build projector (orthonormal in-plane basis + 2D corners, AABB, winding) ---
     QuadProjector build_quad_projector(const PlaneEq& pl, const tdouble3* corner_looped);

    // Project arbitrary point into projector’s 2D space
     QuadProjector::d2 project_point_onto_quad(const QuadProjector& P, const tdouble3& x);

    // Half-space edge test for convex quad in 2D
     bool point_inside_convex_quad2D(const QuadProjector& P,
        const QuadProjector::d2& q,
        double eps);
	template<class PosT>
	void DSfindPointsNearestToMeasurePlanes(const StDeformStrucData* deformstrucdata,
		unsigned DeformStrucCount, unsigned& DSNMeasPlanes, unsigned& DSNPartMeasPlanes,
		llong& MemCpuFixed, std::vector<std::string>& DSMeasPlOutFiles,
		unsigned*& DSMeasPlnCnt, const typecode* DSCode, const PosT* DSPos0,
		unsigned*& DSMeasPlnPart, int MapNdeformstruc, JLog2* Log, bool simulate2d=false)
	{
		// Reset counters
		DSNMeasPlanes = 0;
		DSNPartMeasPlanes = 0;

		// Count total planes
		for (unsigned bodyid = 0; bodyid < DeformStrucCount; ++bodyid)
			DSNMeasPlanes += deformstrucdata[bodyid].nmeasplane;

		// Handle no planes case
		if (DSNMeasPlanes < 1) {
			if (DSMeasPlnCnt) { delete[] DSMeasPlnCnt;  DSMeasPlnCnt = nullptr; }
			if (DSMeasPlnPart) { delete[] DSMeasPlnPart; DSMeasPlnPart = nullptr; }
			DSMeasPlOutFiles.clear();
			return;
		}

		// Precompute plane equations and ordered corners
		std::vector<measplane::PlaneEq> peq(DSNMeasPlanes);
		std::vector<std::array<tdouble3, 4>> corners4_ordered(DSNMeasPlanes);
		std::vector<unsigned> planeBodyId(DSNMeasPlanes, unsigned(-1));

		// (Re)allocate per-plane counters and output filenames
		if (DSMeasPlnCnt) { delete[] DSMeasPlnCnt; DSMeasPlnCnt = nullptr; }
		DSMeasPlnCnt = new unsigned[DSNMeasPlanes];
		MemCpuFixed += sizeof(unsigned) * static_cast<size_t>(DSNMeasPlanes);
		DSMeasPlOutFiles.assign(DSNMeasPlanes, std::string{});

		// Fill plane data
		unsigned p = 0;
		for (unsigned bodyid = 0; bodyid < DeformStrucCount; ++bodyid) {
			const StDeformStrucData& body = deformstrucdata[bodyid];
			for (unsigned p2 = 0; p2 < body.nmeasplane; ++p2) {
				tdouble3 corners[4];
				for (int k = 0; k < 4; ++k) corners[k] = body.measplanelist[p2].corners[k];
				peq[p] = measplane::plane_from_4(corners);

				tdouble3 ordered[4];
				measplane::order_corners_loop(peq[p].n, corners, ordered);
				for (int k = 0; k < 4; ++k) corners4_ordered[p][k] = ordered[k];

				DSMeasPlnCnt[p] = 0;
				planeBodyId[p] = bodyid;
				++p;
			}
		}

		// Build projectors
		std::vector<measplane::QuadProjector> proj(DSNMeasPlanes);
		for (unsigned i = 0; i < DSNMeasPlanes; ++i)
			proj[i] = measplane::build_quad_projector(peq[i], corners4_ordered[i].data());

		// Collect per-plane particle ids
		std::vector<std::vector<unsigned>> perPlaneCollected(DSNMeasPlanes);
		unsigned Ntot = 0;

		int OmpThreads = 1;
#ifdef OMP_USE
		OmpThreads = omp_get_max_threads();
#endif

		for (unsigned ip = 0; ip < DSNMeasPlanes; ++ip) {
			const auto& pl = peq[ip];
			const auto& P = proj[ip];
			const unsigned planeBody = planeBodyId[ip];
			if (planeBody >= DeformStrucCount) {
				DSMeasPlnCnt[ip] = 0;
				continue;
			}
			const StDeformStrucData& body = deformstrucdata[planeBody];
			const double halfThickness = body.dp;
			if (!(halfThickness > 0.0)) {
				DSMeasPlnCnt[ip] = 0;
				continue;
			}
			const double layerTol = min(0.0000001 * body.dp, halfThickness);

			std::vector<std::vector<std::pair<unsigned, double>>> tls(static_cast<size_t>(OmpThreads));

#ifdef OMP_USE
#pragma omp parallel
#endif
			{
				int tid = 0;
#ifdef OMP_USE
				tid = omp_get_thread_num();
#endif
				auto& buf = tls[static_cast<size_t>(tid)];
				buf.clear();
				buf.reserve(5120);

#ifdef OMP_USE
#pragma omp for schedule(static)
#endif
				for (int p1 = 0; p1 < MapNdeformstruc; ++p1) {
					const unsigned bodyid = CODE_GetIbodyDeformStruc(DSCode[p1]);
					if (bodyid != planeBody) continue;

					const tdouble3 posp1 = TDouble3(DSPos0[p1].x, DSPos0[p1].y, DSPos0[p1].z);
					const double s = measplane::point_plane_signed_distance(pl, posp1);
					const double dist = std::abs(s);
					//if (!(dist < halfThickness)) continue;
					if (s < -halfThickness || s >= 0.0) continue;
					const auto q2 = measplane::project_point_onto_quad(P, posp1);
					if (measplane::point_inside_convex_quad2D(P, q2, ALMOSTZERO))
						buf.emplace_back(static_cast<unsigned>(p1), dist);
				}
			} // parallel

			std::size_t candidateCount = 0;
			for (const auto& v : tls) candidateCount += v.size();

			if (!candidateCount) {
				DSMeasPlnCnt[ip] = 0;
				continue;
			}

			double bestDist = DBL_MAX;
			for (const auto& v : tls) {
				for (const auto& cand : v) bestDist = min(bestDist, cand.second);
			}

			if (!(bestDist < DBL_MAX)) {
				DSMeasPlnCnt[ip] = 0;
				continue;
			}

			const double acceptDist = min(bestDist + layerTol, halfThickness);

			auto& target = perPlaneCollected[ip];
			target.clear();
			target.reserve(candidateCount);
			for (const auto& v : tls) {
				for (const auto& cand : v) {
					if (cand.second <= acceptDist) target.push_back(cand.first);
				}
			}

			const unsigned countP = static_cast<unsigned>(target.size());
			DSMeasPlnCnt[ip] = countP;
			Ntot += countP;
		}

		// Flatten into DSMeasPlnPart
		if (DSMeasPlnPart) { delete[] DSMeasPlnPart; DSMeasPlnPart = nullptr; }
		DSMeasPlnPart = (Ntot ? new unsigned[Ntot] : nullptr);
		MemCpuFixed += sizeof(unsigned) * static_cast<size_t>(Ntot);

		unsigned written = 0;
		for (unsigned ip = 0; ip < DSNMeasPlanes && written < Ntot; ++ip) {
			const auto& blk = perPlaneCollected[ip];
			const unsigned able = std::min<unsigned>(static_cast<unsigned>(blk.size()), Ntot - written);
			if (able) {
				std::copy(blk.begin(), blk.begin() + able, DSMeasPlnPart + written);
				written += able;
			}
		}

		DSNPartMeasPlanes = Ntot;
		Log->Print(std::string("  Measuring planes and particles set for ") + fun::IntStr(Ntot) + std::string(" particles"));
	}


}

namespace defstrucbc
{
	void computeBestFitPlane(const std::vector<tdouble3>& pts, tdouble3& origin, tdouble3& normal, const bool simulate2D);
	void computeLocalAxes(const tdouble3& planeNormal, tdouble3& axis1, tdouble3& axis2);
	tdouble2 projectPointToLocal(const tdouble3& p, const tdouble3& origin, const tdouble3& axis1, const tdouble3& axis2);
	std::vector<tdouble2> computeConvexHull2D(std::vector<tdouble2> points, const double offsetDistance);
	void computeConvexHull(const std::vector<tdouble3>& pts,
		const tdouble3& origin, const tdouble3& axis1, const tdouble3& axis2,
		std::vector<tdouble2>& polygon, const double offsetDistance);
	bool pointInRegion2D(double u, double v,
		const std::vector<tdouble2>& regionPolygon,
		double tol, const double almostzero);
	void prepareBCRegion(tbcregion& region, const double offsetDistance, const double almostzero, const bool simulate2d);
	void SetBoundaryConditions( const bool simulate2D, const unsigned deformStrucCount, StDeformStrucData* deformstrucdata,
		const unsigned mapNdeformstruc, const unsigned npb, const unsigned dsnpsurf, const typecode* dscode, const typecode* codec, 
		const float* dsKerSumVol, const unsigned* dssurfpartlist, tbcstruc* dspartfbc, tbcstruc* dspartvbc, const std::function<tdouble3(unsigned)>& getPos0, 
		const std::function<tdouble3(unsigned)>& getPosc, const double dp, unsigned& outCount, JUserExpressionList* UserExpressions
	);
}
#endif
