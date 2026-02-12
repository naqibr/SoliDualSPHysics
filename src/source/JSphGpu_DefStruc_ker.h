
#ifndef _JSphGpu_DefStruc_ker_
#define _JSphGpu_DefStruc_ker_

#include "DualSphDef.h"
#include "TypesDef_GPU.h"
#include "JCellDivDataGpu.h"
#include "JCellDivGpuSingle.h"
#include "FunctionsBasic_iker.h"
#include "JSphGpu_ExpressionParser.h"

namespace cudefstr {

	float ReduMinFloat(unsigned ndata, unsigned inidata, float* data, float* resu);
	float ReduMinFloat_w(unsigned ndata, unsigned inidata, float4* data, float* resu);

	bool DSHasNormalsg(unsigned npb, const typecode* code, const float3* boundnormals);
	unsigned DSCountOrgParticlesg(unsigned npb, const typecode* code);
	void DSCalcRidpg(unsigned npb, unsigned* deformstrucridp, const typecode* code);

	int DSCountMappedParticlesg(unsigned casendeformstruc, bool simulate2D, unsigned* deformstrucridpg, typecode* codeg, StDeformStrucData* deformstrucdatag);

	void DSGenMappedParticles(const unsigned casendeformstruc, const unsigned* deformstrucridpg,
		const typecode* codeg, StDeformStrucData* deformstrucdatag, float4* dspos0, float2* dsposorg0xyg, float* dsposorg0zg,
		unsigned* dsparentg, typecode* dscodeg, bool simulate2D);
	//void DSCalcIbodyRidp(const unsigned deformstruccount, const StDeformStrucData* deformstrucdatag,
	//	const int mapndeformstruc, const typecode* dscodeg, unsigned* dsibodyridpg);

	void DSDetermineMapCenters(const unsigned casendeformstruc, const unsigned* DeformStrucRidpg,
		const float2* DSPosOrg0xyg, const float* DSPosOrg0zg, const typecode* Codeg,
		const StDeformStrucData* DeformStrucDatag, const StDivDataGpu DSDivData, const typecode* DSCodeg,
		const float4* DSPos0g, const unsigned* DSDcellg, unsigned* DSBestChildg);

	unsigned DSCountTotalPairs(const int MapNdeformstruc, const float4* DSPos0, const unsigned* DSParent,
		const typecode* Codec, const StDeformStrucData* DeformStrucData, uint2* DSPairN,
		const StDivDataGpu DSDivData, const bool simulate2d);

	void DSCalcKers(const int MapNdeformstruc, const TpKernel TKernel, const bool Simulate2D, const typecode* DSCodeg,
		const float4* DSPos0g, const StDeformStrucData* DeformStrucDatag, const StDivDataGpu DSDivData, const unsigned* DSDcellg,
		uint2* DSPairNSg, float* DSKerg, float4* DSKerDerLapg,
		unsigned* DSPairJg, float* DSKerSumVolg);

	template<bool Simulate2D> void DSCalcKers_ct(const int MapNdeformstruc, const TpKernel TKernel, const typecode* DSCodeg,
		const float4* DSPos0g, const StDeformStrucData* DeformStrucDatag, const StDivDataGpu DSDivData, const unsigned* DSDcellg,
		uint2* DSPairNg, float* DSKerg, float4* DSKerDerLapg,
		unsigned* DSPairJg, float* DSKerSumVolg);

	void DSFindSurfParticles(const int MapNdeformstruc, const float4* DSPos0g,
		const typecode* DSCodeg, const uint2* DSPairNSg, const unsigned* DSPairJg, const float* DSKerg,
		const StDeformStrucData* DeformStrucDatag, int& DSNpSurf, unsigned*& DSSurfPartList, llong& MemGpuFixed);

	void DSInitFieldVars(const TpStep TStep, const int MapNdeformstruc, const typecode* DSCodeg,
		const StDeformStrucData* DeformStrucDatag, const tbcstruc* DSPartVBCg, tbcstruc* DSPartFBCg, tphibc* DSPartPhiBCg, const float* DSKerSumVolg,
		float4* DSAcclg, float3* DSFlForceg, float4* DSDispPhig, float* DSEqPlasticg, tmatrix3f* DSPlasticStraing, float4* DSDefGradg2D, float4* DSDefGradg3D,
		float2* DSDefPk, float4* DSVelg, float4* DSVelPreg, const float4* DSPos0g, float4* DSPhiTdatag,
		float2* DSDispCorxzg, float* DSDispCoryg, bool Simulate2D, bool DSFrac,
		JUserExpressionListGPU* UserExpressionsg, const float TimeStep, const tfloat3 Gravity);

	double DSCalcMaxInitTimeStep(const int MapNdeformstruc, const unsigned DeformStrucCount, const typecode* DSCodeg,
		StDeformStrucData* DeformStrucDatac, const StDeformStrucData* DeformStrucDatag, const uint2* DSPairNSg,
		const float4* DSPos0g, const float4* DSDispPhig, const float4* DSVelg, const float4* DSAcclg, const tbcstruc* DSPartFBCg,
		const tfloat3 Gravity, const unsigned* DSPairJg, JUserExpressionListGPU* UserExpressionsg, const float TimeStep);

	float DSInteraction_Forces(const int np, unsigned bodycnt, const StDeformStrucIntData* DeformStrucDatag,
		const StDeformStrucIntArraysg DSIntArraysg, unsigned* DSDcellg, tfloat3 gravity, bool simulate2d,
		const tdouble3 DomRealPosMin, const tdouble3 DomRealPosMax, const tdouble3 DSDomPosMin, const float DScellsize, 
		const unsigned DSDomCellCode, StDivDataGpu DSDivData, JCellDivGpuSingle* DSCellDivSingle, JDsTimersGpu* Timersg,
		const int DSNpSurf, const unsigned* DSSurfPartListg, const float* DSKerSumg, JUserExpressionListGPU* UserExpressionsg,
		const float3* DSFlForceg, const float dstime, const float dsdt);
	void DSCompSemImplEuler(const int np, unsigned bodycnt, const float DSStepDt,
		const StDeformStrucIntData* DeformStrucDatag, const StDeformStrucIntArraysg DSIntArraysg, bool simulate2d,
		JUserExpressionListGPU* UserExpressionsg, const float dstime);
	void DSCompSympPre(const int np, unsigned bodycnt, const float dtm,
		const StDeformStrucIntData* DeformStrucDatag, const StDeformStrucIntArraysg DSIntArraysg, bool simulate2d,
		JUserExpressionListGPU* UserExpressionsg, const float dstime);
	void DSCompSympCor(const int np, unsigned bodycnt, const float dt,
		const StDeformStrucIntData* DeformStrucDatag, const StDeformStrucIntArraysg DSIntArraysg, bool simulate2d,
		JUserExpressionListGPU* UserExpressionsg, const float dstime);

	void DSUpdate_OrgVelPos(unsigned CaseNdeformstruc, const unsigned* DeformStrucRidpg,
		const unsigned* DSBestChildg, const float2* DSPosOrg0xyg, const float* DSPosOrg0zg,
		const float4* DSDispPhig, double2* Posxyg, double* Poszg, const bool simulate2d,
		float4* Velrhopg, float4* DSVelg);
	
	void DSInteractionForcesDEM(bool simulate2d, unsigned MapNdeformstruc, unsigned bodycnt,
		const StDeformStrucIntData* DeformStrucDatag,
		const StDeformStrucIntArraysg arrays, StDivDataGpu DivDatag, const typecode* Codeg,
		const double2* Posxyg, const double* Poszg, const float4* velrhop, float3* DSFlForceg,
		const float Dp, const float dtdem, const unsigned* DSSurfPartListg, const float DSContPowerCoeff);

	void DSCalcEnergiesCauchyStress(const int np, unsigned bodycnt, const StDeformStrucIntData* DeformStrucDatag,
		const StDeformStrucIntArraysg DSIntArraysg, const bool simulate2d, tfloat3* energies, tsymatrix3f* dscauchystress,
		float* __restrict__ DSKerSumVolg, tfloat3* particleenergies = nullptr);

	template<bool simulate2d>
	__device__ tfloat3 DScalcenergiesp1(const matrix3f defgrad, const float4 velp1,
		const float phfp1, const uint2 pairns, const StDeformStrucIntArraysg arrays,
		const float eqplastic, const  tmatrix3f plastic_dev, const StDeformStrucIntData body);


	void CheckExpressionValues(const JUserExpressionListGPU* d_expr_list, float* d_results, const int expression_id,
		const float time, const float dt, int num_particles, float dp);
	void ValidateExpressions(JUserExpressionListGPU* expr_list, JUserExpressionList* UserExpressions, const int expid, const float time, StDeformStrucData* defstrucdatac);
}
#endif