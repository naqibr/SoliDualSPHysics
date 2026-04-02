//HEAD_DSPH
/*
 <DUALSPHYSICS>  Copyright (c) 2020 by Dr Jose M. Dominguez et al. (see http://dual.sphysics.org/index.php/developers/). 

 EPHYSLAB Environmental Physics Laboratory, Universidade de Vigo, Ourense, Spain.
 School of Mechanical, Aerospace and Civil Engineering, University of Manchester, Manchester, U.K.

 This file is part of DualSPHysics. 

 DualSPHysics is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License 
 as published by the Free Software Foundation; either version 2.1 of the License, or (at your option) any later version.
 
 DualSPHysics is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details. 

 You should have received a copy of the GNU Lesser General Public License along with DualSPHysics. If not, see <http://www.gnu.org/licenses/>. 
*/

/// \file JSphCpuSingle.h \brief Declares the class \ref JSphCpuSingle.

#ifndef _JSphCpuSingle_
#define _JSphCpuSingle_

#include "DualSphDef.h"
#include "JSphCpu.h"
#include "JSphDeformStruc.h"  //<vs_deformstruc>
//##############################################################################
//# JSphCpuSingle
//##############################################################################
/// \brief Defines the attributes and functions used only in Single-Core implementation.

class JSphCpuSingle : public JSphCpu
{
protected:
  JCellDivCpuSingle* CellDivSingle;
  JCellDivCpuSingle* DSCellDivSingle;
  
  llong GetAllocMemoryCpu()const;
  void UpdateMaxValues();
  void LoadConfig(const JSphCfgRun *cfg);
  void ConfigDomain();

  void ResizeParticlesSize(unsigned newsize,float oversize,bool updatedivide);
  unsigned PeriodicMakeList(unsigned np,unsigned pini,bool stable,unsigned nmax,tdouble3 perinc,const tdouble3 *pos,const typecode *code,unsigned *listp)const;
  void PeriodicDuplicatePos(unsigned pnew,unsigned pcopy,bool inverse,double dx,double dy,double dz,tuint3 cellmax,tdouble3 *pos,unsigned *dcell)const;
  void PeriodicDuplicateVerlet(unsigned np,unsigned pini,tuint3 cellmax,tdouble3 perinc,const unsigned *listp
    ,unsigned *idp,typecode *code,unsigned *dcell,tdouble3 *pos,tfloat4 *velrhop,tsymatrix3f *spstau,tfloat4 *velrhopm1)const;
  void PeriodicDuplicateSymplectic(unsigned np,unsigned pini,tuint3 cellmax,tdouble3 perinc,const unsigned *listp
    ,unsigned *idp,typecode *code,unsigned *dcell,tdouble3 *pos,tfloat4 *velrhop,tsymatrix3f *spstau,tdouble3 *pospre,tfloat4 *velrhoppre)const;
  void PeriodicDuplicateNormals(unsigned np,unsigned pini,tuint3 cellmax
    ,tdouble3 perinc,const unsigned *listp,tfloat3 *motionvel,tfloat3 *normals)const;
  void RunPeriodic();

  void RunCellDivide(bool updateperiodic);
  void DSRunCellDivide();
  void DSRunSurfCellDivide(unsigned DSNpSurf, unsigned* DSDcellSurfc);

  void AbortBoundOut();
  void Interaction_Forces(TpInterStep tinterstep);
  void MdbcBoundCorrection();

  double ComputeAceMax()const;
  template<bool checkcode> inline double ComputeAceMaxSeq(unsigned np,const tfloat3* ace,const typecode *code)const;
  template<bool checkcode> inline double ComputeAceMaxOmp(unsigned np,const tfloat3* ace,const typecode *code)const;
  
  void RunInitialDDTRamp(); //<vs_ddramp>
 
  template<bool rundefstruc, bool simulate2d, bool defsttm, TpKernel tkernel> double ComputeStep(){
      return(TStep==STEP_Verlet? 
          ComputeStep_Ver<rundefstruc, simulate2d, defsttm, tkernel>():
          ComputeStep_Sym<rundefstruc, simulate2d, defsttm, tkernel>()); }
  template<bool rundefstruc, bool simulate2d, bool defsttm, TpKernel tkernel> double ComputeStep_Sym();
  template<bool rundefstruc, bool simulate2d, bool defsttm, TpKernel tkernel> double ComputeStep_Ver();

  inline tfloat3 FtPeriodicDist(const tdouble3 &pos,const tdouble3 &center,float radius)const;
  void FtCalcForcesSum(unsigned cf,tfloat3 &face,tfloat3 &fomegaace)const;
  void FtCalcForces(StFtoForces *ftoforces)const;
  void FtCalcForcesRes(double dt,const StFtoForces *ftoforces,StFtoForcesRes *ftoforcesres)const;
  void FtApplyImposedVel(StFtoForcesRes *ftoforcesres)const;
  void FtApplyConstraints(StFtoForces *ftoforces,StFtoForcesRes *ftoforcesres)const;
  void RunFloating(double dt,bool predictor);
  void RunGaugeSystem(double timestep,bool saveinput=false);

  void ComputePips(bool run);
  
  void SaveData();
  void SaveExtraData();
  void FinishRun(bool stop);


  // Deformable structure functions
  void DSPreTimeInt();
  bool DSHasNormals(unsigned npb, const typecode* code, const tfloat3* boundnormals)const;
  unsigned DSCountOrgParticles(unsigned npb, const typecode* code)const;
  void DSCalcRidp(unsigned npb, unsigned* defstrucridp, const typecode* code)const;
  unsigned DSCountMappedParticles(StDeformStrucData* deformstrucdata) const;
  void DSGenMappedParticles(StDeformStrucData* deformstrucdata) const;
  void DSPerformCellDiv(StDeformStrucData* deformstrucdata);
  void DSCalcIbodyRidp(StDeformStrucData* deformstrucdata) const;
  void DSDetermineMapCenters(StDeformStrucData* deformstrucdata)const;
  void DSFindSurfParticles();
  void DSSetBoundCond(StDeformStrucData* deformstrucdata)const;
  void DSInitFieldVars(StDeformStrucData* deformstrucdata, const double currenttime);
  //void DSfindPointsNearestToMeasurePlanes(StDeformStrucData* deformstrucdata);
  void DSCalcMaxInitTimeStep(StDeformStrucData* deformstrucdata);
  void DSCompSympPre(const double dtm);
  void DSCompSympCor(const double dtm, const double currenttime);
  void DSCompSemImplEuler(double dt, const double currenttime);
  template<bool simulate2d, bool defsttm> void DSComputeStep_Sym(const double dt, double starttime);
  template<bool simulate2d, bool defsttm> void DSComputeStep_Ver(const double dt);
  template<bool simulate2d> 
  void DSInteraction_Forces(const double currenttime, const double dsdt);
  void DSInteractionForcesDEM(const double dtdem)const;
public:
  JSphCpuSingle();
  ~JSphCpuSingle();
  void Run(std::string appname,const JSphCfgRun *cfg,JLog2 *log);
  template<bool rundefstruc, bool simulate2d, bool defsttm, TpKernel tkernel> void TimeLoop(bool& partoutstop);
  template<bool simulate2d, bool defsttm, TpKernel tkernel> void TimeLoop_c2(bool& partoutstop);
  template<bool simulate2d, bool defsttm> void TimeLoop_c1(bool& partoutstop);
  template<bool simulate2d> void TimeLoop_c0(bool& partoutstop);
//-Code for InOut in JSphCpuSingle_InOut.cpp
//--------------------------------------------
protected:
  void InOutInit(double timestepini);
  void InOutIgnoreFluidDef(const std::vector<unsigned> &mkfluidlist);
  void InOutCheckProximity(unsigned newnp);
  void InOutComputeStep(double stepdt);
  void InOutUpdatePartsData(double timestepnew);
  void InOutExtrapolateData(unsigned inoutcount,const int *inoutpart);
};


#endif


