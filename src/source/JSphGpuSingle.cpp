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

/// \file JSphGpuSingle.cpp \brief Implements the class \ref JSphGpuSingle.

#include "JSphGpuSingle.h"
#include "JCellDivGpuSingle.h"
#include "JArraysGpu.h"
#include "JSphMk.h"
#include "JPartsLoad4.h"
#include "Functions.h"
#include "JXml.h"
#include "JDsMotion.h"
#include "JDsViscoInput.h"
#include "JWaveGen.h"
#include "JMLPistons.h"
#include "JRelaxZones.h"
#include "JChronoObjects.h"
#include "JDsMooredFloatings.h"
#include "JDsFtForcePoints.h"
#include "JDsOutputTime.h"
#include "JTimeControl.h"
#include "JSphGpu_ker.h"
#include "JSphGpuSimple_ker.h"
#include "JDsGaugeSystem.h"
#include "JSphInOut.h"
#include "JSphDeformStruc.h"  //<vs_deformstruc>
#include "JFtMotionSave.h"  //<vs_ftmottionsv>
#include "JLinearValue.h"
#include "JDataArrays.h"
#include "JDebugSphGpu.h"
#include "JSphShifting.h"
#include "JDsPips.h"
#include "JDsExtraData.h"
#include "FunctionsCuda.h"

#include <climits>

#include <chrono>
using namespace std;
//==============================================================================
/// Constructor.
//==============================================================================
JSphGpuSingle::JSphGpuSingle() :JSphGpu(false) {
	ClassName = "JSphGpuSingle";
	CellDivSingle = NULL;
}

//==============================================================================
/// Destructor.
//==============================================================================
JSphGpuSingle::~JSphGpuSingle() {
	DestructorActive = true;
	delete CellDivSingle; CellDivSingle = NULL;
}

//==============================================================================
/// Returns the memory allocated to the CPU.
/// Devuelve la memoria reservada en CPU.
//==============================================================================
llong JSphGpuSingle::GetAllocMemoryCpu()const {
	llong s = JSphGpu::GetAllocMemoryCpu();
	//-Allocated in other objects.
	if (CellDivSingle)s += CellDivSingle->GetAllocMemoryCpu();
	return(s);
}

//==============================================================================
/// Returns the memory allocated to the GPU.
/// Devuelve la memoria reservada en GPU.
//==============================================================================
llong JSphGpuSingle::GetAllocMemoryGpu()const {
	llong s = JSphGpu::GetAllocMemoryGpu();
	//-Allocated in other objects.
	if (CellDivSingle)s += CellDivSingle->GetAllocMemoryGpu();
	return(s);
}

//==============================================================================
/// Returns the GPU memory allocated or used for particles
/// Devuelve la memoria GPU reservada o usada para particulas.
//==============================================================================
llong JSphGpuSingle::GetMemoryGpuNp()const {
	llong s = JSphGpu::GetAllocMemoryGpu();
	//-Allocated in other objects.
	if (CellDivSingle)s += CellDivSingle->GetAllocMemoryGpuNp();
	return(s);
}

//==============================================================================
/// Returns the GPU memory allocated or used for cells.
/// Devuelve la memoria GPU reservada o usada para celdas.
//==============================================================================
llong JSphGpuSingle::GetMemoryGpuNct()const {
	llong s = CellDivSingle->GetAllocMemoryGpuNct();
	return(CellDivSingle->GetAllocMemoryGpuNct());
}

//==============================================================================
/// Updates the maximum values of memory, particles and cells.
/// Actualiza los valores maximos de memory, particles y cells.
//==============================================================================
void JSphGpuSingle::UpdateMaxValues() {
	const llong mcpu = GetAllocMemoryCpu();
	const llong mgpu = GetAllocMemoryGpu();
	MaxNumbers.memcpu = max(MaxNumbers.memcpu, mcpu);
	MaxNumbers.memgpu = max(MaxNumbers.memgpu, mgpu);
	MaxNumbers.particles = max(MaxNumbers.particles, Np);
	if (CellDivSingle)MaxNumbers.cells = max(MaxNumbers.cells, CellDivSingle->GetNct());
}

//==============================================================================
/// Loads the configuration of the execution.
/// Carga la configuracion de ejecucion.
//==============================================================================
void JSphGpuSingle::LoadConfig(const JSphCfgRun* cfg) {
	//-Loads general configuration.
	JSph::LoadConfig(cfg);
	if (UserExpressions) {
		JUserExpressionListGPU* h_UserExpressionsg = new JUserExpressionListGPU(*UserExpressions);
		cudaMalloc((void**)&UserExpressionsg, sizeof(JUserExpressionListGPU)); MemGpuFixed += sizeof(JUserExpressionListGPU);
		cudaMemcpy(UserExpressionsg, h_UserExpressionsg, sizeof(JUserExpressionListGPU), cudaMemcpyHostToDevice);
	}
	//-Checks compatibility of selected options.
	Log->Print("**Special case configuration is loaded");
}

//==============================================================================
/// Configuration of the current domain.
/// Configuracion del dominio actual.
//==============================================================================
void JSphGpuSingle::ConfigDomain() {
	//-Configure cell map division (defines ScellDiv, Scell, Map_Cells). 
	ConfigCellDivision();
	//-Computes the number of particles.
	Np = PartsLoaded->GetCount(); Npb = CaseNpb; NpbOk = Npb;
	//-Allocates memory for arrays with fixed size (motion and floating bodies).
	AllocGpuMemoryFixed();
	//-Allocates GPU memory for particles.
	AllocGpuMemoryParticles(Np, 0);
	//-Allocates memory on the CPU.
	AllocCpuMemoryFixed();
	AllocCpuMemoryParticles(Np);

	//-Copies particle data.
	memcpy(AuxPos, PartsLoaded->GetPos(), sizeof(tdouble3) * Np);
	memcpy(Idp, PartsLoaded->GetIdp(), sizeof(unsigned) * Np);
	memcpy(Velrhop, PartsLoaded->GetVelRhop(), sizeof(tfloat4) * Np);

	//-Computes radius of floating bodies.
	if (CaseNfloat && PeriActive != 0 && !PartBegin)CalcFloatingRadius(Np, AuxPos, Idp);
	//-Configures floating motion data storage with high frequency. //<vs_ftmottionsv>  
	if (FtMotSave)ConfigFtMotionSave(Np, AuxPos, Idp);                 //<vs_ftmottionsv>  

	//-Configures Multi-Layer Pistons according particles. | Configura pistones Multi-Layer segun particulas.
	if (MLPistons)MLPistons->PreparePiston(Dp, Np, Idp, AuxPos);

	//-Loads Code of the particles.
	LoadCodeParticles(Np, Idp, Code);

	//-Load normals for boundary particles (fixed and moving).
	tfloat3* boundnormal = NULL;
	if (UseNormals) {
		boundnormal = new tfloat3[Np];
		LoadBoundNormals(Np, Npb, Idp, Code, boundnormal);
	}

	//-Creates PartsInit object with initial particle data for automatic configurations.
	CreatePartsInit(Np, AuxPos, Code);

	//-Runs initialization operations from XML.
	RunInitialize(Np, Npb, AuxPos, Idp, Code, Velrhop, boundnormal);
	if (UseNormals)ConfigBoundNormals(Np, Npb, AuxPos, Idp, boundnormal);

	//-Computes MK domain for boundary and fluid particles.
	MkInfo->ComputeMkDomains(Np, AuxPos, Code);

	//-Sets local domain of the simulation within Map_Cells and computes DomCellCode.
	//-Establece dominio de simulacion local dentro de Map_Cells y calcula DomCellCode.
	SelecDomain(TUint3(0, 0, 0), Map_Cells);
	//-Computes inital cell of the particles and checks if there are unexpected excluded particles.
	//-Calcula celda inicial de particulas y comprueba si hay excluidas inesperadas.
	LoadDcellParticles(Np, Code, AuxPos, Dcell);

	//-Uploads particle data on the GPU.
	ReserveBasicArraysGpu();
	for (unsigned p = 0; p < Np; p++) { Posxy[p] = TDouble2(AuxPos[p].x, AuxPos[p].y); Posz[p] = AuxPos[p].z; }
	ParticlesDataUp(Np, boundnormal);
	delete[] boundnormal; boundnormal = NULL;
	//-Uploads constants on the GPU.
	ConstantDataUp();

	//-Creates object for Celldiv on the GPU and selects a valid cellmode.
	//-Crea objeto para divide en GPU y selecciona un cellmode valido.
	CellDivSingle = new JCellDivGpuSingle(Stable, FtCount != 0, PeriActive, KernelSize2, PosCellSize
		, CellDomFixed, CellMode, Scell, Map_PosMin, Map_PosMax, Map_Cells, CaseNbound, CaseNfixed, CaseNpb, DirOut);
	CellDivSingle->DefineDomain(DomCellCode, DomCelIni, DomCelFin, DomPosMin, DomPosMax);
	ConfigCellDiv((JCellDivGpu*)CellDivSingle);

	ConfigBlockSizes(false, PeriActive != 0);
	ConfigSaveData(0, 1, "");

	//-Reorders particles according to cells.
	//-Reordena particulas por celda.
	BoundChanged = true;
	RunCellDivide(true);
}

//==============================================================================
/// Resizes the allocated space for particles on the CPU and the GPU measuring
/// the time spent with TMG_SuResizeNp. At the end updates the division.
///
/// Redimensiona el espacio reservado para particulas en CPU y GPU midiendo el
/// tiempo consumido con TMG_SuResizeNp. Al terminar actualiza el divide.
//==============================================================================
void JSphGpuSingle::ResizeParticlesSize(unsigned newsize, float oversize, bool updatedivide) {
	Timersg->TmStart(TMG_SuResizeNp, false);
	newsize += (oversize > 0 ? unsigned(oversize * newsize) : 0);
	FreeCpuMemoryParticles();
	CellDivSingle->FreeMemoryGpu();
	DivData = DivDataGpuNull();
	ResizeGpuMemoryParticles(newsize);
	AllocCpuMemoryParticles(newsize);
	Timersg->TmStop(TMG_SuResizeNp, false);
	if (updatedivide)RunCellDivide(true);
}

//==============================================================================
/// Creates duplicate particles for periodic conditions.
/// Creates new periodic particles and marks the old ones to be ignored.
/// The new particles are lccated from the value of Np, first the NpbPer for 
/// boundaries and then the NpfPer for the fluids. The Np output also contains 
/// the new periodic particles.
///
/// Crea particulas duplicadas de condiciones periodicas.
/// Crea nuevas particulas periodicas y marca las viejas para ignorarlas.
/// Las nuevas periodicas se situan a partir del Np de entrada, primero las NpbPer
/// de contorno y despues las NpfPer fluidas. El Np de salida contiene tambien las
/// nuevas periodicas.
//==============================================================================
void JSphGpuSingle::RunPeriodic() {
	Timersg->TmStart(TMG_SuPeriodic, false);
	//-Stores the current number of periodic particles.
	//-Guarda numero de periodicas actuales.
	NpfPerM1 = NpfPer;
	NpbPerM1 = NpbPer;
	//-Marks current periodic particles to be ignored.
	//-Marca periodicas actuales para ignorar.
	cusph::PeriodicIgnore(Np, Codeg);
	//-Creates new periodic particles.
	//-Crea las nuevas periodicas.
	const unsigned npb0 = Npb;
	const unsigned npf0 = Np - Npb;
	const unsigned np0 = Np;
	NpbPer = NpfPer = 0;
	BoundChanged = true;
	for (unsigned ctype = 0; ctype < 2; ctype++) {//-0:bound, 1:fluid+floating.
		//-Computes the particle range to be examined (bound and fluid).
		//-Calcula rango de particulas a examinar (bound o fluid).
		const unsigned pini = (ctype ? npb0 : 0);
		const unsigned num = (ctype ? npf0 : npb0);
		//-Searches for periodic zones on each axis (X, Y and Z).
		//-Busca periodicas en cada eje (X, Y e Z).
		for (unsigned cper = 0; cper < 3; cper++)if ((cper == 0 && PeriX) || (cper == 1 && PeriY) || (cper == 2 && PeriZ)) {
			tdouble3 perinc = (cper == 0 ? PeriXinc : (cper == 1 ? PeriYinc : PeriZinc));
			//-First searches in the list of new periodic particles and then in the initial particle list (necessary for periodic zones in more than one axis)
			//-Primero busca en la lista de periodicas nuevas y despues en la lista inicial de particulas (necesario para periodicas en mas de un eje).
			for (unsigned cblock = 0; cblock < 2; cblock++) {//-0:new periodic particles, 1:original periodic particles
				const unsigned nper = (ctype ? NpfPer : NpbPer);  //-number of new periodic particles for the type currently computed (bound or fluid). | Numero de periodicas nuevas del tipo a procesar.
				const unsigned pini2 = (cblock ? pini : Np - nper);
				const unsigned num2 = (cblock ? num : nper);
				//-Repeats search if the available memory was insufficient and had to be increased.
				//-Repite la busqueda si la memoria disponible resulto insuficiente y hubo que aumentarla.
				bool run = true;
				while (run && num2) {
					//-Allocates memory to create the periodic particle list.
					//-Reserva memoria para crear lista de particulas periodicas.
					unsigned* listpg = ArraysGpu->ReserveUint();
					unsigned nmax = GpuParticlesSize - 1; //-Maximum number of particles that can be included in the list. | Numero maximo de particulas que caben en la lista.
					//-Generates list of new periodic particles
					if (Np >= 0x80000000)Run_Exceptioon("The number of particles is too big.");//-Because the last bit is used to mark the reason the new periodical is created. | Porque el ultimo bit se usa para marcar el sentido en que se crea la nueva periodica. 
					unsigned count = cusph::PeriodicMakeList(num2, pini2, Stable, nmax, Map_PosMin, Map_PosMax, perinc, Posxyg, Poszg, Codeg, listpg);
					//-Resizes the memory size for the particles if there is not sufficient space and repeats the serach process.
					//-Redimensiona memoria para particulas si no hay espacio suficiente y repite el proceso de busqueda.
					if (count > nmax || !CheckGpuParticlesSize(count + Np)) {
						ArraysGpu->Free(listpg); listpg = NULL;
						Timersg->TmStop(TMG_SuPeriodic, false);
						ResizeParticlesSize(Np + count, PERIODIC_OVERMEMORYNP, false);
						Timersg->TmStart(TMG_SuPeriodic, false);
					}
					else {
						run = false;
						//-Create new periodic particles duplicating the particles from the list
						//-Crea nuevas particulas periodicas duplicando las particulas de la lista.
						if (TStep == STEP_Verlet)cusph::PeriodicDuplicateVerlet(count, Np, DomCells, perinc, listpg, Idpg, Codeg, Dcellg, Posxyg, Poszg, Velrhopg, SpsTaug, VelrhopM1g);
						if (TStep == STEP_Symplectic) {
							if ((PosxyPreg || PoszPreg || VelrhopPreg) && (!PosxyPreg || !PoszPreg || !VelrhopPreg))Run_Exceptioon("Symplectic data is invalid.");
							cusph::PeriodicDuplicateSymplectic(count, Np, DomCells, perinc, listpg, Idpg, Codeg, Dcellg, Posxyg, Poszg, Velrhopg, SpsTaug, PosxyPreg, PoszPreg, VelrhopPreg);
						}
						if (UseNormals)cusph::PeriodicDuplicateNormals(count, Np, listpg, BoundNormalg, MotionVelg);

						//-Frees memory and updates the particle number.
						//-Libera lista y actualiza numero de particulas.
						ArraysGpu->Free(listpg); listpg = NULL;
						Np += count;
						//-Updated number of new periodic particles.
						//-Actualiza numero de periodicas nuevas.
						if (!ctype)NpbPer += count;
						else NpfPer += count;
					}
				}
			}
		}
	}
	Timersg->TmStop(TMG_SuPeriodic, true);
	Check_CudaErroor("Failed in creation of periodic particles.");
}

//==============================================================================
/// Executes divide of particles in cells.
/// Ejecuta divide de particulas en celdas.
//==============================================================================
void JSphGpuSingle::RunCellDivide(bool updateperiodic) {
	//JDebugSphGpu::SaveVtk("_DG_Divide_Pre.vtk",Nstep,0,Np,"all",this);

	DivData = DivDataGpuNull();
	if (CaseNdeformstruc > 0) BoundChanged = true;
	//-Creates new periodic particles and marks the old ones to be ignored.
	//-Crea nuevas particulas periodicas y marca las viejas para ignorarlas.
	if (updateperiodic && PeriActive)RunPeriodic();

	//-Initiates Divide.
	CellDivSingle->Divide(Npb, Np - Npb - NpbPer - NpfPer, NpbPer, NpfPer, BoundChanged
		, Dcellg, Codeg, Posxyg, Poszg, Idpg, Timersg);
	DivData = CellDivSingle->GetCellDivData();
	//-Sorts particle data. | Ordena datos de particulas.
	Timersg->TmStart(TMG_NlSortData, false);
	{
		unsigned* idpg = ArraysGpu->ReserveUint();
		typecode* codeg = ArraysGpu->ReserveTypeCode();
		unsigned* dcellg = ArraysGpu->ReserveUint();
		double2* posxyg = ArraysGpu->ReserveDouble2();
		double* poszg = ArraysGpu->ReserveDouble();
		float4* velrhopg = ArraysGpu->ReserveFloat4();
		CellDivSingle->SortBasicArrays(Idpg, Codeg, Dcellg, Posxyg, Poszg, Velrhopg, idpg, codeg, dcellg, posxyg, poszg, velrhopg);
		swap(Idpg, idpg);           ArraysGpu->Free(idpg);
		swap(Codeg, codeg);         ArraysGpu->Free(codeg);
		swap(Dcellg, dcellg);       ArraysGpu->Free(dcellg);
		swap(Posxyg, posxyg);       ArraysGpu->Free(posxyg);
		swap(Poszg, poszg);         ArraysGpu->Free(poszg);
		swap(Velrhopg, velrhopg);   ArraysGpu->Free(velrhopg);
	}
	if (TStep == STEP_Verlet) {
		float4* velrhopg = ArraysGpu->ReserveFloat4();
		CellDivSingle->SortDataArrays(VelrhopM1g, velrhopg);
		swap(VelrhopM1g, velrhopg);   ArraysGpu->Free(velrhopg);
	}
	else if (TStep == STEP_Symplectic && (PosxyPreg || PoszPreg || VelrhopPreg)) { //-In reality, only necessary in the corrector not the predictor step??? | En realidad solo es necesario en el divide del corrector, no en el predictor??? 
		if (!PosxyPreg || !PoszPreg || !VelrhopPreg)Run_Exceptioon("Symplectic data is invalid.");
		double2* posxyg = ArraysGpu->ReserveDouble2();
		double* poszg = ArraysGpu->ReserveDouble();
		float4* velrhopg = ArraysGpu->ReserveFloat4();
		CellDivSingle->SortDataArrays(PosxyPreg, PoszPreg, VelrhopPreg, posxyg, poszg, velrhopg);
		swap(PosxyPreg, posxyg);      ArraysGpu->Free(posxyg);
		swap(PoszPreg, poszg);        ArraysGpu->Free(poszg);
		swap(VelrhopPreg, velrhopg);  ArraysGpu->Free(velrhopg);
	}
	if (TVisco == VISCO_LaminarSPS) {
		tsymatrix3f* spstaug = ArraysGpu->ReserveSymatrix3f();
		CellDivSingle->SortDataArrays(SpsTaug, spstaug);
		swap(SpsTaug, spstaug);  ArraysGpu->Free(spstaug);
	}
	if (UseNormals) {
		float3* boundnormalg = ArraysGpu->ReserveFloat3();
		CellDivSingle->SortDataArrays(BoundNormalg, boundnormalg);
		swap(BoundNormalg, boundnormalg); ArraysGpu->Free(boundnormalg);
		if (MotionVelg) {
			float3* motionvelg = ArraysGpu->ReserveFloat3();
			CellDivSingle->SortDataArrays(MotionVelg, motionvelg);
			swap(MotionVelg, motionvelg); ArraysGpu->Free(motionvelg);
		}
	}
	if (DeformStruc && DeformStrucRidpg) CellDivSingle->UpdateIndices(CaseNdeformstruc, DeformStrucRidpg);
	//-Collect divide data. | Recupera datos del divide.
	Np = CellDivSingle->GetNpFinal();
	Npb = CellDivSingle->GetNpbFinal();
	NpbOk = Npb - CellDivSingle->GetNpbIgnore();

	//-Update PosCellg[] according to current position of particles.
	cusphs::UpdatePosCell(Np, Map_PosMin, PosCellSize, Posxyg, Poszg, PosCellg, NULL);

	//-Manages excluded particles fixed, moving and floating before aborting the execution.
	if (CellDivSingle->GetNpbOut())AbortBoundOut();

	//-Collect position of floating particles. | Recupera posiciones de floatings.
	if (CaseNfloat)cusph::CalcRidp(PeriActive != 0, Np - Npb, Npb, CaseNpb, CaseNpb + CaseNfloat, Codeg, Idpg, FtRidpg);
	Timersg->TmStop(TMG_NlSortData, false);

	//-Control of excluded particles (only fluid because excluded boundary are checked before).
	//-Gestion de particulas excluidas (solo fluid porque las boundary excluidas se comprueban antes).
	Timersg->TmStart(TMG_NlOutCheck, false);
	unsigned npfout = CellDivSingle->GetNpfOut();
	if (npfout) {
		ParticlesDataDown(npfout, Np, true, false);
		AddParticlesOut(npfout, Idp, AuxPos, AuxVel, AuxRhop, Code);
	}
	Timersg->TmStop(TMG_NlOutCheck, true);
	BoundChanged = false;
}

//------------------------------------------------------------------------------
/// Manages excluded particles fixed, moving and floating before aborting the execution.
/// Gestiona particulas excluidas fixed, moving y floating antes de abortar la ejecucion.
//------------------------------------------------------------------------------
void JSphGpuSingle::AbortBoundOut() {
	const unsigned nboundout = CellDivSingle->GetNpbOut();
	//-Get data of excluded boundary particles.
	ParticlesDataDown(nboundout, Np, true, false);
	//-Shows excluded particles information and aborts execution.
	JSph::AbortBoundOut(Log, nboundout, Idp, AuxPos, AuxVel, AuxRhop, Code);
}

//==============================================================================
/// Interaction for force computation.
/// Interaccion para el calculo de fuerzas.
//==============================================================================
void JSphGpuSingle::Interaction_Forces(TpInterStep interstep) {
	if (TBoundary == BC_MDBC && (MdbcCorrector || interstep != INTERSTEP_SymCorrector))MdbcBoundCorrection(); //-Boundary correction for mDBC.
	InterStep = interstep;
	PreInteraction_Forces();
	float3* dengradcorr = NULL;

	Timersg->TmStart(TMG_CfForces, true);
	const bool lamsps = (TVisco == VISCO_LaminarSPS);
	unsigned bsfluid = BlockSizes.forcesfluid;
	unsigned bsbound = BlockSizes.forcesbound;

	//-Interaction Fluid-Fluid/Bound & Bound-Fluid.
	const StInterParmsg parms = StrInterParmsg(Simulate2D
		, Symmetry //<vs_syymmetry>
		, TKernel, FtMode
		, lamsps, TDensity, ShiftingMode
		, Visco * ViscoBoundFactor, Visco
		, bsbound, bsfluid, Np, Npb, NpbOk
		, 0, Nstep, DivData, Dcellg
		, Posxyg, Poszg, PosCellg, Velrhopg, Idpg, Codeg
		, FtoMasspg, SpsTaug, dengradcorr
		, ViscDtg, Arg, Aceg, Deltag
		, SpsGradvelg
		, ShiftPosfsg
		, NULL, NULL);
	cusph::Interaction_Forces(parms);

	//-Interaction DEM Floating-Bound & Floating-Floating. //(DEM)
	if (UseDEM)cusph::Interaction_ForcesDem(BlockSizes.forcesdem, CaseNfloat
		, DivData, Dcellg, FtRidpg, DemDatag, FtoMasspg, float(DemDtForce)
		, PosCellg, Velrhopg, Codeg, Idpg, ViscDtg, Aceg, NULL);

	//<vs_deformstruc_ini>
	//-Interaction deformable structure-deformable structure.
	/*if(DeformStruc){
	  Timersg->TmStart(TMG_SuDeformStruc,false);
	  const StInterParmsDeformStrucg parmsfs=StrInterParmsDeformStrucg(Simulate2D,TKernel,(TVisco==VISCO_LaminarSPS)
		  ,Visco*ViscoBoundFactor,CaseNdeformstruc,DivData,Dcellg
		  ,PosCellg,Velrhopg,Codeg
		  ,DeformStrucDatag,DeformStrucRidpg,PosCell0g,NumPairsg,PairIdxg,KerCorrg,DeformStrucDtg,DefGradg,Aceg,NULL);
	  cusph::Interaction_ForcesDeformStruc(parmsfs);
	  Timersg->TmStop(TMG_SuDeformStruc,false);
	}*/
	//<vs_deformstruc_end>

	//-For 2D simulations always overrides the 2nd component (Y axis).
	//-Para simulaciones 2D anula siempre la 2nd componente.
	if (Simulate2D)cusph::Resety(Np - Npb, Npb, Aceg);

	//-Computes Tau for Laminar+SPS.
	if (lamsps)cusph::ComputeSpsTau(Np, Npb, SpsSmag, SpsBlin, Velrhopg, SpsGradvelg, SpsTaug);

	//-Applies DDT.
	if (Deltag)cusph::AddDelta(Np - Npb, Deltag + Npb, Arg + Npb);//-Adds the Delta-SPH correction for the density. | Anhade correccion de Delta-SPH a Arg[]. 
	cudaDeviceSynchronize();
	Check_CudaErroor("Failed while executing kernels of interaction.");

	//-Calculates maximum value of ViscDt.
	if (Np)ViscDtMax = cusph::ReduMaxFloat(Np, 0, ViscDtg, CellDivSingle->GetAuxMem(cusph::ReduMaxFloatSize(Np)));
	//-Calculates maximum value of Ace (periodic particles are ignored). ViscDtg is used like auxiliary memory.
	AceMax = ComputeAceMax(ViscDtg);

	//<vs_deformstruc_ini>
	//-Calculates maximum value of DeformStrucDt.
	//if(CaseNdeformstruc){
	//  Timersg->TmStart(TMG_SuDeformStruc,false);
	//  DeformStrucDtMax=cusph::ReduMaxFloat(CaseNdeformstruc,0,DeformStrucDtg,CellDivSingle->GetAuxMem(cusph::ReduMaxFloatSize(CaseNdeformstruc)));
	//  Timersg->TmStop(TMG_SuDeformStruc,false);
	//}
	//<vs_deformstruc_end>

	Timersg->TmStop(TMG_CfForces, true);
	Check_CudaErroor("Failed in reduction of viscdt.");
}

//==============================================================================
/// Calculates extrapolated data on boundary particles from fluid domain for mDBC.
/// Calcula datos extrapolados en el contorno para mDBC.
//==============================================================================
void JSphGpuSingle::MdbcBoundCorrection() {
	Timersg->TmStart(TMG_CfPreForces, false);
	const unsigned n = (UseNormalsFt ? Np : NpbOk);
	cusph::Interaction_MdbcCorrection(TKernel, Simulate2D, SlipMode, MdbcFastSingle
		, n, CaseNbound, MdbcThreshold, DivData, Map_PosMin, Posxyg, Poszg, PosCellg, Codeg
		, Idpg, BoundNormalg, MotionVelg, Velrhopg);
	Timersg->TmStop(TMG_CfPreForces, false);
}


//==============================================================================
/// Returns the maximum value of  (ace.x^2 + ace.y^2 + ace.z^2) from Acec[].
/// Devuelve valor maximo de (ace.x^2 + ace.y^2 + ace.z^2) a partir de Acec[].
//==============================================================================
double JSphGpuSingle::ComputeAceMax(float* auxmem) {
	const bool check = (PeriActive != 0 || InOut != NULL);
	float acemax = 0;
	//const unsigned pini=(CaseNdeformstruc? 0: Npb);
	const unsigned pini = Npb;
	if (!check)cusph::ComputeAceMod(Np - pini, Aceg + pini, auxmem);//-Without periodic conditions. | Sin condiciones periodicas.
	else cusph::ComputeAceMod(Np - pini, Codeg + pini, Aceg + pini, auxmem);//-With periodic conditions ignores the periodic particles. | Con condiciones periodicas ignora las particulas periodicas.
	if (Np - pini)acemax = cusph::ReduMaxFloat(Np - pini, 0, auxmem, CellDivSingle->GetAuxMem(cusph::ReduMaxFloatSize(Np - pini)));
	return(sqrt(double(acemax)));
}

//<vs_ddramp_ini>
//==============================================================================
/// Applies initial DDT ramp.
//==============================================================================
void JSphGpuSingle::RunInitialDDTRamp() {
	if (TimeStep < DDTRamp.x) {
		if ((Nstep % 10) == 0) {//-DDTkh value is updated every 10 calculation steps.
			if (TimeStep <= DDTRamp.y)DDTkh = KernelSize * float(DDTRamp.z);
			else {
				const double tt = TimeStep - DDTRamp.y;
				const double tr = DDTRamp.x - DDTRamp.y;
				DDTkh = KernelSize * float(((tr - tt) / tr) * (DDTRamp.z - DDTValue) + DDTValue);
			}
			ConstantDataUp(); //-Updates value in constant memory of GPU.
		}
	}
	else {
		if (DDTkh != DDTkhCte) {
			CSP.ddtkh = DDTkh = DDTkhCte;
			ConstantDataUp();
		}
		DDTRamp.x = 0;
	}
}//<vs_ddramp_end>

//==============================================================================
/// Particle interaction and update of particle data according to
/// the computed forces using the Verlet time stepping scheme.
///
/// Realiza interaccion y actualizacion de particulas segun las fuerzas 
/// calculadas en la interaccion usando Verlet.
//==============================================================================
template<bool rundefstruc, bool simulate2d, bool defsttm, TpKernel tkernel>
double JSphGpuSingle::ComputeStep_Ver() {
	Interaction_Forces(INTERSTEP_Verlet);			//-Interaction.
	const double dt = DtVariable(true);				//-Calculate new dt.
	if (CaseNmoving)CalcMotion(dt);					//-Calculate motion for moving bodies.
	DemDtForce = dt;								//(DEM)
	if (Shifting)RunShifting(dt);					//-Shifting.
	ComputeVerlet(dt);								//-Update particles using Verlet (periodic particles become invalid).
	if (rundefstruc)
	{
		Timersg->TmStart(TMG_SuDeformStruc, false);
		DSComputeStep_Ver<simulate2d, defsttm>(dt);
		BoundChanged = true;
		Timersg->TmStop(TMG_SuDeformStruc, false);
	}
	if (CaseNfloat)RunFloating(dt, false);     //-Control of floating bodies.
	PosInteraction_Forces();                 //-Free memory used for interaction.
	if (Damping)RunDamping(dt, Np, Npb, Posxyg, Poszg, Codeg, Velrhopg); //-Aplies Damping.
	if (RelaxZones)RunRelaxZone(dt);          //-Generate waves using RZ.
	return(dt);
}

//==============================================================================
/// Particle interaction and update of particle data according to
/// the computed forces using the Symplectic time stepping scheme.
///
/// Realiza interaccion y actualizacion de particulas segun las fuerzas 
/// calculadas en la interaccion usando Symplectic.
//==============================================================================
template<bool rundefstruc, bool simulate2d, bool defsttm, TpKernel tkernel>
double JSphGpuSingle::ComputeStep_Sym() {

	const double dt = SymplecticDtPre;
	const double halfdt = dt * .5;
	if (CaseNmoving)CalcMotion(dt);               //-Calculate motion for moving bodies.
	//-Predictor
	//-----------
	DemDtForce = dt * 0.5f;                          //(DEM)

	Interaction_Forces(INTERSTEP_SymPredictor);  //-Interaction.

	const double ddt_p = DtVariable(false);        //-Calculate dt of predictor step.
	if (Shifting)RunShifting(halfdt);              //-Shifting.
	ComputeSymplecticPre(dt);                    //-Apply Symplectic-Predictor to particles (periodic particles become invalid).

	if (CaseNfloat)RunFloating(halfdt, true);       //-Control of floating bodies.

	//-Computes new position and velocity for deformable structures.
	if (rundefstruc) {
		Timersg->TmStart(TMG_SuDeformStruc, false);
		DSComputeStep_Sym<simulate2d, defsttm>(halfdt, TimeStep);
		Timersg->TmStop(TMG_SuDeformStruc, false);
	}

	PosInteraction_Forces();                     //-Free memory used for interaction.

	//-Corrector
	//-----------
	DemDtForce = dt;                               //(DEM)

	RunCellDivide(true);

	Interaction_Forces(INTERSTEP_SymCorrector);  //-Interaction.

	const double ddt_c = DtVariable(true);         //-Calculate dt of corrector step.
	if (Shifting)RunShifting(dt);                 //-Shifting.
	ComputeSymplecticCorr(dt);                   //-Apply Symplectic-Corrector to particles (periodic particles become invalid).
	if (CaseNfloat)RunFloating(dt, false);         //-Control of floating bodies.

	//-Computes new position and velocity for deformable structures.
	if (rundefstruc) {
		Timersg->TmStart(TMG_SuDeformStruc, false);
		DSComputeStep_Sym<simulate2d, defsttm>(halfdt, TimeStep + halfdt);
		Timersg->TmStop(TMG_SuDeformStruc, false);
	}

	PosInteraction_Forces();                     //-Free memory used for interaction.
	if (Damping)RunDamping(dt, Np, Npb, Posxyg, Poszg, Codeg, Velrhopg); //-Aplies Damping.
	if (RelaxZones)RunRelaxZone(dt);              //-Generate waves using RZ.

	SymplecticDtPre = min(ddt_p, ddt_c);            //-Calculate dt for next ComputeStep.
	return(dt);
}

//==============================================================================
/// Updates information in FtObjs[] copying data from GPU.
/// Actualiza informacion en FtObjs[] copiando los datos en GPU.
//==============================================================================
void JSphGpuSingle::UpdateFtObjs() {
	if (FtCount && FtObjsOutdated) {
		tdouble3* fcen = FtoAuxDouble6;
		tfloat3* fang = FtoAuxFloat15;
		tfloat3* fvellin = fang + FtCount;
		tfloat3* fvelang = fvellin + FtCount;
		tfloat3* facelin = fvelang + FtCount;
		tfloat3* faceang = facelin + FtCount;
		cudaMemcpy(fcen, FtoCenterg, sizeof(double3) * FtCount, cudaMemcpyDeviceToHost);
		cudaMemcpy(fang, FtoAnglesg, sizeof(float3) * FtCount, cudaMemcpyDeviceToHost);
		cudaMemcpy(fvellin, FtoVelAceg, sizeof(float3) * FtCount * 4, cudaMemcpyDeviceToHost);
		for (unsigned cf = 0; cf < FtCount; cf++) {
			FtObjs[cf].center = fcen[cf];
			FtObjs[cf].angles = fang[cf];
			FtObjs[cf].fvel = fvellin[cf];
			FtObjs[cf].fomega = fvelang[cf];
			FtObjs[cf].facelin = facelin[cf];
			FtObjs[cf].faceang = faceang[cf];
		}
	}
	FtObjsOutdated = false;
}

//==============================================================================
/// Applies imposed velocity.
/// Aplica velocidad predefinida.
//==============================================================================
void JSphGpuSingle::FtApplyImposedVel(float3* ftoforcesresg)const {
	tfloat3* ftoforcesresc = NULL;
	for (unsigned cf = 0; cf < FtCount; cf++)if (!FtObjs[cf].usechrono && (FtLinearVel[cf] != NULL || FtAngularVel[cf] != NULL)) {
		const tfloat3 v1 = (FtLinearVel[cf] != NULL ? FtLinearVel[cf]->GetValue3f(TimeStep) : TFloat3(FLT_MAX));
		const tfloat3 v2 = (FtAngularVel[cf] != NULL ? FtAngularVel[cf]->GetValue3f(TimeStep) : TFloat3(FLT_MAX));
		if (!ftoforcesresc && (v1 != TFloat3(FLT_MAX) || v2 != TFloat3(FLT_MAX))) {
			//-Copies data on GPU memory to CPU memory.
			ftoforcesresc = FtoAuxFloat15;
			cudaMemcpy(ftoforcesresc, ftoforcesresg, sizeof(tfloat3) * FtCount * 2, cudaMemcpyDeviceToHost);
		}
		unsigned cfpos = cf * 2 + 1;
		if (v1.x != FLT_MAX)ftoforcesresc[cfpos].x = v1.x;
		if (v1.y != FLT_MAX)ftoforcesresc[cfpos].y = v1.y;
		if (v1.z != FLT_MAX)ftoforcesresc[cfpos].z = v1.z;
		cfpos--;
		if (v2.x != FLT_MAX)ftoforcesresc[cfpos].x = v2.x;
		if (v2.y != FLT_MAX)ftoforcesresc[cfpos].y = v2.y;
		if (v2.z != FLT_MAX)ftoforcesresc[cfpos].z = v2.z;
	}
	//-Updates data on GPU memory.
	if (ftoforcesresc != NULL) {
		cudaMemcpy(ftoforcesresg, ftoforcesresc, sizeof(tfloat3) * FtCount * 2, cudaMemcpyHostToDevice);
	}
}

//==============================================================================
/// Process floating objects.
/// Procesa floating objects.
//==============================================================================
void JSphGpuSingle::RunFloating(double dt, bool predictor) {
	Timersg->TmStart(TMG_SuFloating, false);
	if (TimeStep >= FtPause) {//-Operator >= is used because when FtPause=0 in symplectic-predictor, code would not enter here. | Se usa >= pq si FtPause es cero en symplectic-predictor no entraria.

		//-Adds external forces (ForcePoints, Moorings, external file) to FtoForces[].
		if (ForcePoints != NULL || FtLinearForce != NULL) {
			StFtoForces* ftoforces = (StFtoForces*)FtoAuxFloat15;
			memset(ftoforces, 0, sizeof(StFtoForces) * FtCount);
			//-Loads sum of linear and angular forces from ForcePoints and Moorings.
			if (ForcePoints)ForcePoints->GetFtForcesSum(ftoforces);
			//-Adds the external forces.
			if (FtLinearForce != NULL) {
				for (unsigned cf = 0; cf < FtCount; cf++) {
					ftoforces[cf].face = ftoforces[cf].face + GetFtExternalForceLin(cf, TimeStep);
					ftoforces[cf].fomegaace = ftoforces[cf].fomegaace + GetFtExternalForceAng(cf, TimeStep);
				}
			}
			//-Copies data to GPU memory.
			cudaMemcpy(FtoForcesg, ftoforces, sizeof(StFtoForces) * FtCount, cudaMemcpyHostToDevice);
		}
		else {
			//-Initialises forces of floatings when no external forces are applied.
			cudaMemset(FtoForcesg, 0, sizeof(StFtoForces) * FtCount);
		}

		//-Calculate forces summation (face,fomegaace) starting from floating particles and add in FtoForcesg[].
		cusph::FtCalcForcesSum(PeriActive != 0, FtCount, FtoDatpg, FtoCenterg, FtRidpg, Posxyg, Poszg, Aceg, FtoForcesg);

		//-Computes final acceleration from particles and from external forces in FtoForcesg[].
		cusph::FtCalcForces(FtCount, Gravity, FtoMassg, FtoAnglesg, FtoInertiaini8g, FtoInertiaini1g, FtoForcesg);

		//-Calculate data to update floatings / Calcula datos para actualizar floatings.
		cusph::FtCalcForcesRes(FtCount, Simulate2D, dt, FtoVelAceg, FtoCenterg, FtoForcesg, FtoForcesResg, FtoCenterResg);
		//-Applies imposed velocity.
		if (FtLinearVel != NULL)FtApplyImposedVel(FtoForcesResg);
		//-Applies motion constraints.
		if (FtConstraints)cusph::FtApplyConstraints(FtCount, FtoConstraintsg, FtoForcesg, FtoForcesResg);

		//-Saves face and fomegace for debug.
		if (SaveFtAce) {
			StFtoForces* ftoforces = (StFtoForces*)FtoAuxFloat15;
			cudaMemcpy(ftoforces, FtoForcesg, sizeof(tfloat3) * FtCount * 2, cudaMemcpyDeviceToHost);
			SaveFtAceFun(dt, predictor, ftoforces);
		}

		//-Run floating with Chrono library.
		if (ChronoObjects) {
			Timersg->TmStop(TMG_SuFloating, false);
			Timersg->TmStart(TMG_SuChrono, false);
			//-Export data / Exporta datos.
			tfloat3* ftoforces = FtoAuxFloat15;
			cudaMemcpy(ftoforces, FtoForcesg, sizeof(tfloat3) * FtCount * 2, cudaMemcpyDeviceToHost);
			for (unsigned cf = 0; cf < FtCount; cf++)if (FtObjs[cf].usechrono) {
				ChronoObjects->SetFtData(FtObjs[cf].mkbound, ftoforces[cf * 2], ftoforces[cf * 2 + 1]);
			}
			//-Applies the external velocities to each floating body of Chrono.
			if (FtLinearVel != NULL)ChronoFtApplyImposedVel();
			//-Calculate data using Chrono / Calcula datos usando Chrono.
			ChronoObjects->RunChrono(Nstep, TimeStep, dt, predictor);
			//-Load calculated data by Chrono / Carga datos calculados por Chrono.
			tdouble3* ftocenter = FtoAuxDouble6;
			cudaMemcpy(ftocenter, FtoCenterResg, sizeof(tdouble3) * FtCount, cudaMemcpyDeviceToHost);//-Necesario para cargar datos de floatings sin chrono.
			cudaMemcpy(ftoforces, FtoForcesResg, sizeof(tfloat3) * FtCount * 2, cudaMemcpyDeviceToHost);//-Necesario para cargar datos de floatings sin chrono.
			for (unsigned cf = 0; cf < FtCount; cf++)if (FtObjs[cf].usechrono)ChronoObjects->GetFtData(FtObjs[cf].mkbound, ftocenter[cf], ftoforces[cf * 2 + 1], ftoforces[cf * 2]);
			cudaMemcpy(FtoCenterResg, ftocenter, sizeof(tdouble3) * FtCount, cudaMemcpyHostToDevice);
			cudaMemcpy(FtoForcesResg, ftoforces, sizeof(float3) * FtCount * 2, cudaMemcpyHostToDevice);
			Timersg->TmStop(TMG_SuChrono, false);
			Timersg->TmStart(TMG_SuFloating, false);
		}

		//-Apply movement around floating objects / Aplica movimiento sobre floatings.
		cusph::FtUpdate(PeriActive != 0, predictor, FtCount, dt, FtoDatpg, FtoForcesResg, FtoCenterResg
			, FtRidpg, FtoCenterg, FtoAnglesg, FtoVelAceg, Posxyg, Poszg, Dcellg, Velrhopg, Codeg);

		//-Stores floating data.
		if (!predictor) {
			FtObjsOutdated = true;
			//-Updates floating normals for mDBC.
			if (UseNormalsFt) {
				tdouble3* fcen = FtoAuxDouble6;
				tfloat3* fang = FtoAuxFloat15;
				cudaMemcpy(fcen, FtoCenterg, sizeof(double3) * FtCount, cudaMemcpyDeviceToHost);
				cudaMemcpy(fang, FtoAnglesg, sizeof(float3) * FtCount, cudaMemcpyDeviceToHost);
				for (unsigned cf = 0; cf < FtCount; cf++) {
					const StFloatingData fobj = FtObjs[cf];
					FtObjs[cf].center = fcen[cf];
					FtObjs[cf].angles = fang[cf];
					const tdouble3 dang = ToTDouble3(FtObjs[cf].angles - fobj.angles) * TODEG;
					const tdouble3 cen = FtObjs[cf].center;
					JMatrix4d mat;
					mat.Move(cen);
					mat.Rotate(dang);
					mat.Move(fobj.center * -1);
					cusph::FtNormalsUpdate(fobj.count, fobj.begin - CaseNpb, mat.GetMatrix(), FtRidpg, BoundNormalg);
				}
			}
			//<vs_ftmottionsv_ini>
			if (FtMotSave && FtMotSave->CheckTime(TimeStep + dt)) {
				UpdateFtObjs(); //-Updates floating information on CPU memory.
				FtMotSave->SaveFtDataGpu(TimeStep + dt, Nstep + 1, FtObjs, Np, Posxyg, Poszg, FtRidpg);
			}
			//<vs_ftmottionsv_end>
		}
	}
	//-Update data of points in FtForces and calculates motion data of affected floatings.
	Timersg->TmStop(TMG_SuFloating, false);
	if (!predictor && ForcePoints) {
		Timersg->TmStart(TMG_SuMoorings, false);
		UpdateFtObjs(); //-Updates floating information on CPU memory.
		ForcePoints->UpdatePoints(TimeStep, dt, FtObjs);
		if (Moorings)Moorings->ComputeForces(Nstep, TimeStep, dt, ForcePoints);
		ForcePoints->ComputeForcesSum();
		Timersg->TmStop(TMG_SuMoorings, false);
	}
}

//==============================================================================
/// Runs calculations in configured gauges.
/// Ejecuta calculos en las posiciones de medida configuradas.
//==============================================================================
void JSphGpuSingle::RunGaugeSystem(double timestep, bool saveinput) {
	if (!Nstep || GaugeSystem->GetCount()) {
		Timersg->TmStart(TMG_SuGauges, false);
		//const bool svpart=(TimeStep>=TimePartNext);
		GaugeSystem->CalculeGpu(timestep, DivData
			, NpbOk, Npb, Np, Posxyg, Poszg, Codeg, Idpg, Velrhopg, saveinput);
		Timersg->TmStop(TMG_SuGauges, false);
	}
}

//==============================================================================
/// Compute PIPS information of current particles.
/// Calcula datos de PIPS de particulas actuales.
//==============================================================================
void JSphGpuSingle::ComputePips(bool run) {
	if (run || DsPips->CheckRun(Nstep)) {
		TimerSim.Stop();
		const double timesim = TimerSim.GetElapsedTimeD() / 1000.;
		const unsigned sauxmemg = ArraysGpu->GetArraySize();
		unsigned* auxmemg = ArraysGpu->ReserveUint();
		DsPips->ComputeGpu(Nstep, TimeStep, timesim, Np, Npb, NpbOk
			, DivData, Dcellg, PosCellg, sauxmemg, auxmemg);
		ArraysGpu->Free(auxmemg);
	}
}

//==============================================================================
/// Initialises execution of simulation.
/// Inicia ejecucion de simulacion.
//==============================================================================
void JSphGpuSingle::Run(std::string appname, const JSphCfgRun* cfg, JLog2* log) {
	if (!cfg || !log)return;
	AppName = appname; Log = log; CfgRun = cfg;

	//-Selection of GPU.
	//-------------------
	SelecDevice(cfg->GpuId);
	//-Configures timers.
	//-------------------
	Timersg->Config(cfg->SvTimers);
	Timersg->TmStart(TMG_Init, false);

	//-Load parameters and values of input. | Carga de parametros y datos de entrada.
	//--------------------------------------------------------------------------------
	LoadConfig(cfg);
	LoadCaseParticles();
	VisuConfig();
	ConfigDomain();
	ConfigRunMode();
	VisuParticleSummary();

	//-Initialisation of execution variables. | Inicializacion de variables de ejecucion.
	//------------------------------------------------------------------------------------
	InitRunGpu();
	RunGaugeSystem(TimeStep, true);
	if (InOut)InOutInit(TimeStepIni);

	if (DeformStruc) {
		DSPreTimeInt();
	}

	FreePartsInit();
	UpdateMaxValues();
	PrintAllocMemory(GetAllocMemoryCpu(), GetAllocMemoryGpu());
	SaveData();

	Timersg->ResetTimes();
	Timersg->TmStop(TMG_Init, false);
	if (Log->WarningCount())Log->PrintWarningList("\n[WARNINGS]", "");
	PartNstep = -1; Part++;

	//-Main Loop.
	//------------
	bool partoutstop = false;
	TimerSim.Start();
	TimerPart.Start();

	Log->Print(string("\n[Initialising simulation (") + RunCode + ")  " + fun::GetDateTime() + "]");
	if (DsPips)ComputePips(true);
	if (DeformStruc) PrintHeadPart(true);
	else PrintHeadPart(false);
	if (Simulate2D) TimeLoop_c0<true>(partoutstop);
	else TimeLoop_c0<false>(partoutstop);

	TimerSim.Stop(); TimerTot.Stop();

	//-End of Simulation.
	//--------------------
	FinishRun(partoutstop);
}

template<bool rundefstruc, bool simulate2d, bool defsttm, TpKernel tkernel>
void JSphGpuSingle::TimeLoop(bool& partoutstop) {
	JTimeControl tc("30,60,300,600");//-Shows information at 0.5, 1, 5 y 10 minutes (before first PART).
	while (TimeStep < TimeMax) {
		InterStep = (TStep == STEP_Symplectic ? INTERSTEP_SymPredictor : INTERSTEP_Verlet);
		if (ViscoTime)Visco = ViscoTime->GetVisco(float(TimeStep));
		if (DDTRamp.x)RunInitialDDTRamp(); //<vs_ddramp>
		double stepdt = ComputeStep<rundefstruc, simulate2d, defsttm, tkernel>();
		RunGaugeSystem(TimeStep + stepdt);
		if (CaseNmoving)RunMotion(stepdt);
		if (InOut)InOutComputeStep(stepdt);
		else RunCellDivide(true);
		TimeStep += stepdt;
		LastDt = stepdt;
		partoutstop = (Np < NpMinimum || !Np);
		if (TimeStep >= TimePartNext || partoutstop) {
			if (partoutstop) {
				Log->PrintWarning("Particles OUT limit reached...");
				TimeMax = TimeStep;
			}
			SaveData();
			Part++;
			PartNstep = Nstep;
			TimeStepM1 = TimeStep;
			TimePartNext = (SvAllSteps ? TimeStep : OutputTime->GetNextTime(TimeStep));
			TimerPart.Start();
		}
		UpdateMaxValues();
		Nstep++;
		const bool laststep = (TimeStep >= TimeMax || (NstepsBreak && Nstep >= NstepsBreak));
		if (DsPips)ComputePips(laststep);
		if (Part <= PartIni + 1 && tc.CheckTime())Log->Print(string("  ") + tc.GetInfoFinish((TimeStep - TimeStepIni) / (TimeMax - TimeStepIni)));
		if (NstepsBreak && Nstep >= NstepsBreak)break; //-For debugging.
	}
}

template<bool simulate2d> void JSphGpuSingle::TimeLoop_c0(bool& partoutstop) {
	if (DeformStruc && DeformStruc->UseUsrTimeStep) {
		TimeLoop_c1<simulate2d, true>(partoutstop);
	}
	else {
		TimeLoop_c1<simulate2d, false>(partoutstop);
	}
}

template<bool simulate2d, bool defsttm> void JSphGpuSingle::TimeLoop_c1(bool& partoutstop) {
	if (TKernel == KERNEL_Cubic) TimeLoop_c2<simulate2d, defsttm, KERNEL_Cubic>(partoutstop);
	else TimeLoop_c2<simulate2d, defsttm, KERNEL_Wendland>(partoutstop);
}

template<bool simulate2d, bool defsttm, TpKernel tkernel> void JSphGpuSingle::TimeLoop_c2(bool& partoutstop) {
	if (DeformStruc) TimeLoop<true, simulate2d, defsttm, tkernel>(partoutstop);
	else TimeLoop<false, simulate2d, defsttm, tkernel>(partoutstop);
}

//==============================================================================
/// Generates files with output data.
/// Genera los ficheros de salida de datos.
//==============================================================================
void JSphGpuSingle::SaveData() {
	const bool save = (SvData != SDAT_None && SvData != SDAT_Info);
	const unsigned npsave = Np - NpbPer - NpfPer; //-Subtracts the periodic particles if they exist. | Resta las periodicas si las hubiera.
	//-Retrieves particle data from the GPU. | Recupera datos de particulas en GPU.
	if (save) {
		Timersg->TmStart(TMG_SuDownData, false);
		unsigned npnormal = ParticlesDataDown(Np, 0, false, PeriActive != 0);
		if (npnormal != npsave)Run_Exceptioon("The number of particles is invalid.");
		Timersg->TmStop(TMG_SuDownData, false);
	}
	//-Retrieve floating object data from the GPU. | Recupera datos de floatings en GPU.
	if (FtCount) {
		Timersg->TmStart(TMG_SuDownData, false);
		UpdateFtObjs();
		Timersg->TmStop(TMG_SuDownData, false);
	}
	//-Collects additional information. | Reune informacion adicional.
	Timersg->TmStart(TMG_SuSavePart, false);
	StInfoPartPlus infoplus;
	memset(&infoplus, 0, sizeof(StInfoPartPlus));
	if (SvData & SDAT_Info) {
		infoplus.nct = CellDivSingle->GetNct();
		infoplus.npbin = NpbOk;
		infoplus.npbout = Npb - NpbOk;
		infoplus.npf = Np - Npb;
		infoplus.npbper = NpbPer;
		infoplus.npfper = NpfPer;
		infoplus.newnp = (InOut ? InOut->GetNewNpPart() : 0);
		infoplus.memorycpualloc = this->GetAllocMemoryCpu();
		infoplus.gpudata = true;
		infoplus.memorynctalloc = infoplus.memorynctused = GetMemoryGpuNct();
		infoplus.memorynpalloc = infoplus.memorynpused = GetMemoryGpuNp();
		TimerSim.Stop();
		infoplus.timesim = TimerSim.GetElapsedTimeD() / 1000.;

		infoplus.defstruc = false;
		if (DeformStruc) {
			infoplus.defstruc = true;
			infoplus.npdefstruc = MapNdeformstruc;
		}
	}
	//-Obtains current domain limits.
	const tdouble3 vdom[2] = { CellDivSingle->GetDomainLimits(true),CellDivSingle->GetDomainLimits(false) };
	//-Stores particle data. | Graba datos de particulas.
	JDataArrays arrays;
	JDataArrays dsarrays;
	AddBasicArrays(arrays, npsave, AuxPos, Idp, AuxVel, AuxRhop);
	if (DeformStruc) {
		tfloat3* defstrucenergies = new tfloat3[DeformStrucCount];
		tsymatrix3f* dscauchystress = new tsymatrix3f[MapNdeformstruc];
		//std::vector<tfloat4> dsvel4(MapNdeformstruc);
		//std::vector<tfloat3> dsvel3(MapNdeformstruc);
		std::vector<tfloat3> dsparticleenergies(MapNdeformstruc);
		cudaMemcpy(DSDispPhic, DSDispPhig, sizeof(tfloat4) * MapNdeformstruc, cudaMemcpyDeviceToHost);
		if (DSPlastic && DSEqPlasticg && DSEqPlasticc)
			cudaMemcpy(DSEqPlasticc, DSEqPlasticg, sizeof(float) * MapNdeformstruc, cudaMemcpyDeviceToHost);
		if (DSPlastic && DSPlasticStraing && DSPlasticStraingc)
			cudaMemcpy(DSPlasticStraingc, DSPlasticStraing, sizeof(tmatrix3f) * MapNdeformstruc, cudaMemcpyDeviceToHost);
		cudaMemcpy(DSFlForcec, DSFlForceg, sizeof(tfloat3) * MapNdeformstruc, cudaMemcpyDeviceToHost);
		//cudaMemcpy(dsvel4.data(), DSVelg, sizeof(tfloat4) * MapNdeformstruc, cudaMemcpyDeviceToHost);
		//for (unsigned i = 0; i < MapNdeformstruc; i++) dsvel3[i] = TFloat3(dsvel4[i].x, dsvel4[i].y, dsvel4[i].z);
		cudefstr::DSCalcEnergiesCauchyStress(MapNdeformstruc, DeformStrucCount, DefStrucIntDatag, DSIntArraysg,
			Simulate2D, defstrucenergies, dscauchystress, DSKerSumVolg, dsparticleenergies.data());
		AddBasicArraysDS(dsarrays, MapNdeformstruc, DSPos0c, DSDispPhic, (DSPlastic ? DSEqPlasticc : nullptr), (DSPlastic ? DSPlasticStraingc : nullptr),
			DSParentc, DSCodec, DSiBodyRidpc, DeformStrucCount, defstrucenergies, dscauchystress, nullptr /*dsvel3.data()*/, dsparticleenergies.data(),
			DSNMeasPlanes, DSMeasPlnCntc, DSNPartMeasPlanes, DSMeasPlnPartc, DSKerSumVolc, DSFlForcec);
		//for (int i = 0; i < MapNdeformstruc; i++) {
		//	if (DSFlForcec[i].x || DSFlForcec[i].y || DSFlForcec[i].z) std::cout << "\n" << DSFlForcec[i].x << "\t" << DSFlForcec[i].y << "\t" << DSFlForcec[i].z;
		//}
		JSph::SaveData(npsave, arrays, dsarrays, 1, vdom, &infoplus, DefStrucIntDatac, true, DSNMeasPlanes, DSNPartMeasPlanes);
		delete[] dscauchystress; dscauchystress = nullptr;
	}
	else {
		JSph::SaveData(npsave, arrays, dsarrays, 1, vdom, &infoplus, DefStrucIntDatac, true, DSNMeasPlanes, DSNPartMeasPlanes);
	}
	if (UseNormals && SvNormals)SaveVtkNormalsGpu("normals/Normals.vtk", Part, npsave, Npb, Posxyg, Poszg, Idpg, BoundNormalg);
	//-Save extra data.
	if (SvExtraDataBi4)SaveExtraData();
	Timersg->TmStop(TMG_SuSavePart, false);
}

//==============================================================================
/// Displays and stores final summary of the execution.
/// Muestra y graba resumen final de ejecucion.
//==============================================================================
void JSphGpuSingle::SaveExtraData() {
	const bool svextra = (BoundNormalg != NULL);
	if (svextra && SvExtraDataBi4->CheckSave(Part)) {
		SvExtraDataBi4->InitPartData(Part, TimeStep, Nstep);
		//-Saves normals of mDBC.
		unsigned* idp = NULL;
		tfloat3* nor = NULL;
		typecode* code = NULL;
		if (BoundNormalg) {
			const unsigned nsize = (UseNormalsFt ? Np : Npb);
			idp = fcuda::ToHostUint(0, nsize, Idpg);
			nor = fcuda::ToHostFloat3(0, nsize, BoundNormalg);
			if (PeriActive) {
#ifdef CODE_SIZE4
				code = fcuda::ToHostUint(0, nsize, Codeg);
#else
				code = fcuda::ToHostWord(0, nsize, Codeg);
#endif
			}
			SvExtraDataBi4->AddNormals(UseNormalsFt, Np, Npb, idp, code, nor);
		}
		//-Saves file.
		SvExtraDataBi4->SavePartData();
	}
}

//==============================================================================
/// Displays and stores final summary of the execution.
/// Muestra y graba resumen final de ejecucion.
//==============================================================================
void JSphGpuSingle::FinishRun(bool stop) {
	float tsim = TimerSim.GetElapsedTimeF() / 1000.f, ttot = TimerTot.GetElapsedTimeF() / 1000.f;
	JSph::ShowResume(stop, tsim, ttot, true, "");
	Log->Print(" ");
	string hinfo, dinfo;
	if (SvTimers) {
		Timersg->ShowTimes("[GPU Timers]", Log);
		Timersg->GetTimersInfo(hinfo, dinfo);
	}
	if (SvRes)SaveRes(tsim, ttot, hinfo, dinfo);
	Log->PrintFilesList();
	Log->PrintWarningList();
	VisuRefs();
}

//==============================================================================
/// Executes divide of particles in cells for deformable structures.
//==============================================================================
void JSphGpuSingle::DSRunCellDivide()
{
	DSDivData = DivDataGpuNull();
	DSCellDivSingle->Divide(MapNdeformstruc, 0, 0, 0, true, DSDcellg, DSCodeg, NULL, NULL, NULL, Timersg);
	DSDivData = DSCellDivSingle->GetCellDivData();
}

void JSphGpuSingle::DSPerformCellDiv(StDeformStrucData* defstrucdatac) {
	Log->Print(std::string("  Applying initial deformable solid cell division..."));

	float kernelsize2 = defstrucdatac[0].kernelsize2;
	float ksize = defstrucdatac[0].kernelsize;
	tdouble3 dommin = defstrucdatac[0].min;
	tdouble3 dommax = defstrucdatac[0].max;
	for (unsigned bodyid = 0; bodyid < DeformStrucCount; bodyid++) {
		StDeformStrucData& body = defstrucdatac[bodyid];
		if (body.kernelsize > ksize) ksize = body.kernelsize;
		if (body.kernelsize2 > kernelsize2) kernelsize2 = body.kernelsize2;

		if (body.min.x < dommin.x) dommin.x = body.min.x;
		if (body.min.y < dommin.y) dommin.y = body.min.y;
		if (body.min.z < dommin.z) dommin.z = body.min.z;

		if (body.max.x > dommax.x) dommax.x = body.max.x;
		if (body.max.y > dommax.y) dommax.y = body.max.y;
		if (body.max.z > dommax.z) dommax.z = body.max.z;
	}

	DScellsize = ksize;
	DScellsize = DScellsize / ScellDiv;

	DSMap_PosMax = dommax + DScellsize;
	DSMap_PosMin = dommin - DScellsize;

	DSMap_Size = DSMap_PosMax - DSMap_PosMin;
	DSMap_Cells = TUint3(unsigned(ceil(DSMap_Size.x / DScellsize)),
		unsigned(ceil(DSMap_Size.y / DScellsize)),
		unsigned(ceil(DSMap_Size.z / DScellsize)));

	DSSelecDomain(TUint3(0, 0, 0), DSMap_Cells);

	DSLoadDcellParticles(MapNdeformstruc, DSCodec, DSPos0c, DSDispPhic, DSDcellc);

	DSConfigPosCellGpu(ksize);

	cudaMemcpy(DSDcellg, DSDcellc, sizeof(unsigned) * MapNdeformstruc, cudaMemcpyHostToDevice);

	DSCellDivSingle = new JCellDivGpuSingle(Stable, false, false, kernelsize2, DSPosCellSize, true, CellMode, DScellsize
		, DSMap_PosMin, DSMap_PosMax, DSMap_Cells, MapNdeformstruc, 0, MapNdeformstruc, DirOut);

	DSCellDivSingle->DefineDomain(DSDomCellCode, DSDomCelIni, DSDomCelFin, DSDomPosMin, DSDomPosMax);
	DSConfigCellDiv((JCellDivGpu*)DSCellDivSingle);

	DSSaveMapCellsVtk();
	DSRunCellDivide();

	Timersg->TmStart(TMG_NlSortData, false);

	size_t m;
	unsigned* dcellg;			m = sizeof(unsigned) * MapNdeformstruc;	cudaMalloc((void**)&dcellg, m);
	unsigned* parents;			m = sizeof(unsigned) * MapNdeformstruc;	cudaMalloc((void**)&parents, m);
	typecode* codeg;			m = sizeof(typecode) * MapNdeformstruc;	cudaMalloc((void**)&codeg, m);
	float4* pos0g;				m = sizeof(float4) * MapNdeformstruc;	cudaMalloc((void**)&pos0g, m);

	DSCellDivSingle->SortDataArrays(DSParentg, DSCodeg, DSDcellg, DSPos0g, parents, codeg, dcellg, pos0g);

	swap(DSParentg, parents);	cudaFree(parents);
	swap(DSCodeg, codeg);		cudaFree(codeg);
	swap(DSDcellg, dcellg);		cudaFree(dcellg);
	swap(DSPos0g, pos0g);		cudaFree(pos0g);

	cudaMemcpy(DSParentc, DSParentg, sizeof(unsigned) * MapNdeformstruc, cudaMemcpyDeviceToHost);
	cudaMemcpy(DSCodec, DSCodeg, sizeof(typecode) * MapNdeformstruc, cudaMemcpyDeviceToHost);
	cudaMemcpy(DSDcellc, DSDcellg, sizeof(unsigned) * MapNdeformstruc, cudaMemcpyDeviceToHost);
	cudaMemcpy(DSPos0c, DSPos0g, sizeof(float4) * MapNdeformstruc, cudaMemcpyDeviceToHost);

	MapNdeformstruc = DSCellDivSingle->GetNpFinal();
	Log->Print(std::string("  Cell division done for ") + fun::IntStr(MapNdeformstruc) + std::string("  mapped particles."));
	Timersg->TmStop(TMG_NlSortData, false);
}

void JSphGpuSingle::DSSaveInitDomainInfo(StDeformStrucData* defstrucdatac)const
{
	//std::vector<float> tempvar(MapNdeformstruc);
	//std::vector<unsigned> mapcenters(MapNdeformstruc);
	//std::vector<unsigned> surfmark(MapNdeformstruc);
	//std::vector<tfloat3> tempvbc(MapNdeformstruc);
	//std::vector<unsigned> dsbestchild(CaseNdeformstruc);							
	//cudaMemcpy(dsbestchild.data(), DSBestChildg, sizeof(unsigned) * CaseNdeformstruc, cudaMemcpyDeviceToHost);
	//
	/*std::vector<unsigned> dssurfpartList(DSNpSurf);
	cudaMemcpy(dssurfpartList.data(), DSSurfPartListg, sizeof(unsigned) * DSNpSurf, cudaMemcpyDeviceToHost);
	*/
	//std::vector<unsigned> dsparents(MapNdeformstruc);							
	//cudaMemcpy(dsparents.data(), DSParentg, sizeof(unsigned) * MapNdeformstruc, cudaMemcpyDeviceToHost);

	std::vector<tbcstruc> dspartvbc(MapNdeformstruc);
	cudaMemcpy(dspartvbc.data(), DSPartVBCg, sizeof(tbcstruc) * MapNdeformstruc, cudaMemcpyDeviceToHost);

	//std::vector<tbcstruc> dspartfbc(MapNdeformstruc);
	//cudaMemcpy(dspartfbc.data(), DSPartFBCg, sizeof(tbcstruc) * MapNdeformstruc, cudaMemcpyDeviceToHost);

	//std::vector<typecode> dscode(MapNdeformstruc);
	//cudaMemcpy(dscode.data(), DSCodeg, sizeof(typecode) * MapNdeformstruc, cudaMemcpyDeviceToHost);
	//
	std::vector<tfloat4> dspos4(MapNdeformstruc);
	cudaMemcpy(dspos4.data(), DSPos0g, sizeof(float4) * MapNdeformstruc, cudaMemcpyDeviceToHost);
	std::vector<tdouble3> dspos0(MapNdeformstruc);

	std::vector<tfloat4> dsvel04(MapNdeformstruc);
	cudaMemcpy(dsvel04.data(), DSVelg, sizeof(float4) * MapNdeformstruc, cudaMemcpyDeviceToHost);
	std::vector<tdouble3> dsvel(MapNdeformstruc);

	std::vector<tuint2> dspairns(MapNdeformstruc);
	cudaMemcpy(dspairns.data(), DSPairNSg, sizeof(uint2) * MapNdeformstruc, cudaMemcpyDeviceToHost);
	std::vector<unsigned> dspairn(MapNdeformstruc);
	//std::vector<unsigned> dspairstart(MapNdeformstruc);

	//std::vector<float>dskersumg(MapNdeformstruc);
	//cudaMemcpy(dskersumg.data(), DSKerSumVolg, sizeof(float) * MapNdeformstruc, cudaMemcpyDeviceToHost);
	//std::vector<double> dskersum(MapNdeformstruc);

	//std::vector<tfloat4> dskerderlap(DSNumPairs);
	//cudaMemcpy(dskerderlap.data(), DSKerDerLapg, sizeof(float4) * DSNumPairs, cudaMemcpyDeviceToHost);
	//std::vector<tdouble3> dskerder(DSNumPairs);
	//std::vector<double> dskerlap(DSNumPairs);
	//std::vector<unsigned> dspairj(DSNumPairs);
	//cudaMemcpy(dspairj.data(), DSPairJg, sizeof(unsigned) * DSNumPairs, cudaMemcpyDeviceToHost);
	//std::vector<float>dskerg(DSNumPairs);
	//cudaMemcpy(dskerg.data(), DSKerg, sizeof(float) * DSNumPairs, cudaMemcpyDeviceToHost);
	//std::vector<double> dsker(DSNumPairs);
	//
#ifdef OMP_USE
#pragma omp parallel for schedule (static) if(MapNdeformstruc>OMP_LIMIT_COMPUTESTEPDEFSTRUC)
#endif
	for (int p1 = 0; p1 < MapNdeformstruc; p1++)
	{
		tfloat4 dsposp1 = dspos4[p1];
		dspos0[p1] = TDouble3(dsposp1.x, dsposp1.y, dsposp1.z);
		dsposp1 = dsvel04[p1];
		dsvel[p1] = TDouble3(dsposp1.x, dsposp1.y, dsposp1.z);
		tuint2 pairnsp1 = dspairns[p1];
		dspairn[p1] = pairnsp1.x;
		//dspairstart[p1] = pairnsp1.y;
		//dskersum[p1] = dskersumg[p1];
	}

	//#ifdef OMP_USE
	//#pragma omp parallel for schedule (static) if(DSNumPairs>OMP_LIMIT_COMPUTESTEPDEFSTRUC)
	//#endif
	//	for (int p1 = 0; p1 < int(DSNumPairs); p1++)
	//	{
	//		tfloat4 dskerderp1 = dskerderlap[p1];
	//		dskerder[p1] = TDouble3(dskerderp1.x, dskerderp1.y, dskerderp1.z);
	//		dskerlap[p1] = dskerderp1.w;
	//		dsker[p1] = dskerg[p1];
	//	}
	DeformStruc->DSSaveInitDomainInfo(Simulate2D, MapNdeformstruc, DSNpSurf, CaseNdeformstruc,
		nullptr, dspartvbc.data(), nullptr, nullptr, dspos0.data(), dsvel.data(),
		dspairn.data(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
		nullptr, DeformStrucRidpc, nullptr, DSiBodyRidpc,
		DSMeasPlnPartc, DSMeasPlnCntc, DSNMeasPlanes, DefStrucIntDatac, defstrucdatac, DirOut, UserExpressions);
	//DeformStruc->DSSaveInitDomainInfo(Simulate2D, MapNdeformstruc, DSNpSurf, CaseNdeformstruc,
	//	dsparents.data(), dspartvbc.data(), dspartfbc.data(), dscode.data(), dspos0.data(), dsvel.data(),
	//	dspairn.data(), dspairstart.data(), dspairj.data(), dsker.data(), dskerder.data(), dskerlap.data(), dskersum.data(),
	//	dssurfpartList.data(), DeformStrucRidpc, dsbestchild.data(), DSiBodyRidpc, DefStrucIntDatac, defstrucdatac, DirOut);
}

void JSphGpuSingle::DSSetBoundCond(StDeformStrucData* deformstrucdata) const
{

	Log->Print(std::string("  Deformable solid boundary conditions are being set..."));
	std::vector<unsigned> surflistc(DSNpSurf);
	cudaMemcpy(surflistc.data(), DSSurfPartListg, sizeof(unsigned) * DSNpSurf, cudaMemcpyDeviceToHost);
	std::vector<tbcstruc> dspartfbc(MapNdeformstruc);
	std::vector<tbcstruc> dspartvbc(MapNdeformstruc);
	unsigned cnt = 0;
	// Lambda functions to access position data (converting from GPU format)
	auto getPos0 = [this](unsigned i) -> tdouble3 {
		return TDouble3(DSPos0c[i].x, DSPos0c[i].y, DSPos0c[i].z);
		};
	auto getPosc = [this](unsigned i) -> tdouble3 {
		tdouble3 pos;
		pos.x = Posxy[i].x;
		pos.y = Posxy[i].y;
		pos.z = Posz[i];
		return pos;
		};

	// Call shared function
	defstrucbc::SetBoundaryConditions(
		Simulate2D, DeformStrucCount, deformstrucdata, MapNdeformstruc, Npb, DSNpSurf, DSCodec, Code, DSKerSumVolc,
		surflistc.data(), dspartfbc.data(), dspartvbc.data(), getPos0, getPosc, Dp, cnt, UserExpressions);

	// Process phi boundary conditions using compact 8-byte structure
	std::vector<tphibc> dspartphibc_compact(MapNdeformstruc, make_tphibc0());
	unsigned phibc_cnt = 0;
	for (unsigned bodyid = 0; bodyid < DeformStrucCount; bodyid++) {
		StDeformStrucData& body = deformstrucdata[bodyid];
		if (!body.fracture || body.nphibc == 0) continue;

		for (int p1 = 0; p1 < MapNdeformstruc; p1++) {
			unsigned bd = CODE_GetIbodyDeformStruc(DSCodec[p1]);
			if (bd != bodyid) continue;

			for (unsigned k = 0; k < body.nphibc; k++) {
				const tbcstrucbody& bc = body.bcphi[k];

				tphibc& phibc = dspartphibc_compact[p1];
				phibc.exprid = DSBC_GET_PHI_EXPRID(bc.flags);
				phibc.flags = 1;
				phibc_cnt++;
				break;
			}
		}
	}

	cudaMemcpy(DSPartVBCg, dspartvbc.data(), sizeof(tbcstruc) * MapNdeformstruc, cudaMemcpyHostToDevice);
	cudaMemcpy(DSPartFBCg, dspartfbc.data(), sizeof(tbcstruc) * MapNdeformstruc, cudaMemcpyHostToDevice);
	if (DSFracture && DSPartPhiBCg)
		cudaMemcpy(DSPartPhiBCg, dspartphibc_compact.data(), sizeof(tphibc) * MapNdeformstruc, cudaMemcpyHostToDevice);

	Log->Print(std::string("  Deformable structure boundary conditions set for ") + fun::IntStr(cnt) + std::string(" particles"));
	if (phibc_cnt > 0)
		Log->Print(std::string("  Phi boundary conditions set for ") + fun::IntStr(phibc_cnt) + std::string(" particles"));
}

void JSphGpuSingle::DSCalcIbodyRidp(StDeformStrucData* deformstrucdata)const {

	deformstrucdata[0].npstart = 0;
	for (unsigned bodyid = 0; bodyid < DeformStrucCount; bodyid++) {

		if (bodyid > 0) deformstrucdata[bodyid].npstart = deformstrucdata[bodyid - 1].npstart + deformstrucdata[bodyid - 1].npbody;
		unsigned np = 0;
		unsigned npstart = deformstrucdata[bodyid].npstart;
		for (int p1 = 0; p1 < MapNdeformstruc; p1++) {
			unsigned bd = CODE_GetIbodyDeformStruc(DSCodec[p1]);
			if (bd != bodyid) continue;
			DSiBodyRidpc[npstart + np] = p1;
			np++;
		}
	}
}

//==============================================================================
/// Performs Pre-time integration calculations for the deformable structures.
//==============================================================================
void JSphGpuSingle::DSPreTimeInt()
{
	cudaError_t cuerr = cudaGetLastError();
	if (cuerr != cudaSuccess) {
		printf("CUDA Error before initialising Deformable Structures: %s\n", cudaGetErrorString(cuerr));
		exit(0);
	}
	Timersg->TmStart(TMG_SuDeformStruc, false);
	Log->Print("\nInitialising Deformable Structures...");

	//-Report initial GPU memory status
	size_t freeMem, totalMem;
	cudaMemGetInfo(&freeMem, &totalMem);
	Log->Print(fun::PrintStr("  Initial GPU memory: %.2f MB free of %.2f MB total (%.1f%% used)",
		freeMem / (1024.0 * 1024.0), totalMem / (1024.0 * 1024.0),
		100.0 * (1.0 - freeMem / (double)totalMem)));

	DeformStrucCount = DeformStruc->GetCount();
	const float kernelkorg = GetKernelKFactor();

	DSFracture = false;
	DSPlastic = false;
	DSNstep = 0;
	DSPartNstep = 0;
	DSDtModif = 0;
	DSDtModifWrn = 1;
	DSContPowerCoeff = DeformStruc->ContPowerCoeff;

	cudaMemcpy(Code, Codeg, sizeof(typecode) * Npb, cudaMemcpyDeviceToHost);
	DeformStruc->DSConfigCode(Npb, Code, MkInfo);
	cudaMemcpy(Codeg, Code, sizeof(typecode) * Npb, cudaMemcpyHostToDevice);

	if (TBoundary == BC_MDBC && cudefstr::DSHasNormalsg(Npb, Codeg, BoundNormalg))
		Run_Exceptioon("mDBC normals are not permitted to be set for a deformable structure.");
	size_t m = 0;
	StDeformStrucData* defstrucdatag, * defstrucdatac;
	m = sizeof(StDeformStrucIntData) * DeformStrucCount; cudaMalloc((void**)&DefStrucIntDatag, m); MemGpuFixed += m;
	DefStrucIntDatac = (StDeformStrucIntData*)malloc(sizeof(StDeformStrucIntData) * DeformStrucCount);
	MemCpuFixed += DeformStrucCount * sizeof(StDeformStrucIntData);
	m = sizeof(StDeformStrucData) * DeformStrucCount; cudaMalloc((void**)&defstrucdatag, m);
	defstrucdatac = new StDeformStrucData[DeformStrucCount]; MemCpuFixed += (sizeof(StDeformStrucData) * DeformStrucCount);
	for (unsigned bodyid = 0; bodyid < DeformStrucCount; bodyid++) {
		JSphDeformStrucBody& body = *DeformStruc->List[bodyid];
		StDeformStrucData& DSdata = defstrucdatac[bodyid];
		DSdata.vol0 = body.GetPartVol(); DSdata.rho0 = body.GetDensity(); DSdata.youngmod = body.GetYoungMod();
		DSdata.poissratio = body.GetPoissRatio(); DSdata.gc = body.GetGc(); DSdata.constitmodel = body.GetConstModel();
		DSdata.lamemu = body.GetLameMu(); DSdata.lamelambda = body.GetLameLmbda(); DSdata.lamebulk = body.GetLameBulk();
		DSdata.czero = body.GetSoundSpeed(); DSdata.fracture = body.GetFracture(); DSdata.pfLim = body.GetPfLim();
		DSdata.lenscale = body.GetLenScale(); DSdata.mkbound = body.MkBound; DSdata.avfactor1 = body.GetAvFactor1();
		DSdata.avfactor2 = body.GetAvFactor2(); DSdata.dp = body.GetDp();
		DSdata.mapfact = body.GetMapfact(); DSdata.nvbc = body.GetNvBC(); DSdata.nfbc = body.GetNfBC(); DSdata.nphibc = body.GetNphiBC(); DSdata.nnotch = body.GetNnotch();
		DSdata.particlemass = DSdata.vol0 * DSdata.rho0;
		DSdata.nmeasplane = body.GetNMeasPlane(); DSdata.yieldstress = body.GetYieldStress(); DSdata.hardening = body.GetHardening();
		DSdata.kfric = body.GetKFric(); DSdata.restcoeff = body.GetRestCoeff(); DSdata.nbsrange = body.GetNBsfact();
		memcpy(DSdata.bcvel, body.GetVBClist(), sizeof(tbcstrucbody) * DSdata.nvbc);
		memcpy(DSdata.bcforce, body.GetFBClist(), sizeof(tbcstrucbody) * DSdata.nfbc);
		memcpy(DSdata.bcphi, body.GetPhiBClist(), sizeof(tbcstrucbody) * DSdata.nphibc);
		memcpy(DSdata.notchlist, body.GetNotchList(), sizeof(plane4Nstruc) * DSdata.nnotch);
		memcpy(DSdata.measplanelist, body.GetMeasPlaneList(), sizeof(plane4Nstruc) * DSdata.nmeasplane);

		DSdata.kernelh = body.GetKernelh();
		DSdata.kernelsize = body.GetKernelsize();
		DSdata.kernelsize2 = DSdata.kernelsize * DSdata.kernelsize;
		DSdata.selfkern = DSGetSelfKernBody(DSdata.kernelh);
		if (DSdata.fracture) DSFracture = true;
		if (DSdata.constitmodel == CONSTITMODEL_J2) {
			DSdata.fracture = false;
			DSPlastic = true;
		}
		else if (DSdata.fracture) DSFracture = true;
		DSdata.tau = (1.0f - DSdata.poissratio * DSdata.poissratio) / DSdata.youngmod;
		DSdata.npbody = 0; DSdata.npstart = 0;
		DSdata.min = TDouble3(DBL_MAX, DBL_MAX, DBL_MAX);
		DSdata.max = TDouble3(-DBL_MAX, -DBL_MAX, -DBL_MAX);
	}

	cudaMemcpy(defstrucdatag, defstrucdatac, sizeof(StDeformStrucData) * DeformStrucCount, cudaMemcpyHostToDevice);

	CaseNdeformstruc = cudefstr::DSCountOrgParticlesg(Npb, Codeg);

	m = sizeof(unsigned) * CaseNdeformstruc;        cudaMalloc((void**)&DeformStrucRidpg, m);	MemGpuFixed += m;
	m = sizeof(unsigned) * CaseNdeformstruc;		cudaMalloc((void**)&DSBestChildg, m);		MemGpuFixed += m;
	m = sizeof(float2) * CaseNdeformstruc;			cudaMalloc((void**)&DSPosOrg0xyg, m);		MemGpuFixed += m;
	m = sizeof(float) * CaseNdeformstruc;			cudaMalloc((void**)&DSPosOrg0zg, m);		MemGpuFixed += m;
	DSPosOrg0xyc = new tfloat2[CaseNdeformstruc];		MemCpuFixed += (sizeof(tfloat2) * CaseNdeformstruc);
	DSPosOrg0zc = new float[CaseNdeformstruc];			MemCpuFixed += (sizeof(float) * CaseNdeformstruc);
	DeformStrucRidpc = new unsigned[CaseNdeformstruc];	MemCpuFixed += (sizeof(unsigned) * CaseNdeformstruc);

	cudefstr::DSCalcRidpg(Npb, DeformStrucRidpg, Codeg);
	cudaMemcpy(DeformStrucRidpc, DeformStrucRidpg, sizeof(unsigned) * CaseNdeformstruc, cudaMemcpyDeviceToHost);

	DSGatherArray(CaseNdeformstruc, DeformStrucRidpc, Posxy, DSPosOrg0xyc);
	DSGatherArray(CaseNdeformstruc, DeformStrucRidpc, Posz, DSPosOrg0zc);

	cudaMemcpy(DSPosOrg0xyg, DSPosOrg0xyc, sizeof(float2) * CaseNdeformstruc, cudaMemcpyHostToDevice);
	cudaMemcpy(DSPosOrg0zg, DSPosOrg0zc, sizeof(float) * CaseNdeformstruc, cudaMemcpyHostToDevice);

	MapNdeformstruc = cudefstr::DSCountMappedParticlesg(CaseNdeformstruc, Simulate2D, DeformStrucRidpg, Codeg, defstrucdatag);
	cuerr = cudaGetLastError();
	if (cuerr != cudaSuccess) {
		printf("CUDA Error before allocating Deformable Structures CPU GPU memories: %s\n", cudaGetErrorString(cuerr));
		exit(0);
	}
	DSAllocGpuMemoryFixed();
	DSAllocCpuMemoryFixed();
	cuerr = cudaGetLastError();
	if (cuerr != cudaSuccess) {
		printf("CUDA Error after allocating Deformable Structures CPU GPU memories: %s\n", cudaGetErrorString(cuerr));
		exit(0);
	}
	cudefstr::DSGenMappedParticles(CaseNdeformstruc, DeformStrucRidpg, Codeg,
		defstrucdatag, DSPos0g, DSPosOrg0xyg, DSPosOrg0zg, DSParentg, DSCodeg, Simulate2D);

	cudaMemcpy(defstrucdatac, defstrucdatag, sizeof(StDeformStrucData) * DeformStrucCount, cudaMemcpyDeviceToHost);
	defstrucdatac[0].npstart = 0;
	for (unsigned bodyid = 0; bodyid < DeformStrucCount; ++bodyid) {
		StDeformStrucData& body = defstrucdatac[bodyid];
		body.min -= body.dp * 0.5;
		body.max += body.dp * 0.5;
		body.mass = body.vol0 * body.rho0 * body.npbody;
		if (bodyid > 0) defstrucdatac[bodyid].npstart = defstrucdatac[bodyid - 1].npstart + defstrucdatac[bodyid - 1].npbody;
	}

	cudaMemcpy(defstrucdatag, defstrucdatac, sizeof(StDeformStrucData) * DeformStrucCount, cudaMemcpyHostToDevice);

	Log->Print(fun::PrintStr("  Number of particles generated for mapped domains:"));
	for (unsigned bodyid = 0; bodyid < DeformStrucCount; ++bodyid) {
		StDeformStrucData& body = defstrucdatac[bodyid];
		Log->Print(fun::PrintStr("    Deformable Structure %u (mkbound %u): %u", bodyid, body.mkbound, body.npbody));
	}
	Log->Print(fun::PrintStr("  Total mass of deformable bodies:"));
	for (unsigned bodyid = 0; bodyid < DeformStrucCount; ++bodyid) {
		StDeformStrucData& body = defstrucdatac[bodyid];
		Log->Print(fun::PrintStr("    Deformable Structure %u (mkbound %u): %f", bodyid, body.mkbound, body.mass));
	}

	cudaMemcpy(DSCodec, DSCodeg, sizeof(typecode) * MapNdeformstruc, cudaMemcpyDeviceToHost);
	cudaMemcpy(DSDispPhic, DSDispPhig, sizeof(float4) * MapNdeformstruc, cudaMemcpyDeviceToHost);
	cudaMemcpy(DSPos0c, DSPos0g, sizeof(float4) * MapNdeformstruc, cudaMemcpyDeviceToHost);

	DSPerformCellDiv(defstrucdatac);

	DSCalcIbodyRidp(defstrucdatac);
	cudaMemcpy(DSiBodyRidpg, DSiBodyRidpc, sizeof(unsigned) * MapNdeformstruc, cudaMemcpyHostToDevice);

	cudefstr::DSDetermineMapCenters(CaseNdeformstruc, DeformStrucRidpg, DSPosOrg0xyg, DSPosOrg0zg, Codeg,
		defstrucdatag, DSDivData, DSCodeg, DSPos0g, DSDcellg, DSBestChildg);

	DSNumPairs = cudefstr::DSCountTotalPairs(MapNdeformstruc, DSPos0g, DSParentg, DSCodeg,
		defstrucdatag, DSPairNSg, DSDivData, Simulate2D);

	if (fabs(DSNumPairs - fabs(DSNumPairs)) > 0.5)
		Run_Exceptioon("Number of deformable structure pairs is very high. Decrease kernel size or increase particle distancing.");
	else Log->Print(std::string("  Total deformable solid pairs=") + fun::IntStr(DSNumPairs));

	cudaMemcpy(DSParentc, DSParentg, sizeof(unsigned) * MapNdeformstruc, cudaMemcpyDeviceToHost);
	cuerr = cudaGetLastError();
	if (cuerr != cudaSuccess) {
		printf("CUDA Error before allocating neighbour CPU GPU memories: %s\n", cudaGetErrorString(cuerr));
		exit(0);
	}
	m = sizeof(float) * DSNumPairs;			cudaMalloc((void**)&DSKerg, m);			MemGpuFixed += m;
	m = sizeof(float4) * DSNumPairs;		cudaMalloc((void**)&DSKerDerLapg, m);	MemGpuFixed += m;
	m = sizeof(unsigned) * DSNumPairs;		cudaMalloc((void**)&DSPairJg, m);		MemGpuFixed += m;
	m = sizeof(float) * ((MapNdeformstruc / SPHBSIZE + 1) + (MapNdeformstruc / (SPHBSIZE * SPHBSIZE) + SPHBSIZE));
	cudaMalloc(&DSblockMinMax, m); MemGpuFixed += m;
	cuerr = cudaGetLastError();
	if (cuerr != cudaSuccess) {
		printf("CUDA Error after allocating neighbour CPU GPU memories: %s\n", cudaGetErrorString(cuerr));
		exit(0);
	}
	cudefstr::DSCalcKers(MapNdeformstruc, TKernel, Simulate2D, DSCodeg, DSPos0g,
		defstrucdatag, DSDivData, DSDcellg, DSPairNSg, DSKerg,
		DSKerDerLapg, DSPairJg, DSKerSumVolg);
	cudaMemcpy(DSKerSumVolc, DSKerSumVolg, sizeof(float) * MapNdeformstruc, cudaMemcpyDeviceToHost);

	cudefstr::DSFindSurfParticles(MapNdeformstruc, DSPos0g, DSCodeg, DSPairNSg, DSPairJg, DSKerg,
		defstrucdatag, DSNpSurf, DSSurfPartListg, MemGpuFixed);
	cuerr = cudaGetLastError();
	if (cuerr != cudaSuccess) {
		printf("CUDA Error after DSFindSurfParticles: %s\n", cudaGetErrorString(cuerr));
		exit(0);
	}
	Log->Print(std::string("  Total surface particles=") + fun::IntStr(DSNpSurf));

	for (unsigned bodyid = 0; bodyid < DeformStrucCount; bodyid++) {
		StDeformStrucData& DSdata = defstrucdatac[bodyid];
		StDeformStrucIntData& dsintdata = DefStrucIntDatac[bodyid];
		dsintdata = StDeformStrucIntData(DSdata.npbody, DSdata.npstart, DSdata.mkbound, DSdata.pfLim,
			DSdata.fracture, DSdata.lamelambda,
			DSdata.lamebulk, DSdata.lamemu, DSdata.vol0, DSdata.rho0,
			DSdata.czero, DSdata.gc, float(DSdata.lenscale), DSdata.avfactor1, DSdata.avfactor2,
			DSdata.mass, DSdata.kfric, DSdata.yieldstress, DSdata.hardening, DSdata.tau, DSdata.restcoeff, DSdata.mapfact,
			DSdata.kernelh, DSdata.constitmodel, DSdata.nmeasplane, float(DSdata.dp), DSdata.youngmod, DSdata.nbsrange, DSdata.selfkern);
	}
	cudaMemcpy(DefStrucIntDatag, DefStrucIntDatac, sizeof(StDeformStrucIntData) * DeformStrucCount, cudaMemcpyHostToDevice);

	DSSetBoundCond(defstrucdatac);
	cuerr = cudaGetLastError();
	if (cuerr != cudaSuccess) {
		printf("CUDA Error after DSSetBoundCond: %s\n", cudaGetErrorString(cuerr));
		exit(0);
	}
	cudefstr::DSInitFieldVars(TStep, MapNdeformstruc, DSCodeg, defstrucdatag, DSPartVBCg, DSPartFBCg, DSPartPhiBCg, DSKerSumVolg, DSAcclg, DSFlForceg, DSDispPhig, DSEqPlasticg, DSPlasticStraing,
		DSDefGradg2D, DSDefGradg3D, DSDefPk, DSVelg, DSVelPreg, DSPos0g, DSPhiTdatag, DSDispCorxzg, DSDispCoryg,
		Simulate2D, DSFracture, UserExpressionsg, float(TimeStep), Gravity);
	cuerr = cudaGetLastError();
	if (cuerr != cudaSuccess) {
		printf("CUDA Error after DSInitFieldVars: %s\n", cudaGetErrorString(cuerr));
		exit(0);
	}
	DSMaxDt = cudefstr::DSCalcMaxInitTimeStep(MapNdeformstruc, DeformStrucCount, DSCodeg, defstrucdatac, defstrucdatag, DSPairNSg,
		DSPos0g, DSDispPhig, DSVelg, DSAcclg, DSPartFBCg, Gravity, DSPairJg, UserExpressionsg, float(TimeStep));

	cuerr = cudaGetLastError();
	if (cuerr != cudaSuccess) {
		printf("CUDA Error after DSCalcMaxInitTimeStep: %s\n", cudaGetErrorString(cuerr));
		exit(0);
	}
	measplane::DSfindPointsNearestToMeasurePlanes(defstrucdatac, DeformStrucCount, DSNMeasPlanes,
		DSNPartMeasPlanes, MemCpuFixed, DSMeasPlOutFiles, DSMeasPlnCntc, DSCodec, DSPos0c,
		DSMeasPlnPartc, MapNdeformstruc, Log, Simulate2D);
	cuerr = cudaGetLastError();
	if (cuerr != cudaSuccess) {
		printf("CUDA Error after DSfindPointsNearestToMeasurePlanes: %s\n", cudaGetErrorString(cuerr));
		exit(0);
	}
	DSIntArraysg = StDeformStrucIntArraysg(DSCodeg, DSPairJg, DSPairNSg,
		DSKerg, DSPartVBCg, DSKerDerLapg, DSPos0g, DSPartFBCg, DSPartPhiBCg, DSDispPhig, DSEqPlasticg, DSPlasticStraing, DSDefGradg2D,
		DSPiolKirg2D, DSDefGradg3D, DSPiolKirg3D, DSPhiTdatag, DSAcclg, DSVelg, DSVelPreg,
		DSDefPk, DSDispCorxzg, DSDispCoryg, DSblockMinMax, DeformStruc);

	DSSaveInitDomainInfo(defstrucdatac);
	cuerr = cudaGetLastError();
	if (cuerr != cudaSuccess) {
		printf("CUDA Error after DSSaveInitDomainInfo: %s\n", cudaGetErrorString(cuerr));
		exit(0);
	}
	//cudefstr::ValidateExpressions(UserExpressionsg, UserExpressions, 1, 1.0, defstrucdatac);

	Log->Print(std::string("  Maximum deformable structure timestep (initial)=") + fun::DoubleStr(DSMaxDt) + std::string(" s"));
	Log->Print("Computational domain for deformable structure(s) is set");
	Log->Print("");
	
	cudaMemGetInfo(&freeMem, &totalMem);
	Log->Print(fun::PrintStr("GPU memory after DSPreTimeInt: %.2f MB free of %.2f MB total (%.1f%% used)",
		freeMem / (1024.0 * 1024.0), totalMem / (1024.0 * 1024.0),
		100.0 * (1.0 - freeMem / (double)totalMem)));
	cuerr = cudaGetLastError();
	if (cuerr != cudaSuccess) Run_ExceptioonCuda(cuerr, fun::PrintStr(" >> Cuda failed during initialization of deformable structure(s)."));

	Timersg->TmStop(TMG_SuDeformStruc, false);



	/*float tt = 0.0;
	float stepdt = 1.0e-9f;
	std::cout << "\n\nStart";
	auto start = std::chrono::high_resolution_clock::now();
	for (int ii = 0; ii <= 1000; ii++) {
		DSMaxDt = cudefstr::DSInteraction_Forces(MapNdeformstruc, DeformStrucCount, DefStrucIntDatag, DSIntArraysg,
			DSDcellg, Gravity, true, false, DSDomRealPosMin, DSDomRealPosMax, DSDomPosMin, DScellsize,
			DSDomCellCode, DSDivData, DSCellDivSingle, Timersg, float(DeformStruc->ContactR), float(DeformStruc->ContactP), \
			DSNpSurf, DSSurfPartListg, UserExpressionsg, tt);
		cudefstr::DSCompSemImplEuler(MapNdeformstruc, DeformStrucCount, stepdt, DefStrucIntDatag, DSIntArraysg, Simulate2D, UserExpressionsg, tt);
		tt += stepdt;
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << "Time taken for 1000 steps: " << duration.count() * 1.0e-6 << " secs" << std::endl;
	std::cout.flush();
	cudaMemcpy(DSDispPhic, DSDispPhig, sizeof(tfloat4) * MapNdeformstruc, cudaMemcpyDeviceToHost);
	JDataArrays dsarrays;
	tfloat3* defstrucenergies = new tfloat3[DeformStrucCount];
	tsymatrix3f* dscauchystress = new tsymatrix3f[MapNdeformstruc];
	AddBasicArraysDS(dsarrays, MapNdeformstruc, DSPos0c, DSDispPhic, DSParentc, DSCodec, DSiBodyRidpc,
		DeformStrucCount, defstrucenergies, dscauchystress);
	DSSaveDataf(MapNdeformstruc, dsarrays, DefStrucIntDatac, trackparticleid);
	std::cout << "\n ======================================== > " << Part << "\t" << TimeStep;

	exit(0);*/
}



template<bool simulate2d, bool defsttm>
void JSphGpuSingle::DSComputeStep_Ver(const double dt)
{
	double subdt = 0.0;
	while (subdt < dt) {
		double remaining = dt - subdt;
		if (remaining <= ALMOSTZERO) break;
		double dsdt = defsttm ? DeformStruc->UseUsrTimeStep : CFLnumber * DSMaxDt;
		if (dsdt < DtMin) {
			dsdt = DtMin; DSDtModif++;
			if (DSDtModif >= DSDtModifWrn) {
				Log->PrintfWarning("%d DTs adjusted to DtMin for deformable structure (t:%g, nstep:%u)", DSDtModif, TimeStep, DSNstep);
				DSDtModifWrn *= 10;
			}
		}
		if (dsdt > remaining) dsdt = remaining;

		if (DeformStrucCount > 1) cusph::DSInteraction_ForcesDem(Simulate2D, MapNdeformstruc, DeformStrucCount, DefStrucIntDatag, DSIntArraysg, DivData,
			Codeg, Posxyg, Poszg, Velrhopg, DSFlForceg, float(Dp), float(dsdt), DSSurfPartListg, DSContPowerCoeff,
			DSParentg, DeformStrucRidpg, Dcellg);

		DSMaxDt = cudefstr::DSInteraction_Forces(MapNdeformstruc, DeformStrucCount, DefStrucIntDatag, DSIntArraysg,
			DSDcellg, Gravity, simulate2d, DSDomRealPosMin, DSDomRealPosMax, DSDomPosMin, DScellsize,
			DSDomCellCode, DSDivData, DSCellDivSingle, Timersg,
			DSNpSurf, DSSurfPartListg, DSKerSumVolg, UserExpressionsg, DSFlForceg, float(subdt + TimeStep), float(dsdt));
		cudefstr::DSCompSemImplEuler(MapNdeformstruc, DeformStrucCount, float(dsdt), DefStrucIntDatag, DSIntArraysg, Simulate2D, UserExpressionsg, float(subdt + TimeStep));

		cudefstr::DSUpdate_OrgVelPos(CaseNdeformstruc, DeformStrucRidpg, DSBestChildg, DSPosOrg0xyg,
			DSPosOrg0zg, DSDispPhig, Posxyg, Poszg, simulate2d, Velrhopg, DSVelg);

		subdt += dsdt;
		DSNstep++;
	}
	cudaDeviceSynchronize();
	std::cout.flush();
}

//========================================================================================
/// Performs deformable structures' calculations for 1 time step of the main solver.
//========================================================================================
template<bool simulate2d, bool defsttm>
void JSphGpuSingle::DSComputeStep_Sym(const double dt, double starttime) {
	float subdt = 0.0;
	float currenttime = float(starttime);
	while (subdt < dt) {
		const float remaining = float(dt - subdt);
		if (remaining <= ALMOSTZERO) break;

		float dsdt = float(defsttm ? DeformStruc->UseUsrTimeStep : CFLnumber * DSMaxDt);
		if (dsdt < float(DtMin)) {
			dsdt = float(DtMin); DSDtModif++;
			if (DSDtModif >= DSDtModifWrn) {
				Log->PrintfWarning("%d DTs adjusted to DtMin for deformable structure (t:%g, nstep:%u)", DSDtModif, TimeStep, DSNstep);
				DSDtModifWrn *= 10;
			}
		}

		if (dsdt > remaining) dsdt = remaining;
		const float dtm = dsdt * 0.5f;

		/*if (DeformStrucCount > 1)
			cudefstr::DSInteractionForcesDEM(Simulate2D, MapNdeformstruc, DeformStrucCount, DefStrucIntDatag, DSIntArraysg, DivData,
				Codeg, Posxyg, Poszg, Velrhopg, DSFlForceg, float(Dp), float(dsdt), DSSurfPartListg, DSContPowerCoeff,
				Dcellg);*/
		if (DeformStrucCount > 1) cusph::DSInteraction_ForcesDem(Simulate2D, MapNdeformstruc, DeformStrucCount, DefStrucIntDatag, DSIntArraysg, DivData,
			Codeg, Posxyg, Poszg, Velrhopg, DSFlForceg, float(Dp), float(dsdt), DSSurfPartListg, DSContPowerCoeff,
			DSParentg, DeformStrucRidpg, Dcellg);
		cudefstr::DSCompSympPre(MapNdeformstruc, DeformStrucCount, dtm, DefStrucIntDatag, DSIntArraysg, Simulate2D, UserExpressionsg, currenttime);

		DSMaxDt = cudefstr::DSInteraction_Forces(MapNdeformstruc, DeformStrucCount, DefStrucIntDatag, DSIntArraysg,
			DSDcellg, Gravity, simulate2d, DSDomRealPosMin, DSDomRealPosMax, DSDomPosMin, DScellsize,
			DSDomCellCode, DSDivData, DSCellDivSingle, Timersg,
			DSNpSurf, DSSurfPartListg, DSKerSumVolg, UserExpressionsg, DSFlForceg, currenttime, dsdt);

		cudefstr::DSCompSympCor(MapNdeformstruc, DeformStrucCount, dtm, DefStrucIntDatag, DSIntArraysg, Simulate2D, UserExpressionsg, currenttime);

		cudefstr::DSUpdate_OrgVelPos(CaseNdeformstruc, DeformStrucRidpg, DSBestChildg, DSPosOrg0xyg,
			DSPosOrg0zg, DSDispPhig, Posxyg, Poszg, simulate2d, Velrhopg, DSVelg);

		subdt += dsdt;
		currenttime += dsdt;
		DSNstep++;
	}
}


