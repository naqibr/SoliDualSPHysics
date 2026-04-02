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

/// \file JSphDeformStruc.cpp \brief Implements the class \ref JSphDeformStruc.
/// 
/// This file implements the deformable structure framework for SoliDualSPHysics.
/// 
/// \authors Dr. Naqib Rahimi, Dr. Georgios Moutsanidis

#include "JSphDeformStruc.h"
#include "JLog2.h"
#include "JXml.h"
#include "FunSphKernel.h"
#include "FunSphKernelsCfg.h"
#include "JSphMk.h"
#include "JDsDcell.h"


//##############################################################################
//# JSphDeformStrucBody
//##############################################################################
//========================================================================================
/// Constructor.
//========================================================================================
JSphDeformStrucBody::JSphDeformStrucBody(const double dpb,const bool simulate2d,
unsigned idbody, word mkbound, float density, double youngmod, double poissratio, TpConstitModel constitmodel, float avfactor1, float avfactor2,
bool fracturei, float Gci, unsigned mapfac, float pflimiti, float pflenscale, float restcoef, float kfric,
double usrlambda, double usrMu, double usrBulk, double yieldstress, double hardening,
tbcstrucbody* bcvel, unsigned nmbcvel, tbcstrucbody* bcforce, unsigned nmbcforce, tbcstrucbody* bcphi, unsigned nmbcphi,
plane4Nstruc* notchlist, unsigned nmnotch, plane4Nstruc* measplist, unsigned nmmeasp, unsigned nbsrangei)
:Log(AppInfo.LogPtr()), IdBody(idbody), MkBound(mkbound),Dp(dpb)
{
	ClassName = "JSphDeformStrucBody";
	Reset();
	PartVol = static_cast<float> (simulate2d ? Dp * Dp : Dp * Dp * Dp);
	Density = density;
	RestCoeff = restcoef;
	KFric = kfric;
	YieldStress = static_cast<float>(yieldstress);
	Hardening = static_cast<float>(hardening);
	ConstitModel = constitmodel;
	AvFactor1 = avfactor1;
	AvFactor2 = avfactor2;
	Fracture = (ConstitModel != CONSTITMODEL_J2) && fracturei;
	Gc = Gci;
	MapFact = mapfac;
	NBsrange = nbsrangei;
	PfLim = (pflimiti > 1 ? 1 : pflimiti);
	LenScale = pflenscale;
	calc_elastic_const(simulate2d, usrlambda, usrMu, usrBulk, youngmod, poissratio);
	cZero = static_cast<float> (sqrt((LameBulk + 4.0/3.0 * LameMu) / density));

	NvBC = nmbcvel; if (NvBC) for (unsigned i = 0; i < NvBC; i++) BodyVBClist[i] = bcvel[i];
	NfBC = nmbcforce; if (NfBC) for (unsigned i = 0; i < NfBC; i++) BodyFBClist[i] = bcforce[i];
	NphiBC = nmbcphi; if (NphiBC) for (unsigned i = 0; i < NphiBC; i++) BodyPhiBClist[i] = bcphi[i];
	Nnotch = nmnotch; if (Nnotch) for (unsigned i = 0; i < Nnotch; i++) BodyNList[i] = notchlist[i];
	Nmeaspl = nmmeasp; if (Nmeaspl) for (unsigned i = 0; i < Nmeaspl; i++) BodyMPList[i] = measplist[i];
}

void JSphDeformStrucBody::calc_elastic_const(bool simulate2d,
	double usrlambda, double usrMu,
	double usrBulk,
	double youngmod, double poissratio)
{
	double dim = simulate2d ? 2.0 : 3.0;
	//double dim = 3;
	auto from_lambda_mu = [&](double lam, double mu) {
		LameLmbda = static_cast<float>(lam);
		LameMu = static_cast<float>(mu);
		LameBulk = static_cast<float>(lam + 2.0 * mu / dim);

		const double den = (dim - 1.0) * lam + 2.0 * mu;
		YoungMod = static_cast<float>(2.0 * mu * (dim * lam + 2.0 * mu) / den);
		PoissRatio = static_cast<float>(lam / den);
		};

	if (usrlambda && usrMu) {
		from_lambda_mu(usrlambda, usrMu);
	}
	else if (usrBulk && usrMu) {
		const double mu = usrMu;
		const double Kd = usrBulk;
		const double lam = Kd - 2.0 * mu / dim;
		from_lambda_mu(lam, mu);
	}
	else if (usrBulk && usrlambda) {
		const double lam = usrlambda;
		const double Kd = usrBulk;
		const double mu = 0.5 * dim * (Kd - lam);
		from_lambda_mu(lam, mu);
	}
	else {
		const double E = youngmod;
		const double nu = poissratio;

		const double mu = E / (2.0 * (1.0 + nu));
		const double lam = E * nu / ((1.0 + nu) * (1.0 - (dim - 1.0) * nu));
		from_lambda_mu(lam, mu);
	}
}

//========================================================================================
/// Destructor.
//========================================================================================
JSphDeformStrucBody::~JSphDeformStrucBody() {
	DestructorActive = true;
	Reset();
}

//========================================================================================
/// Initialisation of variables.
//========================================================================================
void JSphDeformStrucBody::Reset() {
	BoundCode = 0;
	PartVol = 0;
	Density = 0;
	YoungMod = 0;
	PoissRatio = 0;
	ConstitModel = CONSTITMODEL_SVK;
	AvFactor1 = AvFactor2 = 0;
	Fracture = false;
	Gc = 0.0;
	MapFact = 0;
	YieldStress = 0;
	Hardening = 0;
	PfLim = 0.0;
	LenScale = 0.0;
	NvBC = 0;
	NfBC = 0;
	NphiBC = 0;
	Nnotch = 0;
	for (unsigned i = 0; i < MAX_VELBC_DEFSTRUC; i++) BodyVBClist[i] = make_tbcstrucbody0();
	for (unsigned i = 0; i < MAX_FORCEBC_DEFSTRUC; i++) BodyFBClist[i] = make_tbcstrucbody0();
	for (unsigned i = 0; i < MAX_PHIBC_DEFSTRUC; i++) BodyPhiBClist[i] = make_tbcstrucbody0();
	for (unsigned i = 0; i < MAX_NOTCH_DEFSTRUC; i++) BodyNList[i] = TNotchStrucNull();
}

//========================================================================================
/// Configures BoundCode.
//========================================================================================
void JSphDeformStrucBody::ConfigBoundCode(typecode boundcode) {
	if (BoundCode)Run_Exceptioon(fun::PrintStr("BoundCode was already configured for mkbound=%u.", MkBound));
	BoundCode = boundcode;
}

//========================================================================================
// Configures the codes for mkbound(s) associated with boundary conditions of deformable structures 
//========================================================================================
void JSphDeformStrucBody::ConfigBCCodeBody(const JSphMk* mkinfo) {

	for (unsigned i = 0; i < NfBC; i++) {
		const unsigned cmk = mkinfo->GetMkBlockByMkBound(BodyFBClist[i].mkid);
		if (cmk < mkinfo->Size()) BodyFBClist[i].mkcode = mkinfo->Mkblock(cmk)->Code;
		else if (DSBC_GET_FORCETYPE(BodyFBClist[i].flags) == DSBC_FORCETYPE_SURFACE) {
			if (DSBC_GET_X_IS_EXPR(BodyFBClist[i].flags) | DSBC_GET_Z_IS_EXPR(BodyFBClist[i].flags) | DSBC_GET_Z_IS_EXPR(BodyFBClist[i].flags)) continue;
			Run_Exceptioon(fun::PrintStr("mkid=%u for bcforce in deformable structure body with mkbound=%u is not a valid Mk boundary.", BodyFBClist[i].mkid, MkBound));
		}
	}

	for (unsigned i = 0; i < NvBC; i++) {
		const unsigned cmk = mkinfo->GetMkBlockByMkBound(BodyVBClist[i].mkid);
		if (cmk < mkinfo->Size()) BodyVBClist[i].mkcode = mkinfo->Mkblock(cmk)->Code;
		else if (DSBC_GET_SURFACE(BodyVBClist[i].flags)) {
			if (DSBC_GET_X_IS_EXPR(BodyFBClist[i].flags) | DSBC_GET_Z_IS_EXPR(BodyFBClist[i].flags) | DSBC_GET_Z_IS_EXPR(BodyFBClist[i].flags)) continue;
			Run_Exceptioon(fun::PrintStr("mkid=%u for bcvel in deformable structure body with mkbound=%u is not a valid Mk boundary.", BodyVBClist[i].mkid, MkBound));
		}
	}
}


//========================================================================================
/// Loads lines with configuration information.
//========================================================================================
void JSphDeformStrucBody::GetConfig(std::vector<std::string>& lines)const {
	lines.push_back(fun::PrintStr("    Particle map factor: %i", GetMapfact()));
	lines.push_back(fun::PrintStr("    Particle distance: %g", GetDp()));
	lines.push_back(fun::PrintStr("    KernelH: %g", GetKernelh()));
	lines.push_back(fun::PrintStr("    KernelSize: %g", GetKernelsize()));
	lines.push_back(fun::PrintStr("    Particle volume: %g", GetPartVol()));
	lines.push_back(fun::PrintStr("    Particle mass: %g", GetPartMass()));
	lines.push_back(fun::PrintStr("    Density: %g", GetDensity()));
	lines.push_back(fun::PrintStr("    Young's modulus: %g", GetYoungMod()));
	lines.push_back(fun::PrintStr("    Shear modulus: %g", GetLameMu()));
	lines.push_back(fun::PrintStr("    Bulk modulus: %g", GetLameBulk()));
	lines.push_back(fun::PrintStr("    Lame parameter (lambda): %g", GetLameLmbda()));
	lines.push_back(fun::PrintStr("    Poisson ratio: %g", GetPoissRatio()));
	lines.push_back(fun::PrintStr("    Sound speed: %g", GetSoundSpeed()));
	lines.push_back(fun::PrintStr("    Restitution coefficient: %g", GetRestCoeff()));
	lines.push_back(fun::PrintStr("    Kinetic friction coefficient: %g", GetKFric()));
	lines.push_back(fun::PrintStr("    Artificial viscosity factor1: %g, factor2: %g", GetAvFactor1(), GetAvFactor2()));
	if (ConstitModel == CONSTITMODEL_J2) {
		lines.push_back("    Fracture: 0 (disabled for plasticity)");
	}
	else if (GetFracture()) {
		lines.push_back(fun::PrintStr("    Fracture: %i", GetFracture()));
		lines.push_back(fun::PrintStr("    Gc: %g", GetGc()));
		lines.push_back(fun::PrintStr("    Phase field length scale: %g", GetLenScale()));
		lines.push_back(fun::PrintStr("    Phase field length scale limit: %g", GetPfLim()));
	}

	std::string cmodelstr;
	if (ConstitModel == CONSTITMODEL_SVK) cmodelstr = "St. Venant Kirchhoff";
	else if (ConstitModel == CONSTITMODEL_NH) cmodelstr = "Neo-Hookean";
	else cmodelstr = "J2 elasto-plastic";
	lines.push_back(fun::PrintStr("    Constitutive model: %s", cmodelstr.c_str()));
	if (ConstitModel == CONSTITMODEL_J2) {
		lines.push_back(fun::PrintStr("    Yield stress: %g", GetYieldStress()));
		lines.push_back(fun::PrintStr("    Hardening modulus: %g", GetHardening()));
	}
	unsigned nmbc = GetNvBC();
	if (nmbc) {
		const tbcstrucbody* bcvel = GetVBClist();
		lines.push_back(fun::PrintStr("    Velocity boundary conditions (Total:%u):", nmbc));
		for (unsigned i = 0; i < nmbc; i++) {
			std::ostringstream curline;
			if(DSBC_GET_SURFACE(bcvel[i].flags)) curline << "      surface near mkid=\"" << bcvel[i].mkid << "\":";
			else curline << "      body:";
			if(DSBC_GET_X_FLAG(bcvel[i].flags)) curline << " x=\"" << bcvel[i].x << "\"";
			else if (DSBC_GET_X_IS_EXPR(bcvel[i].flags)) curline << " xe_id=\"" << DSBC_GET_X_EXPRID(bcvel[i].flags) << "\"";

			if (DSBC_GET_Y_FLAG(bcvel[i].flags)) curline << " y=\"" << bcvel[i].y << "\"";
			else if (DSBC_GET_Y_IS_EXPR(bcvel[i].flags)) curline << " ye_id=\"" << DSBC_GET_Y_EXPRID(bcvel[i].flags) << "\"";

			if (DSBC_GET_Z_FLAG(bcvel[i].flags)) curline << " z=\"" << bcvel[i].z << "\"";
			else if (DSBC_GET_Z_IS_EXPR(bcvel[i].flags)) curline << " ze_id=\"" << DSBC_GET_Z_EXPRID(bcvel[i].flags) << "\"";

			curline << " starts at t=\"" << max(0.0f,bcvel[i].tst) << "\"";
			if(bcvel[i].tend < 0.99 * FLT_MAX) curline << " ends at t=\"" <<bcvel[i].tend<< "\"";
			else curline << " ends at t=\"TimeMax\"";
			const std::string& curlinestr = curline.str();
			const char* finalcurline = curlinestr.c_str();
			lines.push_back(fun::PrintStr(finalcurline));
		}
	}
	nmbc = GetNfBC();
	if (nmbc) {
		const tbcstrucbody* bcforce = GetFBClist();
		lines.push_back(fun::PrintStr("    Force boundary conditions (Total:%u):", nmbc));
		for (unsigned i = 0; i < nmbc; i++) {
			std::ostringstream curline;
			unsigned forcetype = DSBC_GET_FORCETYPE(bcforce[i].flags);
			if (forcetype == DSBC_FORCETYPE_SURFACE) {
				curline << "      surface force (bcforces) near mkid=\"" << bcforce[i].mkid << "\" [N/m^2(3D)|N/m(2D)]:";
			}
			else if (forcetype == DSBC_FORCETYPE_BODY) {
				curline << "      body force (bcforceb) [N/m^3(3D)|N/m^2(2D)]:";
			}
			else if (forcetype == DSBC_FORCETYPE_POINT) {
				curline << "      point force (bcforcep) [N]:";
			}
			else {
				// Fallback for unknown type
				curline << "      force (unknown type):";
			}
			if (DSBC_GET_X_FLAG(bcforce[i].flags)) curline << " x=\"" << bcforce[i].x << "\"";
			else if (DSBC_GET_X_IS_EXPR(bcforce[i].flags)) curline << " xe_id=\"" << DSBC_GET_X_EXPRID(bcforce[i].flags) << "\"";

			if (DSBC_GET_Y_FLAG(bcforce[i].flags)) curline << " y=\"" << bcforce[i].y << "\"";
			else if (DSBC_GET_Y_IS_EXPR(bcforce[i].flags)) curline << " ye_id=\"" << DSBC_GET_Y_EXPRID(bcforce[i].flags) << "\"";

			if (DSBC_GET_Z_FLAG(bcforce[i].flags)) curline << " z=\"" << bcforce[i].z << "\"";
			else if (DSBC_GET_Z_IS_EXPR(bcforce[i].flags)) curline << " ze_id=\"" << DSBC_GET_Z_EXPRID(bcforce[i].flags) << "\"";

			curline << " starts at t=\"" << max(0.0f, bcforce[i].tst) << "\"";
			if (bcforce[i].tend < 0.99 * FLT_MAX) curline << " ends at t=\"" << bcforce[i].tend << "\"";
			else curline << " ends at t=\"TimeMax\"";
			const std::string& curlinestr = curline.str();
			const char* finalcurline = curlinestr.c_str();
			lines.push_back(fun::PrintStr(finalcurline));
		}
	}
	nmbc = GetNnotch();
	if (nmbc) {
		const plane4Nstruc* notchlist = GetNotchList();
		lines.push_back(fun::PrintStr("    Pre-existing cracks (Total:%u):", nmbc));
		for (unsigned i = 0; i < nmbc; i++) {
			lines.push_back(fun::PrintStr("      Notch: %u", i + 1));
			for (unsigned j = 0; j < 4; j++) {
				std::ostringstream curline;
				curline << "        p" << j + 1 << ":  x=\"" << notchlist[i].corners[j].x << "\"";
				curline << " y=\"" << notchlist[i].corners[j].y << "\"";
				curline << " z=\"" << notchlist[i].corners[j].z << "\"";
				const std::string& curlinestr = curline.str();
				const char* finalcurline = curlinestr.c_str();
				lines.push_back(fun::PrintStr(finalcurline));
			}
		}
	}
	nmbc = GetNMeasPlane();
	if (nmbc) {
		const plane4Nstruc* measureplanes = GetMeasPlaneList();
		lines.push_back(fun::PrintStr("    Measuring planes (Total:%u):", nmbc));
		for (unsigned i = 0; i < nmbc; i++) {
			lines.push_back(fun::PrintStr("      Plane: %u", i + 1));
			for (unsigned j = 0; j < 4; j++) {
				std::ostringstream curline;
				curline << "        p" << j + 1 << ":  x=\"" << measureplanes[i].corners[j].x << "\"";
				curline << " y=\"" << measureplanes[i].corners[j].y << "\"";
				curline << " z=\"" << measureplanes[i].corners[j].z << "\"";
				const std::string& curlinestr = curline.str();
				const char* finalcurline = curlinestr.c_str();
				lines.push_back(fun::PrintStr(finalcurline));
			}
		}
	}
}

//==================================================================================================================
//##############################################################################
//# JSphDeformStruc
//##############################################################################
//========================================================================================
/// Constructor.
//========================================================================================
JSphDeformStruc::JSphDeformStruc(bool simulate2d, double dp, JXml* sxml, const std::string& place, const JSphMk* mkinfo)
	:Log(AppInfo.LogPtr())
{
	ClassName = "JSphDeformStruc";
	Reset();
	LoadXml(sxml, place, simulate2d, dp);
	ConfigBoundCode(mkinfo);
}

void JSphDeformStruc::CheckUserExpForBC(JUserExpressionList* UserExpressions)
{

	for (unsigned bodyid = 0; bodyid < GetCount(); bodyid++) {
		
		JSphDeformStrucBody& DSdata = *List[bodyid]; 
		for (unsigned ii = 0; ii < DSdata.GetNvBC(); ii++) {
			if (DSBC_GET_X_IS_EXPR(DSdata.GetVBClist()[ii].flags)) {
				if (UserExpressions) {
					if (UserExpressions->GetById(DSBC_GET_X_EXPRID(DSdata.GetVBClist()[ii].flags))) continue;
				}
				Run_Exceptioon(fun::PrintStr("VelX boundary condition on defstruc mk=\"%u\" set as expression id=\"%u\" but expression was not found.", DSdata.MkBound, DSBC_GET_X_EXPRID(DSdata.GetVBClist()[ii].flags)));
			}
			if (DSBC_GET_Y_IS_EXPR(DSdata.GetVBClist()[ii].flags)) {
				if (UserExpressions) {
					if (UserExpressions->GetById(DSBC_GET_Y_EXPRID(DSdata.GetVBClist()[ii].flags))) continue;
				}
				Run_Exceptioon(fun::PrintStr("VelY boundary condition on defstruc mk=\"%u\" set as expression id=\"%u\" but expression was not found.", DSdata.MkBound, DSBC_GET_Y_EXPRID(DSdata.GetVBClist()[ii].flags)));
			}
			if (DSBC_GET_Z_IS_EXPR(DSdata.GetVBClist()[ii].flags)) {
				if (UserExpressions) {
					if (UserExpressions->GetById(DSBC_GET_Z_EXPRID(DSdata.GetVBClist()[ii].flags))) continue;
				}
				Run_Exceptioon(fun::PrintStr("VelZ boundary condition on defstruc mk=\"%u\" set as expression id=\"%u\" but expression was not found.", DSdata.MkBound, DSBC_GET_Z_EXPRID(DSdata.GetVBClist()[ii].flags)));
			}
		}
		for (unsigned ii = 0; ii < DSdata.GetNfBC(); ii++) {
			if (DSBC_GET_X_IS_EXPR(DSdata.GetFBClist()[ii].flags)) {
				if (UserExpressions) {
					if (UserExpressions->GetById(DSBC_GET_X_EXPRID(DSdata.GetFBClist()[ii].flags))) continue;
				}
				Run_Exceptioon(fun::PrintStr("ForceX boundary condition on defstruc mk=\"%u\" set as expression id=\"%u\" but expression was not found.", DSdata.MkBound, DSBC_GET_X_EXPRID(DSdata.GetFBClist()[ii].flags)));
			}
			if (DSBC_GET_Y_IS_EXPR(DSdata.GetFBClist()[ii].flags)) {
				if (UserExpressions) {
					if (UserExpressions->GetById(DSBC_GET_Y_EXPRID(DSdata.GetFBClist()[ii].flags))) continue;
				}
				Run_Exceptioon(fun::PrintStr("ForceY boundary condition on defstruc mk=\"%u\" set as expression id=\"%u\" but expression was not found.", DSdata.MkBound, DSBC_GET_Y_EXPRID(DSdata.GetFBClist()[ii].flags)));
			}
			if (DSBC_GET_Z_IS_EXPR(DSdata.GetFBClist()[ii].flags)) {
				if (UserExpressions) {
					if (UserExpressions->GetById(DSBC_GET_Z_EXPRID(DSdata.GetFBClist()[ii].flags))) continue;
				}
				Run_Exceptioon(fun::PrintStr("ForceZ boundary condition on defstruc mk=\"%u\" set as expression id=\"%u\" but expression was not found.", DSdata.MkBound, DSBC_GET_Z_EXPRID(DSdata.GetFBClist()[ii].flags)));
			}
		}
		// Check phi BC expressions
		for (unsigned ii = 0; ii < DSdata.GetNphiBC(); ii++) {
			const unsigned expr_id = DSBC_GET_PHI_EXPRID(DSdata.GetPhiBClist()[ii].flags);
			if (UserExpressions) {
				if (UserExpressions->GetById(expr_id)) continue;
			}
			Run_Exceptioon(fun::PrintStr("Phi boundary condition on defstruc mk=\"%u\" set as expression id=\"%u\" but expression was not found.", DSdata.MkBound, expr_id));
		}
	}
}
//========================================================================================
/// Destructor.
//========================================================================================
JSphDeformStruc::~JSphDeformStruc() {
	DestructorActive = true;
	Reset();
}

//========================================================================================
/// Initialisation of variables.
//========================================================================================
void JSphDeformStruc::Reset() {
	for (unsigned c = 0; c < GetCount(); c++)delete List[c];
	List.clear();
}

//========================================================================================
/// Returns true if mkbound value is already configured.
//========================================================================================
bool JSphDeformStruc::ExistMk(word mkbound)const {
	bool ret = false;
	for (unsigned c = 0; c < List.size() && !ret; c++)ret = (List[c]->MkBound == mkbound);
	return(ret);
}

//========================================================================================
/// Loads conditions of XML object.
//========================================================================================
void JSphDeformStruc::LoadXml(const JXml* sxml, const std::string& place, const bool simulate2d, const double dporg) {
	TiXmlNode* node = sxml->GetNodeSimple(place);
	if (!node)Run_Exceptioon(std::string("Cannot find the element \'") + place + "\'.");

	if (sxml->CheckNodeActive(node))ReadXml(sxml, node->ToElement(), simulate2d, dporg);

}

//========================================================================================
/// Reads list of configurations in the XML node.
//========================================================================================
void JSphDeformStruc::ReadXml(const JXml* sxml, TiXmlElement* lis, const bool simulate2d, const double dporg) {
	//-Loads deformable structure body elements.
	//user expressions
	// Enhanced trim with full whitespace handling
	
	const unsigned idmax = MAX_DEFORMSTRUC_NUM - 1;

	UseUsrTimeStep = sxml->ReadElementDoublePos(lis, "timestep", "value", true, 0);
	if (UseUsrTimeStep > 0) {
		Log->PrintWarning(fun::PrintStr("Using user-defined timestep: %g seconds for deformable structures.", UseUsrTimeStep));
	}
	ContPowerCoeff = float(sxml->ReadElementDoublePos(lis, "contcoeff", "value", true, 1.0));
	if (ContPowerCoeff < 1.0f) {
		Log->PrintWarning("Contact power coefficient for deformable structures was less than 1.0, clamping to 1.0.");
		ContPowerCoeff = 1.0f;
	}
	TiXmlElement* ele = lis->FirstChildElement("deformstrucbody");
	while (ele) {
		if (sxml->CheckElementActive(ele)) {

			const unsigned id = GetCount();

			if (id > idmax)Run_Exceptioon("Maximum number of deformable structure bodies has been reached.");
			word mkbound = sxml->GetAttributeWord(ele, "mkbound");
			if (ExistMk(mkbound))Run_Exceptioon(fun::PrintStr("An input already exists for the same mkbound=%u.", mkbound));

			double usrlambda = sxml->ReadElementDoublePos(ele, "u_lambda", "value", true, NULL);
			double usrMu = sxml->ReadElementDoublePos(ele, "u_mu", "value", true, NULL);
			double usrBulk = sxml->ReadElementDoublePos(ele, "u_bulk", "value", true, NULL);
			float density = sxml->ReadElementFloatPos(ele, "density", "value");
			bool askyoung = (usrlambda && usrMu || usrlambda && usrBulk || usrMu && usrBulk);
			if (askyoung) {
				Log->PrintWarning(fun::PrintStr("Elastic constants directly specified for mkbound=%u. Young's modulus and Poisson's ratio will be computed.", mkbound));
			}
			double youngmod = sxml->ReadElementDoublePos(ele, "youngmod", "value", askyoung, 1.0);
			double poissratio = sxml->ReadElementDoublePos(ele, "poissratio", "value", askyoung, 0.333);
			double yieldstress = 0.0;
			double hardening = 0.0;
			TpConstitModel constitmodel;
			switch (sxml->ReadElementUnsigned(ele, "constitmodel", "value",true, 0)) {
				case 1: constitmodel = CONSTITMODEL_SVK;  break;
				case 2: constitmodel = CONSTITMODEL_NH;  break;
				case 3: constitmodel = CONSTITMODEL_J2;  break;
				default: constitmodel = CONSTITMODEL_SVK; 
					Log->PrintWarning(fun::PrintStr("Constitutive model not specified for mkbound=%u, using default (St. Venant Kirchhoff).", mkbound));
			}
			if (constitmodel == CONSTITMODEL_J2) {
				yieldstress = sxml->ReadElementDoublePos(ele, "yieldstress", "value");
				hardening = sxml->ReadElementDouble(ele, "hardening", "value", true, 0.0);
				if (hardening == 0.0) {
					Log->PrintWarning(fun::PrintStr("Hardening modulus not specified for mkbound=%u with plasticity, using perfectly plastic behavior (hardening=0).", mkbound));
			}
			}
			float avfactor1 = float(0.2);
			float avfactor2 = float(0.0);
			avfactor1 = sxml->ReadElementFloatPos(ele, "artvisc", "factor1", true, -1);
			avfactor2 = sxml->ReadElementFloatPos(ele, "artvisc", "factor2", true, -1);
			if (avfactor1 == -1.0f && avfactor2 == -1.0f) {
				avfactor1 = float(0.2);
				avfactor2 = float(0.0);
				Log->PrintWarning(fun::PrintStr("Artificial viscosity (artvisc) not specified for mkbound=%u, using defaults (factor1=0.2, factor2=0.0).", mkbound));
			}
			if(avfactor1==-1.0f) avfactor1 = float(0.2);
			if (avfactor2 == -1.0f) avfactor2 = float(0.0);
			bool fracturei = sxml->ReadElementBool(ele, "fracture", "value", true, false);
			if (constitmodel == CONSTITMODEL_J2) {
				fracturei = false;
				Log->PrintWarning(fun::PrintStr("Fracture (fracture) is set true for mkbound=%u but plasticity constitmodel is chosen. Disabling fracture.", mkbound));
			}
			float Gci = 0.0, pflimiti = 0.0, pflenscale = 0.0;

			float restcoef = sxml->ReadElementFloatPos(ele, "restcoef", "value", true, 0.5);
			float kfric = sxml->ReadElementFloatPos(ele, "kfric", "value", true, 0.0);
			if (restcoef == 0.5f) {
				Log->PrintWarning(fun::PrintStr("Restitution coefficient (restcoef) not specified for mkbound=%u, using default (0.5).", mkbound));
			}
			if (kfric == 0.0f) {
				Log->PrintWarning(fun::PrintStr("Kinetic friction coefficient (kfric) not specified for mkbound=%u, using default (0.0 - frictionless).", mkbound));
			}

			if (fracturei) {
				Gci = sxml->ReadElementFloatPos(ele, "Gc", "value");
				if (!Gci) Run_Exceptioon(fun::PrintStr("Fracture modeling is on, Gc value needed for mkbound=%u ", mkbound));
				pflimiti = sxml->ReadElementFloatPos(ele, "pflim", "value", true, float(0.1));
				if (pflimiti == 0.1f) {
					Log->PrintWarning(fun::PrintStr("Phase field limit not specified for mkbound=%u, using default (0.1).", mkbound));
				}
				if (pflimiti > 1.0f) {
					Log->PrintWarning(fun::PrintStr("Phase field limit for mkbound=%u exceeds 1.0 (%g), clamping to 1.0.", mkbound, pflimiti));
				}
				pflenscale = sxml->ReadElementFloatPos(ele, "pflenscale", "value");
			}
			
			unsigned nbsrangei = sxml->ReadElementUnsigned(ele, "nbsrange", "value", true, 20);

			if (nbsrangei < 1) {
				Log->PrintWarning(fun::PrintStr("Neighbor search range for mkbound=%u was less than 1, clamping to 1.", mkbound));
				nbsrangei = 1;
			}
			unsigned dxmfacti = sxml->ReadElementUnsigned(ele, "mapfac", "value", true, 1);
			tbcstrucbody* tmbcvel={}; tbcstrucbody* tmbcforce = {}; tbcstrucbody* tmbcphi = {};
			unsigned nmbcvel = 0; unsigned nmbcforce = 0; unsigned nmbcphi = 0; unsigned nmnotch = 0; unsigned nmmeasureplane = 0;
			unsigned countbcv = sxml->CountElements(ele, "bcvel");

			if (countbcv) {
				if (countbcv > MAX_VELBC_DEFSTRUC)
					Run_Exceptioon(fun::PrintStr("Number of velocity boundary conditions for mkbound=%u is higher than maximum allowed value=%u.", mkbound, MAX_VELBC_DEFSTRUC));

				tmbcvel = new tbcstrucbody[countbcv]; for (unsigned i = 0; i < countbcv; i++) tmbcvel[i] = make_tbcstrucbody0();
				nmbcvel = sxml->ReadArrayBCstruc(ele, "bcvel", tmbcvel, countbcv);
			}
			unsigned countbcf = sxml->CountElements(ele, "bcforce");
			if (countbcf) {
				if (countbcf > MAX_FORCEBC_DEFSTRUC)
					Run_Exceptioon(fun::PrintStr("Number of force boundary conditions for mkbound=%u is higher than maximum allowed value=%u.", mkbound, MAX_FORCEBC_DEFSTRUC));
				tmbcforce = new tbcstrucbody[countbcf]; for (unsigned i = 0; i < countbcf; i++) tmbcforce[i] = make_tbcstrucbody0();
				nmbcforce = sxml->ReadArrayBCstruc(ele, "bcforce", tmbcforce, countbcf);
			}

			if (!countbcv && !countbcf) Log->PrintWarning(fun::PrintStr("No velocity or force boundary condition set for mkbound=%u.", mkbound));
			unsigned countbcphi = sxml->CountElements(ele, "restrictphi");
			if (countbcphi) {
				if (!fracturei) {
					Log->PrintWarning(fun::PrintStr("Phi boundary conditions specified for mkbound=%u but fracture is disabled. Ignoring phi BCs.", mkbound));
					countbcphi = 0;
				}
				else {
					if (countbcphi > MAX_PHIBC_DEFSTRUC)
						Run_Exceptioon(fun::PrintStr("Number of phi boundary conditions for mkbound=%u is higher than maximum allowed value=%u.", mkbound, MAX_PHIBC_DEFSTRUC));
					
					tmbcphi = new tbcstrucbody[countbcphi];
					TiXmlElement* bcphi = ele->FirstChildElement("restrictphi");
					unsigned bcphi_count = 0;
					while (bcphi && bcphi_count < countbcphi) {
						if (sxml->CheckElementActive(bcphi)) {
							tbcstrucbody bc = make_tbcstrucbody0();

							// Read expression ID (required)
							const char* ex = bcphi->Attribute("value");
							if (!ex) Run_Exceptioon(fun::PrintStr("Phi BC for mkbound=%u requires 'value' attribute with expression ID", mkbound));
							unsigned expr_id = atoi(ex);

							// Store expression ID in flags
							// Expression returns restore value (0-1) and handles temporal/spatial logic
							DSBC_SET_PHI_ACTIVE(bc.flags, true);
							DSBC_SET_PHI_EXPRID(bc.flags, expr_id);

							tmbcphi[bcphi_count++] = bc;
						}
						bcphi = bcphi->NextSiblingElement("restrictphi");
					}
					nmbcphi = bcphi_count;
				}
			}

			plane4Nstruc* tmnotch = {};
			unsigned countnotch = sxml->CountElements(ele, "notch");
			if (countnotch) {
				if (countnotch > MAX_NOTCH_DEFSTRUC)
					Run_Exceptioon(fun::PrintStr("Number of preexisting notches for mkbound=%u is higher than maximum allowed value=%u.", mkbound, MAX_NOTCH_DEFSTRUC));
				tmnotch = new plane4Nstruc[countnotch]; for (unsigned i = 0; i < countnotch; i++) tmnotch[i] = TNotchStrucNull();
				nmnotch = sxml->ReadArrayNotchstruc(ele, "notch", tmnotch, countnotch);
			}
			plane4Nstruc* measureplanes = {};
			unsigned countplanes = sxml->CountElements(ele, "measureplane");
			if (countplanes) {
				if (countplanes > MAX_MEASUREPLANE_DEFSTRUC)
					Run_Exceptioon(fun::PrintStr("Number of measuring planes for mkbound=%u is higher than maximum allowed value=%u.", mkbound, MAX_MEASUREPLANE_DEFSTRUC));
				measureplanes = new plane4Nstruc[countplanes]; for (unsigned i = 0; i < countplanes; i++) measureplanes[i] = TNotchStrucNull();
				nmmeasureplane = sxml->ReadArrayNotchstruc(ele, "measureplane", measureplanes, countplanes);
			}
			const double dpb = dporg / dxmfacti;
			JSphDeformStrucBody* body = new JSphDeformStrucBody(dpb, simulate2d, id, mkbound, density, youngmod, poissratio, constitmodel, avfactor1, avfactor2,
				fracturei, Gci, dxmfacti, pflimiti, pflenscale, restcoef, kfric,
				usrlambda, usrMu, usrBulk, yieldstress, hardening, 
				tmbcvel, nmbcvel, tmbcforce, nmbcforce, tmbcphi, nmbcphi, tmnotch, nmnotch, measureplanes, nmmeasureplane, nbsrangei);

			List.push_back(body);
			
			if (countbcv) delete[] tmbcvel;
			if (countbcf) delete[] tmbcforce;
			if (countbcphi) delete[] tmbcphi;

		}
		ele = ele->NextSiblingElement("deformstrucbody");
	}
}

//========================================================================================
/// Configures particle codings for deformable structures.
//========================================================================================
void JSphDeformStruc::DSConfigCode(const unsigned Npb, typecode* Codec, const JSphMk* MkInfo) {
	int npbint = static_cast<int>(Npb);
	for (unsigned c = 0; c < GetCount(); c++) {
		typecode bcode = List[c]->GetBoundCode();
#ifdef OMP_USE
#pragma omp parallel for schedule (static) if(npbint>OMP_LIMIT_COMPUTESTEPDEFSTRUC)
#endif
		for (int p = 0; p < npbint; p++)
			if (Codec[p] == bcode)
				Codec[p] = typecode(CODE_ToDeformStrucDeform(Codec[p], List[c]->IdBody));
	}
	for (unsigned c = 0; c < GetCount(); c++) List[c]->ConfigBCCodeBody(MkInfo);
}

//========================================================================================
/// Configures BoundCode for each body.
//========================================================================================
void JSphDeformStruc::ConfigBoundCode(const JSphMk* mkinfo) {
	for (unsigned c = 0; c < GetCount(); c++) {
		const unsigned cmk = mkinfo->GetMkBlockByMkBound(List[c]->MkBound);
		if (cmk < mkinfo->Size() && (CODE_IsMoving(mkinfo->Mkblock(cmk)->Code))) {
			List[c]->ConfigBoundCode(mkinfo->Mkblock(cmk)->Code);
		}
		else Run_Exceptioon(fun::PrintStr("MkBound value for mkbound=%u is not a valid Mk moving boundary.", List[c]->MkBound));
	}
}

//========================================================================================
/// Shows object configuration using Log.
//========================================================================================
void JSphDeformStruc::VisuConfig(std::string txhead, std::string txfoot) {
	if (!txhead.empty())Log->Print(txhead);
	for (unsigned c = 0; c < GetCount(); c++) {
		Log->Printf("  Deformable Structure %u (mkbound:%u):", List[c]->IdBody, List[c]->MkBound);
		std::vector<std::string> lines;
		List[c]->GetConfig(lines);
		Log->Print(lines);
	}
	if (!txfoot.empty())Log->Print(txfoot);
}

//========================================================================================
/// Get maximum initial sound speed across all deformable structures.
//========================================================================================
double JSphDeformStruc::GetInitialDtMin() {
	double cs0 = DBL_MAX;
	for (unsigned c = 0; c < GetCount(); c++)cs0 = min(cs0, double(List[c]->GetKernelh()/List[c]->GetSoundSpeed()));
	return cs0;
}



//========================================================================================
/// Save TLSPH informations such as number of neighbours, accuracy of 
/// Kernels and its derivatives, Boundary conditions, etc
//========================================================================================
void JSphDeformStruc::DSSaveInitDomainInfo(const bool simulate2d, const int np, const int dsnpsurf, const int casendeformstruc, 
	const unsigned* dsparent, const tbcstruc* dspartvbc, const tbcstruc* dspartfbc,
	const typecode* dscodec, const tdouble3* dspos0, const tdouble3* dsvel,
	const unsigned* dspairn, const unsigned* dspairstart, const unsigned* dspairj, const double* dsker, 
	const tdouble3* dskerderv0, const double* dskerlapc, const float* dskersumvol, const unsigned* dssurfpartlist,
	const unsigned* deformstrucridp, const unsigned* dsbestchild, const unsigned* dsibodyridp,
	const unsigned* dsmeaspartlist, const unsigned* dsmeasplpartnum, const unsigned dsmeasplnum,
	const StDeformStrucIntData* defstrucintdata, const StDeformStrucData* deformstrucdata, const std::string dirout,
	JUserExpressionList* UserExpressions)const
{
	int measptot = 0;
	for (unsigned i = 0; i < dsmeasplnum; i++) measptot += dsmeasplpartnum[i];
	std::vector<unsigned>tempvar2(np);
#ifdef OMP_USE
#pragma omp parallel for schedule (static)
#endif
		for (int p = 0; p < np; p++)
		{
			tempvar2[p] = 0;
		}
#ifdef OMP_USE
#pragma omp parallel for schedule (static)
#endif
		for (int p1 = 0; p1 < measptot; p1++)
		{
			unsigned p = dsmeaspartlist[p1];
			tempvar2[p] = 1;
		}
		std::vector<tdouble3> normals(np);
		const unsigned planeCount = dsmeasplnum;
		std::vector<unsigned> particlePlaneId(np, 0u);
		std::vector<tdouble3> particlePlaneNormal(np, TDouble3(0.0, 0.0, 0.0));
		std::vector<unsigned> planeBody(planeCount, UINT_MAX);
		std::vector<plane4Nstruc> planeGeometry(planeCount);
		std::vector<tdouble3> planeNormals(planeCount, TDouble3(0.0, 0.0, 0.0));
		if (planeCount) {
			unsigned planeIdx = 0;
			for (unsigned bodyid = 0; bodyid < GetCount() && planeIdx < planeCount; ++bodyid) {
				const StDeformStrucData& body = deformstrucdata[bodyid];
				for (unsigned local = 0; local < body.nmeasplane && planeIdx < planeCount; ++local) {
					planeBody[planeIdx] = bodyid;
					planeGeometry[planeIdx] = body.measplanelist[local];
					const auto planeEq = measplane::plane_from_4(planeGeometry[planeIdx].corners);
					planeNormals[planeIdx] = planeEq.n;
					++planeIdx;
				}
			}
			if (dsmeaspartlist && dsmeasplpartnum) {
				unsigned offset = 0;
				for (unsigned planeIdx = 0; planeIdx < planeCount; ++planeIdx) {
					const unsigned count = dsmeasplpartnum[planeIdx];
					for (unsigned c = 0; c < count; ++c) {
						const unsigned pid = dsmeaspartlist[offset + c];
						if (pid < static_cast<unsigned>(np) && !particlePlaneId[pid]) {
							particlePlaneId[pid] = planeIdx + 1u;
							particlePlaneNormal[pid] = planeNormals[planeIdx];
						}
					}
					offset += count;
					if (offset >= static_cast<unsigned>(measptot)) break;
				}
			}
		}
	//std::vector<double>tempvar(np);
	//std::vector<unsigned>randfield(np);
	//std::vector<unsigned>mapcenters(np);
	//std::vector<double>approaxvar(np);
	//std::vector<tdouble3>dvardx(np);
	//std::vector<double>nabla(np);
	//std::vector<float>tempfbc(np);
	
//	{
//#ifdef OMP_USE
//#pragma omp parallel for schedule (static) if(np>OMP_LIMIT_COMPUTESTEPDEFSTRUC)
//#endif
//		for (int p1 = 0; p1 < np; p1++)
//		{
//			randfield[p1] = 0;
//		}

//#ifdef OMP_USE
//#pragma omp parallel for schedule (static) if(dsnpsurf>OMP_LIMIT_COMPUTESTEPDEFSTRUC)
//#endif
//		for (int p = 0; p < dsnpsurf; p++)
//		{
//			unsigned pid = dssurfpartlist[p];
//			randfield[pid] = 1;
//		}
//#ifdef OMP_USE
//#pragma omp parallel for schedule (static) if(casendeformstruc>OMP_LIMIT_COMPUTESTEPDEFSTRUC)
//#endif
//		for (int p = 0; p < int(casendeformstruc); p++)
//		{
//			unsigned por = deformstrucridp[p];
//			unsigned pid = dsbestchild[p];
//			mapcenters[pid] = por;
//		}
//		
//#ifdef OMP_USE
//#pragma omp parallel for schedule (static) if(np>OMP_LIMIT_COMPUTESTEPDEFSTRUC)
//#endif
//		for (int p = 0; p < np; p++)
//		{
//			tempvbc[p].x = dspartvbc[p].x;
//			tempvbc[p].y = dspartvbc[p].y;
//			tempvbc[p].z = dspartvbc[p].z;
//		}
//	}

	const unsigned planeVertexCount = planeCount * 4u;
	const unsigned estimatedTotalPoints = static_cast<unsigned>(np) + planeVertexCount;

	//std::vector<tfloat3> tempfbc;
	//tempfbc.reserve(estimatedTotalPoints);
	//std::vector<float> truncfactor;
	//truncfactor.reserve(estimatedTotalPoints);
	std::vector<tdouble3> posOut;
	posOut.reserve(estimatedTotalPoints);
	std::vector<unsigned> pairnOut;
	pairnOut.reserve(estimatedTotalPoints);
	//std::vector<tdouble3> normalsOut;
	//normalsOut.reserve(estimatedTotalPoints);
	std::vector<tdouble3> velOut;
	velOut.reserve(estimatedTotalPoints);
	//std::vector<unsigned> measurePlaneMaskOut;
	//measurePlaneMaskOut.reserve(estimatedTotalPoints);
	//std::vector<unsigned> entityTypeOut;
	//entityTypeOut.reserve(estimatedTotalPoints);
	std::vector<unsigned> measurePlaneIdOut;
	measurePlaneIdOut.reserve(estimatedTotalPoints);
	//std::vector<unsigned> measurePlaneCornerOut;
	//measurePlaneCornerOut.reserve(estimatedTotalPoints);
	//std::vector<unsigned> measurePlaneBodyOut;
	//measurePlaneBodyOut.reserve(estimatedTotalPoints);
	//std::vector<tdouble3> measurePlaneNormalOut;
	//measurePlaneNormalOut.reserve(estimatedTotalPoints);
//#ifdef OMP_USE
//#pragma omp parallel for schedule (static) if(np>OMP_LIMIT_COMPUTESTEPDEFSTRUC)
//#endif
//	for (int p1 = 0; p1 < np; p1++)
//	{
//		const unsigned bodyid = CODE_GetIbodyDeformStruc(dscodec[p1]);
//		const StDeformStrucData& body = deformstrucdata[bodyid];
//		const double vol0p1 = body.vol0;
//		tdouble3 pos0p1 = dspos0[p1];
//		tdouble3 posp1 = pos0p1 + TDouble3(0.0, 0.0, 0.0);
//		truncfactor[p1] = float(vol0p1 / dskersumvol[p1]);
//		
//		tempfbc[p1].x = dspartfbc[p1].x;
//		tempfbc[p1].y = dspartfbc[p1].y;
//		tempfbc[p1].z = dspartfbc[p1].z;
//	}
	for (int p = 0; p < np; ++p) {
		const size_t idx = static_cast<size_t>(p);
		posOut.push_back(dspos0 ? dspos0[p] : TDouble3(0.0, 0.0, 0.0));
		pairnOut.push_back(dspairn ? dspairn[p] : 0u);
		//normalsOut.push_back(normals[p]);
		velOut.push_back(dsvel ? dsvel[p] : TDouble3(0.0, 0.0, 0.0));
		//measurePlaneMaskOut.push_back(tempvar2[p]);
		//entityTypeOut.push_back(0u);
		measurePlaneIdOut.push_back(idx < particlePlaneId.size() ? particlePlaneId[idx] : 0u);
		//measurePlaneCornerOut.push_back(0u);
		//if (dscodec) measurePlaneBodyOut.push_back(CODE_GetIbodyDeformStruc(dscodec[p]) + 1u);
		//else measurePlaneBodyOut.push_back(0u);
		//measurePlaneNormalOut.push_back(idx < particlePlaneNormal.size() ? particlePlaneNormal[idx] : TDouble3(0.0, 0.0, 0.0));
	}

	if (planeCount) {
		for (unsigned planeIdx = 0; planeIdx < planeCount; ++planeIdx) {
			const unsigned planeId = planeIdx + 1u;
			const unsigned bodyLabel = (planeBody[planeIdx] != UINT_MAX) ? planeBody[planeIdx] + 1u : 0u;
			//const tdouble3 normal = planeNormals[planeIdx];
			for (unsigned corner = 0; corner < 4u; ++corner) {
				posOut.push_back(planeGeometry[planeIdx].corners[corner]);
				pairnOut.push_back(0u);
				//truncfactor.push_back(0);
				//normalsOut.push_back(normal);
				velOut.push_back(TDouble3(0.0, 0.0, 0.0));
				//tempfbc.push_back(TFloat3(0.0, 0.0, 0.0));
				//measurePlaneMaskOut.push_back(0u);
				//entityTypeOut.push_back(1u);
				measurePlaneIdOut.push_back(planeId);
				//measurePlaneCornerOut.push_back(corner + 1u);
				//measurePlaneBodyOut.push_back(bodyLabel);
				//measurePlaneNormalOut.push_back(normal);
			}
		}
	}

	const unsigned totalPoints = static_cast<unsigned>(posOut.size());

	JDataArrays arraysdefstruc;
	arraysdefstruc.AddArray("Pos", totalPoints, posOut.data());
	arraysdefstruc.AddArray("PairN", totalPoints, pairnOut.data());
	arraysdefstruc.AddArray("Init_Vel", totalPoints, velOut.data());
	//if (measptot || planeVertexCount) arraysdefstruc.AddArray("MeasurePlanes", totalPoints, measurePlaneMaskOut.data());
	if (planeCount) {
		arraysdefstruc.AddArray("MeasurePlaneId", totalPoints, measurePlaneIdOut.data());
		//arraysdefstruc.AddArray("MeasurePlaneCorner", totalPoints, measurePlaneCornerOut.data());
		//arraysdefstruc.AddArray("MeasurePlaneBody", totalPoints, measurePlaneBodyOut.data());
		//arraysdefstruc.AddArray("MeasurePlaneNormal", totalPoints, measurePlaneNormalOut.data());
		//arraysdefstruc.AddArray("EntityType", totalPoints, entityTypeOut.data());
	}
	//arraysdefstruc.AddArray("BestChild", dfnp, mpcent.data());
	//arraysdefstruc.AddArray("Parents", dfnp, parents.data());
	//arraysdefstruc.AddArray("Field", dfnp, tpvar.data());
	//arraysdefstruc.AddArray("Field_Approx", dfnp, apvar.data());
	//arraysdefstruc.AddArray("Field_Derivatives_Approx", dfnp, bdvardx.data());
	//arraysdefstruc.AddArray("Field_Derivatives_Actual", dfnp, deract.data());
	//arraysdefstruc.AddArray("Field_Approx_LapLacian", dfnp, bnab.data());
	//arraysdefstruc.AddArray("Surface", np, randfield.data());
	//arraysdefstruc.AddArray("TruncationFactor", totalPoints, truncfactor.data());
	//arraysdefstruc.AddArray("ForceBCstoAcc", totalPoints, tempfbc.data());

	//arraysdefstruc.AddArray("Force_BoundaryConditions", np, dspartfbc);
	//arraysdefstruc.AddArray("Vel_BoundaryConditions", np, dspartvbc);

	const std::string fileds = "TLSPH_INFO/DefStrucDomainInfo.vtk";
	JVtkLib::SaveVtkData(dirout + fileds, arraysdefstruc, "Pos");

	arraysdefstruc.Reset();
	//-Creates lines.
	JVtkLib sh;
	plane4Nstruc notchlist[MAX_NOTCH_DEFSTRUC];
	bool notchfile = false;
	double yy = 0.0;

	for (unsigned bodyid = 0; bodyid < GetCount(); bodyid++) {
		StDeformStrucData body = deformstrucdata[bodyid];
		const unsigned notchnm = body.nnotch;
		if (simulate2d) yy = body.min.y;
		const double dp05 = 0.5 * body.dp;
		if (notchnm) {
			std::copy(body.notchlist, body.notchlist + notchnm, std::begin(notchlist));
			for (unsigned surfi = 0; surfi < notchnm; surfi++) {
				if (simulate2d) {
					notchlist[surfi].corners[0].y = yy - dp05;
					notchlist[surfi].corners[1].y = yy - dp05;

					notchlist[surfi].corners[2].y = yy + dp05;
					notchlist[surfi].corners[3].y = yy + dp05;
				}
				sh.AddShapeQuadWire(notchlist[surfi].corners[0], notchlist[surfi].corners[1],
					notchlist[surfi].corners[2], notchlist[surfi].corners[3], surfi + 1);
			}
			notchfile = true;
		}
	}

	if (notchfile) {
		const std::string file = dirout + "TLSPH_INFO/Pre_Existing_Cracks.vtk";
		Log->AddFileInfo(file, "Saves the preexisting cracks of the deformable objects.");
		sh.SaveShapeVtk(file, "cracks");
	}

}

namespace measplane
{
	 PlaneEq plane_from_4(const tdouble3* corner)
	{
		const tdouble3 p0 = corner[0];
		const tdouble3 v01 = corner[1] - p0;
		const tdouble3 v02 = corner[2] - p0;
		const tdouble3 v03 = corner[3] - p0;

		tdouble3 n1 = fmath::CrossVec3(v01, v02);
		tdouble3 n2 = fmath::CrossVec3(v02, v03);
		tdouble3 n3 = fmath::CrossVec3(v03, v01);
		tdouble3 ns = TDouble3(n1.x + n2.x + n3.x,
			n1.y + n2.y + n3.y,
			n1.z + n2.z + n3.z);

		double m = fmath::NormVec3(ns);
		if (m < ALMOSTZERO) {
			ns = fmath::CrossVec3(v01, v02);
			m = fmath::NormVec3(ns);
			if (m < ALMOSTZERO) {
				throw std::runtime_error("Measuring plane: corners degenerate or nearly collinear");
			}
		}
		tdouble3 n = fmath::NormalizeVec3(ns);
		const double d = -fmath::DotVec3(n, p0);
		return { n, d };
	}

	// Signed distance: >0 is in front of plane along +n
	 double point_plane_signed_distance(const PlaneEq& pl, const tdouble3& x)
	{
		return fmath::DotVec3(pl.n, x) + pl.d;
	}

	// --- Order 4 corners into a proper loop (CCW in a temporary projection) ---
	 void order_corners_loop(const tdouble3& n_hint,
		const tdouble3* in4,
		tdouble3* out4)
	{
		// Build a temporary stable basis from n_hint
		tdouble3 n = fmath::NormalizeVec3(n_hint);
		const tdouble3 a = (std::fabs(n.x) < 0.9)
			? TDouble3(1.0, 0.0, 0.0)
			: TDouble3(0.0, 1.0, 0.0);
		const tdouble3 u = fmath::NormalizeVec3(fmath::CrossVec3(n, a));
		const tdouble3 v = fmath::CrossVec3(n, u);

		// centroid
		tdouble3 c = TDouble3(0, 0, 0);
		for (int i = 0; i < 4; ++i) {
			c.x += in4[i].x; c.y += in4[i].y; c.z += in4[i].z;
		}
		c.x *= 0.25; c.y *= 0.25; c.z *= 0.25;

		// project & sort by polar angle around centroid
		struct P2 { int idx; double ang; };
		P2 p2[4];
		for (int i = 0; i < 4; ++i) {
			const tdouble3 r = in4[i] - c;
			const double x = fmath::DotVec3(u, r);
			const double y = fmath::DotVec3(v, r);
			p2[i] = { i, std::atan2(y, x) };
		}
		std::sort(p2, p2 + 4, [](const P2& A, const P2& B) { return A.ang < B.ang; });
		for (int k = 0; k < 4; ++k) out4[k] = in4[p2[k].idx];
	}

	// --- Build projector (orthonormal in-plane basis + 2D corners, AABB, winding) ---
	 QuadProjector build_quad_projector(const PlaneEq& pl, const tdouble3* corner_looped)
	{
		QuadProjector P{};
		P.p0 = corner_looped[0];
		P.n = pl.n;
		P.d = pl.d;

		// In-plane basis
		const tdouble3 a = (std::fabs(P.n.x) < 0.9)
			? TDouble3(1.0, 0.0, 0.0)
			: TDouble3(0.0, 1.0, 0.0);
		P.u = fmath::NormalizeVec3(fmath::CrossVec3(P.n, a));
		P.v = fmath::CrossVec3(P.n, P.u);

		// Project corners and compute 2D AABB
		P.minx = P.miny = DBL_MAX;
		P.maxx = P.maxy = -DBL_MAX;

		for (int i = 0; i < 4; ++i) {
			const tdouble3 r = corner_looped[i] - P.p0;
			const double x = fmath::DotVec3(P.u, r);
			const double y = fmath::DotVec3(P.v, r);
			P.c2[i] = { x, y };
			P.minx = min(P.minx, x); P.maxx = max(P.maxx, x);
			P.miny = min(P.miny, y); P.maxy = max(P.maxy, y);
		}

		double area2 = 0.0;
		for (int i = 0; i < 4; ++i) {
			const auto& A = P.c2[i];
			const auto& B = P.c2[(i + 1) & 3];
			area2 += (A.x * B.y - A.y * B.x);
		}
		P.ccw = (area2 > 0.0);
		return P;
	}

	// Project arbitrary point into projector’s 2D space
	 QuadProjector::d2 project_point_onto_quad(const QuadProjector& P, const tdouble3& x)
	{
		const tdouble3 r = x - P.p0;
		return { fmath::DotVec3(P.u, r), fmath::DotVec3(P.v, r) };
	}

	// Half-space edge test for convex quad in 2D
	 bool point_inside_convex_quad2D(const QuadProjector& P,
		const QuadProjector::d2& q,
		double eps)
	{
		// Quick AABB reject
		if (q.x < P.minx - eps || q.x > P.maxx + eps ||
			q.y < P.miny - eps || q.y > P.maxy + eps)
			return false;

		// Consistent edge tests
		for (int i = 0; i < 4; ++i) {
			const auto& A = P.c2[i];
			const auto& B = P.c2[(i + 1) & 3];
			const double ex = B.x - A.x, ey = B.y - A.y;
			const double wx = q.x - A.x, wy = q.y - A.y;
			const double crossz = ex * wy - ey * wx; // z of 2D cross

			if (P.ccw) { if (crossz < -eps) return false; }
			else { if (crossz > +eps) return false; }
		}
		return true;
	}
}

namespace defstrucbc
{
#include "FunctionsMath.h"
	void computeBestFitPlane(const std::vector<tdouble3>& pts, tdouble3& origin, tdouble3& normal, const bool simulate2D)
	{
		origin = { 0, 0, 0 };
		for (const auto& p : pts) {
			origin.x += p.x; origin.y += p.y; origin.z += p.z;
		}
		double npts = static_cast<double>(pts.size());
		origin = { origin.x / npts, origin.y / npts, origin.z / npts };

		double covXX = 0, covYY = 0, covZZ = 0;
		for (const auto& p : pts) {
			tdouble3 d = { p.x - origin.x, p.y - origin.y, p.z - origin.z };
			covXX += d.x * d.x; covYY += d.y * d.y; covZZ += d.z * d.z;
		}
		if (simulate2D) {
			if (covXX <= covZZ)
				normal = { 1, 0, 0 };
			else
				normal = { 0, 0, 1 };
		}
		else {
			if (covXX <= covYY && covXX <= covZZ)
				normal = { 1, 0, 0 };
			else if (covYY <= covXX && covYY <= covZZ)
				normal = { 0, 1, 0 };
			else
				normal = { 0, 0, 1 };
		}

		normal = fmath::NormalizeVec3(normal);
	}

	void computeLocalAxes(const tdouble3& planeNormal, tdouble3& axis1, tdouble3& axis2)
	{
		tdouble3 arbitrary = (std::fabs(planeNormal.x) < 0.9) ? tdouble3{ 1,0,0 } : tdouble3{ 0,1,0 };
		axis1 = fmath::NormalizeVec3(fmath::CrossVec3(planeNormal, arbitrary));
		axis2 = fmath::NormalizeVec3(fmath::CrossVec3(planeNormal, axis1));
	}

	tdouble2 projectPointToLocal(const tdouble3& p, const tdouble3& origin, const tdouble3& axis1, const tdouble3& axis2)
	{
		tdouble3 d = { p.x - origin.x, p.y - origin.y, p.z - origin.z };
		double u = fmath::DotVec3(d, axis1);
		double v = fmath::DotVec3(d, axis2);
		return TDouble2(u, v);
	}

	std::vector<tdouble2> offsetPolygon(const std::vector<tdouble2>& poly, double offset, const double almostzero)
	{
		std::vector<tdouble2> offsetPoly;
		size_t n = poly.size();
		if (n < 3)
			return poly;
		for (size_t i = 0; i < n; i++) {
			size_t prev = (i + n - 1) % n;
			size_t next = (i + 1) % n;

			double dx1 = poly[i].x - poly[prev].x;
			double dy1 = poly[i].y - poly[prev].y;
			double dx2 = poly[next].x - poly[i].x;
			double dy2 = poly[next].y - poly[i].y;
			double len1 = std::hypot(dx1, dy1);
			double len2 = std::hypot(dx2, dy2);
			if (len1 == 0 || len2 == 0)
				continue;
			dx1 /= len1; dy1 /= len1;
			dx2 /= len2; dy2 /= len2;

			double nx1 = dy1;
			double ny1 = -dx1;
			double nx2 = dy2;
			double ny2 = -dx2;

			double x1 = poly[i].x + offset * nx1;
			double y1 = poly[i].y + offset * ny1;
			double x2 = poly[i].x + offset * nx2;
			double y2 = poly[i].y + offset * ny2;

			double det = dx1 * dy2 - dy1 * dx2;
			if (abs(det) < almostzero) {
				offsetPoly.push_back({ (x1 + x2) * 0.5, (y1 + y2) * 0.5 });
			}
			else {
				double t = ((x2 - x1) * dy2 - (y2 - y1) * dx2) / det;
				double ix = x1 + t * dx1;
				double iy = y1 + t * dy1;
				offsetPoly.push_back({ ix, iy });
			}
		}
		return offsetPoly;
	}

	std::vector<tdouble2> computeConvexHull2D(std::vector<tdouble2> points, const double offsetDistance)
	{

		std::sort(points.begin(), points.end(),
			[](const tdouble2& a, const tdouble2& b) {
				return (a.x < b.x ||
					(a.x == b.x && a.y < b.y));
			});

		std::vector<tdouble2> hull;
		for (const auto& pt : points) {
			while (hull.size() >= 2) {
				auto& p1 = hull[hull.size() - 2];
				auto& p2 = hull[hull.size() - 1];
				double cross = (p2.x - p1.x) * (pt.y - p2.y)
					- (p2.y - p1.y) * (pt.x - p2.x);
				if (cross <= 0)
					hull.pop_back();
				else
					break;
			}
			hull.push_back(pt);
		}

		size_t lowerSize = hull.size();
		for (int i = int(points.size()) - 2; i >= 0; i--) {
			while (hull.size() > lowerSize) {
				auto& p1 = hull[hull.size() - 2];
				auto& p2 = hull[hull.size() - 1];
				double cross = (p2.x - p1.x) * (points[i].y - p2.y)
					- (p2.y - p1.y) * (points[i].x - p2.x);
				if (cross <= 0)
					hull.pop_back();
				else
					break;
			}
			hull.push_back(points[i]);
		}

		hull.pop_back();
		return hull;
	}

	void computeConvexHull(const std::vector<tdouble3>& pts, const tdouble3& origin, const tdouble3& axis1, const tdouble3& axis2, std::vector<tdouble2>& polygon, const double offsetDistance)
	{

		std::vector<tdouble2> projPts;

		for (const auto& p : pts)
			projPts.push_back(projectPointToLocal(p, origin, axis1, axis2));

		polygon = computeConvexHull2D(projPts, offsetDistance);

	}

	bool pointInRegion2D(double u, double v, const std::vector<tdouble2>& regionPolygon, double tol, const double almostzero)
	{
		if (regionPolygon.size() >= 3) {

			bool inside = false;
			size_t n = regionPolygon.size();
			for (size_t i = 0, j = n - 1; i < n; j = i++) {
				double xi = regionPolygon[i].x, yi = regionPolygon[i].y;
				double xj = regionPolygon[j].x, yj = regionPolygon[j].y;
				bool intersect = ((yi > v) != (yj > v)) &&
					(u < (xj - xi) * (v - yi) / (yj - yi + almostzero) + xi);
				if (intersect)
					inside = !inside;
			}

			return inside;
		}
		else if (regionPolygon.size() == 2) {
			double x0 = regionPolygon[0].x, y0 = regionPolygon[0].y;
			double x1 = regionPolygon[1].x, y1 = regionPolygon[1].y;
			double dx = x1 - x0, dy = y1 - y0;
			double lenSq = dx * dx + dy * dy;
			double t = ((u - x0) * dx + (v - y0) * dy) / (lenSq + almostzero);
			if (t < 0) t = 0; if (t > 1) t = 1;
			double projX = x0 + t * dx;
			double projY = y0 + t * dy;
			double distSq = (u - projX) * (u - projX) + (v - projY) * (v - projY);
			return (distSq <= tol * tol);
		}
		else if (regionPolygon.size() == 1) {
			double dx = u - regionPolygon[0].x;
			double dy = v - regionPolygon[0].y;
			return ((dx * dx + dy * dy) <= tol * tol);
		}
		return false;
	}

	void prepareBCRegion(tbcregion& region, const double offsetDistance, const double almostzero, const bool simulate2d)
	{

		if (region.pts.data()) {
			if (region.pts.size() >= 3) {

				computeBestFitPlane(region.pts, region.planeOrigin, region.planeNormal, simulate2d);

				computeLocalAxes(region.planeNormal, region.axis1, region.axis2);

				computeConvexHull(region.pts, region.planeOrigin, region.axis1, region.axis2, region.polygon, offsetDistance);

				region.polygon = offsetPolygon(region.polygon, offsetDistance, almostzero);
			}
			else if (region.pts.size() == 2) {
				region.planeOrigin = { (region.pts[0].x + region.pts[1].x) * 0.5,
										(region.pts[0].y + region.pts[1].y) * 0.5,
										(region.pts[0].z + region.pts[1].z) * 0.5 };
				tdouble3 diff = { region.pts[1].x - region.pts[0].x,
									region.pts[1].y - region.pts[0].y,
									region.pts[1].z - region.pts[0].z };
				region.axis1 = fmath::NormalizeVec3(diff);
				region.axis2 = { -region.axis1.z, 0, region.axis1.x };
				region.planeNormal = fmath::NormalizeVec3(fmath::CrossVec3(region.axis1, region.axis2));
				region.polygon.clear();
				region.polygon.push_back(projectPointToLocal(region.pts[0], region.planeOrigin, region.axis1, region.axis2));
				region.polygon.push_back(projectPointToLocal(region.pts[1], region.planeOrigin, region.axis1, region.axis2));
			}
			else if (region.pts.size() == 1) {

				region.planeOrigin = region.pts[0];
				region.planeNormal = { 0,0,1 };
				region.axis1 = { 1,0,0 };
				region.axis2 = { 0,1,0 };
				region.polygon.clear();
				region.polygon.push_back({ 0,0 });
			}

		}

	}
	void SetBoundaryConditions( const bool simulate2D, const unsigned deformStrucCount, StDeformStrucData* deformstrucdata,
		const unsigned mapNdeformstruc, const unsigned npb, const unsigned dsnpsurf, const typecode* dscode,
		const typecode* codec, const float* dsKerSumVol, const unsigned* dssurfpartlist, tbcstruc* dspartfbc,
		tbcstruc* dspartvbc, const std::function<tdouble3(unsigned)>& getPos0, const std::function<tdouble3(unsigned)>& getPosc,
		const double dp, unsigned& outCount, JUserExpressionList* UserExpressions)
	{
		std::fill(dspartfbc, dspartfbc + mapNdeformstruc, make_bcstruc0());
		std::fill(dspartvbc, dspartvbc + mapNdeformstruc, make_bcstruc0());

		std::vector<std::vector<unsigned>> boundaryparts(deformStrucCount);

#ifdef OMP_USE
#pragma omp parallel for
#endif
		for (int ib = 0; ib < int(npb); ib++) {
			typecode code = codec[ib];
			for (unsigned bodyid = 0; bodyid < deformStrucCount; bodyid++) {
				StDeformStrucData& body = deformstrucdata[bodyid];

				for (unsigned k = 0; k < body.nvbc; k++) {
					typecode bc_code = body.bcvel[k].mkcode;
					if (code != bc_code) continue;
#pragma omp critical
					boundaryparts[bodyid].push_back(ib);
				}
				for (unsigned k = 0; k < body.nfbc; k++) {
					typecode bc_code = body.bcforce[k].mkcode;
					if (code != bc_code) continue;
#pragma omp critical
					boundaryparts[bodyid].push_back(ib);
				}
			}
		}

		unsigned cnt = 0;

		// Apply non-surface (body and point) boundary conditions
#ifdef OMP_USE
#pragma omp parallel for reduction(+:cnt)
#endif
		for (int p1 = 0; p1 < int(mapNdeformstruc); p1++) {
			const typecode codep1 = dscode[p1];
			const unsigned bodyid = CODE_GetIbodyDeformStruc(codep1);
			StDeformStrucData& body = deformstrucdata[bodyid];
			float surfacefact = 1.0f / (body.vol0 * (1.0f / dsKerSumVol[p1] - body.selfkern));
			bool cntflag = false;
			tbcstruc dsvolbc = make_bcstruc0();
			tbcstruc dsforcebc = make_bcstruc0();
			const tdouble3 pos0p1 = getPos0(p1);

			// Velocity BC
			for (unsigned k = 0; k < body.nvbc; k++) {
				if (!DSBC_GET_SURFACE(body.bcvel[k].flags)) {
					if (simulate2D) {
						DSBC_SET_Y_FLAG(body.bcvel[k].flags, false);
						DSBC_SET_Y_IS_EXPR(body.bcvel[k].flags, false);
						body.bcvel[k].y = 0.0f;
					}
					dsvolbc.flags = body.bcvel[k].flags;
					dsvolbc.x = body.bcvel[k].x;
					dsvolbc.y = body.bcvel[k].y;
					dsvolbc.z = body.bcvel[k].z;
					dsvolbc.tst = body.bcvel[k].tst;
					dsvolbc.tend = body.bcvel[k].tend;
					cntflag = true;
				}
			}

			// Force BC
			for (unsigned k = 0; k < body.nfbc; k++) {
				if (!DSBC_GET_SURFACE(body.bcforce[k].flags)) {
					unsigned forcetype = DSBC_GET_FORCETYPE(body.bcforce[k].flags);
					float conversionFactor = 1.0f;
					if (forcetype == DSBC_FORCETYPE_POINT) {
						conversionFactor = 1.0f / body.particlemass;
					}
					else if (forcetype == DSBC_FORCETYPE_SURFACE) {
						conversionFactor = float(1.0f / (body.dp * body.rho0) * surfacefact);
					}
					//else if (forcetype == DSBC_FORCETYPE_BODY) {
					//	conversionFactor = 1.0f / body.rho0;
					//}
					dsforcebc.flags = body.bcforce[k].flags;
					if (!DSBC_GET_X_IS_EXPR(dsforcebc.flags)) {
						dsforcebc.x = body.bcforce[k].x * conversionFactor;
					}
					if (!DSBC_GET_Y_IS_EXPR(dsforcebc.flags)) {
						dsforcebc.y = body.bcforce[k].y * conversionFactor;
					}
					if (!DSBC_GET_Z_IS_EXPR(dsforcebc.flags)) {
						dsforcebc.z = body.bcforce[k].z * conversionFactor;
					}

					if (simulate2D) {
						DSBC_SET_Y_FLAG(body.bcforce[k].flags, false);
						DSBC_SET_Y_IS_EXPR(body.bcforce[k].flags, false);
						body.bcforce[k].y = 0.0f;

						DSBC_SET_Y_FLAG(dsforcebc.flags, false);
						DSBC_SET_Y_IS_EXPR(dsforcebc.flags, false);
						dsforcebc.y = 0.0f;
					}
					dsforcebc.tst = body.bcforce[k].tst;
					dsforcebc.tend = body.bcforce[k].tend;
					cntflag = true;
				}
			}
			dspartvbc[p1] = dsvolbc;
			dspartfbc[p1] = dsforcebc;
			if (cntflag) cnt++;
		}

		// Apply surface boundary conditions
#ifdef OMP_USE
#pragma omp parallel for reduction(+:cnt)
#endif
		for (int p = 0; p < int(dsnpsurf); p++) {
			const unsigned p1 = dssurfpartlist[p];
			const typecode codep1 = dscode[p1];
			const unsigned bodyid = CODE_GetIbodyDeformStruc(codep1);
			StDeformStrucData& body = deformstrucdata[bodyid];
			const tdouble3 pospb = getPos0(p1);
			double proxdist2 = (0.5 * dp + 0.5 * body.dp) * (0.5 * dp + 0.5 * body.dp) + 0.2501 * dp * dp;
			float surfacefact = 1.0f / (body.vol0 * (1.0f / dsKerSumVol[p1] - body.selfkern));
			bool cntflag = false;
			tbcstruc dsvolbc = dspartvbc[p1];
			tbcstruc dsforcebc = dspartfbc[p1];
			bool applybcf = false;
			bool applybcv = false;
			for (unsigned ib0 = 0; ib0 < boundaryparts[bodyid].size(); ib0++) {
				unsigned ib = boundaryparts[bodyid][ib0];
				typecode code = codec[ib];
				if (code == codep1) continue;
				const tdouble3 posporg = getPosc(ib);
				const tdouble3 drx = posporg - pospb;
				double rr2 = fmath::DotVec3(drx, drx);
				if (rr2 > proxdist2) continue;

				// Velocity BC
				for (unsigned k = 0; k < body.nvbc; k++) {
					if (DSBC_GET_SURFACE(body.bcvel[k].flags)) {
						typecode bc_code = body.bcvel[k].mkcode;
						if (code != bc_code) continue;
						if (simulate2D) {
							DSBC_SET_Y_FLAG(body.bcvel[k].flags, false);
							DSBC_SET_Y_IS_EXPR(body.bcvel[k].flags, false);
							body.bcvel[k].y = 0.0f;
						}
						dsvolbc.flags = body.bcvel[k].flags;
						dsvolbc.x = body.bcvel[k].x;
						dsvolbc.y = body.bcvel[k].y;
						dsvolbc.z = body.bcvel[k].z;
						dsvolbc.tst = body.bcvel[k].tst;
						dsvolbc.tend = body.bcvel[k].tend;
						cntflag = true;
						applybcv = true;
					}
				}

				// Force BC
				for (unsigned k = 0; k < body.nfbc; k++) {
					if (DSBC_GET_SURFACE(body.bcforce[k].flags)) {
						typecode bc_code = body.bcforce[k].mkcode;
						if (code != bc_code) continue;
						unsigned forcetype = DSBC_GET_FORCETYPE(body.bcforce[k].flags);
						float conversionFactor = 1.0f;
						if (forcetype == DSBC_FORCETYPE_POINT) {
							conversionFactor = 1.0f / body.particlemass;
						}
						else if (forcetype == DSBC_FORCETYPE_SURFACE) {
							conversionFactor = float(1.0f / (body.dp * body.rho0) * surfacefact);
						}
						//else if (forcetype == DSBC_FORCETYPE_BODY) {
						//	conversionFactor = 1.0f / body.rho0;
						//}
						dsforcebc.flags = body.bcforce[k].flags;
						if (!DSBC_GET_X_IS_EXPR(dsforcebc.flags)) {
							dsforcebc.x = body.bcforce[k].x * conversionFactor;
						}
						if (!DSBC_GET_Y_IS_EXPR(dsforcebc.flags)) {
							dsforcebc.y = body.bcforce[k].y * conversionFactor;
						}
						if (!DSBC_GET_Z_IS_EXPR(dsforcebc.flags)) {
							dsforcebc.z = body.bcforce[k].z * conversionFactor;
						}
					
						if (simulate2D) {
							DSBC_SET_Y_FLAG(body.bcforce[k].flags, false);
							DSBC_SET_Y_IS_EXPR(body.bcforce[k].flags, false);
							body.bcforce[k].y = 0.0f;
							DSBC_SET_Y_FLAG(dsforcebc.flags, false);
							DSBC_SET_Y_IS_EXPR(dsforcebc.flags, false);
							dsforcebc.y = 0.0f;
						}
						dsforcebc.tst = body.bcforce[k].tst;
						dsforcebc.tend = body.bcforce[k].tend;
						applybcf = true;
						cntflag = true;
					}
				}
			}
			if (applybcv) dspartvbc[p1] = dsvolbc;
			if (applybcf) dspartfbc[p1] = dsforcebc;
			if (cntflag) cnt++;
		}

		outCount = cnt;
	}
}


