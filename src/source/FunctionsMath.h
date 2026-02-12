//HEAD_DSCODES
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

//:#############################################################################
//:# Descripcion:
//:# =============
//:# Conjunto de funciones tipicas de geometria y demas.
//:#
//:# Cambios:
//:# =========
//:# - Implementacion. (19-03-2013)
//:# - Metodos para calcular area de un triangulo. (01-04-2013)
//:# - Nuevos metodos para interpolacion lineal y bilineal. (08-05-2013)
//:# - Nuevas funciones trigonometricas. (20-08-2015)
//:# - Nuevas funcion DistLine(). (15-03-2016)
//:# - Nuevas funciones PointPlane() y PlanePtVec(). (23-03-2016)
//:# - Nuevas funciones. (05-04-2016)
//:# - Nuevas funciones para matrices. (24-01-2017)
//:# - En el calculo de la matriz inversa puedes pasarle el determinante. (08-02-2017)
//:# - Nuevas funciones IntersecPlaneLine(). (08-09-2016)
//:# - Nuevas funciones MulMatrix3x3(), TrasMatrix3x3() y RotMatrix3x3(). (29-11-2017)
//:# - Nueva funcion VecOrthogonal(). (10-08-2018)
//:# - Nueva funciones Rect3d2pt(), RectPosX(), RectPosY(), RectPosZ(). (21-08-2018)
//:# - Nuevas funciones VecOrthogonal2(). (05-10-2018)
//:# - Se mueven las funciones de geometria 2D y 3D a los nuevos ficheros 
//:#   FunctionsGeo2d.h y FunctionsGeoed.h respectivamente. (08-02-2019)
//:# - Nueva funcion CalcRoundPos(). (02-03-2021)
//:#############################################################################

/// \file FunctionsMath.h \brief Declares basic/general math functions.

#ifndef _FunctionsMath_
#define _FunctionsMath_

#include "TypesDef.h"
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include "DualSphDef.h"

/// Implements a set of basic/general math functions.
namespace fmath {

	//==============================================================================
	/// Devuelve la interpolacion lineal de dos valores.
	/// Returns the linear interpolation value.
	//==============================================================================
	inline double InterpolationLinear(double x, double x0, double x1, double v0, double v1) {
		const double fx = (x - x0) / (x1 - x0);
		return(fx * (v1 - v0) + v0);
	}

	//==============================================================================
	/// Devuelve la interpolacion lineal de dos valores.
	/// Returns the linear interpolation value.
	//==============================================================================
	inline float InterpolationLinear(float x, float x0, float x1, float v0, float v1) {
		const float fx = (x - x0) / (x1 - x0);
		return(fx * (v1 - v0) + v0);
	}

	//==============================================================================
	/// Devuelve la interpolacion bilineal de cuatro valores que forman un cuadrado.
	/// Returns the bilinear interpolation of four values that form a square.
	//==============================================================================
	inline double InterpolationBilinear(double x, double y, double px, double py, double dx, double dy, double vxy, double vxyy, double vxxy, double vxxyy) {
		double vy0 = InterpolationLinear(x, px, px + dx, vxy, vxxy);
		double vy1 = InterpolationLinear(x, px, px + dx, vxyy, vxxyy);
		return(InterpolationLinear(y, py, py + dy, vy0, vy1));
	}

	//<vs_deformstruc_ini>
	//------------------------------------------------------------------------------
	/// Calcula el determinante de una matriz de 2x2.
	/// Returns the determinant of a 2x2 matrix.
	//------------------------------------------------------------------------------
	inline float Determinant2x2(const tmatrix3f& d) {
		return(d.a11 * d.a33 - d.a13 * d.a31);
	}

	inline double Determinant2x2(const tmatrix3d& d) {
		return(d.a11 * d.a33 - d.a13 * d.a31);
	}

	inline double Determinant2x2(const tmatrix2d& d) {
		return(d.a11 * d.a22 - d.a12 * d.a21);
	}

	inline float Determinant2x2(const tmatrix2f& d) {
		return(d.a11 * d.a22 - d.a12 * d.a21);
	}

	//------------------------------------------------------------------------------
	/// Devuelve la matriz inversa de una matriz de 2x2.
	/// Returns the inverse matrix of a 2x2 matrix.
	//------------------------------------------------------------------------------
	inline tmatrix3f InverseMatrix2x2(const tmatrix3f& d, const float det) {
		tmatrix3f inv;
		if (det) {
			inv.a11 = d.a33 / det;
			inv.a12 = 0;
			inv.a13 = -d.a13 / det;
			inv.a21 = 0;
			inv.a22 = 0.0f;
			inv.a23 = 0;
			inv.a31 = -d.a31 / det;
			inv.a32 = 0;
			inv.a33 = d.a11 / det;
		}
		else {
			inv = TMatrix3f(0);
			//inv.a22 = 1.0f;
		}
		return(inv);
	}

	inline tmatrix3d InverseMatrix2x2(const tmatrix3d& d, const double det) {
		tmatrix3d inv;
		if (det) {
			inv.a11 = d.a33 / det;
			inv.a12 = 0;
			inv.a13 = -d.a13 / det;
			inv.a21 = 0;
			inv.a22 = 0.0;
			inv.a23 = 0;
			inv.a31 = -d.a31 / det;
			inv.a32 = 0;
			inv.a33 = d.a11 / det;
		}
		else {
			inv = TMatrix3d(0);
			//inv.a22 = 1.0;
		}
		return(inv);
	}

	//==============================================================================
	/// Devuelve la matriz inversa de una matriz de 2x2.
	/// Returns the inverse matrix of a 2x2 matrix.
	//==============================================================================
	inline tmatrix3f InverseMatrix2x2(const tmatrix3f& d) {
		return(InverseMatrix2x2(d, Determinant2x2(d)));
	}

	inline tmatrix3d InverseMatrix2x2(const tmatrix3d& d) {
		return(InverseMatrix2x2(d, Determinant2x2(d)));
	}

	//<vs_deformstruc_end>

	//==============================================================================
	/// Calcula el determinante de una matriz de 3x3.
	/// Returns the determinant of a 3x3 matrix.
	//==============================================================================
	inline double Determinant3x3(const tmatrix3d& d) {
		return(d.a11 * d.a22 * d.a33 + d.a12 * d.a23 * d.a31 + d.a13 * d.a21 * d.a32 - d.a31 * d.a22 * d.a13 - d.a32 * d.a23 * d.a11 - d.a33 * d.a21 * d.a12);
	}

	//==============================================================================
	/// Calcula el determinante de una matriz de 3x3.
	/// Returns the determinant of a 3x3 matrix.
	//==============================================================================
	inline float Determinant3x3(const tmatrix3f& d) {
		return(d.a11 * d.a22 * d.a33 + d.a12 * d.a23 * d.a31 + d.a13 * d.a21 * d.a32 - d.a31 * d.a22 * d.a13 - d.a32 * d.a23 * d.a11 - d.a33 * d.a21 * d.a12);
	}

	//==============================================================================
	/// Calcula el determinante de una matriz simetrica de 3x3.
	/// Returns the determinant of a 3x3 symmetric matrix.
	//==============================================================================
	inline float Determinant3x3(const tsymatrix3f& d) {
		return(d.xx * (d.yy * d.zz - d.yz * d.yz) +
			d.xy * (d.yz * d.xz - d.xy * d.zz) +
			d.xz * (d.xy * d.yz - d.yy * d.xz));
	}

	//==============================================================================
	/// Calcula el determinante de una matriz simetrica de 4x4.
	/// Returns the determinant of a 4x4 symmetric matrix.
	//==============================================================================
	inline float Determinant4x4(const tsymatrix4f& d) {
		return(d.a11 * (d.a22 * d.a33 * d.a44 + d.a23 * d.a34 * d.a24 + d.a24 * d.a23 * d.a34 - d.a22 * d.a34 * d.a34 - d.a23 * d.a23 * d.a44 - d.a24 * d.a33 * d.a24) +
			d.a12 * (d.a12 * d.a34 * d.a34 + d.a23 * d.a13 * d.a44 + d.a24 * d.a33 * d.a14 - d.a12 * d.a33 * d.a44 - d.a23 * d.a34 * d.a14 - d.a24 * d.a13 * d.a34) +
			d.a13 * (d.a12 * d.a23 * d.a44 + d.a22 * d.a34 * d.a14 + d.a24 * d.a13 * d.a24 - d.a12 * d.a34 * d.a24 - d.a22 * d.a13 * d.a44 - d.a24 * d.a23 * d.a14) +
			d.a14 * (d.a12 * d.a33 * d.a24 + d.a22 * d.a13 * d.a34 + d.a23 * d.a23 * d.a14 - d.a12 * d.a23 * d.a34 - d.a22 * d.a33 * d.a14 - d.a23 * d.a13 * d.a24));
	}

	//==============================================================================
	/// Devuelve la matriz inversa de una matriz de 3x3.
	/// Returns the inverse matrix of a 3x3 matrix.
	//==============================================================================
	inline tmatrix3f InverseMatrix3x3(const tmatrix3f& d, const float det) {
		tmatrix3f inv;
		if (det) {
			inv.a11 = (d.a22 * d.a33 - d.a23 * d.a32) / det;
			inv.a12 = -(d.a12 * d.a33 - d.a13 * d.a32) / det;
			inv.a13 = (d.a12 * d.a23 - d.a13 * d.a22) / det;
			inv.a21 = -(d.a21 * d.a33 - d.a23 * d.a31) / det;
			inv.a22 = (d.a11 * d.a33 - d.a13 * d.a31) / det;
			inv.a23 = -(d.a11 * d.a23 - d.a13 * d.a21) / det;
			inv.a31 = (d.a21 * d.a32 - d.a22 * d.a31) / det;
			inv.a32 = -(d.a11 * d.a32 - d.a12 * d.a31) / det;
			inv.a33 = (d.a11 * d.a22 - d.a12 * d.a21) / det;
		}
		else inv = TMatrix3f(0);
		return(inv);
	}

	//==============================================================================
	/// Devuelve la matriz inversa de una matriz de 3x3.
	/// Returns the inverse matrix of a 3x3 matrix.
	//==============================================================================
	inline tmatrix3f InverseMatrix3x3(const tmatrix3f& d) {
		return(InverseMatrix3x3(d, Determinant3x3(d)));
	}

	//==============================================================================
	/// Devuelve la matriz inversa de una matriz de 3x3.
	/// Returns the inverse matrix of a 3x3 matrix.
	//==============================================================================
	inline tmatrix3d InverseMatrix3x3(const tmatrix3d& d, const double det) {
		tmatrix3d inv;
		if (det) {
			inv.a11 = (d.a22 * d.a33 - d.a23 * d.a32) / det;
			inv.a12 = -(d.a12 * d.a33 - d.a13 * d.a32) / det;
			inv.a13 = (d.a12 * d.a23 - d.a13 * d.a22) / det;
			inv.a21 = -(d.a21 * d.a33 - d.a23 * d.a31) / det;
			inv.a22 = (d.a11 * d.a33 - d.a13 * d.a31) / det;
			inv.a23 = -(d.a11 * d.a23 - d.a13 * d.a21) / det;
			inv.a31 = (d.a21 * d.a32 - d.a22 * d.a31) / det;
			inv.a32 = -(d.a11 * d.a32 - d.a12 * d.a31) / det;
			inv.a33 = (d.a11 * d.a22 - d.a12 * d.a21) / det;
		}
		else inv = TMatrix3d(0);
		return(inv);
	}

	//==============================================================================
	/// Devuelve la matriz inversa de una matriz de 3x3.
	/// Returns the inverse matrix of a 3x3 matrix.
	//==============================================================================
	inline tmatrix3d InverseMatrix3x3(const tmatrix3d& d) {
		return(InverseMatrix3x3(d, Determinant3x3(d)));
	}

	inline tmatrix3d SafeInvDefgrad(const tmatrix3d& F,
		bool isPlastic,
		double& J_out)
	{
		const double a = F.a11, b = F.a12, c = F.a13;
		const double d = F.a21, e = F.a22, f = F.a23;
		const double g = F.a31, h = F.a32, i = F.a33;

		const double C11 = (e * i - f * h);
		const double C12 = -(d * i - f * g);
		const double C13 = (d * h - e * g);

		const double C21 = -(b * i - c * h);
		const double C22 = (a * i - c * g);
		const double C23 = -(a * h - b * g);

		const double C31 = (b * f - c * e);
		const double C32 = -(a * f - c * d);
		const double C33 = (a * e - b * d);

		double J = a * C11 + b * C12 + c * C13;

		double J_use = J;

		if (abs(J_use) < ALMOSTZERO) {
			J_use = (J_use >= 0.0 ? ALMOSTZERO : -ALMOSTZERO);
		}

		if (isPlastic) {
			const double Jmin_plastic = 0.1;
			const double Jmax_plastic = 10.0;

			if (J_use > 0.0) {
				if (J_use < Jmin_plastic) J_use = Jmin_plastic;
				if (J_use > Jmax_plastic) J_use = Jmax_plastic;
			}
			else {
				J_use = 1.0;
			}
		}

		const double invJ = 1.0 / J_use;

		tmatrix3d Finv;
		Finv.a11 = C11 * invJ;
		Finv.a12 = C21 * invJ;
		Finv.a13 = C31 * invJ;

		Finv.a21 = C12 * invJ;
		Finv.a22 = C22 * invJ;
		Finv.a23 = C32 * invJ;

		Finv.a31 = C13 * invJ;
		Finv.a32 = C23 * invJ;
		Finv.a33 = C33 * invJ;

		J_out = J;
		return Finv;
	}
	//==============================================================================
	/// Calcula el determinante de una matriz de 4x4.
	/// Returns the determinant of a 4x4 matrix.
	//==============================================================================
	inline double Determinant4x4(const tmatrix4d& d) {
		return(d.a14 * d.a23 * d.a32 * d.a41 - d.a13 * d.a24 * d.a32 * d.a41 -
			d.a14 * d.a22 * d.a33 * d.a41 + d.a12 * d.a24 * d.a33 * d.a41 +
			d.a13 * d.a22 * d.a34 * d.a41 - d.a12 * d.a23 * d.a34 * d.a41 -
			d.a14 * d.a23 * d.a31 * d.a42 + d.a13 * d.a24 * d.a31 * d.a42 +
			d.a14 * d.a21 * d.a33 * d.a42 - d.a11 * d.a24 * d.a33 * d.a42 -
			d.a13 * d.a21 * d.a34 * d.a42 + d.a11 * d.a23 * d.a34 * d.a42 +
			d.a14 * d.a22 * d.a31 * d.a43 - d.a12 * d.a24 * d.a31 * d.a43 -
			d.a14 * d.a21 * d.a32 * d.a43 + d.a11 * d.a24 * d.a32 * d.a43 +
			d.a12 * d.a21 * d.a34 * d.a43 - d.a11 * d.a22 * d.a34 * d.a43 -
			d.a13 * d.a22 * d.a31 * d.a44 + d.a12 * d.a23 * d.a31 * d.a44 +
			d.a13 * d.a21 * d.a32 * d.a44 - d.a11 * d.a23 * d.a32 * d.a44 -
			d.a12 * d.a21 * d.a33 * d.a44 + d.a11 * d.a22 * d.a33 * d.a44);
	}

	//==============================================================================
	/// Calcula el determinante de una matriz de 4x4.
	/// Returns the determinant of a 4x4 matrix.
	//==============================================================================
	inline float Determinant4x4(const tmatrix4f& d) {
		return(d.a14 * d.a23 * d.a32 * d.a41 - d.a13 * d.a24 * d.a32 * d.a41 -
			d.a14 * d.a22 * d.a33 * d.a41 + d.a12 * d.a24 * d.a33 * d.a41 +
			d.a13 * d.a22 * d.a34 * d.a41 - d.a12 * d.a23 * d.a34 * d.a41 -
			d.a14 * d.a23 * d.a31 * d.a42 + d.a13 * d.a24 * d.a31 * d.a42 +
			d.a14 * d.a21 * d.a33 * d.a42 - d.a11 * d.a24 * d.a33 * d.a42 -
			d.a13 * d.a21 * d.a34 * d.a42 + d.a11 * d.a23 * d.a34 * d.a42 +
			d.a14 * d.a22 * d.a31 * d.a43 - d.a12 * d.a24 * d.a31 * d.a43 -
			d.a14 * d.a21 * d.a32 * d.a43 + d.a11 * d.a24 * d.a32 * d.a43 +
			d.a12 * d.a21 * d.a34 * d.a43 - d.a11 * d.a22 * d.a34 * d.a43 -
			d.a13 * d.a22 * d.a31 * d.a44 + d.a12 * d.a23 * d.a31 * d.a44 +
			d.a13 * d.a21 * d.a32 * d.a44 - d.a11 * d.a23 * d.a32 * d.a44 -
			d.a12 * d.a21 * d.a33 * d.a44 + d.a11 * d.a22 * d.a33 * d.a44);
	}

	//==============================================================================
	/// Devuelve la matriz inversa de una matriz de 4x4.
	/// Returns the inverse matrix of a 4x4 matrix.
	//==============================================================================
	inline tmatrix4f InverseMatrix4x4(const tmatrix4f& d, const float det) {
		tmatrix4f inv;
		if (det) {
			inv.a11 = (d.a22 * (d.a33 * d.a44 - d.a34 * d.a43) + d.a23 * (d.a34 * d.a42 - d.a32 * d.a44) + d.a24 * (d.a32 * d.a43 - d.a33 * d.a42)) / det;
			inv.a21 = (d.a21 * (d.a34 * d.a43 - d.a33 * d.a44) + d.a23 * (d.a31 * d.a44 - d.a34 * d.a41) + d.a24 * (d.a33 * d.a41 - d.a31 * d.a43)) / det;
			inv.a31 = (d.a21 * (d.a32 * d.a44 - d.a34 * d.a42) + d.a22 * (d.a34 * d.a41 - d.a31 * d.a44) + d.a24 * (d.a31 * d.a42 - d.a32 * d.a41)) / det;
			inv.a41 = (d.a21 * (d.a33 * d.a42 - d.a32 * d.a43) + d.a22 * (d.a31 * d.a43 - d.a33 * d.a41) + d.a23 * (d.a32 * d.a41 - d.a31 * d.a42)) / det;
			inv.a12 = (d.a12 * (d.a34 * d.a43 - d.a33 * d.a44) + d.a13 * (d.a32 * d.a44 - d.a34 * d.a42) + d.a14 * (d.a33 * d.a42 - d.a32 * d.a43)) / det;
			inv.a22 = (d.a11 * (d.a33 * d.a44 - d.a34 * d.a43) + d.a13 * (d.a34 * d.a41 - d.a31 * d.a44) + d.a14 * (d.a31 * d.a43 - d.a33 * d.a41)) / det;
			inv.a32 = (d.a11 * (d.a34 * d.a42 - d.a32 * d.a44) + d.a12 * (d.a31 * d.a44 - d.a34 * d.a41) + d.a14 * (d.a32 * d.a41 - d.a31 * d.a42)) / det;
			inv.a42 = (d.a11 * (d.a32 * d.a43 - d.a33 * d.a42) + d.a12 * (d.a33 * d.a41 - d.a31 * d.a43) + d.a13 * (d.a31 * d.a42 - d.a32 * d.a41)) / det;
			inv.a13 = (d.a12 * (d.a23 * d.a44 - d.a24 * d.a43) + d.a13 * (d.a24 * d.a42 - d.a22 * d.a44) + d.a14 * (d.a22 * d.a43 - d.a23 * d.a42)) / det;
			inv.a23 = (d.a11 * (d.a24 * d.a43 - d.a23 * d.a44) + d.a13 * (d.a21 * d.a44 - d.a24 * d.a41) + d.a14 * (d.a23 * d.a41 - d.a21 * d.a43)) / det;
			inv.a33 = (d.a11 * (d.a22 * d.a44 - d.a24 * d.a42) + d.a12 * (d.a24 * d.a41 - d.a21 * d.a44) + d.a14 * (d.a21 * d.a42 - d.a22 * d.a41)) / det;
			inv.a43 = (d.a11 * (d.a23 * d.a42 - d.a22 * d.a43) + d.a12 * (d.a21 * d.a43 - d.a23 * d.a41) + d.a13 * (d.a22 * d.a41 - d.a21 * d.a42)) / det;
			inv.a14 = (d.a12 * (d.a24 * d.a33 - d.a23 * d.a34) + d.a13 * (d.a22 * d.a34 - d.a24 * d.a32) + d.a14 * (d.a23 * d.a32 - d.a22 * d.a33)) / det;
			inv.a24 = (d.a11 * (d.a23 * d.a34 - d.a24 * d.a33) + d.a13 * (d.a24 * d.a31 - d.a21 * d.a34) + d.a14 * (d.a21 * d.a33 - d.a23 * d.a31)) / det;
			inv.a34 = (d.a11 * (d.a24 * d.a32 - d.a22 * d.a34) + d.a12 * (d.a21 * d.a34 - d.a24 * d.a31) + d.a14 * (d.a22 * d.a31 - d.a21 * d.a32)) / det;
			inv.a44 = (d.a11 * (d.a22 * d.a33 - d.a23 * d.a32) + d.a12 * (d.a23 * d.a31 - d.a21 * d.a33) + d.a13 * (d.a21 * d.a32 - d.a22 * d.a31)) / det;
		}
		else inv = TMatrix4f(0);
		return(inv);
	}

	//==============================================================================
	/// Devuelve la matriz inversa de una matriz de 4x4.
	/// Returns the inverse matrix of a 4x4 matrix.
	//==============================================================================
	inline tmatrix4f InverseMatrix4x4(const tmatrix4f& d) {
		return(InverseMatrix4x4(d, Determinant4x4(d)));
	}

	//==============================================================================
	/// Devuelve la matriz inversa de una matriz de 4x4.
	/// Returns the inverse matrix of a 4x4 matrix.
	//==============================================================================
	inline tmatrix4d InverseMatrix4x4(const tmatrix4d& d, const double det) {
		tmatrix4d inv;
		if (det) {
			inv.a11 = (d.a22 * (d.a33 * d.a44 - d.a34 * d.a43) + d.a23 * (d.a34 * d.a42 - d.a32 * d.a44) + d.a24 * (d.a32 * d.a43 - d.a33 * d.a42)) / det;
			inv.a21 = (d.a21 * (d.a34 * d.a43 - d.a33 * d.a44) + d.a23 * (d.a31 * d.a44 - d.a34 * d.a41) + d.a24 * (d.a33 * d.a41 - d.a31 * d.a43)) / det;
			inv.a31 = (d.a21 * (d.a32 * d.a44 - d.a34 * d.a42) + d.a22 * (d.a34 * d.a41 - d.a31 * d.a44) + d.a24 * (d.a31 * d.a42 - d.a32 * d.a41)) / det;
			inv.a41 = (d.a21 * (d.a33 * d.a42 - d.a32 * d.a43) + d.a22 * (d.a31 * d.a43 - d.a33 * d.a41) + d.a23 * (d.a32 * d.a41 - d.a31 * d.a42)) / det;
			inv.a12 = (d.a12 * (d.a34 * d.a43 - d.a33 * d.a44) + d.a13 * (d.a32 * d.a44 - d.a34 * d.a42) + d.a14 * (d.a33 * d.a42 - d.a32 * d.a43)) / det;
			inv.a22 = (d.a11 * (d.a33 * d.a44 - d.a34 * d.a43) + d.a13 * (d.a34 * d.a41 - d.a31 * d.a44) + d.a14 * (d.a31 * d.a43 - d.a33 * d.a41)) / det;
			inv.a32 = (d.a11 * (d.a34 * d.a42 - d.a32 * d.a44) + d.a12 * (d.a31 * d.a44 - d.a34 * d.a41) + d.a14 * (d.a32 * d.a41 - d.a31 * d.a42)) / det;
			inv.a42 = (d.a11 * (d.a32 * d.a43 - d.a33 * d.a42) + d.a12 * (d.a33 * d.a41 - d.a31 * d.a43) + d.a13 * (d.a31 * d.a42 - d.a32 * d.a41)) / det;
			inv.a13 = (d.a12 * (d.a23 * d.a44 - d.a24 * d.a43) + d.a13 * (d.a24 * d.a42 - d.a22 * d.a44) + d.a14 * (d.a22 * d.a43 - d.a23 * d.a42)) / det;
			inv.a23 = (d.a11 * (d.a24 * d.a43 - d.a23 * d.a44) + d.a13 * (d.a21 * d.a44 - d.a24 * d.a41) + d.a14 * (d.a23 * d.a41 - d.a21 * d.a43)) / det;
			inv.a33 = (d.a11 * (d.a22 * d.a44 - d.a24 * d.a42) + d.a12 * (d.a24 * d.a41 - d.a21 * d.a44) + d.a14 * (d.a21 * d.a42 - d.a22 * d.a41)) / det;
			inv.a43 = (d.a11 * (d.a23 * d.a42 - d.a22 * d.a43) + d.a12 * (d.a21 * d.a43 - d.a23 * d.a41) + d.a13 * (d.a22 * d.a41 - d.a21 * d.a42)) / det;
			inv.a14 = (d.a12 * (d.a24 * d.a33 - d.a23 * d.a34) + d.a13 * (d.a22 * d.a34 - d.a24 * d.a32) + d.a14 * (d.a23 * d.a32 - d.a22 * d.a33)) / det;
			inv.a24 = (d.a11 * (d.a23 * d.a34 - d.a24 * d.a33) + d.a13 * (d.a24 * d.a31 - d.a21 * d.a34) + d.a14 * (d.a21 * d.a33 - d.a23 * d.a31)) / det;
			inv.a34 = (d.a11 * (d.a24 * d.a32 - d.a22 * d.a34) + d.a12 * (d.a21 * d.a34 - d.a24 * d.a31) + d.a14 * (d.a22 * d.a31 - d.a21 * d.a32)) / det;
			inv.a44 = (d.a11 * (d.a22 * d.a33 - d.a23 * d.a32) + d.a12 * (d.a23 * d.a31 - d.a21 * d.a33) + d.a13 * (d.a21 * d.a32 - d.a22 * d.a31)) / det;
		}
		else inv = TMatrix4d(0);
		return(inv);
	}

	//==============================================================================
	/// Devuelve la matriz inversa de una matriz de 4x4.
	/// Returns the inverse matrix of a 4x4 matrix.
	//==============================================================================
	inline tmatrix4d InverseMatrix4x4(const tmatrix4d& d) {
		return(InverseMatrix4x4(d, Determinant4x4(d)));
	}


	//==============================================================================
	/// Returns the product of 2 matrices of 3x3.
	//==============================================================================
	inline tmatrix3f MulMatrix3x3(const tmatrix3f& a, const tmatrix3f& b) {
		return(TMatrix3f(
			a.a11 * b.a11 + a.a12 * b.a21 + a.a13 * b.a31, a.a11 * b.a12 + a.a12 * b.a22 + a.a13 * b.a32, a.a11 * b.a13 + a.a12 * b.a23 + a.a13 * b.a33,
			a.a21 * b.a11 + a.a22 * b.a21 + a.a23 * b.a31, a.a21 * b.a12 + a.a22 * b.a22 + a.a23 * b.a32, a.a21 * b.a13 + a.a22 * b.a23 + a.a23 * b.a33,
			a.a31 * b.a11 + a.a32 * b.a21 + a.a33 * b.a31, a.a31 * b.a12 + a.a32 * b.a22 + a.a33 * b.a32, a.a31 * b.a13 + a.a32 * b.a23 + a.a33 * b.a33
		));
	}
	inline tmatrix3d MulMatrix3x3(const tmatrix3d& a, const tmatrix3d& b) {
		return(TMatrix3d(
			a.a11 * b.a11 + a.a12 * b.a21 + a.a13 * b.a31, a.a11 * b.a12 + a.a12 * b.a22 + a.a13 * b.a32, a.a11 * b.a13 + a.a12 * b.a23 + a.a13 * b.a33,
			a.a21 * b.a11 + a.a22 * b.a21 + a.a23 * b.a31, a.a21 * b.a12 + a.a22 * b.a22 + a.a23 * b.a32, a.a21 * b.a13 + a.a22 * b.a23 + a.a23 * b.a33,
			a.a31 * b.a11 + a.a32 * b.a21 + a.a33 * b.a31, a.a31 * b.a12 + a.a32 * b.a22 + a.a33 * b.a32, a.a31 * b.a13 + a.a32 * b.a23 + a.a33 * b.a33
		));
	}

	inline double Trace3x3(const tmatrix3d& a) { return a.a11 + a.a22 + a.a33; }
	inline float Trace3x3(const tmatrix3f& a) { return a.a11 + a.a22 + a.a33; }
	//==============================================================================
	/// Multiplies a 3x3 matrix with a 3D vector and returns a 3D vector.
	//==============================================================================
	inline tdouble3 MulMatrix3x3Vec3(const tmatrix3d& mat, const tdouble3& vec) {
		return TDouble3(
			mat.a11 * vec.x + mat.a12 * vec.y + mat.a13 * vec.z,
			mat.a21 * vec.x + mat.a22 * vec.y + mat.a23 * vec.z,
			mat.a31 * vec.x + mat.a32 * vec.y + mat.a33 * vec.z
		);
	}
	inline tdouble3 MulMatrix3x3Vec3(const tdouble3& vec, const tmatrix3d& mat) {
		return TDouble3(
			vec.x * mat.a11 + vec.y * mat.a21 + vec.z * mat.a31, // X component
			vec.x * mat.a12 + vec.y * mat.a22 + vec.z * mat.a32, // Y component
			vec.x * mat.a13 + vec.y * mat.a23 + vec.z * mat.a33  // Z component
		);
	}

	//==============================================================================
	/// Computes the dot product of two 3D double vectors and returns a double value.
	//==============================================================================
	inline double DotVec3(const tdouble3& vec1, const tdouble3& vec2) {
		return vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z;
	}
	/// Computes the dot product of two 3D float vectors and returns a float value.
	inline float DotVec3(const tfloat3& vec1, const tfloat3& vec2) {
		return vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z;
	}

	inline float DotVecf3(const tdouble3& vec1, const tdouble3& vec2) {
		return float(vec1.x) * float(vec2.x) + float(vec1.y) * float(vec2.y) + float(vec1.z) * float(vec2.z);
	}

	inline tdouble3 CrossVec3(const tdouble3& a, const tdouble3& b) {
		return { a.y * b.z - a.z * b.y,
				a.z * b.x - a.x * b.z,
				a.x * b.y - a.y * b.x };
	}

	inline tfloat3 CrossVec3(const tfloat3& a, const tfloat3& b) {
		return { a.y * b.z - a.z * b.y,
				a.z * b.x - a.x * b.z,
				a.x * b.y - a.y * b.x };
	}

	inline double NormVec3(const tdouble3& a) {
		return std::sqrt(DotVec3(a, a));
	}

	inline float NormVec3(const tfloat3& a) {
		return std::sqrt(DotVec3(a, a));
	}

	inline tdouble3 NormalizeVec3(const tdouble3& a) {
		double n = NormVec3(a);
		return (n < DBL_MIN) ? a : tdouble3{ a.x / n, a.y / n, a.z / n };
	}
	
	inline tfloat3 NormalizeVec3(const tfloat3& a) {
		float n = NormVec3(a);
		return (n < DBL_MIN) ? a : tfloat3{ a.x / n, a.y / n, a.z / n };
	}
	//==============================================================================
	/// Computes the dyadic product of two 3D vectors and returns a 3x3 matrix.
	//==============================================================================
	inline tmatrix3d DyadicVec3(const tdouble3& vec1, const tdouble3& vec2) {
		return TMatrix3d(
			vec1.x * vec2.x, vec1.x * vec2.y, vec1.x * vec2.z,
			vec1.y * vec2.x, vec1.y * vec2.y, vec1.y * vec2.z,
			vec1.z * vec2.x, vec1.z * vec2.y, vec1.z * vec2.z
		);
	}
	inline tmatrix3f DyadicVec3(const tfloat3& vec1, const tfloat3& vec2) {
		return TMatrix3f(
			vec1.x * vec2.x, vec1.x * vec2.y, vec1.x * vec2.z,
			vec1.y * vec2.x, vec1.y * vec2.y, vec1.y * vec2.z,
			vec1.z * vec2.x, vec1.z * vec2.y, vec1.z * vec2.z
		);
	}
	//==============================================================================
	/// Devuelve traspuesta de matriz 3x3.
	/// Returns the transpose from matrix 3x3.
	//==============================================================================
	inline tmatrix3f TrasMatrix3x3(const tmatrix3f& a) {
		return(TMatrix3f(
			a.a11, a.a21, a.a31,
			a.a12, a.a22, a.a32,
			a.a13, a.a23, a.a33
		));
	}
	inline tmatrix3d TrasMatrix3x3(const tmatrix3d& a) {
		return(TMatrix3d(
			a.a11, a.a21, a.a31,
			a.a12, a.a22, a.a32,
			a.a13, a.a23, a.a33
		));
	}
	//==============================================================================
	/// Devuelve la matriz de rotacion.
	/// Returns the rotation matrix.
	//==============================================================================
	inline tmatrix3f RotMatrix3x3(const tfloat3& ang) {
		const float cosx = cos(ang.x), cosy = cos(ang.y), cosz = cos(ang.z);
		const float sinx = sin(ang.x), siny = sin(ang.y), sinz = sin(ang.z);
		return(TMatrix3f(
			cosy * cosz, -cosy * sinz, siny,
			sinx * siny * cosz + cosx * sinz, -sinx * siny * sinz + cosx * cosz, -sinx * cosy,
			-cosx * siny * cosz + sinx * sinz, cosx * siny * sinz + sinx * cosz, cosx * cosy
		));
	}


	//==============================================================================
	/// Returns cotangent of angle in radians.
	//==============================================================================
	inline double cot(double z) { return(1.0 / tan(z)); }

	//==============================================================================
	/// Returns hyperbolic cotangent of angle in radians.
	//==============================================================================
	inline double coth(double z) { return(1.0 / tanh(z)); }
	//inline double coth(double z){ return(cosh(z) / sinh(z)); }

	//==============================================================================
	/// Returns secant of angle in radians.
	//==============================================================================
	inline double sec(double z) { return(1.0 / cos(z)); }

	//==============================================================================
	/// Returns cosecant of input angle in radians.
	//==============================================================================
	inline double csc(double z) { return(1.0 / sin(z)); }

	//==============================================================================
	// Calculates position set at specific intervals (dp) from a start (posmin).
	//==============================================================================
	inline double CalcRoundPos(double pos, double posmin, double dp) {
		const int posi = int(round((pos - posmin) / dp));
		return(posmin + dp * posi);
	}

	template <typename T>
	inline T array_max(const T* arr, int sz) {
		if (sz <= 0) return T(); // Return default-constructed T if empty
		T max_val = arr[0];
#ifdef OMP_USE
#pragma omp parallel if (sz > OMP_LIMIT_COMPUTESTEPDEFSTRUC)
		{
			T local_max = max_val;
#pragma omp for schedule(static)
			for (int p = 0; p < sz; ++p) {
				if (local_max < arr[p])
					local_max = arr[p];
			}
#pragma omp critical
			{
				if (max_val < local_max)
					max_val = local_max;
			}
		}
#else
		for (int p = 0; p < sz; ++p) {
			if (max_val < arr[p])
				max_val = arr[p];
		}
#endif
		return max_val;
	}

	template <typename T>
	inline T array_min(const T* arr, int sz) {
		if (sz <= 0) return T(); // Return default-constructed T if empty
		T min_val = arr[0];
#ifdef OMP_USE
#pragma omp parallel if (sz > OMP_LIMIT_COMPUTESTEPDEFSTRUC)
		{
			T local_min = min_val;
#pragma omp for schedule(static)
			for (int p = 0; p < sz; ++p) {
				if (local_min > arr[p])
					local_min = arr[p];
			}
#pragma omp critical
			{
				if (min_val > local_min)
					min_val = local_min;
			}
		}
#else
		for (int p = 0; p < sz; ++p) {
			if (min_val > arr[p])
				min_val = arr[p];
		}
#endif
		return min_val;
	}

	inline tdouble3 array_max(const tdouble3* arr, int sz) {
		if (sz <= 0) return tdouble3();
		double max_x = arr[0].x, max_y = arr[0].y, max_z = arr[0].z;
#ifdef OMP_USE
#pragma omp parallel if (sz > OMP_LIMIT_COMPUTESTEPDEFSTRUC)
		{
			double local_x = max_x, local_y = max_y, local_z = max_z;
#pragma omp for schedule(static)
			for (int p = 0; p < sz; ++p) {
				if (local_x < arr[p].x)
					local_x = arr[p].x;
				if (local_y < arr[p].y)
					local_y = arr[p].y;
				if (local_z < arr[p].z)
					local_z = arr[p].z;
			}
#pragma omp critical
			{
				if (max_x < local_x)
					max_x = local_x;
				if (max_y < local_y)
					max_y = local_y;
				if (max_z < local_z)
					max_z = local_z;
			}
		}
#else
		for (int p = 0; p < sz; ++p) {
			if (max_x < arr[p].x)
				max_x = arr[p].x;
			if (max_y < arr[p].y)
				max_y = arr[p].y;
			if (max_z < arr[p].z)
				max_z = arr[p].z;
		}
#endif
		return TDouble3(max_x, max_y, max_z);
	}

	inline tdouble3 array_min(const tdouble3* arr, int sz) {
		if (sz <= 0) return tdouble3();
		double min_x = arr[0].x, min_y = arr[0].y, min_z = arr[0].z;
#ifdef OMP_USE
#pragma omp parallel if (sz > OMP_LIMIT_COMPUTESTEPDEFSTRUC)
		{
			double local_x = min_x, local_y = min_y, local_z = min_z;
#pragma omp for schedule(static)
			for (int p = 0; p < sz; ++p) {
				if (local_x > arr[p].x)
					local_x = arr[p].x;
				if (local_y > arr[p].y)
					local_y = arr[p].y;
				if (local_z > arr[p].z)
					local_z = arr[p].z;
			}
#pragma omp critical
			{
				if (min_x > local_x)
					min_x = local_x;
				if (min_y > local_y)
					min_y = local_y;
				if (min_z > local_z)
					min_z = local_z;
			}
		}
#else
		for (int p = 0; p < sz; ++p) {
			if (min_x > arr[p].x)
				min_x = arr[p].x;
			if (min_y > arr[p].y)
				min_y = arr[p].y;
			if (min_z > arr[p].z)
				min_z = arr[p].z;
		}
#endif
		return TDouble3(min_x, min_y, min_z);
	}

	inline tfloat3 array_max(const tfloat3* arr, int sz) {
		if (sz <= 0) return tfloat3();
		float max_x = arr[0].x, max_y = arr[0].y, max_z = arr[0].z;
#ifdef OMP_USE
#pragma omp parallel if (sz > OMP_LIMIT_COMPUTESTEPDEFSTRUC)
		{
			float local_x = max_x, local_y = max_y, local_z = max_z;
#pragma omp for schedule(static)
			for (int p = 0; p < sz; ++p) {
				if (local_x < arr[p].x)
					local_x = arr[p].x;
				if (local_y < arr[p].y)
					local_y = arr[p].y;
				if (local_z < arr[p].z)
					local_z = arr[p].z;
			}
#pragma omp critical
			{
				if (max_x < local_x)
					max_x = local_x;
				if (max_y < local_y)
					max_y = local_y;
				if (max_z < local_z)
					max_z = local_z;
			}
		}
#else
		for (int p = 0; p < sz; ++p) {
			if (max_x < arr[p].x)
				max_x = arr[p].x;
			if (max_y < arr[p].y)
				max_y = arr[p].y;
			if (max_z < arr[p].z)
				max_z = arr[p].z;
		}
#endif
		return TFloat3(max_x, max_y, max_z);
	}

	inline tfloat3 array_min(const tfloat3* arr, int sz) {
		if (sz <= 0) return tfloat3();
		float min_x = arr[0].x, min_y = arr[0].y, min_z = arr[0].z;
#ifdef OMP_USE
#pragma omp parallel if (sz > OMP_LIMIT_COMPUTESTEPDEFSTRUC)
		{
			float local_x = min_x, local_y = min_y, local_z = min_z;
#pragma omp for schedule(static)
			for (int p = 0; p < sz; ++p) {
				if (local_x > arr[p].x)
					local_x = arr[p].x;
				if (local_y > arr[p].y)
					local_y = arr[p].y;
				if (local_z > arr[p].z)
					local_z = arr[p].z;
			}
#pragma omp critical
			{
				if (min_x > local_x)
					min_x = local_x;
				if (min_y > local_y)
					min_y = local_y;
				if (min_z > local_z)
					min_z = local_z;
			}
		}
#else
		for (int p = 0; p < sz; ++p) {
			if (min_x > arr[p].x)
				min_x = arr[p].x;
			if (min_y > arr[p].y)
				min_y = arr[p].y;
			if (min_z > arr[p].z)
				min_z = arr[p].z;
		}
#endif
		return TFloat3(min_x, min_y, min_z);
	}


	//========================================================================================
/// Decomposes a 3x3 matrix into positive and negative parts using eigenvalue decomposition
//========================================================================================
	template<bool simulate2d>
	tmatrix3d DSEigenDecompose(const tmatrix3d Amat)
	{

		tdouble3 eigenvalues;
		tmatrix3d Qmat;
		if (simulate2d) {

			const double bb = Amat.a11 + Amat.a33;
			const double cc = Amat.a11 * Amat.a33 - Amat.a13 * Amat.a13;
			double delta_arg = bb * bb - 4.0 * cc;
			if (delta_arg < 0.0) delta_arg = 0.0;
			const double delta = sqrt(delta_arg);
			eigenvalues.x = (bb - delta) * 0.5;
			eigenvalues.z = (bb + delta) * 0.5;
			eigenvalues.y = 0.0;

			auto normalise = [](double& vx, double& vz) {
				const double norm = sqrt(vx * vx + vz * vz);
				if (norm > 1.0e-20) {
					vx /= norm;
					vz /= norm;
					return true;
				}
				return false;
				};

			double v1x = -Amat.a13;
			double v1z = Amat.a11 - eigenvalues.x;
			if (!normalise(v1x, v1z)) {
				v1x = 1.0;
				v1z = 0.0;
			}

			double v2x = -Amat.a13;
			double v2z = Amat.a11 - eigenvalues.z;
			if (!normalise(v2x, v2z)) {
				v2x = -v1z;
				v2z = v1x;
			}
			else {
				const double dot = v1x * v2x + v1z * v2z;
				v2x -= dot * v1x;
				v2z -= dot * v1z;
				if (!normalise(v2x, v2z)) {
					v2x = -v1z;
					v2z = v1x;
				}
			}

			Qmat = { v1x, 0.0, v2x,
					0.0, 1.0, 0.0,
					v1z, 0.0, v2z };
		}
		else {

			// Compute characteristic polynomial coefficients
			double p1 = Amat.a12 * Amat.a12 + Amat.a13 * Amat.a13 + Amat.a23 * Amat.a23;
			if (p1 < 1.0e-20) {  // Diagonal matrix case
				eigenvalues = { Amat.a11, Amat.a22, Amat.a33 };
				Qmat = { 1, 0, 0,  0, 1, 0,  0, 0, 1 }; // Identity matrix
			}
			else {
				double q = (Amat.a11 + Amat.a22 + Amat.a33) / 3.0;
				double p2 = (Amat.a11 - q) * (Amat.a11 - q) + (Amat.a22 - q) * (Amat.a22 - q) + (Amat.a33 - q) * (Amat.a33 - q) + 2 * p1;
				double p = sqrt(p2 / 6.0);

				tmatrix3d B = {
					(Amat.a11 - q) / p, Amat.a12 / p, Amat.a13 / p,
					Amat.a21 / p, (Amat.a22 - q) / p, Amat.a23 / p,
					Amat.a31 / p, Amat.a32 / p, (Amat.a33 - q) / p
				};

				double detB = B.a11 * (B.a22 * B.a33 - B.a23 * B.a32)
					- B.a12 * (B.a21 * B.a33 - B.a23 * B.a31)
					+ B.a13 * (B.a21 * B.a32 - B.a22 * B.a31);

				double r = detB / 2.0;
				double phi = (r <= -1) ? PI / 3.0 : (r >= 1) ? 0.0 : acos(r) / 3.0;

				double lambda1 = q + 2 * p * cos(phi);
				double lambda3 = q + 2 * p * cos(phi + (2.0 * PI / 3.0));
				double lambda2 = 3 * q - lambda1 - lambda3;

				if (lambda1 < lambda2) std::swap(lambda1, lambda2);
				if (lambda1 < lambda3) std::swap(lambda1, lambda3);
				if (lambda2 < lambda3) std::swap(lambda2, lambda3);

				eigenvalues = { lambda1, lambda2, lambda3 };

				auto computeEigenvector = [&](double lambda, double& v1, double& v2, double& v3) {
					tmatrix3d M = {
						Amat.a11 - lambda, Amat.a12, Amat.a13,
						Amat.a21, Amat.a22 - lambda, Amat.a23,
						Amat.a31, Amat.a32, Amat.a33 - lambda
					};

					v1 = M.a22 * M.a33 - M.a23 * M.a32;
					v2 = M.a13 * M.a32 - M.a12 * M.a33;
					v3 = M.a12 * M.a23 - M.a13 * M.a22;

					double norm = sqrt(v1 * v1 + v2 * v2 + v3 * v3);
					if (norm > 1.0e-20) {
						v1 /= norm; v2 /= norm; v3 /= norm;
					}
					};

				computeEigenvector(lambda1, Qmat.a11, Qmat.a21, Qmat.a31);
				computeEigenvector(lambda2, Qmat.a12, Qmat.a22, Qmat.a32);
				computeEigenvector(lambda3, Qmat.a13, Qmat.a23, Qmat.a33);
			}
		}

		tmatrix3d Lp = TMatrix3d(0);
		Lp.a11 = (eigenvalues.x > 0.0 ? eigenvalues.x : 0.0);
		Lp.a22 = (eigenvalues.y > 0.0 ? eigenvalues.y : 0.0);
		Lp.a33 = (eigenvalues.z > 0.0 ? eigenvalues.z : 0.0);
		return fmath::MulMatrix3x3(Qmat, fmath::MulMatrix3x3(Lp, fmath::TrasMatrix3x3(Qmat)));
	}

}

#endif


