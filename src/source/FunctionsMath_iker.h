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

/// \file FunctionsMath_iker.h \brief Implements basic/general math functions for the GPU executions.

#include "TypesDef.h"
#include "TypesDef_GPU.h"
#include <cuda_runtime_api.h>

namespace cumath {

	//------------------------------------------------------------------------------
	/// Resuelve punto en el plano.
	/// Solves point in the plane.
	//------------------------------------------------------------------------------
	__device__ double PointPlane(const float4& pla, const double3& pt) {
		return(pt.x * pla.x + pt.y * pla.y + pt.z * pla.z + pla.w);
	}

	//------------------------------------------------------------------------------
	/// Resuelve punto en el plano.
	/// Solves point in the plane.
	//------------------------------------------------------------------------------
	__device__ float PointPlane(const float4& pla, float px, float py, float pz) {
		return(pla.x * px + pla.y * py + pla.z * pz + pla.w);
	}

	//------------------------------------------------------------------------------
	/// Returns the distance between a point and a plane.
	/// Devuelve la distancia entre un punto y un plano.
	//------------------------------------------------------------------------------
	__device__ double DistPlaneSign(const float4& pla, const double3& pt) {
		return(PointPlane(pla, pt) / sqrt(pla.x * pla.x + pla.y * pla.y + pla.z * pla.z));
	}

	//------------------------------------------------------------------------------
	/// Returns the distance between a point and a plane.
	/// Devuelve la distancia entre un punto y un plano.
	//------------------------------------------------------------------------------
	__device__ float KerDistPlaneSign(const float4& pla, float px, float py, float pz) {
		return(PointPlane(pla, px, py, pz) / sqrt(pla.x * pla.x + pla.y * pla.y + pla.z * pla.z));
	}

	//------------------------------------------------------------------------------
	/// Returns the distance between a point and a plane.
	/// Devuelve la distancia entre un punto y un plano.
	//------------------------------------------------------------------------------
	__device__ double DistPlane(const float4& pla, const double3& pt) {
		return(fabs(DistPlaneSign(pla, pt)));
	}

	//------------------------------------------------------------------------------
	/// Initializes matrix to zero.
	/// Inicializa matriz a cero.
	//------------------------------------------------------------------------------
	__device__ void Tmatrix3fReset(tmatrix3f& m) {
		m.a11 = m.a12 = m.a13 = m.a21 = m.a22 = m.a23 = m.a31 = m.a32 = m.a33 = 0;
	}

	//------------------------------------------------------------------------------
	/// Initializes matrix to zero.
	/// Inicializa matriz a cero.
	//------------------------------------------------------------------------------
	__device__ void Tmatrix3dReset(tmatrix3d& m) {
		m.a11 = m.a12 = m.a13 = m.a21 = m.a22 = m.a23 = m.a31 = m.a32 = m.a33 = 0;
	}

	//------------------------------------------------------------------------------
	/// Initializes matrix to zero.
	/// Inicializa matriz a cero.
	//------------------------------------------------------------------------------
	__device__ void Tmatrix4fReset(tmatrix4f& m) {
		m.a11 = m.a12 = m.a13 = m.a14 = m.a21 = m.a22 = m.a23 = m.a24 = m.a31 = m.a32 = m.a33 = m.a34 = m.a41 = m.a42 = m.a43 = m.a44 = 0;
	}

	//------------------------------------------------------------------------------
	/// Initializes matrix to zero.
	/// Inicializa matriz a cero.
	//------------------------------------------------------------------------------
	__device__ void Tmatrix4dReset(tmatrix4d& m) {
		m.a11 = m.a12 = m.a13 = m.a14 = m.a21 = m.a22 = m.a23 = m.a24 = m.a31 = m.a32 = m.a33 = m.a34 = m.a41 = m.a42 = m.a43 = m.a44 = 0;
	}

	//<vs_deformstruc_ini>
	//------------------------------------------------------------------------------
	/// Calcula el determinante de una matriz de 2x2.
	/// Returns the determinant of a 2x2 matrix.
	//------------------------------------------------------------------------------
	__device__ float Determinant2x2(const tmatrix3f& d) {
		return(d.a11 * d.a33 - d.a13 * d.a31);
	}

	//------------------------------------------------------------------------------
	/// Devuelve la matriz inversa de una matriz de 2x2.
	/// Returns the inverse matrix of a 2x2 matrix.
	//------------------------------------------------------------------------------
	__device__ tmatrix3f InverseMatrix2x2(const tmatrix3f& d, const float det) {
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
			Tmatrix3fReset(inv);
			//inv.a22 = 1.0f;
		}
		return(inv);
	}

	//==============================================================================
	/// Devuelve la matriz inversa de una matriz de 2x2.
	/// Returns the inverse matrix of a 2x2 matrix.
	//==============================================================================
	__device__ tmatrix3f InverseMatrix2x2(const tmatrix3f& d) {
		return(InverseMatrix2x2(d, Determinant2x2(d)));
	}
	//<vs_deformstruc_end>

	//------------------------------------------------------------------------------
	/// Calcula el determinante de una matriz de 3x3.
	/// Returns the determinant of a 3x3 matrix.
	//------------------------------------------------------------------------------
	__device__ float Determinant3x3(const tmatrix3f& d) {
		return(d.a11 * d.a22 * d.a33 + d.a12 * d.a23 * d.a31 + d.a13 * d.a21 * d.a32 - d.a31 * d.a22 * d.a13 - d.a32 * d.a23 * d.a11 - d.a33 * d.a21 * d.a12);
	}

	//------------------------------------------------------------------------------
	/// Calcula el determinante de una matriz de 3x3.
	/// Returns the determinant of a 3x3 matrix.
	//------------------------------------------------------------------------------
	__device__ double Determinant3x3dbl(const tmatrix3f& d) {
		return(double(d.a11) * double(d.a22) * double(d.a33) + double(d.a12) * double(d.a23) * double(d.a31) + double(d.a13) * double(d.a21) * double(d.a32) - double(d.a31) * double(d.a22) * double(d.a13) - double(d.a32) * double(d.a23) * double(d.a11) - double(d.a33) * double(d.a21) * double(d.a12));
	}

	//------------------------------------------------------------------------------
	/// Calcula el determinante de una matriz de 3x3.
	/// Returns the determinant of a 3x3 matrix.
	//------------------------------------------------------------------------------
	__device__ double Determinant3x3(const tmatrix3d& d) {
		return(d.a11 * d.a22 * d.a33 + d.a12 * d.a23 * d.a31 + d.a13 * d.a21 * d.a32 - d.a31 * d.a22 * d.a13 - d.a32 * d.a23 * d.a11 - d.a33 * d.a21 * d.a12);
	}

	//------------------------------------------------------------------------------
	/// Devuelve la matriz inversa de una matriz de 3x3.
	/// Returns the inverse matrix of a 3x3 matrix.
	//------------------------------------------------------------------------------
	__device__ tmatrix3f InverseMatrix3x3(const tmatrix3f& d, const float det) {
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
		else Tmatrix3fReset(inv);
		return(inv);
	}

	//------------------------------------------------------------------------------
	/// Devuelve la matriz inversa de una matriz de 3x3.
	/// Returns the inverse matrix of a 3x3 matrix.
	//------------------------------------------------------------------------------
	__device__ tmatrix3f InverseMatrix3x3dbl(const tmatrix3f& d, const double det) {
		tmatrix3f inv;
		if (det) {
			inv.a11 = float((double(d.a22) * double(d.a33) - double(d.a23) * double(d.a32)) / det);
			inv.a12 = float(-(double(d.a12) * double(d.a33) - double(d.a13) * double(d.a32)) / det);
			inv.a13 = float((double(d.a12) * double(d.a23) - double(d.a13) * double(d.a22)) / det);
			inv.a21 = float(-(double(d.a21) * double(d.a33) - double(d.a23) * double(d.a31)) / det);
			inv.a22 = float((double(d.a11) * double(d.a33) - double(d.a13) * double(d.a31)) / det);
			inv.a23 = float(-(double(d.a11) * double(d.a23) - double(d.a13) * double(d.a21)) / det);
			inv.a31 = float((double(d.a21) * double(d.a32) - double(d.a22) * double(d.a31)) / det);
			inv.a32 = float(-(double(d.a11) * double(d.a32) - double(d.a12) * double(d.a31)) / det);
			inv.a33 = float((double(d.a11) * double(d.a22) - double(d.a12) * double(d.a21)) / det);
		}
		else Tmatrix3fReset(inv);
		return(inv);
	}

	//------------------------------------------------------------------------------
	/// Devuelve la matriz inversa de una matriz de 3x3.
	/// Returns the inverse matrix of a 3x3 matrix.
	//------------------------------------------------------------------------------
	__device__ tmatrix3d InverseMatrix3x3(const tmatrix3d& d, const double det) {
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
		else Tmatrix3dReset(inv);
		return(inv);
	}

	//==============================================================================
	/// Devuelve la matriz inversa de una matriz de 3x3.
	/// Returns the inverse matrix of a 3x3 matrix.
	//==============================================================================
	__device__ tmatrix3f InverseMatrix3x3(const tmatrix3f& d) {
		return(InverseMatrix3x3(d, Determinant3x3(d)));
	}

	//==============================================================================
	/// Devuelve la matriz inversa de una matriz de 3x3.
	/// Returns the inverse matrix of a 3x3 matrix.
	//==============================================================================
	__device__ tmatrix3d InverseMatrix3x3(const tmatrix3d& d) {
		return(InverseMatrix3x3(d, Determinant3x3(d)));
	}

	//------------------------------------------------------------------------------
	/// Calcula el determinante de una matriz de 4x4.
	/// Returns the determinant of a 4x4 matrix.
	//------------------------------------------------------------------------------
	__device__ float Determinant4x4(const tmatrix4f& d) {
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

	//------------------------------------------------------------------------------
	/// Calcula el determinante de una matriz de 4x4.
	/// Returns the determinant of a 4x4 matrix.
	//------------------------------------------------------------------------------
	__device__ double Determinant4x4dbl(const tmatrix4f& d) {
		return(double(d.a14) * double(d.a23) * double(d.a32) * double(d.a41) - double(d.a13) * double(d.a24) * double(d.a32) * double(d.a41) -
			double(d.a14) * double(d.a22) * double(d.a33) * double(d.a41) + double(d.a12) * double(d.a24) * double(d.a33) * double(d.a41) +
			double(d.a13) * double(d.a22) * double(d.a34) * double(d.a41) - double(d.a12) * double(d.a23) * double(d.a34) * double(d.a41) -
			double(d.a14) * double(d.a23) * double(d.a31) * double(d.a42) + double(d.a13) * double(d.a24) * double(d.a31) * double(d.a42) +
			double(d.a14) * double(d.a21) * double(d.a33) * double(d.a42) - double(d.a11) * double(d.a24) * double(d.a33) * double(d.a42) -
			double(d.a13) * double(d.a21) * double(d.a34) * double(d.a42) + double(d.a11) * double(d.a23) * double(d.a34) * double(d.a42) +
			double(d.a14) * double(d.a22) * double(d.a31) * double(d.a43) - double(d.a12) * double(d.a24) * double(d.a31) * double(d.a43) -
			double(d.a14) * double(d.a21) * double(d.a32) * double(d.a43) + double(d.a11) * double(d.a24) * double(d.a32) * double(d.a43) +
			double(d.a12) * double(d.a21) * double(d.a34) * double(d.a43) - double(d.a11) * double(d.a22) * double(d.a34) * double(d.a43) -
			double(d.a13) * double(d.a22) * double(d.a31) * double(d.a44) + double(d.a12) * double(d.a23) * double(d.a31) * double(d.a44) +
			double(d.a13) * double(d.a21) * double(d.a32) * double(d.a44) - double(d.a11) * double(d.a23) * double(d.a32) * double(d.a44) -
			double(d.a12) * double(d.a21) * double(d.a33) * double(d.a44) + double(d.a11) * double(d.a22) * double(d.a33) * double(d.a44));
	}

	//------------------------------------------------------------------------------
	/// Calcula el determinante de una matriz de 4x4.
	/// Returns the determinant of a 4x4 matrix.
	//------------------------------------------------------------------------------
	__device__ double Determinant4x4(const tmatrix4d& d) {
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

	//------------------------------------------------------------------------------
	/// Devuelve la matriz inversa de una matriz de 4x4.
	/// Returns the inverse matrix of a 4x4 matrix.
	//------------------------------------------------------------------------------
	__device__ tmatrix4f InverseMatrix4x4(const tmatrix4f& d, const float det) {
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
		else Tmatrix4fReset(inv);
		return(inv);
	}

	//------------------------------------------------------------------------------
	/// Devuelve la matriz inversa de una matriz de 4x4.
	/// Returns the inverse matrix of a 4x4 matrix.
	//------------------------------------------------------------------------------
	__device__ tmatrix4f InverseMatrix4x4dbl(const tmatrix4f& d, const double det) {
		tmatrix4f inv;
		if (det) {
			inv.a11 = (double(d.a22) * (double(d.a33) * double(d.a44) - double(d.a34) * double(d.a43)) + double(d.a23) * (double(d.a34) * double(d.a42) - double(d.a32) * double(d.a44)) + double(d.a24) * (double(d.a32) * double(d.a43) - double(d.a33) * double(d.a42))) / det;
			inv.a21 = (double(d.a21) * (double(d.a34) * double(d.a43) - double(d.a33) * double(d.a44)) + double(d.a23) * (double(d.a31) * double(d.a44) - double(d.a34) * double(d.a41)) + double(d.a24) * (double(d.a33) * double(d.a41) - double(d.a31) * double(d.a43))) / det;
			inv.a31 = (double(d.a21) * (double(d.a32) * double(d.a44) - double(d.a34) * double(d.a42)) + double(d.a22) * (double(d.a34) * double(d.a41) - double(d.a31) * double(d.a44)) + double(d.a24) * (double(d.a31) * double(d.a42) - double(d.a32) * double(d.a41))) / det;
			inv.a41 = (double(d.a21) * (double(d.a33) * double(d.a42) - double(d.a32) * double(d.a43)) + double(d.a22) * (double(d.a31) * double(d.a43) - double(d.a33) * double(d.a41)) + double(d.a23) * (double(d.a32) * double(d.a41) - double(d.a31) * double(d.a42))) / det;
			inv.a12 = (double(d.a12) * (double(d.a34) * double(d.a43) - double(d.a33) * double(d.a44)) + double(d.a13) * (double(d.a32) * double(d.a44) - double(d.a34) * double(d.a42)) + double(d.a14) * (double(d.a33) * double(d.a42) - double(d.a32) * double(d.a43))) / det;
			inv.a22 = (double(d.a11) * (double(d.a33) * double(d.a44) - double(d.a34) * double(d.a43)) + double(d.a13) * (double(d.a34) * double(d.a41) - double(d.a31) * double(d.a44)) + double(d.a14) * (double(d.a31) * double(d.a43) - double(d.a33) * double(d.a41))) / det;
			inv.a32 = (double(d.a11) * (double(d.a34) * double(d.a42) - double(d.a32) * double(d.a44)) + double(d.a12) * (double(d.a31) * double(d.a44) - double(d.a34) * double(d.a41)) + double(d.a14) * (double(d.a32) * double(d.a41) - double(d.a31) * double(d.a42))) / det;
			inv.a42 = (double(d.a11) * (double(d.a32) * double(d.a43) - double(d.a33) * double(d.a42)) + double(d.a12) * (double(d.a33) * double(d.a41) - double(d.a31) * double(d.a43)) + double(d.a13) * (double(d.a31) * double(d.a42) - double(d.a32) * double(d.a41))) / det;
			inv.a13 = (double(d.a12) * (double(d.a23) * double(d.a44) - double(d.a24) * double(d.a43)) + double(d.a13) * (double(d.a24) * double(d.a42) - double(d.a22) * double(d.a44)) + double(d.a14) * (double(d.a22) * double(d.a43) - double(d.a23) * double(d.a42))) / det;
			inv.a23 = (double(d.a11) * (double(d.a24) * double(d.a43) - double(d.a23) * double(d.a44)) + double(d.a13) * (double(d.a21) * double(d.a44) - double(d.a24) * double(d.a41)) + double(d.a14) * (double(d.a23) * double(d.a41) - double(d.a21) * double(d.a43))) / det;
			inv.a33 = (double(d.a11) * (double(d.a22) * double(d.a44) - double(d.a24) * double(d.a42)) + double(d.a12) * (double(d.a24) * double(d.a41) - double(d.a21) * double(d.a44)) + double(d.a14) * (double(d.a21) * double(d.a42) - double(d.a22) * double(d.a41))) / det;
			inv.a43 = (double(d.a11) * (double(d.a23) * double(d.a42) - double(d.a22) * double(d.a43)) + double(d.a12) * (double(d.a21) * double(d.a43) - double(d.a23) * double(d.a41)) + double(d.a13) * (double(d.a22) * double(d.a41) - double(d.a21) * double(d.a42))) / det;
			inv.a14 = (double(d.a12) * (double(d.a24) * double(d.a33) - double(d.a23) * double(d.a34)) + double(d.a13) * (double(d.a22) * double(d.a34) - double(d.a24) * double(d.a32)) + double(d.a14) * (double(d.a23) * double(d.a32) - double(d.a22) * double(d.a33))) / det;
			inv.a24 = (double(d.a11) * (double(d.a23) * double(d.a34) - double(d.a24) * double(d.a33)) + double(d.a13) * (double(d.a24) * double(d.a31) - double(d.a21) * double(d.a34)) + double(d.a14) * (double(d.a21) * double(d.a33) - double(d.a23) * double(d.a31))) / det;
			inv.a34 = (double(d.a11) * (double(d.a24) * double(d.a32) - double(d.a22) * double(d.a34)) + double(d.a12) * (double(d.a21) * double(d.a34) - double(d.a24) * double(d.a31)) + double(d.a14) * (double(d.a22) * double(d.a31) - double(d.a21) * double(d.a32))) / det;
			inv.a44 = (double(d.a11) * (double(d.a22) * double(d.a33) - double(d.a23) * double(d.a32)) + double(d.a12) * (double(d.a23) * double(d.a31) - double(d.a21) * double(d.a33)) + double(d.a13) * (double(d.a21) * double(d.a32) - double(d.a22) * double(d.a31))) / det;
		}
		else Tmatrix4fReset(inv);
		return(inv);
	}

	//------------------------------------------------------------------------------
	/// Devuelve la matriz inversa de una matriz de 4x4.
	/// Returns the inverse matrix of a 4x4 matrix.
	//------------------------------------------------------------------------------
	__device__ tmatrix4d InverseMatrix4x4(const tmatrix4d& d, const double det) {
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
		else Tmatrix4dReset(inv);
		return(inv);
	}

	//==============================================================================
	/// Devuelve producto de 2 matrices de 3x3.
	/// Returns the product of 2 matrices of 3x3.
	//==============================================================================
	__device__ tmatrix3f MulMatrix3x3(const tmatrix3f& a, const tmatrix3f& b) {
		tmatrix3f ret;
		ret.a11 = a.a11 * b.a11 + a.a12 * b.a21 + a.a13 * b.a31;
		ret.a12 = a.a11 * b.a12 + a.a12 * b.a22 + a.a13 * b.a32;
		ret.a13 = a.a11 * b.a13 + a.a12 * b.a23 + a.a13 * b.a33;
		ret.a21 = a.a21 * b.a11 + a.a22 * b.a21 + a.a23 * b.a31;
		ret.a22 = a.a21 * b.a12 + a.a22 * b.a22 + a.a23 * b.a32;
		ret.a23 = a.a21 * b.a13 + a.a22 * b.a23 + a.a23 * b.a33;
		ret.a31 = a.a31 * b.a11 + a.a32 * b.a21 + a.a33 * b.a31;
		ret.a32 = a.a31 * b.a12 + a.a32 * b.a22 + a.a33 * b.a32;
		ret.a33 = a.a31 * b.a13 + a.a32 * b.a23 + a.a33 * b.a33;
		return(ret);
	}

	//==============================================================================
	/// Devuelve producto de 2 matrices de 3x3.
	/// Returns the product of 2 matrices of 3x3.
	//==============================================================================
	__device__ tmatrix3d MulMatrix3x3(const tmatrix3d& a, const tmatrix3d& b) {
		tmatrix3d ret;
		ret.a11 = a.a11 * b.a11 + a.a12 * b.a21 + a.a13 * b.a31;
		ret.a12 = a.a11 * b.a12 + a.a12 * b.a22 + a.a13 * b.a32;
		ret.a13 = a.a11 * b.a13 + a.a12 * b.a23 + a.a13 * b.a33;
		ret.a21 = a.a21 * b.a11 + a.a22 * b.a21 + a.a23 * b.a31;
		ret.a22 = a.a21 * b.a12 + a.a22 * b.a22 + a.a23 * b.a32;
		ret.a23 = a.a21 * b.a13 + a.a22 * b.a23 + a.a23 * b.a33;
		ret.a31 = a.a31 * b.a11 + a.a32 * b.a21 + a.a33 * b.a31;
		ret.a32 = a.a31 * b.a12 + a.a32 * b.a22 + a.a33 * b.a32;
		ret.a33 = a.a31 * b.a13 + a.a32 * b.a23 + a.a33 * b.a33;
		return(ret);
	}

	//==============================================================================
	/// Devuelve traspuesta de matriz 3x3.
	/// Returns the transpose from matrix 3x3.
	//==============================================================================
	__device__ tmatrix3f TrasMatrix3x3(const tmatrix3f& a) {
		tmatrix3f ret;
		ret.a11 = a.a11;  ret.a12 = a.a21;  ret.a13 = a.a31;
		ret.a21 = a.a12;  ret.a22 = a.a22;  ret.a23 = a.a32;
		ret.a31 = a.a13;  ret.a32 = a.a23;  ret.a33 = a.a33;
		return(ret);
	}

	//==============================================================================
	/// Devuelve traspuesta de matriz 3x3.
	/// Returns the transpose from matrix 3x3.
	//==============================================================================
	__device__ tmatrix3d TrasMatrix3x3(const tmatrix3d& a) {
		tmatrix3d ret;
		ret.a11 = a.a11;  ret.a12 = a.a21;  ret.a13 = a.a31;
		ret.a21 = a.a12;  ret.a22 = a.a22;  ret.a23 = a.a32;
		ret.a31 = a.a13;  ret.a32 = a.a23;  ret.a33 = a.a33;
		return(ret);
	}

	//==============================================================================
	/// Devuelve la matriz de rotacion.
	/// Returns the rotation matrix.
	//==============================================================================
	__device__ tmatrix3f RotMatrix3x3(const float3& ang) {
		const float cosx = cos(ang.x), cosy = cos(ang.y), cosz = cos(ang.z);
		const float sinx = sin(ang.x), siny = sin(ang.y), sinz = sin(ang.z);
		tmatrix3f ret;
		ret.a11 = cosy * cosz;
		ret.a12 = -cosy * sinz;
		ret.a13 = siny;
		ret.a21 = sinx * siny * cosz + cosx * sinz;
		ret.a22 = -sinx * siny * sinz + cosx * cosz;
		ret.a23 = -sinx * cosy;
		ret.a31 = -cosx * siny * cosz + sinx * sinz;
		ret.a32 = cosx * siny * sinz + sinx * cosz;
		ret.a33 = cosx * cosy;
		return(ret);
	}

	//==============================================================================
	/// Devuelve la matriz de rotacion.
	/// Returns the rotation matrix.
	//==============================================================================
	__device__ tmatrix3d RotMatrix3x3(const double3& ang) {
		const double cosx = cos(ang.x), cosy = cos(ang.y), cosz = cos(ang.z);
		const double sinx = sin(ang.x), siny = sin(ang.y), sinz = sin(ang.z);
		tmatrix3d ret;
		ret.a11 = cosy * cosz;
		ret.a12 = -cosy * sinz;
		ret.a13 = siny;
		ret.a21 = sinx * siny * cosz + cosx * sinz;
		ret.a22 = -sinx * siny * sinz + cosx * cosz;
		ret.a23 = -sinx * cosy;
		ret.a31 = -cosx * siny * cosz + sinx * sinz;
		ret.a32 = cosx * siny * sinz + sinx * cosz;
		ret.a33 = cosx * cosy;
		return(ret);
	}


	__device__ __forceinline__ matrix3f DyadicVec3(float3 a, float3 b) {
		matrix3f m;
		m.a11 = a.x * b.x; m.a12 = a.x * b.y; m.a13 = a.x * b.z;
		m.a21 = a.y * b.x; m.a22 = a.y * b.y; m.a23 = a.y * b.z;
		m.a31 = a.z * b.x; m.a32 = a.z * b.y; m.a33 = a.z * b.z;
		return m;
	}

	__device__ __forceinline__ matrix3f DyadicVec43(float4 a, float3 b) {
		matrix3f m;
		m.a11 = a.x * b.x; m.a12 = a.x * b.y; m.a13 = a.x * b.z;
		m.a21 = a.y * b.x; m.a22 = a.y * b.y; m.a23 = a.y * b.z;
		m.a31 = a.z * b.x; m.a32 = a.z * b.y; m.a33 = a.z * b.z;
		return m;
	}

	__device__ __forceinline__ matrix3f DyadicVec34(float3 a, float4 b) {
		matrix3f m;
		m.a11 = a.x * b.x; m.a12 = a.x * b.y; m.a13 = a.x * b.z;
		m.a21 = a.y * b.x; m.a22 = a.y * b.y; m.a23 = a.y * b.z;
		m.a31 = a.z * b.x; m.a32 = a.z * b.y; m.a33 = a.z * b.z;
		return m;
	}

	__device__ __forceinline__ matrix3f Dyadic3Vec44(float4 a, float4 b) {
		matrix3f m;
		m.a11 = a.x * b.x; m.a12 = a.x * b.y; m.a13 = a.x * b.z;
		m.a21 = a.y * b.x; m.a22 = a.y * b.y; m.a23 = a.y * b.z;
		m.a31 = a.z * b.x; m.a32 = a.z * b.y; m.a33 = a.z * b.z;
		return m;
	}

	__device__ __forceinline__ matrix3f TrasMatrix3x3(const matrix3f a) {
		matrix3f ret;
		ret.a11 = a.a11;  ret.a12 = a.a21;  ret.a13 = a.a31;
		ret.a21 = a.a12;  ret.a22 = a.a22;  ret.a23 = a.a32;
		ret.a31 = a.a13;  ret.a32 = a.a23;  ret.a33 = a.a33;
		return(ret);
	}

	__device__ __forceinline__ float Determinant2x2(const matrix3f d) {
		return(d.a11 * d.a33 - d.a13 * d.a31);
	}

	__device__ __forceinline__ matrix3f InverseMatrix2x2(const matrix3f d, const float det) {
		matrix3f inv;
		if (det) {
			inv.a11 = d.a33 / det;
			inv.a12 = 0;
			inv.a13 = -d.a13 / det;
			inv.a21 = 0;
			inv.a22 = 0;
			inv.a23 = 0;
			inv.a31 = -d.a31 / det;
			inv.a32 = 0;
			inv.a33 = d.a11 / det;
		}
		else inv = Matrix3f(0);
		return(inv);
	}

	__device__ __forceinline__ matrix3f InverseMatrix2x2(const matrix3f d) {
		return(InverseMatrix2x2(d, Determinant2x2(d)));
	}

	__device__ __forceinline__ float Determinant3x3(const matrix3f d) {
		return(d.a11 * d.a22 * d.a33 + d.a12 * d.a23 * d.a31 + d.a13 * d.a21 * d.a32 - d.a31 * d.a22 * d.a13 - d.a32 * d.a23 * d.a11 - d.a33 * d.a21 * d.a12);
	}

	__device__ __forceinline__ matrix3f InverseMatrix3x3(const matrix3f d, const float det) {
		matrix3f inv;
		if (fabs(det)>ALMOSTZERO) {
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
		else inv = Matrix3f(0);
		return(inv);
	}

	__device__ __forceinline__ matrix3f InverseMatrix3x3(const matrix3f d) {
		return(InverseMatrix3x3(d, Determinant3x3(d)));
	}
	__device__ __forceinline__ void SymmetrizeMatrix3x3(matrix3f& A)
	{
		const float a12 = 0.5f * (A.a12 + A.a21);
		const float a13 = 0.5f * (A.a13 + A.a31);
		const float a23 = 0.5f * (A.a23 + A.a32);
		A.a12 = a12; A.a21 = a12;
		A.a13 = a13; A.a31 = a13;
		A.a23 = a23; A.a32 = a23;
	}
	__device__ inline matrix3f SafeInvDefgrad(const matrix3f& F,
		bool isPlastic,
		float& J_out)
	{
		const float a = F.a11, b = F.a12, c = F.a13;
		const float d = F.a21, e = F.a22, f = F.a23;
		const float g = F.a31, h = F.a32, i = F.a33;

		const float C11 = (e * i - f * h);
		const float C12 = -(d * i - f * g);
		const float C13 = (d * h - e * g);

		const float C21 = -(b * i - c * h);
		const float C22 = (a * i - c * g);
		const float C23 = -(a * h - b * g);

		const float C31 = (b * f - c * e);
		const float C32 = -(a * f - c * d);
		const float C33 = (a * e - b * d);

		float J = a * C11 + b * C12 + c * C13;

		float J_use = J;

		if (fabsf(J_use) < ALMOSTZERO) {
			J_use = (J_use >= 0.0f ? ALMOSTZERO : -ALMOSTZERO);
		}

		if (isPlastic) {
			const float Jmin_plastic = 0.8f;
			const float Jmax_plastic = 1.2f;

			if (J_use > 0.0f) {
				if (J_use < Jmin_plastic) J_use = Jmin_plastic;
				if (J_use > Jmax_plastic) J_use = Jmax_plastic;
			}
			else {
				J_use = 1.0;
			}
		}

		const float invJ = 1.0f / J_use;

		matrix3f Finv;
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

	__device__ __forceinline__ float DotVec3(float3 a, float3 b) {
		return a.x * b.x + a.y * b.y + a.z * b.z;
	}

	__device__ __forceinline__ float DotVec3(float4 a, float3 b) {
		return a.x * b.x + a.y * b.y + a.z * b.z;
	}

	__device__ __forceinline__ float DotVec3(float3 a, float4 b) {
		return a.x * b.x + a.y * b.y + a.z * b.z;
	}

	__device__ __forceinline__ float Dot3Vec44(float4 a, float4 b) {
		return a.x * b.x + a.y * b.y + a.z * b.z;
	}

	__device__ __forceinline__ float3 DotMatVec3(matrix3f mat, float3 vec) {
		float3 res;
		res.x = mat.a11 * vec.x + mat.a12 * vec.y + mat.a13 * vec.z;
		res.y = mat.a21 * vec.x + mat.a22 * vec.y + mat.a23 * vec.z;
		res.z = mat.a31 * vec.x + mat.a32 * vec.y + mat.a33 * vec.z;
		return res;
	}

	__device__ __forceinline__ float3 DotMat3Vec4(matrix3f mat, float4 vec) {
		float3 res;
		res.x = mat.a11 * vec.x + mat.a12 * vec.y + mat.a13 * vec.z;
		res.y = mat.a21 * vec.x + mat.a22 * vec.y + mat.a23 * vec.z;
		res.z = mat.a31 * vec.x + mat.a32 * vec.y + mat.a33 * vec.z;
		return res;
	}

	__device__ __forceinline__ float3 CrossVec3(const float3 a, const float3 b) {
		return { a.y * b.z - a.z * b.y,
				a.z * b.x - a.x * b.z,
				a.x * b.y - a.y * b.x };
	}

	__device__ __forceinline__ float NormVec3(const float3 a) {
		return sqrtf(DotVec3(a, a));
	}

	__device__ __forceinline__ float3 NormalizeVec3(const float3 a) {
		float n = NormVec3(a);
		return (n < ALMOSTZERO) ? a : float3{ a.x / n, a.y / n, a.z / n };
	}

	__device__ __forceinline__ matrix3f MulMatrix3x3(const matrix3f a, const matrix3f b) {
		matrix3f ret;
		ret.a11 = a.a11 * b.a11 + a.a12 * b.a21 + a.a13 * b.a31;
		ret.a12 = a.a11 * b.a12 + a.a12 * b.a22 + a.a13 * b.a32;
		ret.a13 = a.a11 * b.a13 + a.a12 * b.a23 + a.a13 * b.a33;
		ret.a21 = a.a21 * b.a11 + a.a22 * b.a21 + a.a23 * b.a31;
		ret.a22 = a.a21 * b.a12 + a.a22 * b.a22 + a.a23 * b.a32;
		ret.a23 = a.a21 * b.a13 + a.a22 * b.a23 + a.a23 * b.a33;
		ret.a31 = a.a31 * b.a11 + a.a32 * b.a21 + a.a33 * b.a31;
		ret.a32 = a.a31 * b.a12 + a.a32 * b.a22 + a.a33 * b.a32;
		ret.a33 = a.a31 * b.a13 + a.a32 * b.a23 + a.a33 * b.a33;
		return(ret);
	}

	__device__ float Trace3x3(const matrix3f a) { return a.a11 + a.a22 + a.a33; }

	__device__ __forceinline__ void swap(float& a, float& b) 
	{
		float temp = a;
		a = b;
		b = temp;
	}

	__device__ void computeEigenvector(const matrix3f& Amat, float lambda, float& v1, float& v2, float& v3)
	{
		matrix3f M;
		M.a11 = Amat.a11 - lambda; M.a12 = Amat.a12; M.a13 = Amat.a13;
		M.a21 = Amat.a21; M.a22 = Amat.a22 - lambda; M.a23 = Amat.a23;
		M.a31 = Amat.a31; M.a32 = Amat.a32; M.a33 = Amat.a33 - lambda;

		v1 = M.a22 * M.a33 - M.a23 * M.a32;
		v2 = M.a13 * M.a32 - M.a12 * M.a33;
		v3 = M.a12 * M.a23 - M.a13 * M.a22;

		float norm = sqrtf(v1 * v1 + v2 * v2 + v3 * v3);
		if (norm > ALMOSTZERO) {
			v1 /= norm;
			v2 /= norm;
			v3 /= norm;
		}
		else {
			v1 = v2 = v3 = 0.0f;
		}
	}

	template<bool simulate2d>
	__device__ matrix3f DSEigenDecompose(const matrix3f Amat)
	{
		float3 eigenvalues;
		matrix3f Qmat;
		if (simulate2d) {
			const float bb = Amat.a11 + Amat.a33;
			float delta_arg = bb * bb - 4.0f * (Amat.a11 * Amat.a33 - Amat.a13 * Amat.a13);
			if (delta_arg < 0.0f) delta_arg = 0.0f;
			const float delta = sqrtf(delta_arg);
			eigenvalues.x = (bb - delta) * 0.5f;
			eigenvalues.z = (bb + delta) * 0.5f;
			eigenvalues.y = 0.0f;

			float v1x = -Amat.a13;
			float v1z = Amat.a11 - eigenvalues.x;
			float norm1 = sqrtf(v1x * v1x + v1z * v1z);
			if (norm1 > ALMOSTZERO) {
				v1x /= norm1;
				v1z /= norm1;
			}
			else {
				v1x = 1.0f;
				v1z = 0.0f;
			}

			float v2x = -Amat.a13;
			float v2z = Amat.a11 - eigenvalues.z;
			float norm2 = sqrtf(v2x * v2x + v2z * v2z);
			if (norm2 > ALMOSTZERO) {
				v2x /= norm2;
				v2z /= norm2;
			}
			else {
				v2x = -v1z;
				v2z = v1x;
			}

			const float dot = v1x * v2x + v1z * v2z;
			v2x -= dot * v1x;
			v2z -= dot * v1z;
			float norm2b = sqrtf(v2x * v2x + v2z * v2z);
			if (norm2b > ALMOSTZERO) {
				v2x /= norm2b;
				v2z /= norm2b;
			}
			else {
				v2x = -v1z;
				v2z = v1x;
			}

			Qmat = { v1x, 0.0f, v2x,
					0.0f, 1.0f, 0.0f,
					v1z, 0.0f, v2z };
		}
		else {
			float polycof = Amat.a12 * Amat.a12 + Amat.a13 * Amat.a13 + Amat.a23 * Amat.a23;
			if (polycof < ALMOSTZERO) {
				eigenvalues.x = Amat.a11;
				eigenvalues.y = Amat.a22;
				eigenvalues.z = Amat.a33;
				Qmat = Ident3f();
			}
			else {
				float q = (Amat.a11 + Amat.a22 + Amat.a33) / 3.0;
				float p2 = (Amat.a11 - q) * (Amat.a11 - q) +
					(Amat.a22 - q) * (Amat.a22 - q) +
					(Amat.a33 - q) * (Amat.a33 - q) +
					2 * polycof;
				float p = sqrtf(p2 / 6.0f);

				matrix3f B = { (Amat.a11 - q) / p, Amat.a12 / p, Amat.a13 / p,
					Amat.a21 / p, (Amat.a22 - q) / p, Amat.a23 / p,
					Amat.a31 / p, Amat.a32 / p, (Amat.a33 - q) / p };

				float detB = B.a11 * (B.a22 * B.a33 - B.a23 * B.a32)
					- B.a12 * (B.a21 * B.a33 - B.a23 * B.a31)
					+ B.a13 * (B.a21 * B.a32 - B.a22 * B.a31);

				float r = detB * 0.5f;
				float phi = (r <= -1) ? PI / 3.0f : (r >= 1) ? 0.0f : acosf(r) / 3.0f;

				float lambda1 = q + 2.f * p * cosf(phi);
				float lambda3 = q + 2.f * p * cosf(phi + (2.0f * PI / 3.0f));
				float lambda2 = 3.f * q - lambda1 - lambda3;

				if (lambda1 < lambda2) cumath::swap(lambda1, lambda2);
				if (lambda1 < lambda3) cumath::swap(lambda1, lambda3);
				if (lambda2 < lambda3) cumath::swap(lambda2, lambda3);

				eigenvalues.x = lambda1;
				eigenvalues.y = lambda2;
				eigenvalues.z = lambda3;

				computeEigenvector(Amat, lambda1, Qmat.a11, Qmat.a21, Qmat.a31);
				computeEigenvector(Amat, lambda2, Qmat.a12, Qmat.a22, Qmat.a32);
				computeEigenvector(Amat, lambda3, Qmat.a13, Qmat.a23, Qmat.a33);
			}
		}

		matrix3f Lp = Zero3f();
		Lp.a11 = eigenvalues.x > ALMOSTZERO ? eigenvalues.x : 0.0f;
		Lp.a22 = eigenvalues.y > ALMOSTZERO ? eigenvalues.y : 0.0f;
		Lp.a33 = eigenvalues.z > ALMOSTZERO ? eigenvalues.z : 0.0f;

		return cumath::MulMatrix3x3(Qmat, cumath::MulMatrix3x3(Lp, cumath::TrasMatrix3x3(Qmat)));
	}












	// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------
	static __device__ __forceinline__ matrix3f ZeroMatrix3x3()
	{
		matrix3f A;
		A.a11 = A.a12 = A.a13 = 0.0f;
		A.a21 = A.a22 = A.a23 = 0.0f;
		A.a31 = A.a32 = A.a33 = 0.0f;
		return A;
	}

	static __device__ __forceinline__ matrix3f ScaleMatrix3x3(const matrix3f& A, const float s)
	{
		matrix3f B;
		B.a11 = s * A.a11; B.a12 = s * A.a12; B.a13 = s * A.a13;
		B.a21 = s * A.a21; B.a22 = s * A.a22; B.a23 = s * A.a23;
		B.a31 = s * A.a31; B.a32 = s * A.a32; B.a33 = s * A.a33;
		return B;
	}

	static __device__ __forceinline__ matrix3f AddMatrix3x3(const matrix3f& A, const matrix3f& B)
	{
		matrix3f C;
		C.a11 = A.a11 + B.a11; C.a12 = A.a12 + B.a12; C.a13 = A.a13 + B.a13;
		C.a21 = A.a21 + B.a21; C.a22 = A.a22 + B.a22; C.a23 = A.a23 + B.a23;
		C.a31 = A.a31 + B.a31; C.a32 = A.a32 + B.a32; C.a33 = A.a33 + B.a33;
		return C;
	}

	static __device__ __forceinline__ float TraceMatrix3x3(const matrix3f& A)
	{
		return A.a11 + A.a22 + A.a33;
	}

	static __device__ __forceinline__ matrix3f DeviatoricMatrix3x3(const matrix3f& A)
	{
		const float mean = TraceMatrix3x3(A) / 3.0f;
		matrix3f D = A;
		D.a11 -= mean;
		D.a22 -= mean;
		D.a33 -= mean;
		return D;
	}

	static __device__ __forceinline__ float SymmetricDoubleDotMatrix3x3(const matrix3f& A, const matrix3f& B)
	{
		return
			A.a11 * B.a11 +
			A.a22 * B.a22 +
			A.a33 * B.a33 +
			2.0f * (A.a12 * B.a12 + A.a13 * B.a13 + A.a23 * B.a23);
	}

	static __device__ __forceinline__ float MatrixInfNorm3x3(const matrix3f& A)
	{
		const float r1 = fabsf(A.a11) + fabsf(A.a12) + fabsf(A.a13);
		const float r2 = fabsf(A.a21) + fabsf(A.a22) + fabsf(A.a23);
		const float r3 = fabsf(A.a31) + fabsf(A.a32) + fabsf(A.a33);
		return fmaxf(r1, fmaxf(r2, r3));
	}

	static __device__ __forceinline__ matrix3f MakePlaneStrainF3(const matrix3f& Fin)
	{
		matrix3f F = Fin;
		F.a12 = 0.0f; F.a21 = 0.0f;
		F.a22 = 1.0f; 
		F.a23 = 0.0f; F.a32 = 0.0f;
		return F;
	}

	// Scaling-and-squaring Taylor exponential.
	// Good enough here because A = dgamma * Nflow is usually small.
	static __device__ __forceinline__ matrix3f ExpmMatrix3x3(const matrix3f& A)
	{
		matrix3f B = A;
		int squarings = 0;

		float nrm = MatrixInfNorm3x3(B);
		while (nrm > 0.5f && squarings < 12) {
			B = ScaleMatrix3x3(B, 0.5f);
			nrm *= 0.5f;
			++squarings;
		}

		matrix3f E = Ident3f();
		matrix3f term = Ident3f();

#pragma unroll
		for (int k = 1; k <= 18; ++k) {
			term = cumath::MulMatrix3x3(term, B);
			term = ScaleMatrix3x3(term, 1.0f / float(k));
			E = AddMatrix3x3(E, term);
		}

#pragma unroll
		for (int i = 0; i < 12; ++i) {
			if (i >= squarings) break;
			E = cumath::MulMatrix3x3(E, E);
		}

		return E;
	}

	static __device__ __forceinline__ void ComputeMandelDevAndEqStress(
		const matrix3f& C,
		const float J,
		const matrix3f& Fp,
		const float mu,
		matrix3f& M_dev,
		float& sigma_eq)
	{
		matrix3f Fp_inv = cumath::InverseMatrix3x3(Fp);
		matrix3f Fp_invT = cumath::TrasMatrix3x3(Fp_inv);

		matrix3f tmpCe = cumath::MulMatrix3x3(C, Fp_inv);
		matrix3f Ce = cumath::MulMatrix3x3(Fp_invT, tmpCe);
		cumath::SymmetrizeMatrix3x3(Ce);

		float Jp = fmaxf(cumath::Determinant3x3(Fp), 1e-12f);
		float Je = fmaxf(J / Jp, 1e-12f);
		float Je_pow = powf(Je, -2.0f / 3.0f);

		matrix3f Ce_bar = ScaleMatrix3x3(Ce, Je_pow);
		matrix3f devCe_bar = DeviatoricMatrix3x3(Ce_bar);

		M_dev = ScaleMatrix3x3(devCe_bar, mu);
		cumath::SymmetrizeMatrix3x3(M_dev);

		const float M2 = SymmetricDoubleDotMatrix3x3(M_dev, M_dev);
		sigma_eq = sqrtf(fmaxf(1.5f * M2, 0.0f));
	}

	static __device__ __forceinline__ float J2ResidualAtDgamma(
		const float dgamma,
		const matrix3f& Nflow_trial,
		const matrix3f& Fp_n,
		const matrix3f& C,
		const float J,
		const float mu,
		const float sigma_y0,
		const float H,
		const float alpha_n,
		matrix3f* Fp_out)
	{
		// Exponential-map plastic update
		matrix3f A = ExpmMatrix3x3(ScaleMatrix3x3(Nflow_trial, dgamma));
		matrix3f Fp = cumath::MulMatrix3x3(A, Fp_n);

		// Tiny numerical correction to keep det(Fp)=1
		float detFp = fmaxf(cumath::Determinant3x3(Fp), 1e-12f);
		float siso = powf(detFp, -1.0f / 3.0f);
		Fp = ScaleMatrix3x3(Fp, siso);
		if (!isfinite(detFp) || detFp <= 0.0f) {
			if (Fp_out) *Fp_out = Fp_n;
			return 1e30f;
		}
		matrix3f M_dev;
		float sigma_eq = 0.0f;
		ComputeMandelDevAndEqStress(C, J, Fp, mu, M_dev, sigma_eq);

		if (Fp_out) *Fp_out = Fp;

		const float alpha = alpha_n + dgamma;
		const float sigma_y = fmaxf(sigma_y0 + H * alpha, 0.0f);
		return sigma_eq - sigma_y;
	}
	static __device__ __forceinline__ bool SolveDgammaBracketed(
		const matrix3f& Nflow_trial,
		const matrix3f& Fp_n,
		const matrix3f& C,
		const float J,
		const float mu,
		const float sigma_y0,
		const float H,
		const float alpha_n,
		const float sigma_eq_trial,
		const float sigma_y_trial,
		float* dgamma_sol,
		matrix3f* Fp_sol)
	{
		const float tiny = 1e-12f;

		// Left endpoint: elastic predictor
		float xL = 0.0f;
		float fL = sigma_eq_trial - sigma_y_trial;   // should be > 0 here

		if (!(fL > 0.0f)) {
			// No plastic correction needed
			if (dgamma_sol) *dgamma_sol = 0.0f;
			if (Fp_sol) *Fp_sol = Fp_n;
			return true;
		}

		// Initial right guess: only a heuristic
		float xR = (sigma_eq_trial - sigma_y_trial) / fmaxf(3.0f * mu + H, tiny);
		xR = fmaxf(xR, 1e-10f);

		matrix3f FpR = Fp_n;
		float fR = J2ResidualAtDgamma(
			xR, Nflow_trial, Fp_n, C, J, mu, sigma_y0, H, alpha_n, &FpR);

		// Expand until the root is bracketed: want fL > 0 and fR <= 0
		bool bracketed = (fR <= 0.0f);
#pragma unroll
		for (int k = 0; k < 20 && !bracketed; ++k) {
			xR *= 2.0f;
			fR = J2ResidualAtDgamma(
				xR, Nflow_trial, Fp_n, C, J, mu, sigma_y0, H, alpha_n, &FpR);
			bracketed = (fR <= 0.0f);
		}

		if (!bracketed) {
			// Do NOT accept a fake plastic step.
			if (dgamma_sol) *dgamma_sol = 0.0f;
			if (Fp_sol) *Fp_sol = Fp_n;
			return false;
		}

		// Safeguarded bisection
		matrix3f FpM = Fp_n;
		float xM = 0.0f;
		float fM = 0.0f;

		for (int it = 0; it < 40; ++it) {
			xM = 0.5f * (xL + xR);
			fM = J2ResidualAtDgamma(
				xM, Nflow_trial, Fp_n, C, J, mu, sigma_y0, H, alpha_n, &FpM);

			const float xtol = 1e-8f * (1.0f + xR);
			const float ftol = 1e-6f * fmaxf(sigma_y0 + H * (alpha_n + xM), 1.0f);

			if (fabsf(fM) <= ftol || fabsf(xR - xL) <= xtol) {
				if (dgamma_sol) *dgamma_sol = xM;
				if (Fp_sol) *Fp_sol = FpM;
				return true;
			}

			if (fM > 0.0f) {
				xL = xM;
				fL = fM;
			}
			else {
				xR = xM;
				fR = fM;
				FpR = FpM;
			}
		}

		// Final verification before accepting
		xM = 0.5f * (xL + xR);
		fM = J2ResidualAtDgamma(
			xM, Nflow_trial, Fp_n, C, J, mu, sigma_y0, H, alpha_n, &FpM);

		const float ftol = 1e-6f * fmaxf(sigma_y0 + H * (alpha_n + xM), 1.0f);
		if (fabsf(fM) <= ftol) {
			if (dgamma_sol) *dgamma_sol = xM;
			if (Fp_sol) *Fp_sol = FpM;
			return true;
		}

		if (dgamma_sol) *dgamma_sol = 0.0f;
		if (Fp_sol) *Fp_sol = Fp_n;
		return false;
	}
	static __device__ __forceinline__ float SolveDgammaIllinois(
		const matrix3f& Nflow_trial,
		const matrix3f& Fp_n,
		const matrix3f& C,
		const float J,
		const float mu,
		const float sigma_y0,
		const float H,
		const float alpha_n,
		const float sigma_eq_trial,
		const float sigma_y_trial,
		matrix3f* Fp_sol)
	{
		float xL = 0.0f;
		float fL = sigma_eq_trial - sigma_y_trial;

		float xR = (sigma_eq_trial - sigma_y_trial) / fmaxf(3.0f * mu + H, 1e-12f);
		xR = fmaxf(xR, 1e-8f);

		matrix3f FpR;
		float fR = J2ResidualAtDgamma(
			xR, Nflow_trial, Fp_n, C, J, mu, sigma_y0, H, alpha_n, &FpR);

		const float ftolR = 1e-5f * fmaxf(sigma_y0 + H * (alpha_n + xR), 1.0f);
		if (fabsf(fR) < ftolR) {
			if (Fp_sol) *Fp_sol = FpR;
			return xR;
		}
#pragma unroll
		for (int k = 0; k < 20 && fR > 0.0f; ++k) {
			xR *= 2.0f;
			fR = J2ResidualAtDgamma(
				xR, Nflow_trial, Fp_n, C, J, mu, sigma_y0, H, alpha_n, &FpR);

			const float ftol_expand = 1e-5f * fmaxf(sigma_y0 + H * (alpha_n + xR), 1.0f);
			if (fabsf(fR) < ftol_expand) {
				if (Fp_sol) *Fp_sol = FpR;
				return xR;
			}
		}

		// If we still do not have a bracket, accept best available point rather than pretending convergence
		if (fR > 0.0f) {
			if (Fp_sol) *Fp_sol = FpR;
			return xR;
		}

		matrix3f FpX = FpR;
#pragma unrol
		for (int it = 0; it < 20; ++it) {
			const float denom = (fR - fL);
			float x = (fabsf(denom) > 1e-20f)
				? (xR - fR * (xR - xL) / denom)
				: (0.5f * (xL + xR));

			if (!(x > xL && x < xR) || !isfinite(x)) {
				x = 0.5f * (xL + xR);
			}

			const float f = J2ResidualAtDgamma(
				x, Nflow_trial, Fp_n, C, J, mu, sigma_y0, H, alpha_n, &FpX);

			const float xtol = 1e-7f * (1.0f + xR);
			const float ftol = 1e-5f * fmaxf(sigma_y0 + H * (alpha_n + x), 1.0f);

			if (fabsf(f) < ftol || fabsf(xR - xL) < xtol) {
				if (Fp_sol) *Fp_sol = FpX;
				return x;
			}

			if (f > 0.0f) {
				xL = x;
				fL = f;
				fR *= 0.5f;
			}
			else {
				xR = x;
				fR = f;
				fL *= 0.5f;
				FpR = FpX;
			}
		}

		if (Fp_sol) *Fp_sol = FpR;
		return xR;
	}
}


