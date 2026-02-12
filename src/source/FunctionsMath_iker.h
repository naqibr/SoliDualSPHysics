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

}


