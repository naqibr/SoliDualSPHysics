#pragma once
#include "DualSphDef.h"
#include <cuda_runtime.h>
typedef struct {
    float a11, a12, a13, a21, a22, a23, a31, a32, a33;
} matrix3f;

__device__ __forceinline__ bool operator==(float3 a, float3 b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

__device__ __forceinline__ bool operator!=(float3 a, float3 b) {
    return a.x != b.x || a.y != b.y || a.z != b.z;
}

__device__ __forceinline__ bool operator<(float3 a, float3 b) {
    return a.x < b.x && a.y < b.y && a.z < b.z;
}

__device__ __forceinline__ bool operator>(float3 a, float3 b) {
    return a.x > b.x && a.y > b.y && a.z > b.z;
}

__device__ __forceinline__ bool operator<=(float3 a, float3 b) {
    return a.x <= b.x && a.y <= b.y && a.z <= b.z;
}

__device__ __forceinline__ bool operator>=(float3 a, float3 b) {
    return a.x >= b.x && a.y >= b.y && a.z >= b.z;
}

// Arithmetic operators (float3 with float3)
__device__ __forceinline__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 operator*(float3 a, float3 b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ __forceinline__ float3 operator/(float3 a, float3 b) {
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__device__ __forceinline__ float3 operator+(float3 a, float b) {
    return make_float3(a.x + b, a.y + b, a.z + b);
}

__device__ __forceinline__ float3 operator-(float3 a, float b) {
    return make_float3(a.x - b, a.y - b, a.z - b);
}

__device__ __forceinline__ float3 operator*(float3 a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ __forceinline__ float3 operator/(float3 a, float b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}

__device__ __forceinline__ float3 operator+(float b, float3 a) {
    return make_float3(a.x + b, a.y + b, a.z + b);
}

__device__ __forceinline__ float3 operator-(float b, float3 a) {
    return make_float3(b - a.x, b - a.y, b - a.z);
}

__device__ __forceinline__ float3 operator*(float b, float3 a) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ __forceinline__ float3 operator/(float b, float3 a) {
    return make_float3(b / a.x, b / a.y, b / a.z);
}

__device__ __forceinline__ float3& operator+=(float3& a, float3 b) {
    a.x += b.x; a.y += b.y; a.z += b.z;
    return a;
}

__device__ __forceinline__ float3& operator-=(float3& a, float3 b) {
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
    return a;
}

__device__ __forceinline__ float3& operator*=(float3& a, float3 b) {
    a.x *= b.x; a.y *= b.y; a.z *= b.z;
    return a;
}

__device__ __forceinline__ float3& operator/=(float3& a, float3 b) {
    a.x /= b.x; a.y /= b.y; a.z /= b.z;
    return a;
}

__device__ __forceinline__ float3& operator+=(float3& a, float b) {
    a.x += b; a.y += b; a.z += b;
    return a;
}

__device__ __forceinline__ float3& operator-=(float3& a, float b) {
    a.x -= b; a.y -= b; a.z -= b;
    return a;
}

__device__ __forceinline__ float3& operator*=(float3& a, float b) {
    a.x *= b; a.y *= b; a.z *= b;
    return a;
}

__device__ __forceinline__ float3& operator/=(float3& a, float b) {
    a.x /= b; a.y /= b; a.z /= b;
    return a;
}

__host__ __device__ __forceinline__ matrix3f Matrix3f(float v) { matrix3f m = { v,v,v,v,v,v,v,v,v }; return(m); }
__device__ __forceinline__ matrix3f make_matrix3f(float _a11, float _a12, float _a13, 
    float _a21, float _a22, float _a23, float _a31, float _a32, float _a33)
{
    matrix3f a; 
    a.a11 = _a11; a.a12 = _a12; a.a13 = _a13;
    a.a21 = _a21; a.a22 = _a22; a.a23 = _a23;
    a.a31 = _a31; a.a32 = _a32; a.a33 = _a33;
    return a;
}

__device__ __forceinline__ tmatrix3f make_tmatrix3f(float _a11, float _a12, float _a13,
    float _a21, float _a22, float _a23, float _a31, float _a32, float _a33)
{
    tmatrix3f a;
    a.a11 = _a11; a.a12 = _a12; a.a13 = _a13;
    a.a21 = _a21; a.a22 = _a22; a.a23 = _a23;
    a.a31 = _a31; a.a32 = _a32; a.a33 = _a33;
    return a;
}

__device__ __forceinline__ tmatrix3f make_tmatrix3f(float v)
{
    tmatrix3f a;
    a.a11 = a.a12 = a.a13 = a.a21 = a.a22 = a.a23 = a.a31 = a.a32 = a.a33 = v;
    return a;
}

__device__ __forceinline__ matrix3f Ident3f() {
    return make_matrix3f(1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f);
}

__device__ __forceinline__ matrix3f Zero3f() {
    return make_matrix3f(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
}

__device__ __forceinline__ matrix3f& operator+=(matrix3f& a, matrix3f b) {
    a.a11 += b.a11; a.a12 += b.a12; a.a13 += b.a13;
    a.a21 += b.a21; a.a22 += b.a22; a.a23 += b.a23;
    a.a31 += b.a31; a.a32 += b.a32; a.a33 += b.a33;
    return a;
}

__device__ __forceinline__ matrix3f& operator-=(matrix3f& a, matrix3f b) {
    a.a11 -= b.a11; a.a12 -= b.a12; a.a13 -= b.a13;
    a.a21 -= b.a21; a.a22 -= b.a22; a.a23 -= b.a23;
    a.a31 -= b.a31; a.a32 -= b.a32; a.a33 -= b.a33;
    return a;
}

__device__ __forceinline__ matrix3f operator+(matrix3f a, matrix3f b) {
    return { a.a11 + b.a11, a.a12 + b.a12, a.a13 + b.a13,
             a.a21 + b.a21, a.a22 + b.a22, a.a23 + b.a23,
             a.a31 + b.a31, a.a32 + b.a32, a.a33 + b.a33 };
}

__device__ __forceinline__ matrix3f operator-(matrix3f a, matrix3f b) {
    return { a.a11 - b.a11, a.a12 - b.a12, a.a13 - b.a13,
             a.a21 - b.a21, a.a22 - b.a22, a.a23 - b.a23,
             a.a31 - b.a31, a.a32 - b.a32, a.a33 - b.a33 };
}

__device__ __forceinline__ matrix3f operator*(matrix3f a, float b) {
    matrix3f res;
    res.a11 = a.a11 * b; res.a12 = a.a12 * b; res.a13 = a.a13 * b;
    res.a21 = a.a21 * b; res.a22 = a.a22 * b; res.a23 = a.a23 * b;
    res.a31 = a.a31 * b; res.a32 = a.a32 * b; res.a33 = a.a33 * b;
    return res;
}

__device__ __forceinline__ matrix3f operator*(float b, matrix3f a) {
    matrix3f res;
    res.a11 = a.a11 * b; res.a12 = a.a12 * b; res.a13 = a.a13 * b;
    res.a21 = a.a21 * b; res.a22 = a.a22 * b; res.a23 = a.a23 * b;
    res.a31 = a.a31 * b; res.a32 = a.a32 * b; res.a33 = a.a33 * b;
    return res;
}


__device__ __forceinline__ float4 operator+(float4 a, float4 b) {
    return make_float4(a.x+b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__device__ __forceinline__ float4 operator-(float4 a, float4 b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

enum StGPUOpType {
    OP_CONST, OP_X, OP_Y, OP_Z, OP_T, OP_LOCAL,
    OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_POW,
    OP_SIN, OP_COS, OP_TAN, OP_SQRT, OP_LOG, OP_LN,
    OP_CMP_LT, OP_CMP_GT, OP_CMP_LE, OP_CMP_GE, OP_CMP_EQ, OP_CMP_NE,
    OP_TERNARY
};

struct StGPUOp {
    StGPUOpType type;
    union {
        double const_val;
        int local_idx;
    };
};

struct StGPUExpression {
    StGPUOp* ops;       // Device pointer to operations
    int op_count;
    double* locals;     // Device pointer to local variables
    int local_count;
};
class JSphDeformStruc;
///Structure with the arrays of the deformable structure for time integration.
typedef struct StDeformStrucIntArraysg {
    const JSphDeformStruc* DeformStruc;		///Pointer to deformable structure class
	const typecode* DSCodeg;				///<Indicator of group of particles & other special markers. [MapNdeformstruc].
	const unsigned* DSPairJg;				///<List of j (neighbour) index to each initial neighbourlist of all the mapped domain [DSNumPairs].
	const uint2* DSPairNSg;					///<x:Number of pairs and y: address of neighbours of a particle in the kernelderlap array [MapNdeformstruc].
	const float* DSKerg;					///<List of kernel values between solid structure particles [DSNumPairs].
	const tbcstruc* DSPartVBCg;				///<Particle array with informations of velocity boundary condition [MapNdeformstruc].
	const tbcstruc* DSPartFBCg;				///<Particle Force boundary condition w is empty for now[MapNdeformstruc].
	const tphibc* DSPartPhiBCg;				///<Particle Phi boundary condition (compact 8-byte: exprid + flags) [MapNdeformstruc].
	const float4* DSKerDerLapg;				///<List of kernel x,y,z: derivative and w:laplacian values between solid structure particles [DSNumPairs].
	const float4* DSPos0g;					///<Initial particle positions and w:strain energy in mapped domain [MapNdeformstruc].
    //===================

    float4* DSDispPhig;						///<Particle x,y,z: displacement and w:phase field in mapped domain [MapNdeformstruc].
    float* DSEqPlasticg;					///<Equivalent plastic strain per particle [MapNdeformstruc].
    tmatrix3f* DSPlasticStraing;			///<Deviatoric plastic strain tensor per particle [MapNdeformstruc].
    float4* DSDefGradg2D;					///<Particle 2D (a11,a13,a31,a33) Deformation Gradient in mapped domain [MapNdeformstruc].
    float4* DSPiolKirg2D;					///<Particle 2D (a11,a13,a31,a33) Piolla-Kirchhof Stress in mapped domain [MapNdeformstruc].
    float4* DSDefGradg3D;					///<Particle (a12,a21,a23,a32) Deformation Gradient in mapped domain [MapNdeformstruc].
    float4* DSPiolKirg3D;					///<Particle (a12,a21,a23,a32) Piolla-Kirchhof Stress in mapped domain [MapNdeformstruc].
    float4* DSPhiTdatag;					///<Particle x:Phidot, y:Phiddot, z:History, w:PhidotPre values in mapped domain [MapNdeformstruc].
    float4* DSAcclg;						///<Particle acceleration and w: optimal time step in mapped domain [MapNdeformstruc].
    float4* DSVelg;							///<Particle velocity in mapped domain [MapNdeformstruc].
    float4* DSVelPreg;						///<Particle velocity and w:Phidot in mapped domain [MapNdeformstruc].
    float2* DSDefPk;						///< x: a22 Deformation gradient, y: a22 Pk
    float2* DSDispCorxzg;					///< x,z Displacement correction for crack surface
    float* DSDispCoryg;						///< y Displacement correction for crack surface
    float* DSblockMinMax;					///Array to store any min or max value of deformable structure domain

	StDeformStrucIntArraysg(typecode* _DSCodeg, unsigned* _DSPairJg, uint2* _DSPairNSg,
		float* _DSKerg, tbcstruc* _DSPartVBCg, float4* _DSKerDerLapg,
		float4* _DSPos0g, tbcstruc* _DSPartFBCg, tphibc* _DSPartPhiBCg, float4* _DSDispPhig, float* _DSEqPlasticg, tmatrix3f* _DSPlasticStraing, float4* _DSDefGradg2D,
		float4* _DSPiolKirg2D, float4* _DSDefGradg3D, float4* _DSPiolKirg3D, float4* _DSPhiTdatag,
		float4* _DSAcclg, float4* _DSVelg, float4* _DSVelPreg, float2* _DSDefPk, float2* _DSDispCorxzg,
		float* _DSDispCoryg, float* _DSblockMinMax, JSphDeformStruc* _DeformStruc) :
		DSCodeg(_DSCodeg), DSPairJg(_DSPairJg), DSPairNSg(_DSPairNSg),
		DSKerg(_DSKerg), DSPartVBCg(_DSPartVBCg), DSKerDerLapg(_DSKerDerLapg),
		DSPos0g(_DSPos0g), DSPartFBCg(_DSPartFBCg), DSPartPhiBCg(_DSPartPhiBCg), DSDispPhig(_DSDispPhig), DSDefGradg2D(_DSDefGradg2D),
		DSPiolKirg2D(_DSPiolKirg2D), DSDefGradg3D(_DSDefGradg3D), DSPiolKirg3D(_DSPiolKirg3D), DSPhiTdatag(_DSPhiTdatag),
		DSAcclg(_DSAcclg), DSVelg(_DSVelg), DSVelPreg(_DSVelPreg), DSDefPk(_DSDefPk), DSblockMinMax(_DSblockMinMax),
		DSDispCorxzg(_DSDispCorxzg), DSDispCoryg(_DSDispCoryg), DeformStruc(_DeformStruc), DSEqPlasticg(_DSEqPlasticg), DSPlasticStraing(_DSPlasticStraing)
	{
	}

    StDeformStrucIntArraysg() :
        DSCodeg(nullptr), DSPairJg(nullptr), DSPairNSg(nullptr),
        DSKerg(nullptr), DSPartVBCg(nullptr), DSKerDerLapg(nullptr),
        DSPos0g(nullptr), DSPartFBCg(nullptr), DSPartPhiBCg(nullptr), DSDispPhig(nullptr), DSDefGradg2D(nullptr),
        DSPiolKirg2D(nullptr), DSDefGradg3D(nullptr), DSPiolKirg3D(nullptr), DSPhiTdatag(nullptr),
        DSAcclg(nullptr), DSVelg(nullptr), DSVelPreg(nullptr), DSDefPk(nullptr), 
        DSDispCorxzg(nullptr), DSDispCoryg(nullptr), DSblockMinMax(nullptr), 
        DeformStruc(nullptr), DSEqPlasticg(nullptr), DSPlasticStraing(nullptr)
    {
    }

} StDeformStrucIntArraysg;