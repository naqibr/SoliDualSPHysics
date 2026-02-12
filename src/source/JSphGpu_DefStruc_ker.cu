
#include "JSphGpu_DefStruc_ker.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "Functions.h"
#include <thrust/device_vector.h>
#include <thrust/logical.h>
#include <thrust/count.h> 
#include <thrust/gather.h>
#include <thrust/sort.h>
//#include <thrust/scan.h>

#pragma warning(disable : 4267) //Cancels "warning C4267: conversion from 'size_t' to 'int', possible loss of data"
#pragma warning(disable : 4244) //Cancels "warning C4244: conversion from 'unsigned __int64' to 'unsigned int', possible loss of data"
#pragma warning(disable : 4503) //Cancels "warning C4503: decorated name length exceeded, name was truncated"

class JSphDeformStruc;

namespace cudefstr {
#include "JCellSearch_iker.h"
#include "FunSphKernel_iker.h"
#include "FunSphEos_iker.h"
#include "FunctionsMath_iker.h"

	__device__ void atomicMinDouble(double* address, double val) {
		unsigned long long* addr_as_ull = (unsigned long long*)address;
		unsigned long long old = *addr_as_ull, assumed;
		do {
			assumed = old;
			double old_val = __longlong_as_double(assumed);
			if (val >= old_val) break;
			old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(val));
		} while (assumed != old);
	}

	__device__ void atomicMaxDouble(double* address, double val) {
		unsigned long long* addr_as_ull = (unsigned long long*)address;
		unsigned long long old = *addr_as_ull, assumed;
		do {
			assumed = old;
			double old_val = __longlong_as_double(assumed);
			if (val <= old_val) break;
			old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(val));
		} while (assumed != old);
	}

	template <unsigned blockSize>
	__device__ void KerReduMinFloatWarp(float* sdat, unsigned tid) {
		if (blockSize >= 64) { if (tid < 32) sdat[tid] = fmin(sdat[tid], sdat[tid + 32]); __syncthreads(); }
		if (blockSize >= 32) { if (tid < 16) sdat[tid] = fmin(sdat[tid], sdat[tid + 16]); __syncthreads(); }
		if (blockSize >= 16) { if (tid < 8)  sdat[tid] = fmin(sdat[tid], sdat[tid + 8]);  __syncthreads(); }
		if (blockSize >= 8) { if (tid < 4)  sdat[tid] = fmin(sdat[tid], sdat[tid + 4]);  __syncthreads(); }
		if (blockSize >= 4) { if (tid < 2)  sdat[tid] = fmin(sdat[tid], sdat[tid + 2]);  __syncthreads(); }
		if (blockSize >= 2) { if (tid < 1)  sdat[tid] = fmin(sdat[tid], sdat[tid + 1]);  __syncthreads(); }
	}
	//==============================================================================
	/// Accumulates the minimum of n values of array dat[], storing the result in 
	/// the beginning of res[]. (Many positions of res[] are used as blocks, 
	/// storing the final result in res[0]).
	//==============================================================================
	template <unsigned blockSize> __global__ void KerReduMinFloat(unsigned n, unsigned ini, const float* dat, float* res) {
		extern __shared__ float sdat[];
		unsigned tid = threadIdx.x;
		unsigned c = blockIdx.x * blockDim.x + threadIdx.x;
		sdat[tid] = (c < n ? dat[c + ini] : FLT_MAX);
		__syncthreads();

		if (blockSize >= 512) { if (tid < 256) sdat[tid] = fmin(sdat[tid], sdat[tid + 256]); __syncthreads(); }
		if (blockSize >= 256) { if (tid < 128) sdat[tid] = fmin(sdat[tid], sdat[tid + 128]); __syncthreads(); }
		if (blockSize >= 128) { if (tid < 64)  sdat[tid] = fmin(sdat[tid], sdat[tid + 64]);  __syncthreads(); }

		if (tid < 32) {
			KerReduMinFloatWarp<blockSize>(sdat, tid);
		}
		if (tid == 0) res[blockIdx.x] = sdat[0];
	}

	//==============================================================================
	/// Returns the minimum of an array, using resu[] as auxiliary array.
	/// Size of resu[] must be >= (N/SPHBSIZE+1)+(N/(SPHBSIZE*SPHBSIZE)+SPHBSIZE)
	//==============================================================================
	float ReduMinFloat(unsigned ndata, unsigned inidata, float* data, float* resu) {
		float resf = 0;
		if (ndata >= 1) {
			unsigned n = ndata, ini = inidata;
			unsigned smemSize = SPHBSIZE * sizeof(float);
			dim3 sgrid = GetSimpleGridSize(n, SPHBSIZE);
			unsigned n_blocks = sgrid.x * sgrid.y;
			float* dat = data;
			float* resu1 = resu, * resu2 = resu + n_blocks;
			float* res = resu1;
			while (n > 1) {
				KerReduMinFloat<SPHBSIZE> << <sgrid, SPHBSIZE, smemSize >> > (n, ini, dat, res);
				n = n_blocks; ini = 0;
				sgrid = GetSimpleGridSize(n, SPHBSIZE);
				n_blocks = sgrid.x * sgrid.y;
				if (n > 1) {
					dat = res; res = (dat == resu1 ? resu2 : resu1);
				}
			}
			if (ndata > 1) cudaMemcpy(&resf, res, sizeof(float), cudaMemcpyDeviceToHost);
			else cudaMemcpy(&resf, data, sizeof(float), cudaMemcpyDeviceToHost);
		}
		// Optional Thrust fallback (commented out as in original)
		// else {
		//   thrust::device_ptr<float> dev_ptr(data);
		//   resf = thrust::reduce(dev_ptr, dev_ptr+ndata, FLT_MAX, thrust::minimum<float>());
		// }
		return resf;
	}


	//==============================================================================
	/// Accumulates the sum of n values of array dat[], storing the result in 
	/// the beginning of res[].(Many positions of res[] are used as blocks, 
	/// storing the final result in res[0]).
	//==============================================================================
	template <unsigned blockSize> __global__ void KerReduMinFloat_w(unsigned n, unsigned ini, const float4* dat, float* res) {
		extern __shared__ float sdat[];
		unsigned tid = threadIdx.x;
		unsigned c = blockIdx.x * blockDim.x + threadIdx.x;
		sdat[tid] = (c < n ? dat[c + ini].w : FLT_MAX);
		__syncthreads();
		if (blockSize >= 512) { if (tid < 256)sdat[tid] = min(sdat[tid], sdat[tid + 256]);  __syncthreads(); }
		if (blockSize >= 256) { if (tid < 128)sdat[tid] = min(sdat[tid], sdat[tid + 128]);  __syncthreads(); }
		if (blockSize >= 128) { if (tid < 64) sdat[tid] = min(sdat[tid], sdat[tid + 64]);   __syncthreads(); }
		if (tid < 32)KerReduMinFloatWarp<blockSize>(sdat, tid);
		if (tid == 0)res[blockIdx.x] = sdat[0];
	}

	//==============================================================================
	/// Returns the maximum of an array, using resu[] as auxiliar array.
	/// Size of resu[] must be >= (N/SPHBSIZE+1)+(N/(SPHBSIZE*SPHBSIZE)+SPHBSIZE).
	//==============================================================================
	float ReduMinFloat_w(unsigned ndata, unsigned inidata, float4* data, float* resu) {
		unsigned n = ndata, ini = inidata;
		unsigned smemSize = SPHBSIZE * sizeof(float);
		dim3 sgrid = GetSimpleGridSize(n, SPHBSIZE);
		unsigned n_blocks = sgrid.x * sgrid.y;
		float* dat = NULL;
		float* resu1 = resu, * resu2 = resu + n_blocks;
		float* res = resu1;
		while (n > 1) {
			if (!dat)KerReduMinFloat_w<SPHBSIZE> << <sgrid, SPHBSIZE, smemSize >> > (n, ini, data, res);
			else KerReduMinFloat<SPHBSIZE> << <sgrid, SPHBSIZE, smemSize >> > (n, ini, dat, res);
			n = n_blocks; ini = 0;
			sgrid = GetSimpleGridSize(n, SPHBSIZE);
			n_blocks = sgrid.x * sgrid.y;
			if (n > 1) {
				dat = res; res = (dat == resu1 ? resu2 : resu1);
			}
		}
		float resf;
		if (ndata > 1)cudaMemcpy(&resf, res, sizeof(float), cudaMemcpyDeviceToHost);
		else {
			float4 resf4;
			cudaMemcpy(&resf4, data, sizeof(float4), cudaMemcpyDeviceToHost);
			resf = resf4.w;
		}
		return(resf);
	}

	//==============================================================================
	/// Functor for checking if a deformable structure has a normal defined.
	/// Functor para verificar si una estructura deformable tiene una normal definida.
	//==============================================================================
	struct DeformStrucHasNormal {
		DeformStrucHasNormal(const typecode* code, const float3* boundnormals) :code(code), boundnormals(boundnormals) {};
		__device__ bool operator()(unsigned idx) {
			const float3 bnormal = boundnormals[idx];
			return CODE_IsDeformStrucDeform(code[idx]) && (bnormal.x != 0 || bnormal.y != 0 || bnormal.z != 0);
		}
	private:
		const typecode* code;
		const float3* boundnormals;
	};

	//==============================================================================
	/// Functor for checking if particle is a deformable structure particle.
	//==============================================================================
	struct IsDeformStrucAny { __device__ bool operator()(const typecode& code) { return CODE_IsDeformStrucAny(code); } };

	struct is_non_zero {
		__host__ __device__
			bool operator()(const unsigned& x) const {
			return x != 0;
		}
	};
	//==============================================================================
	/// Checks if any normals are defined for the deformable structure particles.
	//==============================================================================
	bool DSHasNormalsg(unsigned npb, const typecode* code, const float3* boundnormals) {
		if (npb) {
			thrust::counting_iterator<unsigned> idx(0);
			DeformStrucHasNormal pred = DeformStrucHasNormal(code, boundnormals);
			return thrust::any_of(idx, idx + npb, pred);
		}
		return false;
	}

	//==============================================================================
	/// Counts the number of deformable structure particles
	//==============================================================================
	unsigned DSCountOrgParticlesg(unsigned npb, const typecode* code) {
		if (npb) {
			thrust::device_ptr<const typecode> dev_code(code);
			return thrust::count_if(dev_code, dev_code + npb, IsDeformStrucAny());
		}
		return 0;
	}
	void DSCalcRidpg(unsigned npb, unsigned* deformstrucridp, const typecode* code) {
		if (npb) {
			thrust::counting_iterator<unsigned> idx(0);
			thrust::device_ptr<const typecode> dev_code(code);
			thrust::device_ptr<unsigned> dev_deformstrucridp(deformstrucridp);
			thrust::copy_if(idx, idx + npb, dev_code, dev_deformstrucridp, IsDeformStrucAny());
		}
	}


	__global__ void computeIssurfKernel(const int MapNdeformstruc, const float4* __restrict__ DSPos0g,
		const uint2* __restrict__ DSPairNSg, const unsigned* __restrict__ DSPairJg,
		const float* __restrict__ DSKerg, unsigned* issurf_d)
	{
		int p1 = blockIdx.x * blockDim.x + threadIdx.x;
		if (p1 >= MapNdeformstruc) return;

		const uint2 pairns = DSPairNSg[p1];
		float4 pos0p1 = DSPos0g[p1];
		float3 norm = make_float3(0.0f, 0.0f, 0.0f);

		for (unsigned i = 0; i < pairns.x; ++i) {
			unsigned jadd = pairns.y + i;
			unsigned p2 = DSPairJg[jadd];
			float3 dx = make_float3(
				DSPos0g[p2].x - pos0p1.x,
				DSPos0g[p2].y - pos0p1.y,
				DSPos0g[p2].z - pos0p1.z
			);
			float len = sqrtf(cumath::DotVec3(dx, dx));
			if (len > ALMOSTZERO) { // Avoid division by zero
				float k = DSKerg[jadd];
				norm.x += k * dx.x / len;
				norm.y += k * dx.y / len;
				norm.z += k * dx.z / len;
			}
		}

		float NormMag = sqrtf(cumath::DotVec3(norm, norm));
		issurf_d[p1] = (NormMag > 0.25f) ? 1 : 0;
	}

	__global__ void DSCountMappedParticlesKernel(unsigned casendeformstruc, bool simulate2D, const unsigned* deformstrucridpg,
		const typecode* codeg, const StDeformStrucData* deformstrucdatag, int* total)
	{
		unsigned thread_sum = 0;
		const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
		const unsigned total_threads = blockDim.x * gridDim.x;

		for (unsigned p = tid; p < casendeformstruc; p += total_threads) {
			const unsigned p1 = deformstrucridpg[p];
			const unsigned bodyid = CODE_GetIbodyDeformStruc(codeg[p1]);
			const unsigned mapfac = deformstrucdatag[bodyid].mapfact;

			thread_sum += simulate2D ? (mapfac * mapfac) : (mapfac * mapfac * mapfac);
		}

		__shared__ unsigned shared_sum[SPHBSIZE / 32];
		const unsigned lane = threadIdx.x % 32;
		const unsigned wid = threadIdx.x / 32;

		for (int offset = 16; offset > 0; offset >>= 1)
			thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);

		if (lane == 0) shared_sum[wid] = thread_sum;
		__syncthreads();

		if (wid == 0) {
			thread_sum = (lane < (blockDim.x / 32)) ? shared_sum[lane] : 0;

			for (int offset = 16; offset > 0; offset >>= 1)
				thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);

			if (threadIdx.x == 0) atomicAdd(total, thread_sum);
		}
	}

	int DSCountMappedParticlesg(unsigned casendeformstruc, bool simulate2D, unsigned* deformstrucridpg,
		typecode* codeg, StDeformStrucData* deformstrucdatag) {
		int h_total = 0;
		int* d_total;

		cudaMalloc(&d_total, sizeof(int));
		cudaMemset(d_total, 0, sizeof(int));

		dim3 sgridf = GetSimpleGridSize(casendeformstruc, SPHBSIZE);
		DSCountMappedParticlesKernel << <sgridf, SPHBSIZE >> > (casendeformstruc, simulate2D, deformstrucridpg, codeg, deformstrucdatag, d_total);
		cudaDeviceSynchronize();
		cudaMemcpy(&h_total, d_total, sizeof(int), cudaMemcpyDeviceToHost);
		cudaFree(d_total);

		return h_total;
	}


	//========================================================================================
	/// Generates new particle list for the mapped domain of deformable structure.
	/// Makes sure that the generated particles are in order aligned with the 
	/// id of deformable structure body.
	/// This is the domain for structural solution
	//========================================================================================
	__global__ void DSGenMappedParticlesKernel(int cnd, const unsigned* __restrict__ deformstrucridpg,
		const typecode* __restrict__ codeg, StDeformStrucData* __restrict__ deformstrucdatag, float4* dspos0,
		float2* dsposorg0xyg, float* dsposorg0zg, unsigned* dsparentg, typecode* dscodeg,
		bool simulate2D, unsigned* total)
	{
		int p = blockIdx.x * blockDim.x + threadIdx.x;
		if (p >= cnd) return;

		unsigned p1 = deformstrucridpg[p];
		float3 DSposp1;
		DSposp1.x = dsposorg0xyg[p].x;
		DSposp1.y = dsposorg0xyg[p].y;
		DSposp1.z = dsposorg0zg[p];
		typecode DScodep1 = codeg[p1];
		unsigned bodyid = CODE_GetIbodyDeformStruc(DScodep1);
		StDeformStrucData& body = deformstrucdatag[bodyid];
		unsigned mapfac = body.mapfact;
		double partDist = body.dp;

		double xshift = DSposp1.x - 0.5 * mapfac * partDist + 0.5 * partDist;
		double yshift = DSposp1.y - 0.5 * mapfac * partDist + 0.5 * partDist;
		double zshift = DSposp1.z - 0.5 * mapfac * partDist + 0.5 * partDist;

		if (simulate2D) {
			for (unsigned k = 0; k < mapfac; ++k) {
				for (unsigned m = 0; m < mapfac; ++m) {
					tdouble3 posp;
					posp.y = DSposp1.y;
					posp.x = xshift + partDist * m;
					posp.z = zshift + partDist * k;

					unsigned index = atomicAdd(total, 1);

					dspos0[index].x = posp.x;
					dspos0[index].y = posp.y;
					dspos0[index].z = posp.z;

					dsparentg[index] = p1;
					dscodeg[index] = DScodep1;

					atomicAdd(&body.npbody, 1);

					atomicMinDouble(&body.min.x, posp.x);
					atomicMinDouble(&body.min.z, posp.z);
					atomicMinDouble(&body.min.y, posp.y);

					atomicMaxDouble(&body.max.x, posp.x);
					atomicMaxDouble(&body.max.z, posp.z);
					atomicMaxDouble(&body.max.y, posp.y);
				}
			}
		}
		else {
			for (unsigned k = 0; k < mapfac; ++k) {
				for (unsigned l = 0; l < mapfac; ++l) {
					for (unsigned m = 0; m < mapfac; ++m) {
						tdouble3 posp;
						posp.x = xshift + partDist * m;
						posp.y = yshift + partDist * l;
						posp.z = zshift + partDist * k;

						unsigned index = atomicAdd(total, 1);

						dspos0[index].x = posp.x;
						dspos0[index].y = posp.y;
						dspos0[index].z = posp.z;

						dsparentg[index] = p1;
						dscodeg[index] = DScodep1;

						atomicAdd(&body.npbody, 1);

						atomicMinDouble(&body.min.x, posp.x);
						atomicMinDouble(&body.min.z, posp.z);
						atomicMinDouble(&body.min.y, posp.y);

						atomicMaxDouble(&body.max.x, posp.x);
						atomicMaxDouble(&body.max.z, posp.z);
						atomicMaxDouble(&body.max.y, posp.y);
					}
				}
			}
		}
	}

	void DSGenMappedParticles(const unsigned casendeformstruc, const unsigned* deformstrucridpg,
		const typecode* codeg, StDeformStrucData* deformstrucdatag, float4* dspos0, float2* dsposorg0xyg, float* dsposorg0zg,
		unsigned* dsparentg, typecode* dscodeg, bool simulate2D)
	{
		int cnd = static_cast<int>(casendeformstruc);
		if (cnd <= 0) return;

		unsigned* d_total;
		cudaMalloc(&d_total, sizeof(unsigned));
		cudaMemset(d_total, 0, sizeof(unsigned));

		dim3 sgridf = GetSimpleGridSize(casendeformstruc, SPHBSIZE);
		DSGenMappedParticlesKernel << <sgridf, SPHBSIZE >> > (cnd, deformstrucridpg, codeg, deformstrucdatag, dspos0, dsposorg0xyg, dsposorg0zg, dsparentg, dscodeg, simulate2D, d_total);
		cudaDeviceSynchronize();
		cudaFree(d_total);
	}


	__global__ void FillDSiBodyRidpKernel(const typecode* dscodeg, unsigned* dsibodyridpg,
		const StDeformStrucData* deformstrucdatag, unsigned* counters,
		const int mapndeformstruc, const unsigned deformstruccount)
	{
		unsigned p1 = blockIdx.x * blockDim.x + threadIdx.x;
		if (p1 >= mapndeformstruc) return;

		unsigned bodyid = CODE_GetIbodyDeformStruc(dscodeg[p1]);
		if (bodyid >= deformstruccount) return;
		unsigned npstart = deformstrucdatag[bodyid].npstart;
		unsigned pos = atomicAdd(&counters[bodyid], 1);
		dsibodyridpg[npstart + pos] = p1;
	}

	//void DSCalcIbodyRidp(const unsigned deformstruccount, const StDeformStrucData* deformstrucdatag,
	//	const int mapndeformstruc, const typecode* dscodeg, unsigned* dsibodyridpg)
	//{
	//	unsigned* d_counters;
	//	cudaMalloc(&d_counters, deformstruccount * sizeof(unsigned));
	//	cudaMemset(d_counters, 0, deformstruccount * sizeof(unsigned));

	//	dim3 sgridf = GetSimpleGridSize(mapndeformstruc, SPHBSIZE);
	//	FillDSiBodyRidpKernel << <sgridf, SPHBSIZE >> > (dscodeg, dsibodyridpg, deformstrucdatag,
	//		d_counters, mapndeformstruc, deformstruccount);
	//	cudaDeviceSynchronize();
	//	cudaFree(d_counters);
	//}

	__global__ void DSDetermineMapCentersKernel(const unsigned casendeformstruc,
		const unsigned* __restrict__ DeformStrucRidpg, const float2* __restrict__ DSPosOrg0xyg,
		const float* __restrict__ DSPosOrg0zg, const typecode* __restrict__ Codeg,
		const StDeformStrucData* __restrict__ DeformStrucDatag, const StDivDataGpu DSDivData,
		const typecode* __restrict__ DSCodeg, const float4* __restrict__ DSPos0g,
		const unsigned* __restrict__ DSDcellg, unsigned* DSBestChildg)
	{
		const int pds = blockIdx.x * blockDim.x + threadIdx.x;
		if (pds >= casendeformstruc) return;

		const unsigned npid = DeformStrucRidpg[pds];
		float3 posds;
		posds.x = DSPosOrg0xyg[pds].x;
		posds.y = DSPosOrg0xyg[pds].y;
		posds.z = DSPosOrg0zg[pds];

		const unsigned bodyid = CODE_GetIbodyDeformStruc(Codeg[npid]);
		const double dp05 = 0.501 * DeformStrucDatag[bodyid].dp;
		const double dp01 = 0.1 * dp05;

		int ini1, fin1, ini2, fin2, ini3, fin3;
		cunsearch::Initsp(posds.x, posds.y, posds.z, DSDivData.axis, DSDivData.domposmin,
			DSDivData.scell, DSDivData.scelldiv, DSDivData.nc, DSDivData.cellzero,
			ini1, fin1, ini2, fin2, ini3, fin3);

		for (int c3 = ini3; c3 < fin3; c3 += DSDivData.nc.w)for (int c2 = ini2; c2 < fin2; c2 += DSDivData.nc.x) {
			unsigned pini, pfin = 0;
			cunsearch::ParticleRange(c2, c3, ini1, fin1, DSDivData.beginendcell, pini, pfin);
			if (pfin)for (int p2 = pini; p2 < pfin; p2++) {
				const unsigned bodypid = CODE_GetIbodyDeformStruc(DSCodeg[p2]);
				if (bodypid != bodyid) continue;

				const float4 posp1 = DSPos0g[p2];
				float3 dist;
				dist.x = posp1.x - posds.x;
				dist.y = posp1.y - posds.y;
				dist.z = posp1.z - posds.z;

				if (dist.x < -dp01 || dist.y < -dp01 || dist.z < -dp01) continue;

				if (dist.x < dp05 && dist.y < dp05 && dist.z < dp05) {
					DSBestChildg[pds] = static_cast<unsigned>(p2);
				}
			}
		}
	}

	void DSDetermineMapCenters(const unsigned casendeformstruc, const unsigned* DeformStrucRidpg,
		const float2* DSPosOrg0xyg, const float* DSPosOrg0zg, const typecode* Codeg,
		const StDeformStrucData* DeformStrucDatag, const StDivDataGpu DSDivData, const typecode* DSCodeg,
		const float4* DSPos0g, const unsigned* DSDcellg, unsigned* DSBestChildg)
	{
		if (casendeformstruc) {
			dim3 sgridf = GetSimpleGridSize(casendeformstruc, SPHBSIZE);
			DSDetermineMapCentersKernel << <sgridf, SPHBSIZE >> > (casendeformstruc,
				DeformStrucRidpg, DSPosOrg0xyg, DSPosOrg0zg, Codeg, DeformStrucDatag, DSDivData,
				DSCodeg, DSPos0g, DSDcellg, DSBestChildg);
			cudaDeviceSynchronize();
		}
	}

	__device__ bool DSLineSurfIntersect(const float4 part1, const float4 part2, const plane4Nstruc* rect4n, unsigned size4n)
	{
		double3 domainAmin, domainAmax, domainBmin, domainBmax;
		double3 AdjuncVec1, AdjuncVec2, AdjuncVec3;
		double3 LineVec1, LineVec2, LineVec3, xint, tVec1;
		double Adjunc1x, Adjunc2x, Adjunc3x, Adjunc1y, Adjunc2y, Adjunc3y;
		double tv, m11, m21, m31, d;

		m11 = part1.x - part2.x;
		m21 = part1.y - part2.y;
		m31 = part1.z - part2.z;

		domainAmin = { fmin(part1.x, part2.x), fmin(part1.y, part2.y), fmin(part1.z, part2.z) };
		domainAmax = { fmax(part1.x, part2.x), fmax(part1.y, part2.y), fmax(part1.z, part2.z) };

		for (unsigned surfi = 0; surfi < size4n; surfi++) {

			LineVec1.x = rect4n[surfi].corners[1].x - rect4n[surfi].corners[0].x;
			LineVec1.y = rect4n[surfi].corners[1].y - rect4n[surfi].corners[0].y;
			LineVec1.z = rect4n[surfi].corners[1].z - rect4n[surfi].corners[0].z;

			LineVec2.x = rect4n[surfi].corners[2].x - rect4n[surfi].corners[0].x;
			LineVec2.y = rect4n[surfi].corners[2].y - rect4n[surfi].corners[0].y;
			LineVec2.z = rect4n[surfi].corners[2].z - rect4n[surfi].corners[0].z;

			LineVec3.x = rect4n[surfi].corners[3].x - rect4n[surfi].corners[0].x;
			LineVec3.y = rect4n[surfi].corners[3].y - rect4n[surfi].corners[0].y;
			LineVec3.z = rect4n[surfi].corners[3].z - rect4n[surfi].corners[0].z;

			domainBmin = { 1.0e20, 1.0e20, 1.0e20 };
			domainBmax = { -1.0e20, -1.0e20, -1.0e20 };

			for (int i = 0; i < 4; ++i) {
				domainBmin.x = fmin(domainBmin.x, rect4n[surfi].corners[i].x);
				domainBmin.y = fmin(domainBmin.y, rect4n[surfi].corners[i].y);
				domainBmin.z = fmin(domainBmin.z, rect4n[surfi].corners[i].z);
				domainBmax.x = fmax(domainBmax.x, rect4n[surfi].corners[i].x);
				domainBmax.y = fmax(domainBmax.y, rect4n[surfi].corners[i].y);
				domainBmax.z = fmax(domainBmax.z, rect4n[surfi].corners[i].z);
			}

			if (domainAmin.x > domainBmax.x || domainAmin.y > domainBmax.y || domainAmin.z > domainBmax.z ||
				domainAmin.x > domainAmax.x || domainAmin.y > domainAmax.y || domainAmin.z > domainAmax.z) {
				continue;
			}

			Adjunc1x = LineVec1.y * LineVec2.z - LineVec2.y * LineVec1.z;
			Adjunc2x = LineVec1.z * LineVec2.x - LineVec2.z * LineVec1.x;
			Adjunc3x = LineVec1.x * LineVec2.y - LineVec2.x * LineVec1.y;
			Adjunc1y = LineVec2.y * LineVec3.z - LineVec3.y * LineVec2.z;
			Adjunc2y = LineVec2.z * LineVec3.x - LineVec3.z * LineVec2.x;
			Adjunc3y = LineVec2.x * LineVec3.y - LineVec3.x * LineVec2.y;

			AdjuncVec1.y = -m21 * LineVec2.z + LineVec2.y * m31;
			AdjuncVec1.z = m21 * LineVec1.z - LineVec1.y * m31;
			d = m11 * Adjunc1x + LineVec1.x * AdjuncVec1.y + LineVec2.x * AdjuncVec1.z;
			if (fabs(d) < DBL_MIN) {
				continue;
			}
			AdjuncVec2.y = -m31 * LineVec2.x + LineVec2.z * m11;
			AdjuncVec2.z = m31 * LineVec1.x - LineVec1.z * m11;
			AdjuncVec3.y = -m11 * LineVec2.y + LineVec2.x * m21;
			AdjuncVec3.z = m11 * LineVec1.y - LineVec1.x * m21;

			tv = AdjuncVec1.y;
			AdjuncVec1.y = Adjunc2x;
			AdjuncVec2.x = tv;

			tv = AdjuncVec1.z;
			AdjuncVec1.z = Adjunc3x;
			AdjuncVec3.x = tv;

			tv = AdjuncVec2.z;
			AdjuncVec2.z = AdjuncVec3.y;
			AdjuncVec3.y = tv;

			tVec1.x = part1.x - rect4n[surfi].corners[0].x;
			tVec1.y = part1.y - rect4n[surfi].corners[0].y;
			tVec1.z = part1.z - rect4n[surfi].corners[0].z;

			xint.x = (Adjunc1x * tVec1.x + AdjuncVec1.y * tVec1.y + AdjuncVec1.z * tVec1.z) / d;
			xint.y = (AdjuncVec2.x * tVec1.x + AdjuncVec2.y * tVec1.y + AdjuncVec2.z * tVec1.z) / d;
			xint.z = (AdjuncVec3.x * tVec1.x + AdjuncVec3.y * tVec1.y + AdjuncVec3.z * tVec1.z) / d;

			if (xint.x > -DBL_MIN && xint.x < 1 + DBL_MIN &&
				xint.y > -DBL_MIN && xint.y < 1 + DBL_MIN &&
				xint.z > -DBL_MIN && xint.z < 1 + DBL_MIN &&
				xint.y + xint.z < 1 + DBL_MIN) {
				return true;
			}

			AdjuncVec1.y = -m21 * LineVec3.z + LineVec3.y * m31;
			AdjuncVec1.z = m21 * LineVec2.z - LineVec2.y * m31;
			d = m11 * Adjunc1y + LineVec2.x * AdjuncVec1.y + LineVec3.x * AdjuncVec1.z;
			if (std::abs(d) < DBL_MIN) {
				continue;
			}
			AdjuncVec2.y = -m31 * LineVec3.x + LineVec3.z * m11;
			AdjuncVec2.z = m31 * LineVec2.x - LineVec2.z * m11;
			AdjuncVec3.y = -m11 * LineVec3.y + LineVec3.x * m21;
			AdjuncVec3.z = m11 * LineVec2.y - LineVec2.x * m21;

			tv = AdjuncVec1.y;
			AdjuncVec1.y = Adjunc2y;
			AdjuncVec2.x = tv;

			tv = AdjuncVec1.z;
			AdjuncVec1.z = Adjunc3y;
			AdjuncVec3.x = tv;

			tv = AdjuncVec2.z;
			AdjuncVec2.z = AdjuncVec3.y;
			AdjuncVec3.y = tv;

			xint.x = (Adjunc1y * tVec1.x + AdjuncVec1.y * tVec1.y + AdjuncVec1.z * tVec1.z) / d;
			xint.y = (AdjuncVec2.x * tVec1.x + AdjuncVec2.y * tVec1.y + AdjuncVec2.z * tVec1.z) / d;
			xint.z = (AdjuncVec3.x * tVec1.x + AdjuncVec3.y * tVec1.y + AdjuncVec3.z * tVec1.z) / d;

			if (xint.x > -DBL_MIN && xint.x < 1 + DBL_MIN &&
				xint.y > -DBL_MIN && xint.y < 1 + DBL_MIN &&
				xint.z > -DBL_MIN && xint.z < 1 + DBL_MIN &&
				xint.y + xint.z < 1 + DBL_MIN) {
				return true;
			}
		}
		return false;
	}

	__global__ void DSCountTotalPairsKernel(const int MapNdeformstruc, const float4* __restrict__ DSPos0,
		const unsigned* __restrict__ DSParent, const typecode* __restrict__ DSCodeg,
		const StDeformStrucData* __restrict__ DeformStrucData, unsigned* pairn,
		const StDivDataGpu DSDivData, const bool simulate2d)
	{
		int pid = blockIdx.x * blockDim.x + threadIdx.x;
		if (pid >= MapNdeformstruc) return;

		float4 posp1 = DSPos0[pid];
		//unsigned parentp1 = DSParent[pid];
		typecode code = DSCodeg[pid];
		unsigned bodyid = CODE_GetIbodyDeformStruc(code);
		const StDeformStrucData& body = DeformStrucData[bodyid];

		float ksize2 = body.kernelsize2;
		const float factorrr = simulate2d ? 1.42 : 1.74;
		const float distfact = factorrr * body.nbsrange;

		const float dp201 = distfact * body.dp * distfact * body.dp;
		unsigned notchnm = body.nnotch;
		const plane4Nstruc* notchlist = body.notchlist;

		unsigned pairCount = 0;

		int ini1, fin1, ini2, fin2, ini3, fin3;
		cunsearch::Initsp(posp1.x, posp1.y, posp1.z, DSDivData.axis, DSDivData.domposmin,
			DSDivData.scell, DSDivData.scelldiv, DSDivData.nc, DSDivData.cellzero,
			ini1, fin1, ini2, fin2, ini3, fin3);

		for (int c3 = ini3; c3 < fin3; c3 += DSDivData.nc.w) {
			for (int c2 = ini2; c2 < fin2; c2 += DSDivData.nc.x) {
				unsigned pini, pfin = 0;
				cunsearch::ParticleRange(c2, c3, ini1, fin1, DSDivData.beginendcell, pini, pfin);
				if (pfin)for (int p2 = pini; p2 < pfin; p2++) {

					if (p2 == pid) continue;
					if (CODE_GetIbodyDeformStruc(DSCodeg[p2]) != bodyid) continue;
					float4 posp2 = DSPos0[p2];

					const float4 dx = posp1 - posp2;
					const float drxw = dx.x * dx.x + dx.y * dx.y + dx.z * dx.z;
					if (drxw > ksize2) continue;
					if (drxw > dp201) continue;
					if (notchnm && DSLineSurfIntersect(posp1, posp2, notchlist, notchnm)) continue;

					pairCount++;
				}
			}
		}

		pairn[pid] = pairCount;
	}

	__global__ void DSComputePairNS(const int MapNdeformstruc, const unsigned* __restrict__ pairn,
		const unsigned* __restrict__ pairs, uint2* __restrict__ pairns)
	{
		int pid = blockIdx.x * blockDim.x + threadIdx.x;
		if (pid >= MapNdeformstruc) return;
		pairns[pid].x = pairn[pid];
		pairns[pid].y = pairs[pid];
	}

	template<TpKernel tker, bool simulate2d>
	__global__ void DSCalcKersCudaKernel(const int MapNdeformstruc,
		const float4* __restrict__ DSPos0g, const typecode* __restrict__ DSCodeg,
		const StDeformStrucData* __restrict__ DeformStrucDatag, const StDivDataGpu DSDivDatag,
		uint2* __restrict__ DSPairNSg, float* __restrict__ DSKerg, float4* __restrict__ DSKerDerLapg,
		unsigned* __restrict__ DSPairJg, float* __restrict__ DSKerSumVolg,
		const unsigned* __restrict__ DSDcellg)
	{
		int pid = blockIdx.x * blockDim.x + threadIdx.x;
		if (pid >= MapNdeformstruc) return;

		const float4 posp1 = DSPos0g[pid];
		const typecode code = DSCodeg[pid];
		const unsigned pairstarti = DSPairNSg[pid].y;

		const unsigned bodyid = CODE_GetIbodyDeformStruc(code);
		const StDeformStrucData& body = DeformStrucDatag[bodyid];
		const float ksize2 = body.kernelsize2;
		const float factorrr = simulate2d ? 1.42 : 1.74;
		const float distfact = factorrr * body.nbsrange;
		const float dp201 = distfact * body.dp * distfact * body.dp;
		const unsigned notchnm = body.nnotch;
		const plane4Nstruc* notchlist = body.notchlist;
		const float bkernh = body.kernelh;

		float kersum = 0.0f;
		unsigned pairCount = 0;

		int ini1, fin1, ini2, fin2, ini3, fin3;
		cunsearch::Initsp(posp1.x, posp1.y, posp1.z, DSDivDatag.axis, DSDivDatag.domposmin,
			DSDivDatag.scell, DSDivDatag.scelldiv, DSDivDatag.nc, DSDivDatag.cellzero,
			ini1, fin1, ini2, fin2, ini3, fin3);

		for (int c3 = ini3; c3 < fin3; c3 += DSDivDatag.nc.w) {
			for (int c2 = ini2; c2 < fin2; c2 += DSDivDatag.nc.x) {
				unsigned pini, pfin = 0;
				cunsearch::ParticleRange(c2, c3, ini1, fin1, DSDivDatag.beginendcell, pini, pfin);
				if (pfin)for (int p2 = pini; p2 < pfin; p2++) {

					if (p2 == pid) continue;
					if (CODE_GetIbodyDeformStruc(DSCodeg[p2]) != bodyid) continue;
					float4 posp2 = DSPos0g[p2];

					const float4 dx = posp1 - posp2;
					const float drxw = dx.x * dx.x + dx.y * dx.y + dx.z * dx.z;
					if (drxw > ksize2) continue;
					if (drxw > dp201) continue;
					if (notchnm && DSLineSurfIntersect(posp1, posp2, notchlist, notchnm)) continue;

					float fac0 = 0.0f;
					unsigned pairid = pairstarti + pairCount;

					float wabij = cufsph::GetTLSPHKernel_WabFac<tker, simulate2d>(bkernh, drxw, fac0);
					DSKerg[pairid] = wabij;
					DSKerDerLapg[pairid] = make_float4(fac0 * dx.x, fac0 * dx.y, fac0 * dx.z, 0.0f);
					DSPairJg[pairid] = p2;
					kersum += wabij;
					pairCount++;
				}
			}
		}
		kersum += cufsph::GetTLSPHKernel_Wab<tker, simulate2d>(bkernh, 0.0f);
		DSKerSumVolg[pid] = 1.0f / (kersum);

		if (pairCount > 0) {
			matrix3f kercorrp1 = { 0.0f };
			for (unsigned np = 0; np < pairCount; np++) {
				const unsigned pairid = pairstarti + np;
				const unsigned p2 = DSPairJg[pairid];
				float4 drx = DSPos0g[p2] - posp1;
				float4 kerderlap = DSKerDerLapg[pairid];
				kercorrp1 += cumath::Dyadic3Vec44(kerderlap, drx);
			}

			matrix3f kercorr = simulate2d ? cumath::InverseMatrix2x2(kercorrp1) : cumath::InverseMatrix3x3(kercorrp1);

			for (unsigned np = 0; np < pairCount; np++) {
				unsigned pairid = pairstarti + np;
				DSKerg[pairid] /= kersum;
				const unsigned p2 = DSPairJg[pairid];
				const float4 v = DSKerDerLapg[pairid];
				const float3 res = cumath::DotMat3Vec4(kercorr, v);

				float4 drx = DSPos0g[p2] - posp1;
				float rr20 = drx.x * drx.x + drx.y * drx.y + drx.z * drx.z;
				const float lapp1 = (drx.x * res.x + drx.y * res.y + drx.z * res.z) * 2.0f / rr20;
				DSKerDerLapg[pairid] = make_float4(res.x, res.y, res.z, lapp1);
			}
		}

		DSPairNSg[pid].x = pairCount;
	}

	unsigned DSCountTotalPairs(const int MapNdeformstruc, const float4* DSPos0, const unsigned* DSParent,
		const typecode* DSCodeg, const StDeformStrucData* DeformStrucData, uint2* DSPairNS,
		const StDivDataGpu DSDivData, const bool simulate2d)
	{
		unsigned* pairn, * pairs;
		cudaMalloc(&pairn, MapNdeformstruc * sizeof(unsigned));
		cudaMalloc(&pairs, MapNdeformstruc * sizeof(unsigned));

		dim3 sgridf = GetSimpleGridSize(MapNdeformstruc, SPHBSIZE);
		DSCountTotalPairsKernel << <sgridf, SPHBSIZE >> > (MapNdeformstruc, DSPos0, DSParent,
			DSCodeg, DeformStrucData, pairn, DSDivData, simulate2d);

		cudaDeviceSynchronize();
		thrust::exclusive_scan(thrust::device, pairn, pairn + MapNdeformstruc, pairs);
		cudaDeviceSynchronize();
		DSComputePairNS << <sgridf, SPHBSIZE >> > (MapNdeformstruc, pairn, pairs, DSPairNS);
		cudaDeviceSynchronize();
		cudaFree(pairn);
		cudaFree(pairs);

		unsigned totalPairs, lastPairStart, lastPairN;
		cudaMemcpy(&lastPairStart, &DSPairNS[MapNdeformstruc - 1].y, sizeof(unsigned), cudaMemcpyDeviceToHost);
		cudaMemcpy(&lastPairN, &DSPairNS[MapNdeformstruc - 1].x, sizeof(unsigned), cudaMemcpyDeviceToHost);
		totalPairs = lastPairStart + lastPairN;
		return totalPairs;
	}

	template <bool simulate2D>
	void DSCalcKers_ct(const int MapNdeformstruc, const TpKernel TKernel, const typecode* DSCodeg,
		const float4* DSPos0g, const StDeformStrucData* DeformStrucDatag, const StDivDataGpu DSDivData, const unsigned* DSDcellg,
		uint2* DSPairNSg, float* DSKerg, float4* DSKerDerLapg,
		unsigned* DSPairJg, float* DSKerSumVolg)
	{
		if (MapNdeformstruc) {
			dim3 sgridb = GetSimpleGridSize(MapNdeformstruc, SPHBSIZE);
			if (TKernel == KERNEL_Cubic) {
				DSCalcKersCudaKernel<KERNEL_Cubic, simulate2D> << <sgridb, SPHBSIZE >> > (MapNdeformstruc,
					DSPos0g, DSCodeg, DeformStrucDatag, DSDivData, DSPairNSg, DSKerg, DSKerDerLapg,
					DSPairJg, DSKerSumVolg, DSDcellg);
			}
			else {
				DSCalcKersCudaKernel<KERNEL_Wendland, simulate2D> << <sgridb, SPHBSIZE >> > (MapNdeformstruc,
					DSPos0g, DSCodeg, DeformStrucDatag, DSDivData, DSPairNSg, DSKerg, DSKerDerLapg,
					DSPairJg, DSKerSumVolg, DSDcellg);
			}
			cudaDeviceSynchronize();
		}
	}

	void DSCalcKers(const int MapNdeformstruc, const TpKernel TKernel, const bool Simulate2D, const typecode* DSCodeg,
		const float4* DSPos0g, const StDeformStrucData* DeformStrucDatag, const StDivDataGpu DSDivData, const unsigned* DSDcellg,
		uint2* DSPairNSg, float* DSKerg, float4* DSKerDerLapg,
		unsigned* DSPairJg, float* DSKerSumVolg)
	{

		if (Simulate2D)
			DSCalcKers_ct<true>(MapNdeformstruc, TKernel, DSCodeg, DSPos0g,
				DeformStrucDatag, DSDivData, DSDcellg, DSPairNSg, DSKerg,
				DSKerDerLapg, DSPairJg, DSKerSumVolg);
		else
			DSCalcKers_ct<false>(MapNdeformstruc, TKernel, DSCodeg, DSPos0g,
				DeformStrucDatag, DSDivData, DSDcellg, DSPairNSg, DSKerg,
				DSKerDerLapg, DSPairJg, DSKerSumVolg);

	}

	void DSFindSurfParticles(const int MapNdeformstruc, const float4* DSPos0g,
		const typecode* DSCodeg, const uint2* DSPairNSg, const unsigned* DSPairJg, const float* DSKerg,
		const StDeformStrucData* DeformStrucDatag, int& DSNpSurf, unsigned*& DSSurfPartListg, llong& MemGpuFixed)
	{
		dim3 sgridb = GetSimpleGridSize(MapNdeformstruc, SPHBSIZE);//

		unsigned* issurf_d;
		cudaMalloc(&issurf_d, MapNdeformstruc * sizeof(unsigned));

		computeIssurfKernel << <sgridb, SPHBSIZE >> > (MapNdeformstruc, DSPos0g,
			DSPairNSg, DSPairJg, DSKerg, issurf_d);
		cudaDeviceSynchronize();

		thrust::device_ptr<unsigned> d_issurf(issurf_d);
		unsigned dsnpsurf = thrust::reduce(d_issurf, d_issurf + MapNdeformstruc, 0, thrust::plus<unsigned>());

		DSNpSurf = dsnpsurf;

		cudaMalloc(&DSSurfPartListg, DSNpSurf * sizeof(unsigned)); MemGpuFixed += sizeof(unsigned) * DSNpSurf;

		thrust::device_ptr<unsigned> d_output(DSSurfPartListg);
		auto start = thrust::make_counting_iterator<unsigned>(0);
		auto end = thrust::make_counting_iterator<unsigned>(MapNdeformstruc);

		thrust::copy_if(start, end, d_issurf, d_output, is_non_zero());
		cudaDeviceSynchronize();
		cudaFree(issurf_d);
	}


	__global__ void DSInitFieldVarsKernel(const TpStep TStep, const int MapNdeformstruc, const typecode* __restrict__ DSCodeg,
		const StDeformStrucData* __restrict__ DeformStrucDatag, const tbcstruc* __restrict__ DSPartVBCg,
		tbcstruc* __restrict__ DSPartFBCg, const float* __restrict__ DSKerSumVolg, float4* __restrict__ DSAcclg, float3* __restrict__ DSFlForceg, float4* __restrict__ DSDispPhig, float* __restrict__ DSEqPlasticg,
		tmatrix3f* __restrict__ DSPlasticStraing, float4* __restrict__ DSDefGradg2D, float4* __restrict__ DSDefGradg3D,
		float2* __restrict__ DSDefPk, float4* __restrict__ DSVelg, float4* __restrict__ DSVelPreg,
		const float4* __restrict__ DSPos0g, float4* __restrict__ DSPhiTdatag,
		float2* __restrict__ DSDispCorxzg, float* __restrict__ DSDispCoryg,
		bool Simulate2D, bool DSFrac, const float TimeStep,
		JUserExpressionListGPU* UserExpressionsg, const tfloat3 Gravity)
	{
		const int p1 = blockIdx.x * blockDim.x + threadIdx.x;
		if (p1 < MapNdeformstruc) {
			const unsigned bodyid = CODE_GetIbodyDeformStruc(DSCodeg[p1]);
			const StDeformStrucData& body = DeformStrucDatag[bodyid];

			DSDispPhig[p1] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
			if (DSEqPlasticg) DSEqPlasticg[p1] = 0.0f;
			if (DSPlasticStraing) DSPlasticStraing[p1] = { 1.0f,0.0f,0.0f,0.0f,1.0f,0.0f,0.0f,0.0f,1.0f };  // Initialize plastic metric to identity
			DSDefGradg2D[p1] = make_float4(1.0f, 0.0f, 0.0f, 1.0f);
			if (DSFrac) {
				DSPhiTdatag[p1] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
				DSDispCorxzg[p1] = make_float2(0.0f, 0.0f);
				if (!Simulate2D) DSDispCoryg[p1] = 0.0f;
			}
			float4 pos0p1 = DSPos0g[p1];
			float surfacefact = 1.0f / (body.vol0 * (1.0f / DSKerSumVolg[p1] - body.selfkern));
			const tbcstruc bcvel = DSPartVBCg[p1];
			float3 initvel = make_float3(0.0f, 0.0f, 0.0f);
			if (TimeStep > bcvel.tst && TimeStep < bcvel.tend) {
				bool skip;
				float value;
				if (DSBC_GET_X_FLAG(bcvel.flags)) {
					initvel.x = bcvel.x;
				}
				else if (DSBC_GET_X_IS_EXPR(bcvel.flags)) {
					skip = false;
					value = UserExpressionsg->GetById(DSBC_GET_X_EXPRID(bcvel.flags))->Evaluate(make_float3(pos0p1.x, pos0p1.y, pos0p1.z),
						make_float3(pos0p1.x + 0, pos0p1.y + 0, pos0p1.z + 0),
						make_float3(0, 0, 0), TimeStep, 0.0, body.dp, skip);
					if (!skip) initvel.x = value;
				}

				if (DSBC_GET_Y_FLAG(bcvel.flags)) {
					initvel.y = bcvel.y;
				}
				else if (DSBC_GET_Y_IS_EXPR(bcvel.flags)) {
					skip = false;
					value = UserExpressionsg->GetById(DSBC_GET_Y_EXPRID(bcvel.flags))->Evaluate(make_float3(pos0p1.x, pos0p1.y, pos0p1.z),
						make_float3(pos0p1.x + 0, pos0p1.y + 0, pos0p1.z + 0),
						make_float3(0, 0, 0), TimeStep, 0.0, body.dp, skip);
					if (!skip) initvel.y = value;
				}

				if (DSBC_GET_Z_FLAG(bcvel.flags)) {
					initvel.z = bcvel.z;
				}
				else if (DSBC_GET_Z_IS_EXPR(bcvel.flags)) {
					skip = false;
					value = UserExpressionsg->GetById(DSBC_GET_Z_EXPRID(bcvel.flags))->Evaluate(make_float3(pos0p1.x, pos0p1.y, pos0p1.z),
						make_float3(pos0p1.x + 0, pos0p1.y + 0, pos0p1.z + 0),
						make_float3(0, 0, 0), TimeStep, 0.0, body.dp, skip);
					if (!skip) initvel.z = value;
				}
			}
			tbcstruc bcforce = DSPartFBCg[p1];
			if (TimeStep > bcforce.tst && TimeStep < bcforce.tend) {
				bool skip;
				float value;
				float conversionFactor = 1.0f;
				unsigned forcetype = DSBC_GET_FORCETYPE(bcforce.flags);
				if (forcetype == DSBC_FORCETYPE_POINT) {
					conversionFactor = 1.0f / body.particlemass;
				}
				else if (forcetype == DSBC_FORCETYPE_SURFACE) {
					conversionFactor = 1.0f / (body.dp * body.rho0) * surfacefact;
				}
				//else if (forcetype == DSBC_FORCETYPE_BODY) {
				//	conversionFactor = 1.0f / body.rho0;
				//}

				if (DSBC_GET_X_IS_EXPR(bcforce.flags)) {
					skip = false;
					value = UserExpressionsg->GetById(DSBC_GET_X_EXPRID(bcforce.flags))->Evaluate(make_float3(pos0p1.x, pos0p1.y, pos0p1.z),
						make_float3(pos0p1.x + 0, pos0p1.y + 0, pos0p1.z + 0),
						make_float3(0, 0, 0), TimeStep, 0.0, body.dp, skip);
					if (!skip) bcforce.x = value * conversionFactor;
				}

				if (DSBC_GET_Y_IS_EXPR(bcforce.flags)) {
					skip = false;
					value = UserExpressionsg->GetById(DSBC_GET_Y_EXPRID(bcforce.flags))->Evaluate(make_float3(pos0p1.x, pos0p1.y, pos0p1.z),
						make_float3(pos0p1.x + 0, pos0p1.y + 0, pos0p1.z + 0),
						make_float3(0, 0, 0), TimeStep, 0.0, body.dp, skip);
					if (!skip) bcforce.y = value * conversionFactor;
				}

				if (DSBC_GET_Z_IS_EXPR(bcforce.flags)) {
					skip = false;
					value = UserExpressionsg->GetById(DSBC_GET_Z_EXPRID(bcforce.flags))->Evaluate(make_float3(pos0p1.x, pos0p1.y, pos0p1.z),
						make_float3(pos0p1.x + 0, pos0p1.y + 0, pos0p1.z + 0),
						make_float3(0, 0, 0), TimeStep, 0.0, body.dp, skip);
					if (!skip) bcforce.z = value * conversionFactor;
				}
			}
			DSPartFBCg[p1] = bcforce;
			DSAcclg[p1] = make_float4(bcforce.x + Gravity.x, bcforce.y + Gravity.y, bcforce.z + Gravity.z, FLT_MAX);
			DSFlForceg[p1] = { 0.0,0.0,0.0 };
			if (!Simulate2D) {
				DSDefGradg3D[p1] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
				DSDefPk[p1] = make_float2(1.0f, 0.0f);
			}
			DSVelg[p1] = make_float4(initvel.x, initvel.y, initvel.z, 0.0f);

			if (TStep == STEP_Symplectic) DSVelPreg[p1] = make_float4(initvel.x, initvel.y, initvel.z, 0.0f);
		}

	}

	void DSInitFieldVars(const TpStep TStep, const int MapNdeformstruc, const typecode* DSCodeg,
		const StDeformStrucData* DeformStrucDatag, const tbcstruc* DSPartVBCg, tbcstruc* DSPartFBCg, tphibc* DSPartPhiBCg, const float* DSKerSumVolg,
		float4* DSAcclg, float3* DSFlForceg, float4* DSDispPhig, float* DSEqPlasticg, tmatrix3f* DSPlasticStraing, float4* DSDefGradg2D, float4* DSDefGradg3D,
		float2* DSDefPk, float4* DSVelg, float4* DSVelPreg, const float4* DSPos0g, float4* DSPhiTdatag,
		float2* DSDispCorxzg, float* DSDispCoryg, bool Simulate2D, bool DSFrac,
		JUserExpressionListGPU* UserExpressionsg, const float TimeStep, const tfloat3 Gravity)
	{

		dim3 sgridb = GetSimpleGridSize(MapNdeformstruc, SPHBSIZE);
		DSInitFieldVarsKernel << <sgridb, SPHBSIZE >> > (TStep, MapNdeformstruc, DSCodeg, DeformStrucDatag,
			DSPartVBCg, DSPartFBCg, DSKerSumVolg, DSAcclg, DSFlForceg, DSDispPhig, DSEqPlasticg, DSPlasticStraing, DSDefGradg2D, DSDefGradg3D, DSDefPk, DSVelg, DSVelPreg, DSPos0g,
			DSPhiTdatag, DSDispCorxzg, DSDispCoryg, Simulate2D, DSFrac, TimeStep, UserExpressionsg, Gravity);
		cudaDeviceSynchronize();
	}


	__global__ void __launch_bounds__(128)  // Limit threads per block to reduce register usage
		DSCalcMaxInitTimeStepKernel(const int MapNdeformstruc, const typecode* __restrict__ DSCode,
			const StDeformStrucData* __restrict__ DeformStrucData, const float4* __restrict__ DSVel,
			const float4* __restrict__ DSAccl, float* __restrict__ DSTimesteps)
	{
		int p1 = blockIdx.x * blockDim.x + threadIdx.x;

		if (p1 < MapNdeformstruc) {
			const typecode code = DSCode[p1];
			const unsigned bodyid = CODE_GetIbodyDeformStruc(code);

			// Access only the fields we need instead of loading the entire struct
			const float kernelh = DeformStrucData[bodyid].kernelh;
			const float czero = DeformStrucData[bodyid].czero;

			const float4 velp1 = DSVel[p1];
			const float4 acclp1 = DSAccl[p1];

			const float velmag = sqrtf(velp1.x * velp1.x + velp1.y * velp1.y + velp1.z * velp1.z);
			const float dt_av = kernelh / (czero + velmag);

			const float famag = acclp1.x * acclp1.x + acclp1.y * acclp1.y + acclp1.z * acclp1.z;
			const float dtf = (famag > ALMOSTZERO) ? sqrtf(kernelh / sqrtf(famag)) : FLT_MAX;

			DSTimesteps[p1] = fminf(dt_av, dtf);
		}
	}


	double DSCalcMaxInitTimeStep(const int MapNdeformstruc, const unsigned DeformStrucCount, const typecode* DSCodeg,
		StDeformStrucData* DeformStrucDatac, const StDeformStrucData* DeformStrucDatag, const uint2* DSPairNSg,
		const float4* DSPos0g, const float4* DSDispPhig, const float4* DSVelg, const float4* DSAcclg, const tbcstruc* DSPartFBCg,
		const tfloat3 Gravity, const unsigned* DSPairJg, JUserExpressionListGPU* UserExpressionsg, const float TimeStep)
	{
		// Allocate temporary array for timesteps
		float* d_DSTimesteps;
		cudaError_t err = cudaMalloc(&d_DSTimesteps, MapNdeformstruc * sizeof(float));
		if (err != cudaSuccess) {
			printf("CUDA Error allocating timesteps array: %s\n", cudaGetErrorString(err));
			return DBL_MAX;
		}

		// Use block size 128 to match __launch_bounds__
		const int blockSize = 128;
		dim3 sgridb = GetSimpleGridSize(MapNdeformstruc, blockSize);
		unsigned n_blocks = sgridb.x * sgridb.y;
		unsigned resu_size = n_blocks + n_blocks;
		float* d_resu;
		err = cudaMalloc(&d_resu, resu_size * sizeof(float));
		if (err != cudaSuccess) {
			printf("CUDA Error allocating reduction auxiliary array: %s\n", cudaGetErrorString(err));
			cudaFree(d_DSTimesteps);
			return DBL_MAX;
		}

		// Launch kernel with block size 128
		DSCalcMaxInitTimeStepKernel << <sgridb, blockSize >> > (MapNdeformstruc,
			DSCodeg, DeformStrucDatag, DSVelg, DSAcclg, d_DSTimesteps);

		err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("CUDA Error after DSCalcMaxInitTimeStepKernel launch: %s\n", cudaGetErrorString(err));
			cudaFree(d_DSTimesteps);
			cudaFree(d_resu);
			return DBL_MAX;
		}

		cudaDeviceSynchronize();
		err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("CUDA Error after DSCalcMaxInitTimeStepKernel sync: %s\n", cudaGetErrorString(err));
			cudaFree(d_DSTimesteps);
			cudaFree(d_resu);
			return DBL_MAX;
		}

		// Use existing ReduMinFloat function for reduction
		double init_val = (double)ReduMinFloat(MapNdeformstruc, 0, d_DSTimesteps, d_resu);
		
		cudaFree(d_DSTimesteps);
		cudaFree(d_resu);

		for (unsigned bodyid1 = 0; bodyid1 < DeformStrucCount - 1; bodyid1++) {
			StDeformStrucData& body1 = DeformStrucDatac[bodyid1];
			for (unsigned bodyid2 = bodyid1 + 1; bodyid2 < DeformStrucCount; bodyid2++) {
				StDeformStrucData& body2 = DeformStrucDatac[bodyid2];
				double E1 = body1.youngmod;
				double E2 = body2.youngmod;
				double tau1 = body1.tau;
				double tau2 = body2.tau;

				double E_star = 1.0 / (tau1 + tau2);
				double R_star = 0.25 * (body1.dp + body2.dp);

				double kn = 4.0 / 3.0 * E_star * sqrt(R_star);
				double mp = body1.rho0 * body1.vol0;
				double mq = body2.rho0 * body2.vol0;
				double m_eff = (mp * mq) / (mp + mq);

				double delta0 = 0.1 * min(body1.dp, body2.dp);
				double kn_lin = 1.5 * kn * sqrt(delta0);

				double dt_contact = sqrt(m_eff / kn_lin);
				init_val = min(init_val, dt_contact);
			}
		}
		return init_val;
	}


	template<bool simulate2d>
	//__launch_bounds__(SPHBSIZE, 3)
	__global__ void DSComputeDeformGradPKKernel(const int np, unsigned bodycnt,
		const StDeformStrucIntData* __restrict__ DeformStrucDatag, const StDeformStrucIntArraysg arrays)
	{
		extern __shared__ StDeformStrucIntData deformbody[];
		for (int i = threadIdx.x; i < bodycnt; i += blockDim.x) deformbody[i] = DeformStrucDatag[i];
		__syncthreads();
		const int p1 = blockIdx.x * blockDim.x + threadIdx.x;
		if (p1 >= np) return;

		const StDeformStrucIntData body = deformbody[CODE_GetIbodyDeformStruc(arrays.DSCodeg[p1])];
		const float4 dispphip1 = __ldg(&arrays.DSDispPhig[p1]);
		//const float4 pos0p1 = __ldg(&arrays.DSPos0g[p1]);
		const uint2 pairns = __ldg(&arrays.DSPairNSg[p1]);

		matrix3f defgradp1 = Ident3f();

		float4 gradlapphi = make_float4(0.0f, 0.0f, 0.0f, 0.0f);;
		float4 corsum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		//float gradphi2;
		if (dispphip1.w >= body.pflim) {
			if (body.fracture) {
#pragma unroll 8
				for (unsigned pair = 0; pair < pairns.x; ++pair)
				{
					const unsigned p2 = __ldg(&arrays.DSPairJg[pairns.y + pair]);
					const float4 kerderlap = __ldg(&arrays.DSKerDerLapg[pairns.y + pair]);
					const float4 dispphip2 = __ldg(&arrays.DSDispPhig[p2]);

					const float4 disp_diff = make_float4(dispphip2.x - dispphip1.x, dispphip2.y - dispphip1.y,
						dispphip2.z - dispphip1.z, dispphip2.w - dispphip1.w);

					//gradlapphi.x += disp_diff.w * kerderlap.x;
					//gradlapphi.y += disp_diff.w * kerderlap.y;
					//gradlapphi.z += disp_diff.w * kerderlap.z;
					gradlapphi.w += disp_diff.w * kerderlap.w;

					//if (dispphip2.w >= body.pflim) {
					defgradp1.a11 += disp_diff.x * kerderlap.x;
					defgradp1.a13 += disp_diff.x * kerderlap.z;
					defgradp1.a31 += disp_diff.z * kerderlap.x;
					defgradp1.a33 += disp_diff.z * kerderlap.z;

					if (!simulate2d) {
						defgradp1.a12 += disp_diff.x * kerderlap.y;
						defgradp1.a21 += disp_diff.y * kerderlap.x;
						defgradp1.a22 += disp_diff.y * kerderlap.y;
						defgradp1.a23 += disp_diff.y * kerderlap.z;
						defgradp1.a32 += disp_diff.z * kerderlap.y;
					}

					//}
				}
				//gradphi2 = gradlapphi.x * gradlapphi.x + gradlapphi.y * gradlapphi.y + gradlapphi.z * gradlapphi.z;
			}
			else {
#pragma unroll 8
				for (unsigned pair = 0; pair < pairns.x; ++pair)
				{
					const unsigned p2 = __ldg(&arrays.DSPairJg[pairns.y + pair]);
					const float4 kerderlap = __ldg(&arrays.DSKerDerLapg[pairns.y + pair]);
					const float4 dispphip2 = __ldg(&arrays.DSDispPhig[p2]);
					const float4 disp_diff = make_float4(dispphip2.x - dispphip1.x, dispphip2.y - dispphip1.y,
						dispphip2.z - dispphip1.z, dispphip2.w - dispphip1.w);

					defgradp1.a11 += disp_diff.x * kerderlap.x;
					defgradp1.a13 += disp_diff.x * kerderlap.z;
					defgradp1.a31 += disp_diff.z * kerderlap.x;
					defgradp1.a33 += disp_diff.z * kerderlap.z;

					if (!simulate2d) {
						defgradp1.a12 += disp_diff.x * kerderlap.y;
						defgradp1.a21 += disp_diff.y * kerderlap.x;
						defgradp1.a22 += disp_diff.y * kerderlap.y;
						defgradp1.a23 += disp_diff.y * kerderlap.z;
						defgradp1.a32 += disp_diff.z * kerderlap.y;
					}
				}
			}
		}
		else {
#pragma unroll 8
			for (unsigned pair = 0; pair < pairns.x; ++pair)
			{
				const unsigned p2 = __ldg(&arrays.DSPairJg[pairns.y + pair]);
				const float4 kerderlap = __ldg(&arrays.DSKerDerLapg[pairns.y + pair]);
				const float4 dispphip2 = __ldg(&arrays.DSDispPhig[p2]);

				const float4 disp_diff = make_float4(dispphip2.x - dispphip1.x, dispphip2.y - dispphip1.y,
					dispphip2.z - dispphip1.z, dispphip2.w - dispphip1.w);

				//gradlapphi.x += disp_diff.w * kerderlap.x;
				//gradlapphi.y += disp_diff.w * kerderlap.y;
				//gradlapphi.z += disp_diff.w * kerderlap.z;
				gradlapphi.w += disp_diff.w * kerderlap.w;

				//const float4 pos0p2 = __ldg(&arrays.DSPos0g[p2]);
				//float wabv = (pos0p2.x - pos0p1.x) * (pos0p2.x - pos0p1.x) +
				//	(pos0p2.y - pos0p1.y) * (pos0p2.y - pos0p1.y) +
				//	(pos0p2.z - pos0p1.z) * (pos0p2.z - pos0p1.z);
				//if (wabv >= body.kernelh * body.kernelh) continue;
				float weight = __ldg(&arrays.DSKerg[pairns.y + pair]);
				weight = dispphip2.w * dispphip2.w * weight * weight;
				corsum.x += disp_diff.x * weight;
				corsum.y += disp_diff.y * weight;
				corsum.z += disp_diff.z * weight;
				corsum.w += weight;
			}
			//gradphi2 = gradlapphi.x * gradlapphi.x + gradlapphi.y * gradlapphi.y + gradlapphi.z * gradlapphi.z;
			if (corsum.w > ALMOSTZERO) {
				arrays.DSDispCorxzg[p1] = make_float2(corsum.x / corsum.w, corsum.z / corsum.w);
				if (!simulate2d) arrays.DSDispCoryg[p1] = corsum.y / corsum.w;
			}
			else {
				arrays.DSDispCorxzg[p1] = make_float2(0.0f, 0.0f);
				if (!simulate2d) arrays.DSDispCoryg[p1] = 0.0f;
			}

		}

		matrix3f PK2 = Zero3f();
		if (body.constitmodel == CONSTITMODEL_SVK) {
			const matrix3f GL = 0.5 * (cumath::MulMatrix3x3(cumath::TrasMatrix3x3(defgradp1), defgradp1) - Ident3f());
			const float trE = GL.a11 + GL.a22 + GL.a33;
			if (body.fracture) {
				float4 phitimedata = __ldg(&arrays.DSPhiTdatag[p1]);
				if (dispphip1.w < body.pflim) {
					phitimedata.y = 0.0f;
				}
				else {

					const matrix3f Glp = cumath::DSEigenDecompose<simulate2d>(GL);

					const matrix3f Sp = body.lambda * max(trE, 0.0f) * Ident3f() + 2.0f * body.mu * Glp;
					const matrix3f Sn = body.lambda * min(trE, 0.0f) * Ident3f() + 2.0f * body.mu * (GL - Glp);
					PK2 = dispphip1.w * dispphip1.w * Sp + Sn;
					const float Wp = 0.5f * body.lambda * max(trE, 0.0f) * max(trE, 0.0f) + body.mu * cumath::Trace3x3(cumath::MulMatrix3x3(Glp, Glp));
					//const float Wn = 0.5 * body.lambda * min(trE, 0.0f) * min(trE, 0.0f) + body.mu * cumath::Trace3x3(cumath::MulMatrix3x3((GL - Glp), (GL - Glp)));

					if (Wp > phitimedata.w) phitimedata.w = Wp;

					const float fm = 0.5f * body.czero / sqrtf(4.0f * body.gc * body.lc * phitimedata.w + body.gc * body.gc);
					const float term1 = body.gc * (2.0f * body.lc * gradlapphi.w + 0.5f * (1.0f - dispphip1.w) / body.lc) - 2.0f * dispphip1.w * phitimedata.w - phitimedata.x / fm;
					phitimedata.y = 0.5f * body.czero * body.czero * term1 / (body.gc * body.lc);
				}
				arrays.DSPhiTdatag[p1] = phitimedata;
				//energyp1.z = body.gc * (0.25 / body.lc * (1.0 - dispphip1.w) * (1.0 - dispphip1.w) + body.lc * gradphi2);
			}
			else {
				PK2 = body.lambda * trE * Ident3f() + 2.0f * body.mu * GL;
				//energyp1.x = 0.5 * body.lambda * trE * trE + body.mu * cumath::Trace3x3(cumath::MulMatrix3x3(GL, GL));
			}
		}
		else if (body.constitmodel == CONSTITMODEL_NH) {
			const float jac = simulate2d ? cumath::Determinant2x2(defgradp1) : cumath::Determinant3x3(defgradp1);

			if (body.fracture) {
				float4 phitimedata = __ldg(&arrays.DSPhiTdatag[p1]);
				if (dispphip1.w < body.pflim) {
					phitimedata.y = 0.0f;
				}
				else {

					const matrix3f matb = cumath::MulMatrix3x3(cumath::TrasMatrix3x3(defgradp1), defgradp1);
					matrix3f binv;
					if (simulate2d) {
						binv = cumath::InverseMatrix2x2(matb);
						binv.a22 = 1.0f;
					}
					else binv = cumath::InverseMatrix3x3(matb);

					float Wp = 0.5f * body.mu * (powf(jac, -2.0f / 3.0f) * (matb.a11 + matb.a22 + matb.a33) - 3.0f);
					//float Wn = 0.0;
					if (jac >= 1.0) {
						PK2 = dispphip1.w * dispphip1.w * (0.5f * body.bulk * (jac * jac - 1.0f) * binv + powf(jac, -2.0f / 3.0f) * body.mu * (Ident3f() - binv * ((matb.a11 + matb.a22 + matb.a33) / 3.0f)));
						Wp += 0.5f * body.bulk * (0.5f * (jac * jac - 1.0f) - logf(jac));
					}
					else {
						PK2 = 0.5f * body.bulk * (jac * jac - 1.0f) * binv + dispphip1.w * dispphip1.w * (powf(jac, -2.0f / 3.0f) * body.mu * (Ident3f() - binv * ((matb.a11 + matb.a22 + matb.a33) / 3.0f)));
						//Wn = 0.5f * body.bulk * (0.5f * (jac * jac - 1.0f) - logf(jac));
					}

					//energyp1.x = dispphip1.w * dispphip1.w * Wp + Wn;

					if (Wp > phitimedata.w) phitimedata.w = Wp;

					const float fm = 0.5f * body.czero / sqrtf(4.0f * body.gc * body.lc * phitimedata.w + body.gc * body.gc);
					const float term1 = body.gc * (2.0f * body.lc * gradlapphi.w + 0.5f * (1.0f - dispphip1.w) / body.lc) - 2.0f * dispphip1.w * phitimedata.w - phitimedata.x / fm;
					phitimedata.y = 0.5f * body.czero * body.czero * term1 / (body.gc * body.lc);
				}
				arrays.DSPhiTdatag[p1] = phitimedata;
				//energyp1.z = body.gc * (0.25 / body.lc * (1.0 - dispphip1.w) * (1.0 - dispphip1.w) + body.lc * gradphi2);
			}
			else {

				const matrix3f matb = cumath::MulMatrix3x3(cumath::TrasMatrix3x3(defgradp1), defgradp1);
				matrix3f binv;
				if (simulate2d) {
					binv = cumath::InverseMatrix2x2(matb);
					binv.a22 = 1.0;
				}
				else binv = cumath::InverseMatrix3x3(matb);

				PK2 = 0.5f * body.bulk * (jac * jac - 1.0f) * binv + powf(jac, -2.0f / 3.0f) * body.mu * (Ident3f() - binv * ((matb.a11 + matb.a22 + matb.a33) / 3.0f));
				//energyp1.x = 0.5f * body.mu * (powf(jac, -2.0f / 3.0f) * (matb.a11 + matb.a22 + matb.a33) - 3.0f) + 0.5f * body.bulk * (0.5f * (jac * jac - 1.0f) - logf(jac));
			}
		}
		else if (body.constitmodel == CONSTITMODEL_J2) {
			const float body_mu = body.mu;
			const float body_lambda = body.lambda;
			const float body_hard = body.hardening;
			matrix3f defgradT = cumath::TrasMatrix3x3(defgradp1);
			matrix3f cauchy_green = cumath::MulMatrix3x3(defgradT, defgradp1);
			const float jac = simulate2d ? cumath::Determinant2x2(defgradp1) : cumath::Determinant3x3(defgradp1);

			// Load plastic metric tensor C_p (stored in Lagrangian config)
			tmatrix3f Cpt3f = arrays.DSPlasticStraing[p1];
			matrix3f Cp;
			Cp.a11 = Cpt3f.a11; Cp.a12 = Cpt3f.a12; Cp.a13 = Cpt3f.a13;
			Cp.a21 = Cpt3f.a21; Cp.a22 = Cpt3f.a22; Cp.a23 = Cpt3f.a23;
			Cp.a31 = Cpt3f.a31; Cp.a32 = Cpt3f.a32; Cp.a33 = Cpt3f.a33;
			// Compute elastic right Cauchy-Green: C_e = C * C_p^{-1}
			matrix3f Cp_inv;
			if (simulate2d) {
				Cp_inv = cumath::InverseMatrix2x2(Cp);
				Cp_inv.a22 = 1.0f;
			}
			else {
				Cp_inv = cumath::InverseMatrix3x3(Cp);
			}
			matrix3f C_e = cumath::MulMatrix3x3(cauchy_green, Cp_inv);

			// Elastic Jacobian: J_e = sqrt(det(C_e)) = J / J_p
			float detCp = simulate2d ? cumath::Determinant2x2(Cp) : cumath::Determinant3x3(Cp);
			detCp = fmaxf(detCp, 1e-12f);
			const float Jp = sqrtf(detCp);
			const float Je = jac / fmaxf(Jp, 1e-12f);

			// Isochoric elastic right Cauchy-Green: C_e_bar = J_e^{-2/3} * C_e
			const float Je_pow = powf(Je, -2.0f / 3.0f);
			matrix3f C_e_bar = C_e;
			C_e_bar.a11 *= Je_pow; C_e_bar.a12 *= Je_pow; C_e_bar.a13 *= Je_pow;
			C_e_bar.a21 *= Je_pow; C_e_bar.a22 *= Je_pow; C_e_bar.a23 *= Je_pow;
			C_e_bar.a31 *= Je_pow; C_e_bar.a32 *= Je_pow; C_e_bar.a33 *= Je_pow;

			// Deviatoric part of C_e_bar
			const float tr_Ce_bar = C_e_bar.a11 + C_e_bar.a22 + C_e_bar.a33;
			const float mean_Ce_bar = tr_Ce_bar / 3.0f;
			matrix3f dev_Ce_bar = C_e_bar;
			dev_Ce_bar.a11 -= mean_Ce_bar; dev_Ce_bar.a22 -= mean_Ce_bar; dev_Ce_bar.a33 -= mean_Ce_bar;

			// Trial Mandel stress (deviatoric part): M_dev_trial = mu * dev(C_e_bar)
			matrix3f M_dev_trial;
			M_dev_trial.a11 = body_mu * dev_Ce_bar.a11;
			M_dev_trial.a22 = body_mu * dev_Ce_bar.a22;
			M_dev_trial.a33 = body_mu * dev_Ce_bar.a33;
			M_dev_trial.a12 = body_mu * dev_Ce_bar.a12;
			M_dev_trial.a21 = body_mu * dev_Ce_bar.a21;
			M_dev_trial.a13 = body_mu * dev_Ce_bar.a13;
			M_dev_trial.a31 = body_mu * dev_Ce_bar.a31;
			M_dev_trial.a23 = body_mu * dev_Ce_bar.a23;
			M_dev_trial.a32 = body_mu * dev_Ce_bar.a32;

			// von Mises equivalent stress from Mandel stress
			const float M_trial_sq =
				M_dev_trial.a11 * M_dev_trial.a11 +
				M_dev_trial.a22 * M_dev_trial.a22 +
				M_dev_trial.a33 * M_dev_trial.a33 +
				2.0f * (M_dev_trial.a12 * M_dev_trial.a12 +
					M_dev_trial.a13 * M_dev_trial.a13 +
					M_dev_trial.a23 * M_dev_trial.a23);
			const float sigma_eq_trial = sqrtf(fmaxf(1.5f * M_trial_sq, 0.0f));
			float plastic_strain = arrays.DSEqPlasticg[p1];
			const float sigma_yield = fmaxf(body.yieldstress + body_hard * plastic_strain, 0.0f);

			matrix3f M_dev = M_dev_trial;
			matrix3f Cp_new = Cp;
			if (sigma_eq_trial > sigma_yield + ALMOSTZERO) {
				const float denom = 3.0f * body_mu + body_hard * 0.816496580927726f;
				float dgamma = (sigma_eq_trial - sigma_yield) / denom;
				dgamma = fmaxf(dgamma, 0.0f);
				const float ratio = fmaxf(1.0f - (3.0f * body_mu * dgamma) / sigma_eq_trial, 0.0f);
				M_dev.a11 *= ratio; M_dev.a22 *= ratio; M_dev.a33 *= ratio;
				M_dev.a12 *= ratio; M_dev.a21 *= ratio;
				M_dev.a13 *= ratio; M_dev.a31 *= ratio;
				M_dev.a23 *= ratio; M_dev.a32 *= ratio;

				// Update plastic metric: C_p_new = C_p + 2 * dgamma * N * C_p
				// where N is flow direction: N = 1.5 * M_dev / ||M_dev||
				const float norm_M = fmaxf(sqrtf(M_trial_sq), ALMOSTZERO);
				const float factor = 3.0f * dgamma / norm_M;

				matrix3f flow_Cp = cumath::MulMatrix3x3(M_dev_trial, Cp);
				Cp_new.a11 += factor * flow_Cp.a11; Cp_new.a22 += factor * flow_Cp.a22; Cp_new.a33 += factor * flow_Cp.a33;
				Cp_new.a12 += factor * flow_Cp.a12; Cp_new.a21 += factor * flow_Cp.a21;
				Cp_new.a13 += factor * flow_Cp.a13; Cp_new.a31 += factor * flow_Cp.a31;
				Cp_new.a23 += factor * flow_Cp.a23; Cp_new.a32 += factor * flow_Cp.a32;

				float detCpNew = simulate2d ? cumath::Determinant2x2(Cp_new) : cumath::Determinant3x3(Cp_new);
				detCpNew = fmaxf(detCpNew, 1e-12f);
				float s = powf(detCpNew, -1.0f / 3.0f);
				Cp_new.a11 *= s; Cp_new.a12 *= s; Cp_new.a13 *= s;
				Cp_new.a21 *= s; Cp_new.a22 *= s; Cp_new.a23 *= s;
				Cp_new.a31 *= s; Cp_new.a32 *= s; Cp_new.a33 *= s;
				plastic_strain += 0.816496580927726f * dgamma;
			}
			matrix3f C_inv;
			if (simulate2d) {
				C_inv = cumath::InverseMatrix2x2(cauchy_green);
				C_inv.a22 = 1.0f;
			}
			else {
				C_inv = cumath::InverseMatrix3x3(cauchy_green);
			}

			// Bulk modulus: K = lambda + 2*mu/3
			const float bulk = body_lambda + (2.0f / 3.0f) * body_mu;
			const float vol_coeff = 0.5f * bulk * (jac * jac - 1.0f);

			// Deviatoric part: S_dev = J_e^{-1} * C_e^{-1} * M_dev * C_e^{-1}
			// Note: C_e^{-1} = (C * C_p^{-1})^{-1} = C_p * C^{-1}
			// Use updated Cp_new after plastic flow
			matrix3f Cp_Cinv = cumath::MulMatrix3x3(Cp_new, C_inv);
			matrix3f Ce_inv_M = cumath::MulMatrix3x3(Cp_Cinv, M_dev);
			matrix3f S_dev = cumath::MulMatrix3x3(Ce_inv_M, Cp_Cinv);
			const float Je_inv = 1.0f / fmaxf(Je, 1e-10f);
			S_dev.a11 *= Je_inv; S_dev.a22 *= Je_inv; S_dev.a33 *= Je_inv;
			S_dev.a12 *= Je_inv; S_dev.a21 *= Je_inv;
			S_dev.a13 *= Je_inv; S_dev.a31 *= Je_inv;
			S_dev.a23 *= Je_inv; S_dev.a32 *= Je_inv;

			// Total PK2 stress
			PK2 = S_dev;
			PK2.a11 += vol_coeff * C_inv.a11; PK2.a22 += vol_coeff * C_inv.a22; PK2.a33 += vol_coeff * C_inv.a33;
			PK2.a12 += vol_coeff * C_inv.a12; PK2.a21 += vol_coeff * C_inv.a21;
			PK2.a13 += vol_coeff * C_inv.a13; PK2.a31 += vol_coeff * C_inv.a31;
			PK2.a23 += vol_coeff * C_inv.a23; PK2.a32 += vol_coeff * C_inv.a32;

			arrays.DSEqPlasticg[p1] = fmaxf(plastic_strain, 0.0f);
			Cpt3f.a11 = Cp_new.a11; Cpt3f.a12 = Cp_new.a12; Cpt3f.a13 = Cp_new.a13;
			Cpt3f.a21 = Cp_new.a21; Cpt3f.a22 = Cp_new.a22; Cpt3f.a23 = Cp_new.a23;
			Cpt3f.a31 = Cp_new.a31; Cpt3f.a32 = Cp_new.a32; Cpt3f.a33 = Cp_new.a33;
			arrays.DSPlasticStraing[p1] = Cpt3f;
		}
		const matrix3f PK1 = cumath::MulMatrix3x3(defgradp1, PK2);

		arrays.DSDefGradg2D[p1] = make_float4(defgradp1.a11, defgradp1.a13, defgradp1.a31, defgradp1.a33);
		arrays.DSPiolKirg2D[p1] = make_float4(PK1.a11, PK1.a13, PK1.a31, PK1.a33);
		if (!simulate2d) {
			arrays.DSDefGradg3D[p1] = make_float4(defgradp1.a12, defgradp1.a21, defgradp1.a23, defgradp1.a32);
			arrays.DSPiolKirg3D[p1] = make_float4(PK1.a12, PK1.a21, PK1.a23, PK1.a32);
			arrays.DSDefPk[p1] = make_float2(defgradp1.a22, PK1.a22);
		}
	}

	template<bool simulate2d>
	//__launch_bounds__(SPHBSIZE, 3)
	__global__ void DSComputeAcclKernel(const int np, unsigned bodycnt, const StDeformStrucIntData* __restrict__ DeformStrucDatag,
		const StDeformStrucIntArraysg arrays, const float3 gravity, JUserExpressionListGPU* UserExpressionsg,
		const float3* __restrict__ DSFlForceg, const float* __restrict__ DSKerSumVolg, const float dstime, const float dsdt)
	{
		extern __shared__ StDeformStrucIntData deformbody[];
		for (int i = threadIdx.x; i < bodycnt; i += blockDim.x) deformbody[i] = DeformStrucDatag[i];
		__syncthreads();
		const int p1 = blockIdx.x * blockDim.x + threadIdx.x;
		if (p1 < np) {
			const StDeformStrucIntData body = deformbody[CODE_GetIbodyDeformStruc(arrays.DSCodeg[p1])];

			const uint2 pairnsp1 = __ldg(&arrays.DSPairNSg[p1]);
			const float4 pos0p1 = __ldg(&arrays.DSPos0g[p1]);
			const float4 dispp1 = __ldg(&arrays.DSDispPhig[p1]);
			const float4 velp1 = __ldg(&arrays.DSVelg[p1]);
			const float4 piolak2d = __ldg(&arrays.DSPiolKirg2D[p1]);
			matrix3f piolakp1 = make_matrix3f(piolak2d.x, 0.0f, piolak2d.y, 0.0f, 0.0f, 0.0f, piolak2d.z, 0.0f, piolak2d.w);
			const float4 defgrad2d = __ldg(&arrays.DSDefGradg2D[p1]);
			matrix3f defgradF = make_matrix3f(defgrad2d.x, 0.0f, defgrad2d.y, 0.0f, 1.0f, 0.0f, defgrad2d.z, 0.0f, defgrad2d.w);
			matrix3f defgradInv;
			float jacob;
			float surfacefact = 1.0f / (body.vol0 * (1.0f / DSKerSumVolg[p1] - body.selfkern));
			const tbcstruc bcf = arrays.DSPartFBCg[p1];
			float3 forcebnd = make_float3(0.0f, 0.0f, 0.0f);
			const float3 fluidforcep1 = DSFlForceg[p1];
			if (dstime > bcf.tst && dstime < bcf.tend) {
				bool skip;
				float value;
				float conversionFactor = 1.0f;
				unsigned forcetype = DSBC_GET_FORCETYPE(bcf.flags);
				if (forcetype == DSBC_FORCETYPE_POINT) {
					conversionFactor = 1.0f / (body.vol0 * body.rho0) * surfacefact;
				}
				else if (forcetype == DSBC_FORCETYPE_SURFACE) {
					conversionFactor = 1.0f / (body.dp * body.rho0) * surfacefact;
				}
				//else if (forcetype == DSBC_FORCETYPE_BODY) {
				//	conversionFactor = 1.0f / body.rho0;
				//}
				if (DSBC_GET_X_FLAG(bcf.flags)) {
					forcebnd.x = bcf.x;
				}
				else if (DSBC_GET_X_IS_EXPR(bcf.flags)) {
					skip = false;
					value = UserExpressionsg->GetById(DSBC_GET_X_EXPRID(bcf.flags))->Evaluate(make_float3(pos0p1.x, pos0p1.y, pos0p1.z),
						make_float3(pos0p1.x + dispp1.x, pos0p1.y + dispp1.y, pos0p1.z + dispp1.z),
						make_float3(dispp1.x, dispp1.y, dispp1.z), dstime, dsdt, body.dp, skip);
					if (!skip) forcebnd.x = value * conversionFactor;
				}

				if (DSBC_GET_Y_FLAG(bcf.flags)) {
					forcebnd.y = bcf.y;
				}
				else if (DSBC_GET_Y_IS_EXPR(bcf.flags)) {
					skip = false;
					value = UserExpressionsg->GetById(DSBC_GET_Y_EXPRID(bcf.flags))->Evaluate(make_float3(pos0p1.x, pos0p1.y, pos0p1.z),
						make_float3(pos0p1.x + dispp1.x, pos0p1.y + dispp1.y, pos0p1.z + dispp1.z),
						make_float3(dispp1.x, dispp1.y, dispp1.z), dstime, dsdt, body.dp, skip);
					if (!skip) forcebnd.y = value * conversionFactor;
				}

				if (DSBC_GET_Z_FLAG(bcf.flags)) {
					forcebnd.z = bcf.z;
				}
				else if (DSBC_GET_Z_IS_EXPR(bcf.flags)) {
					skip = false;
					value = UserExpressionsg->GetById(DSBC_GET_Z_EXPRID(bcf.flags))->Evaluate(make_float3(pos0p1.x, pos0p1.y, pos0p1.z),
						make_float3(pos0p1.x + dispp1.x, pos0p1.y + dispp1.y, pos0p1.z + dispp1.z),
						make_float3(dispp1.x, dispp1.y, dispp1.z), dstime, dsdt, body.dp, skip);
					if (!skip) forcebnd.z = value * conversionFactor;
				}
			}

			float3 acclp1 = make_float3(0.0f, 0.0f, 0.0f);

			float3 viscacc = make_float3(0.0f, 0.0f, 0.0f);
			float3 hgacc = make_float3(0.0f, 0.0f, 0.0f);

			if (simulate2d) {
				jacob = cumath::Determinant2x2(defgradF);
				defgradInv = cumath::InverseMatrix2x2(defgradF);
				defgradInv.a22 = 1.0f;
			}
			else {
				const float4 piolak3d = __ldg(&arrays.DSPiolKirg3D[p1]);
				const float2 defpk = __ldg(&arrays.DSDefPk[p1]);

				piolakp1.a12 = piolak3d.x; piolakp1.a21 = piolak3d.y;
				piolakp1.a23 = piolak3d.z; piolakp1.a32 = piolak3d.w;
				piolakp1.a22 = defpk.y;

				const float4 defgrad3d = __ldg(&arrays.DSDefGradg3D[p1]);
				defgradF.a12 = defgrad3d.x; defgradF.a21 = defgrad3d.y;
				defgradF.a23 = defgrad3d.z; defgradF.a32 = defgrad3d.w;
				defgradF.a22 = defpk.x;
				jacob = cumath::Determinant3x3(defgradF);
				defgradInv = cumath::InverseMatrix3x3(defgradF);
			}
			if (body.constitmodel == CONSTITMODEL_J2) {
				float epbar = __ldg(&arrays.DSEqPlasticg[p1]);
				defgradInv = cumath::SafeInvDefgrad(defgradF, epbar > SIGN_PLASTIC_FLOW, jacob);
			}

#pragma unroll 8
			for (unsigned pair = 0; pair < pairnsp1.x; pair++)
			{
				const unsigned pairadd = pairnsp1.y + pair;
				const unsigned p2 = __ldg(&arrays.DSPairJg[pairadd]);
				const float4 kerderlap = __ldg(&arrays.DSKerDerLapg[pairadd]);
				const float4 velp2 = __ldg(&arrays.DSVelg[p2]);
				const float4 pos0p2 = __ldg(&arrays.DSPos0g[p2]);
				const float4 dispp2 = __ldg(&arrays.DSDispPhig[p2]);
				const typecode codep2 = __ldg(&arrays.DSCodeg[p2]);

				const float4 piolak2dp2 = __ldg(&arrays.DSPiolKirg2D[p2]);
				matrix3f piolakp2 = make_matrix3f(piolak2dp2.x, 0.0f, piolak2dp2.y, 0.0f, 0.0f, 0.0f, piolak2dp2.z, 0.0f, piolak2dp2.w);

				if (!simulate2d) {
					const float4 piolak3dp2 = __ldg(&arrays.DSPiolKirg3D[p2]);
					const float2 defpk2 = __ldg(&arrays.DSDefPk[p2]);
					piolakp2.a12 = piolak3dp2.x; piolakp2.a21 = piolak3dp2.y;
					piolakp2.a23 = piolak3dp2.z; piolakp2.a32 = piolak3dp2.w;
					piolakp2.a22 = defpk2.y;
				}

				acclp1.x += (piolakp1.a11 + piolakp2.a11) * kerderlap.x + (piolakp1.a12 + piolakp2.a12) * kerderlap.y + (piolakp1.a13 + piolakp2.a13) * kerderlap.z;
				acclp1.y += (piolakp1.a21 + piolakp2.a21) * kerderlap.x + (piolakp1.a22 + piolakp2.a22) * kerderlap.y + (piolakp1.a23 + piolakp2.a23) * kerderlap.z;
				acclp1.z += (piolakp1.a31 + piolakp2.a31) * kerderlap.x + (piolakp1.a32 + piolakp2.a32) * kerderlap.y + (piolakp1.a33 + piolakp2.a33) * kerderlap.z;

				const float3 vel_dif = make_float3(velp2.x - velp1.x, velp2.y - velp1.y, velp2.z - velp1.z);
				const float3 dpos0 = make_float3(pos0p2.x - pos0p1.x, pos0p2.y - pos0p1.y, pos0p2.z - pos0p1.z);
				const float3 currdiff = make_float3(pos0p2.x + dispp2.x - pos0p1.x - dispp1.x, pos0p2.y + dispp2.y - pos0p1.y - dispp1.y, pos0p2.z + dispp2.z - pos0p1.z - dispp1.z);
				const float dist2 = currdiff.x * currdiff.x + currdiff.y * currdiff.y + currdiff.z * currdiff.z;
				const float vijx = vel_dif.x * currdiff.x + vel_dif.y * currdiff.y + vel_dif.z * currdiff.z;
				if (vijx < 0.0f) {
					const float muij = vijx * body.kernelh / (dist2 + 0.01f * body.kernelh * body.kernelh);
					const float q = body.avfactor1 * body.czero * muij - body.avfactor2 * muij * muij;
					viscacc.x += kerderlap.x * q;
					viscacc.y += kerderlap.y * q;
					viscacc.z += kerderlap.z * q;
				}
			}
			const float invrho0 = 1.0f / body.rho0;
			acclp1.x *= invrho0;
			acclp1.y *= invrho0;
			acclp1.z *= invrho0;




			const matrix3f defgradInvT = cumath::TrasMatrix3x3(defgradInv);

			//-> here
			float3 viscforce = cumath::DotMatVec3(defgradInvT, viscacc);
			acclp1.x += viscforce.x * jacob;
			acclp1.y += viscforce.y * jacob;
			acclp1.z += viscforce.z * jacob;

			acclp1.x += forcebnd.x + fluidforcep1.x + gravity.x;
			acclp1.y += forcebnd.y + fluidforcep1.y + gravity.y;
			acclp1.z += forcebnd.z + fluidforcep1.z + gravity.z;

			if (simulate2d) acclp1.y = 0.0f;

			const float dt_av = body.kernelh / (body.czero + sqrtf(velp1.x * velp1.x + velp1.y * velp1.y + velp1.z * velp1.z));
			const float famag = acclp1.x * acclp1.x + acclp1.y * acclp1.y + acclp1.z * acclp1.z;
			float dt_accl = (famag > ALMOSTZERO) ? sqrtf(body.kernelh / sqrtf(famag)) : FLT_MAX;

			if (body.fracture) {
				float4 phidatp1 = arrays.DSPhiTdatag[p1];
				float dphi = phidatp1.x;
				float aphi = phidatp1.y;
				aphi = aphi > ALMOSTZERO ? sqrtf(body.kernelh / aphi) : FLT_MAX;
				dt_accl = fminf(dt_accl, aphi);

				dphi = dphi > double(ALMOSTZERO) ? body.kernelh / dphi : DBL_MAX;
				dt_accl = min(dt_accl, dphi);
			}
			arrays.DSAcclg[p1] = make_float4(acclp1.x, acclp1.y, acclp1.z, fminf(dt_accl, dt_av));
		}
	}

	__global__ void DSLoadDcellParticlesKernel(const int np, unsigned* __restrict__ Dcellsg,
		const StDeformStrucIntArraysg arrays, const tfloat3 DSDomRealPosMin, const tfloat3 DSDomRealPosMax,
		const tfloat3 DSDomPosMin, const float DScellsize, const unsigned DSDomCellCode)
	{
		const int p1 = blockIdx.x * blockDim.x + threadIdx.x;
		if (p1 < np) {
			typecode codeout = CODE_GetSpecialValue(arrays.DSCodeg[p1]);
			if (codeout < CODE_OUTIGNORE) {
				const float4 pos0p1 = __ldg(&arrays.DSPos0g[p1]);
				const float4 dispphip1 = __ldg(&arrays.DSDispPhig[p1]);
				const float3 ps = { pos0p1.x + dispphip1.x , pos0p1.y + dispphip1.y , pos0p1.z + dispphip1.z };
				if (ps.x >= DSDomRealPosMin.x && ps.x < DSDomRealPosMax.x &&
					ps.y >= DSDomRealPosMin.y && ps.y < DSDomRealPosMax.y &&
					ps.z >= DSDomRealPosMin.z && ps.z < DSDomRealPosMax.z) {
					const float dx = ps.x - DSDomPosMin.x;
					const float dy = ps.y - DSDomPosMin.y;
					const float dz = ps.z - DSDomPosMin.z;
					const unsigned cx = unsigned(dx / DScellsize);
					const unsigned cy = unsigned(dy / DScellsize);
					const unsigned cz = unsigned(dz / DScellsize);
					Dcellsg[p1] = DCEL_Cell(DSDomCellCode, cx, cy, cz);
				}
				else {
					printf("ERROR: Particle %d of deformable structure is outside domain!\n", p1);
					Dcellsg[p1] = DCEL_CodeMapOut;
				}
			}
			else {
				printf("ERROR: Particle %d of deformable structure is outside domain!\n", p1);
				Dcellsg[p1] = DCEL_CodeMapOut;
			}
		}
	}

	//========================================================================================
	/// Performs deformable structures' calculations for 1 time step of the local solver.
	//========================================================================================
	//template<bool simulate2d, bool tcontact>
	float DSInteraction_Forces(const int np, unsigned bodycnt, const StDeformStrucIntData* DeformStrucDatag,
		const StDeformStrucIntArraysg DSIntArraysg, unsigned* DSDcellg, tfloat3 gravity, bool simulate2d,
		const tdouble3 DSDomRealPosMin, const tdouble3 DSDomRealPosMax, const tdouble3 DSDomPosMin, const float DScellsize,
		const unsigned DSDomCellCode, StDivDataGpu DSDivData, JCellDivGpuSingle* DSCellDivSingle, JDsTimersGpu* Timersg,
		const int DSNpSurf, const unsigned* DSSurfPartListg, const float* DSKerSumg,
		JUserExpressionListGPU* UserExpressionsg, const float3* DSFlForceg, const float dstime, const float dsdt)
	{
		const float3 Gravity = make_float3(gravity.x, gravity.y, gravity.z);
		dim3 sgridb = GetSimpleGridSize(np, SPHBSIZE);
		const size_t shmem = size_t(bodycnt) * sizeof(StDeformStrucIntData);
		if (simulate2d) {
			DSComputeDeformGradPKKernel<true> << <sgridb, SPHBSIZE, shmem >> > (np, bodycnt, DeformStrucDatag, DSIntArraysg);
			cudaDeviceSynchronize();
			DSComputeAcclKernel<true> << <sgridb, SPHBSIZE, shmem >> > (np, bodycnt, DeformStrucDatag, DSIntArraysg, Gravity, UserExpressionsg, DSFlForceg, DSKerSumg, dstime, dsdt);
		}
		else {
			DSComputeDeformGradPKKernel<false> << <sgridb, SPHBSIZE, shmem >> > (np, bodycnt, DeformStrucDatag, DSIntArraysg);
			cudaDeviceSynchronize();
			DSComputeAcclKernel<false> << <sgridb, SPHBSIZE, shmem >> > (np, bodycnt, DeformStrucDatag, DSIntArraysg, Gravity, UserExpressionsg, DSFlForceg, DSKerSumg, dstime, dsdt);
		}

		cudaDeviceSynchronize();
		float dtmin = ReduMinFloat_w(np, 0, DSIntArraysg.DSAcclg, DSIntArraysg.DSblockMinMax);
		return dtmin;
	}


	//__launch_bounds__(SPHBSIZE, 3)
	template<bool simulate2d>
	__global__ void DSComputeSemiImplicitEulerKernel(const int np, unsigned bodycnt, const float DSStepDt,
		const StDeformStrucIntData* __restrict__ DeformStrucDatag,
		const StDeformStrucIntArraysg arrays, JUserExpressionListGPU* UserExpressionsg, const float dstime)
	{
		extern __shared__ StDeformStrucIntData deformbody[];
		for (int i = threadIdx.x; i < bodycnt; i += blockDim.x) deformbody[i] = DeformStrucDatag[i];
		__syncthreads();
		const int p1 = blockIdx.x * blockDim.x + threadIdx.x;
		if (p1 < np) {
			const StDeformStrucIntData body = deformbody[CODE_GetIbodyDeformStruc(arrays.DSCodeg[p1])];
			const float4 pos0p1 = __ldg(&arrays.DSPos0g[p1]);  // Load once for entire kernel
			float4 accln = __ldg(&arrays.DSAcclg[p1]);
			const float4 velold = __ldg(&arrays.DSVelg[p1]);
			float4 dispphip1 = __ldg(&arrays.DSDispPhig[p1]);
			const tbcstruc bcvel = arrays.DSPartVBCg[p1];

			float4 velp1new;
			velp1new.x = velold.x + accln.x * DSStepDt;
			velp1new.y = velold.y + accln.y * DSStepDt;
			velp1new.z = velold.z + accln.z * DSStepDt;
			velp1new.w = 0.0f;

			if (dstime > bcvel.tst && dstime < bcvel.tend) {
				bool skip;
				float value;
				if (DSBC_GET_X_FLAG(bcvel.flags)) {
					velp1new.x = bcvel.x; accln.x = 0.0;
				}
				else if (DSBC_GET_X_IS_EXPR(bcvel.flags)) {
					skip = false;
					value = UserExpressionsg->GetById(DSBC_GET_X_EXPRID(bcvel.flags))->Evaluate(make_float3(pos0p1.x, pos0p1.y, pos0p1.z),
						make_float3(pos0p1.x + dispphip1.x, pos0p1.y + dispphip1.y, pos0p1.z + dispphip1.z),
						make_float3(dispphip1.x, dispphip1.y, dispphip1.z), dstime, DSStepDt, body.dp, skip);
					if (!skip) {
						velp1new.x = value;
						accln.x = 0.0;
					}
				}

				if (DSBC_GET_Y_FLAG(bcvel.flags)) {
					velp1new.y = bcvel.y; accln.y = 0.0;
				}
				else if (DSBC_GET_Y_IS_EXPR(bcvel.flags)) {
					skip = false;
					value = UserExpressionsg->GetById(DSBC_GET_Y_EXPRID(bcvel.flags))->Evaluate(make_float3(pos0p1.x, pos0p1.y, pos0p1.z),
						make_float3(pos0p1.x + dispphip1.x, pos0p1.y + dispphip1.y, pos0p1.z + dispphip1.z),
						make_float3(dispphip1.x, dispphip1.y, dispphip1.z), dstime, DSStepDt, body.dp, skip);
					if (!skip) {
						velp1new.y = value;
						accln.y = 0.0;
					}
				}

				if (DSBC_GET_Z_FLAG(bcvel.flags)) {
					velp1new.z = bcvel.z; accln.z = 0.0;
				}
				else if (DSBC_GET_Z_IS_EXPR(bcvel.flags)) {
					skip = false;
					value = UserExpressionsg->GetById(DSBC_GET_Z_EXPRID(bcvel.flags))->Evaluate(make_float3(pos0p1.x, pos0p1.y, pos0p1.z),
						make_float3(pos0p1.x + dispphip1.x, pos0p1.y + dispphip1.y, pos0p1.z + dispphip1.z),
						make_float3(dispphip1.x, dispphip1.y, dispphip1.z), dstime, DSStepDt, body.dp, skip);
					if (!skip) {
						velp1new.z = value;
						accln.z = 0.0;
					}
				}
				arrays.DSAcclg[p1] = accln;
			}

			dispphip1.x += velp1new.x * DSStepDt;
			dispphip1.y += velp1new.y * DSStepDt;
			dispphip1.z += velp1new.z * DSStepDt;

			if (body.fracture) {
				float4 phitdatan = __ldg(&arrays.DSPhiTdatag[p1]);

				phitdatan.x += phitdatan.y * DSStepDt;
				dispphip1.w += phitdatan.x * DSStepDt;

				if (dispphip1.w < body.pflim) {
					dispphip1.w = 0.0f;
					phitdatan.x = 0.0f;
					const float2 dispcorxz = __ldg(&arrays.DSDispCorxzg[p1]);
					dispphip1.x += dispcorxz.x;
					dispphip1.z += dispcorxz.y;
					if (!simulate2d)
						dispphip1.y += __ldg(&arrays.DSDispCoryg[p1]);
				}

				// Apply phi boundary condition
				if (arrays.DSPartPhiBCg) {
					const tphibc phibc = arrays.DSPartPhiBCg[p1];
					if (phibc.flags == 1) {
						const float4 pos0p1 = __ldg(&arrays.DSPos0g[p1]);
						const unsigned expr_id = phibc.exprid;
						// Evaluate expression
						bool skip = true;
						const float restore_value = UserExpressionsg->GetById(expr_id)->Evaluate(
							make_float3(pos0p1.x, pos0p1.y, pos0p1.z),
							make_float3(pos0p1.x + dispphip1.x, pos0p1.y + dispphip1.y, pos0p1.z + dispphip1.z),
							make_float3(dispphip1.x, dispphip1.y, dispphip1.z), dstime, DSStepDt, body.dp, skip);

						if (!skip && restore_value >= 0.0f && restore_value <= 1.0f && dispphip1.w < restore_value) {

							dispphip1.w = restore_value;
							phitdatan.x = 0.0f;
						}
					}
				}

				arrays.DSPhiTdatag[p1] = phitdatan;
			}
			arrays.DSVelg[p1] = velp1new;
			arrays.DSDispPhig[p1] = dispphip1;
		}
	}

	void DSCompSemImplEuler(const int np, unsigned bodycnt, const float DSStepDt,
		const StDeformStrucIntData* DeformStrucDatag, const StDeformStrucIntArraysg DSIntArraysg, bool simulate2d,
		JUserExpressionListGPU* UserExpressionsg, const float dstime)
	{
		dim3 sgridb = GetSimpleGridSize(np, SPHBSIZE);
		const size_t shmem = size_t(bodycnt) * sizeof(StDeformStrucIntData);
		if (simulate2d)
			DSComputeSemiImplicitEulerKernel <true> << <sgridb, SPHBSIZE, shmem >> > (np, bodycnt, DSStepDt,
				DeformStrucDatag, DSIntArraysg, UserExpressionsg, dstime);
		else
			DSComputeSemiImplicitEulerKernel <false> << <sgridb, SPHBSIZE, shmem >> > (np, bodycnt, DSStepDt,
				DeformStrucDatag, DSIntArraysg, UserExpressionsg, dstime);
		cudaDeviceSynchronize();
	}

	template<bool simulate2d>
	__global__ void DSComputeSymplecticPreKernel(const int np, unsigned bodycnt, const float dtm,
		const StDeformStrucIntData* __restrict__ DeformStrucDatag,
		const StDeformStrucIntArraysg arrays, JUserExpressionListGPU* UserExpressionsg, const float dstime)
	{
		extern __shared__ StDeformStrucIntData deformbody[];
		for (int i = threadIdx.x; i < bodycnt; i += blockDim.x) deformbody[i] = DeformStrucDatag[i];
		__syncthreads();
		const int p1 = blockIdx.x * blockDim.x + threadIdx.x;
		if (p1 < np) {
			const StDeformStrucIntData body = deformbody[CODE_GetIbodyDeformStruc(arrays.DSCodeg[p1])];
			const float4 velp1 = __ldg(&arrays.DSVelg[p1]);
			const float4 acclp1 = __ldg(&arrays.DSAcclg[p1]);
			float4 dispphip1 = __ldg(&arrays.DSDispPhig[p1]);
			dispphip1.x += velp1.x * dtm;
			dispphip1.y += velp1.y * dtm;
			dispphip1.z += velp1.z * dtm;

			float4 velpre;
			velpre.x = velp1.x + acclp1.x * dtm;
			velpre.y = velp1.y + acclp1.y * dtm;
			velpre.z = velp1.z + acclp1.z * dtm;

			arrays.DSVelg[p1] = make_float4(velpre.x, velpre.y, velpre.z, velp1.w);
			velpre = make_float4(velp1.x, velp1.y, velp1.z, 0.0f);
			if (body.fracture) {
				const float4 phidata = __ldg(&arrays.DSPhiTdatag[p1]);
				dispphip1.w += phidata.x * dtm;
				velpre.w = phidata.x + phidata.y * dtm;

				if (dispphip1.w < body.pflim) {
					dispphip1.w = 0.0f;
					velpre.w = 0.0f;
					const float2 dispcorxz = __ldg(&arrays.DSDispCorxzg[p1]);
					dispphip1.x += dispcorxz.x; dispphip1.z += dispcorxz.y;
					if (!simulate2d) dispphip1.y += __ldg(&arrays.DSDispCoryg[p1]);
				}

				// Apply phi boundary condition in predictor step
				if (arrays.DSPartPhiBCg) {
					const tphibc phibc = arrays.DSPartPhiBCg[p1];
					if (phibc.flags == 1) {
						const float4 pos0p1 = __ldg(&arrays.DSPos0g[p1]);
						const unsigned expr_id = phibc.exprid;
						// Evaluate expression
						bool skip = true;
						const float restore_value = UserExpressionsg->GetById(expr_id)->Evaluate(
							make_float3(pos0p1.x, pos0p1.y, pos0p1.z),
							make_float3(pos0p1.x + dispphip1.x, pos0p1.y + dispphip1.y, pos0p1.z + dispphip1.z),
							make_float3(dispphip1.x, dispphip1.y, dispphip1.z), dstime, dtm, body.dp, skip);

						// Apply BC if condition is true and phi is below restore value
						if (!skip && restore_value >= 0.0f && restore_value <= 1.0f && dispphip1.w < restore_value) {
							dispphip1.w = restore_value;
							velpre.w = 0.0f; // Set phi velocity to zero when restricted
						}
					}
				}

				arrays.DSPhiTdatag[p1].x = velpre.w;
				velpre.w = phidata.x;
			}
			arrays.DSVelPreg[p1] = velpre;
			arrays.DSDispPhig[p1] = dispphip1;
		}
	}


	void DSCompSympPre(const int np, unsigned bodycnt, const float dtm,
		const StDeformStrucIntData* DeformStrucDatag, const StDeformStrucIntArraysg DSIntArraysg, bool simulate2d,
		JUserExpressionListGPU* UserExpressionsg, const float dstime)
	{

		dim3 sgridb = GetSimpleGridSize(np, SPHBSIZE);
		const size_t shmem = size_t(bodycnt) * sizeof(StDeformStrucIntData);
		if (simulate2d)
			DSComputeSymplecticPreKernel <true> << <sgridb, SPHBSIZE, shmem >> > (np, bodycnt, dtm,
				DeformStrucDatag, DSIntArraysg, UserExpressionsg, dstime);
		else
			DSComputeSymplecticPreKernel <false> << <sgridb, SPHBSIZE, shmem >> > (np, bodycnt, dtm,
				DeformStrucDatag, DSIntArraysg, UserExpressionsg, dstime);
		cudaDeviceSynchronize();
	}

	template <bool simulate2d>
	__global__ void DSComputeSymplecticCorKernel(const int np, unsigned bodycnt, const float dtm,
		const StDeformStrucIntData* __restrict__ DeformStrucDatag,
		const StDeformStrucIntArraysg arrays, JUserExpressionListGPU* UserExpressionsg, const float dstime)
	{
		extern __shared__ StDeformStrucIntData deformbody[];
		for (int i = threadIdx.x; i < bodycnt; i += blockDim.x) deformbody[i] = DeformStrucDatag[i];
		__syncthreads();
		const int p1 = blockIdx.x * blockDim.x + threadIdx.x;
		if (p1 < np) {
			const StDeformStrucIntData body = deformbody[CODE_GetIbodyDeformStruc(arrays.DSCodeg[p1])];
			float4 acclp1 = __ldg(&arrays.DSAcclg[p1]);
			const float4 velprep1 = __ldg(&arrays.DSVelPreg[p1]);
			const tbcstruc bcvel = arrays.DSPartVBCg[p1];
			float4 dispphip1 = __ldg(&arrays.DSDispPhig[p1]);

			float4 velp1new;
			velp1new.x = velprep1.x + acclp1.x * dtm * 2.0f;
			velp1new.y = velprep1.y + acclp1.y * dtm * 2.0f;
			velp1new.z = velprep1.z + acclp1.z * dtm * 2.0f;
			velp1new.w = 0.0f;;

			if (dstime > bcvel.tst && dstime < bcvel.tend) {
				const float4 pos0p1 = __ldg(&arrays.DSPos0g[p1]);
				bool skip;
				float value;
				if (DSBC_GET_X_FLAG(bcvel.flags)) {
					velp1new.x = bcvel.x; acclp1.x = 0.0;
				}
				else if (DSBC_GET_X_IS_EXPR(bcvel.flags)) {
					skip = false;
					value = UserExpressionsg->GetById(DSBC_GET_X_EXPRID(bcvel.flags))->Evaluate(make_float3(pos0p1.x, pos0p1.y, pos0p1.z),
						make_float3(pos0p1.x + dispphip1.x, pos0p1.y + dispphip1.y, pos0p1.z + dispphip1.z),
						make_float3(dispphip1.x, dispphip1.y, dispphip1.z), dstime, dtm * 2.0f, body.dp, skip);
					if (!skip) {
						velp1new.x = value;
						acclp1.x = 0.0;
					}
				}

				if (DSBC_GET_Y_FLAG(bcvel.flags)) {
					velp1new.y = bcvel.y; acclp1.y = 0.0;
				}
				else if (DSBC_GET_Y_IS_EXPR(bcvel.flags)) {
					skip = false;
					value = UserExpressionsg->GetById(DSBC_GET_Y_EXPRID(bcvel.flags))->Evaluate(make_float3(pos0p1.x, pos0p1.y, pos0p1.z),
						make_float3(pos0p1.x + dispphip1.x, pos0p1.y + dispphip1.y, pos0p1.z + dispphip1.z),
						make_float3(dispphip1.x, dispphip1.y, dispphip1.z), dstime, dtm * 2.0f, body.dp, skip);
					if (!skip) {
						velp1new.y = value;
						acclp1.y = 0.0;
					}
				}

				if (DSBC_GET_Z_FLAG(bcvel.flags)) {
					velp1new.z = bcvel.z; acclp1.z = 0.0;
				}
				else if (DSBC_GET_Z_IS_EXPR(bcvel.flags)) {
					skip = false;
					value = UserExpressionsg->GetById(DSBC_GET_Z_EXPRID(bcvel.flags))->Evaluate(make_float3(pos0p1.x, pos0p1.y, pos0p1.z),
						make_float3(pos0p1.x + dispphip1.x, pos0p1.y + dispphip1.y, pos0p1.z + dispphip1.z),
						make_float3(dispphip1.x, dispphip1.y, dispphip1.z), dstime, dtm * 2.0f, body.dp, skip);
					if (!skip) {
						velp1new.z = value;
						acclp1.z = 0.0;
					}
				}
				arrays.DSAcclg[p1] = acclp1;
			}

			dispphip1.x += velp1new.x * dtm;
			dispphip1.y += velp1new.y * dtm;
			dispphip1.z += velp1new.z * dtm;

			arrays.DSVelg[p1] = velp1new;
			if (body.fracture) {
				velp1new = __ldg(&arrays.DSPhiTdatag[p1]);
				velp1new.x += velp1new.y * dtm * 2.0f;
				float phidotnew = velprep1.w + velp1new.y * dtm * 2.0f;
				dispphip1.w += phidotnew * dtm;

				if (dispphip1.w < body.pflim) {
					dispphip1.w = 0.0f;
					phidotnew = 0.0f;
					const float2 dispcorxz = __ldg(&arrays.DSDispCorxzg[p1]);
					dispphip1.x += dispcorxz.x; dispphip1.z += dispcorxz.y;
					if (!simulate2d) dispphip1.y += __ldg(&arrays.DSDispCoryg[p1]);
				}

				if (arrays.DSPartPhiBCg) {
					const tphibc phibc = arrays.DSPartPhiBCg[p1];
					if (phibc.flags == 1) {
						const float4 pos0p1 = __ldg(&arrays.DSPos0g[p1]);
						const unsigned expr_id = phibc.exprid;
						// Evaluate expression
						bool skip = true;
						const float restore_value = UserExpressionsg->GetById(expr_id)->Evaluate(
							make_float3(pos0p1.x, pos0p1.y, pos0p1.z),
							make_float3(pos0p1.x + dispphip1.x, pos0p1.y + dispphip1.y, pos0p1.z + dispphip1.z),
							make_float3(dispphip1.x, dispphip1.y, dispphip1.z), dstime, dtm * 2.0f, body.dp, skip);

						if (!skip && restore_value >= 0.0f && restore_value <= 1.0f && dispphip1.w < restore_value) {
							dispphip1.w = restore_value;
							phidotnew = 0.0f;
						}
					}
				}

				arrays.DSPhiTdatag[p1].x = phidotnew;
			}
			arrays.DSDispPhig[p1] = dispphip1;
		}
	}

	void DSCompSympCor(const int np, unsigned bodycnt, const float dtm,
		const StDeformStrucIntData* DeformStrucDatag, const StDeformStrucIntArraysg DSIntArraysg, bool simulate2d,
		JUserExpressionListGPU* UserExpressionsg, const float dstime)
	{

		dim3 sgridb = GetSimpleGridSize(np, SPHBSIZE);
		const size_t shmem = size_t(bodycnt) * sizeof(StDeformStrucIntData);
		if (simulate2d)
			DSComputeSymplecticCorKernel<true> << <sgridb, SPHBSIZE, shmem >> > (np, bodycnt, dtm,
				DeformStrucDatag, DSIntArraysg, UserExpressionsg, dstime);
		else
			DSComputeSymplecticCorKernel<false> << <sgridb, SPHBSIZE, shmem >> > (np, bodycnt, dtm,
				DeformStrucDatag, DSIntArraysg, UserExpressionsg, dstime);
		cudaDeviceSynchronize();
	}


	template <bool simulate2d>
	__global__ void KerDSUpdate_OrgVelPos(unsigned n, const unsigned* DeformStrucRidpg,
		const unsigned* DSBestChildg, const float2* DSPosOrg0xyg, const float* DSPosOrg0zg,
		const float4* DSDispPhig, double2* Posxyg, double* Poszg, float4* Velrhopc, float4* DSVelg)
	{
		unsigned p = blockIdx.x * blockDim.x + threadIdx.x;
		if (p < n) {
			const unsigned pds = DeformStrucRidpg[p];
			const unsigned pdmap = DSBestChildg[p];
			const float4 dispp1 = DSDispPhig[pdmap];
			const float4 velp1 = DSVelg[pdmap];
			const float2 pos0pxy = DSPosOrg0xyg[p];
			double2 posxypds;
			posxypds.x = pos0pxy.x + dispp1.x;
			posxypds.y = pos0pxy.y + dispp1.y;
			Posxyg[pds] = posxypds;
			Velrhopc[pds] = make_float4(velp1.x, velp1.y, velp1.z, Velrhopc[pds].w);
			const float pos0pz = DSPosOrg0zg[p];
			Poszg[pds] = pos0pz + dispp1.z;
		}
	}

	void DSUpdate_OrgVelPos(unsigned CaseNdeformstruc, const unsigned* DeformStrucRidpg,
		const unsigned* DSBestChildg, const float2* DSPosOrg0xyg, const float* DSPosOrg0zg,
		const float4* DSDispPhig, double2* Posxyg, double* Poszg, const bool simulate2d,
		float4* Velrhopg, float4* DSVelg)
	{
		if (CaseNdeformstruc) {
			dim3 sgrid = GetSimpleGridSize(CaseNdeformstruc, SPHBSIZE);
			if (simulate2d)
				KerDSUpdate_OrgVelPos<true> << <sgrid, SPHBSIZE >> > (CaseNdeformstruc, DeformStrucRidpg,
					DSBestChildg, DSPosOrg0xyg, DSPosOrg0zg, DSDispPhig, Posxyg, Poszg, Velrhopg, DSVelg);
			else
				KerDSUpdate_OrgVelPos<false> << <sgrid, SPHBSIZE >> > (CaseNdeformstruc, DeformStrucRidpg,
					DSBestChildg, DSPosOrg0xyg, DSPosOrg0zg, DSDispPhig, Posxyg, Poszg, Velrhopg, DSVelg);
		}
	}

	//template <bool simulate2d>
	//__global__ void DSApplyFluidPressureKer(unsigned MapNdeformstruc, unsigned bodycnt,
	//	const StDeformStrucIntData* __restrict__ DeformStrucDatag,
	//	const StDeformStrucIntArraysg arrays,
	//	StDivDataGpu DivDatag, const typecode* __restrict__ Codeg,
	//	const double2* __restrict__ Posxyg, const double* __restrict__ Poszg,
	//	const float4* __restrict__ velrhop, float3* DSFlForceg,
	//	const float Dp, const float dtdem,
	//	const unsigned* __restrict__ DSSurfPartListg, const float DSContPowerCoeff)
	//{
	//	extern __shared__ StDeformStrucIntData deformbody[];
	//	for (int i = threadIdx.x; i < bodycnt; i += blockDim.x) deformbody[i] = DeformStrucDatag[i];
	//	__syncthreads();
	//	unsigned p1 = blockIdx.x * blockDim.x + threadIdx.x;
	//	if (p1 < MapNdeformstruc) {
	//		float3 acep1 = make_float3(0.0, 0.0, 0.0);
	//		const float4 posp1 = arrays.DSPos0g[p1] + arrays.DSDispPhig[p1];
	//		const float4 velp1 = arrays.DSVelg[p1];
	//		const unsigned bodyid1 = CODE_GetIbodyDeformStruc(arrays.DSCodeg[p1]);
	//		const StDeformStrucIntData body1 = deformbody[bodyid1];
	//		const float mass1 = body1.rho0 * body1.vol0;
	//		const float kfricp1 = body1.kfric;
	//		//const float ftmassp1 = body1.rho0 * body1.vol0;
	//		int ini1, fin1, ini2, fin2, ini3, fin3;
	//		cunsearch::Initsp(posp1.x, posp1.y, posp1.z, DivDatag.axis, DivDatag.domposmin,
	//			DivDatag.scell, DivDatag.scelldiv, DivDatag.nc, DivDatag.cellzero, ini1, fin1, ini2, fin2, ini3, fin3);

	//		for (int c3 = ini3; c3 < fin3; c3 += DivDatag.nc.w) {
	//			for (int c2 = ini2; c2 < fin2; c2 += DivDatag.nc.x) {
	//				const int v = c2 + c3;
	//				for (int c1 = ini1; c1 < fin1; c1++) {
	//					const int2 cbeg = DivDatag.beginendcell[c1 + v];
	//					if (cbeg.y) {
	//						for (int p2 = cbeg.x; p2 < cbeg.y; p2++) {
	//							typecode codep2 = Codeg[p2];
	//							if (CODE_IsDeformStrucAny(codep2)) {
	//								const unsigned bodyid2 = CODE_GetIbodyDeformStruc(codep2);
	//								if (bodyid1 != bodyid2) {
	//									double2 posp2xy = Posxyg[p2];
	//									const float drx = float(posp1.x - posp2xy.x);
	//									const float dry = float(posp1.y - posp2xy.y);
	//									const float drz = float(posp1.z - Poszg[p2]);
	//									const float rr2 = drx * drx + dry * dry + drz * drz;
	//									if (rr2 < ALMOSTZERO) continue;

	//									const StDeformStrucIntData body2 = deformbody[bodyid2];
	//									const float sum_radii = 0.501f * body1.dp + 0.501f * body2.dp;
	//									float Dpmsq = sum_radii * sum_radii;

	//									if (rr2 >= Dpmsq) continue;
	//									const float rad = sqrtf(rr2);

	//									const float over_lap = sum_radii - rad;
	//									if (over_lap >= 0.0f) {
	//										const float4 velrhop2 = velrhop[p2];
	//										const float mass2 = body2.rho0 * body2.vol0;
	//										const float nx = drx / rad, ny = dry / rad, nz = drz / rad;
	//										const float kfricp2 = body2.kfric;
	//										const float R1 = 0.5f * body1.dp;
	//										const float R2 = 0.5f * body2.dp;
	//										const float R_eff = (R1 * R2) / (R1 + R2);
	//										const float E_eff = 1.0f / (body1.tau + body2.tau);

	//										const float kn_hertz = (4.0f / 3.0f) * E_eff * sqrtf(R_eff);

	//										const float dvx = velp1.x - velrhop2.x;
	//										const float dvy = velp1.y - velrhop2.y;
	//										const float dvz = velp1.z - velrhop2.z;

	//										const float vn = dvx * nx + dvy * ny + dvz * nz;
	//										const float k_lin = 2.0f * E_eff * sqrtf(R_eff) * sqrtf(over_lap);
	//										const float m_eff = (mass1 * mass2) / (mass1 + mass2);
	//										const float e = fmaxf(0.5f * (body1.restcoeff + body2.restcoeff), 1e-3f);
	//										const float ln_e = logf(e);
	//										const float zeta = -ln_e / sqrtf(ln_e * ln_e + PI * PI);
	//										const float c = 2.0f * zeta * sqrtf(m_eff * k_lin);

	//										const float normal_force = kn_hertz * over_lap * sqrtf(over_lap) - c * vn;

	//										if (normal_force > 0.0f) {
	//											const float accel_norm = DSContPowerCoeff * normal_force / (body1.rho0 * body1.vol0);
	//											acep1.x += accel_norm * nx;
	//											acep1.y += accel_norm * ny;
	//											acep1.z += accel_norm * nz;
	//										}
	//										//-Tangential.
	//										const float vtx = dvx - vn * nx;
	//										const float vty = dvy - vn * ny;
	//										const float vtz = dvz - vn * nz;
	//										const float vt_sq = vtx * vtx + vty * vty + vtz * vtz;
	//										if (vt_sq > ALMOSTZERO) {
	//											const float vt = sqrtf(vt_sq);
	//											const float tx = vtx / vt;
	//											const float ty = vty / vt;
	//											const float tz = vtz / vt;

	//											const float G_eff = E_eff / 2.6f;
	//											const float tangential_stiffness = 8.0f * G_eff * sqrtf(R_eff * over_lap);

	//											const float mu_eff = 0.5f * (kfricp1 + kfricp2);
	//											const float delta_tangential = vt * dtdem;
	//											const float tangential_force_elastic = tangential_stiffness * delta_tangential;

	//											const float tangential_force_limit = mu_eff * fabsf(normal_force);

	//											float tangential_force;
	//											if (fabsf(tangential_force_elastic) <= tangential_force_limit) {
	//												tangential_force = tangential_force_elastic;
	//											}
	//											else {
	//												tangential_force = copysignf(tangential_force_limit, tangential_force_elastic);
	//											}

	//											const float accel_tang = tangential_force / (body1.rho0 * body1.vol0);
	//											acep1.x += accel_tang * tx;
	//											acep1.y += accel_tang * ty;
	//											acep1.z += accel_tang * tz;
	//											//if(bodyid1==1) printf("\t\t\t 4-> %u %u", bodyid1, bodyid2);
	//										}
	//									}
	//									
	//								}
	//							}
	//						}
	//					}
	//				}
	//			}
	//		}
	//		if (simulate2d) acep1.y = 0;
	//		
	//		DSFlForceg[p1] = make_float3(acep1.x, acep1.y, acep1.z);
	//	}
	//}

	//void DSInteractionForcesDEM(bool simulate2d, unsigned MapNdeformstruc, unsigned bodycnt,
	//	const StDeformStrucIntData* DeformStrucDatag,
	//	const StDeformStrucIntArraysg arrays, StDivDataGpu DivDatag, const typecode* Codeg,
	//	const double2* Posxyg, const double* Poszg, const float4* velrhop, float3* DSFlForceg, 
	//	const float Dp, const float dtdem, const unsigned* DSSurfPartListg, const float DSContPowerCoeff)
	//{
	//	if (MapNdeformstruc) {
	//		dim3 sgrid = GetSimpleGridSize(MapNdeformstruc, SPHBSIZE);
	//		const size_t shmem = size_t(bodycnt) * sizeof(StDeformStrucIntData);
	//		if (simulate2d)
	//			DSApplyFluidPressureKer<true> << <sgrid, SPHBSIZE, shmem >> > (MapNdeformstruc, bodycnt,
	//				DeformStrucDatag, arrays, DivDatag, Codeg, Posxyg, Poszg, velrhop,
	//				DSFlForceg, Dp, dtdem, DSSurfPartListg, DSContPowerCoeff);
	//		else
	//			DSApplyFluidPressureKer<false> << <sgrid, SPHBSIZE, shmem >> > (MapNdeformstruc, bodycnt,
	//				DeformStrucDatag, arrays, DivDatag, Codeg, Posxyg, Poszg, velrhop,
	//				DSFlForceg, Dp, dtdem, DSSurfPartListg, DSContPowerCoeff);
	//	}
	//}

	template<bool simulate2d>
	__device__ tfloat3 DScalcenergiesp1(const matrix3f defgrad, const float4 velp1,
		const float phfp1, const uint2 pairns, const StDeformStrucIntArraysg arrays,
		const float eqplastic, const  tmatrix3f plastic_dev, const StDeformStrucIntData body)
	{
		tfloat3 energies = { 0.0f,0.0f,0.0f };
		if (body.constitmodel == CONSTITMODEL_SVK) {
			const matrix3f GL = 0.5 * (cumath::MulMatrix3x3(cumath::TrasMatrix3x3(defgrad), defgrad) - Ident3f());
			const float trE = GL.a11 + GL.a22 + GL.a33;
			if (body.fracture) {
				if (phfp1 >= body.pflim)
				{
					const matrix3f Glp = cumath::DSEigenDecompose<simulate2d>(GL);
					const float Wp = 0.5f * body.lambda * max(trE, 0.0f) * max(trE, 0.0f) + body.mu * cumath::Trace3x3(cumath::MulMatrix3x3(Glp, Glp));
					const float Wn = 0.5 * body.lambda * min(trE, 0.0f) * min(trE, 0.0f) + body.mu * cumath::Trace3x3(cumath::MulMatrix3x3((GL - Glp), (GL - Glp)));
					energies.x = phfp1 * phfp1 * Wp + Wn;
				}
			}
			else energies.x = 0.5 * body.lambda * trE * trE + body.mu * cumath::Trace3x3(cumath::MulMatrix3x3(GL, GL));
		}
		else if (body.constitmodel == CONSTITMODEL_NH) {
			const float jac = simulate2d ? cumath::Determinant2x2(defgrad) : cumath::Determinant3x3(defgrad);
			const matrix3f matb = cumath::MulMatrix3x3(cumath::TrasMatrix3x3(defgrad), defgrad);
			if (body.fracture) {
				if (phfp1 >= body.pflim)
				{
					float Wp = 0.5f * body.mu * (powf(jac, -2.0f / 3.0f) * (matb.a11 + matb.a22 + matb.a33) - 3.0f);
					float Wn = 0.0;
					if (jac >= 1.0) Wp += 0.5f * body.bulk * (0.5f * (jac * jac - 1.0f) - logf(jac));
					else Wn = 0.5f * body.bulk * (0.5f * (jac * jac - 1.0f) - logf(jac));
					energies.x = phfp1 * phfp1 * Wp + Wn;
				}
			}
			else energies.x = 0.5f * body.mu * (powf(jac, -2.0f / 3.0f) * (matb.a11 + matb.a22 + matb.a33) - 3.0f) + 0.5f * body.bulk * (0.5f * (jac * jac - 1.0f) - logf(jac));
		}
		else if (body.constitmodel == CONSTITMODEL_J2) {
			// Use multiplicative plasticity formulation consistent with stress calculation
			matrix3f defgradT = cumath::TrasMatrix3x3(defgrad);
			matrix3f cauchy_green = cumath::MulMatrix3x3(defgradT, defgrad);
			const float jac = cumath::Determinant3x3(defgrad);

			// plastic_dev here is actually the plastic metric tensor Cp (misnomer from variable name)
			matrix3f Cp;
			Cp.a11 = plastic_dev.a11; Cp.a12 = plastic_dev.a12; Cp.a13 = plastic_dev.a13;
			Cp.a21 = plastic_dev.a21; Cp.a22 = plastic_dev.a22; Cp.a23 = plastic_dev.a23;
			Cp.a31 = plastic_dev.a31; Cp.a32 = plastic_dev.a32; Cp.a33 = plastic_dev.a33;

			// Compute elastic right Cauchy-Green: C_e = C * C_p^{-1}
			matrix3f Cp_inv = cumath::InverseMatrix3x3(Cp);
			matrix3f C_e = cumath::MulMatrix3x3(cauchy_green, Cp_inv);

			// Elastic Jacobian: J_e = sqrt(det(C_e))
			const float det_Ce = cumath::Determinant3x3(C_e);
			const float Je = sqrtf(fmaxf(det_Ce, 1e-10f));

			// Isochoric elastic right Cauchy-Green: C_e_bar = J_e^{-2/3} * C_e
			const float Je_pow = powf(Je, -2.0f / 3.0f);
			matrix3f C_e_bar = C_e;
			C_e_bar.a11 *= Je_pow; C_e_bar.a12 *= Je_pow; C_e_bar.a13 *= Je_pow;
			C_e_bar.a21 *= Je_pow; C_e_bar.a22 *= Je_pow; C_e_bar.a23 *= Je_pow;
			C_e_bar.a31 *= Je_pow; C_e_bar.a32 *= Je_pow; C_e_bar.a33 *= Je_pow;

			// Compute strain energy using Neo-Hookean form for elastic part
			const float tr_Ce_bar = cumath::Trace3x3(C_e_bar);
			const float W_dev = 0.5f * body.mu * (tr_Ce_bar - 3.0f);
			const float W_vol = 0.5f * body.bulk * (0.5f * (Je * Je - 1.0f) - logf(Je));

			energies.x = W_dev + W_vol;

			energies.z = body.yieldstress * eqplastic + 0.5 * body.hardening * eqplastic * eqplastic;
		}

		energies.y = 0.5f * body.rho0 * cumath::Dot3Vec44(velp1, velp1);
		if (body.fracture) {
			float3 gradphi = { 0.0f,0.0f,0.0f };
#pragma unroll 8
			for (unsigned pair = 0; pair < pairns.x; ++pair)
			{
				const unsigned p2 = __ldg(&arrays.DSPairJg[pairns.y + pair]);
				const float4 kerderlap = __ldg(&arrays.DSKerDerLapg[pairns.y + pair]);
				const float4 dispphip2 = __ldg(&arrays.DSDispPhig[p2]);

				gradphi.x += (dispphip2.w - phfp1) * kerderlap.x;
				gradphi.y += (dispphip2.w - phfp1) * kerderlap.y;
				gradphi.z += (dispphip2.w - phfp1) * kerderlap.z;
			}
			energies.z = body.gc * (0.25f / body.lc * (1.0f - phfp1) * (1.0f - phfp1) + body.lc * cumath::DotVec3(gradphi, gradphi));
		}
		return energies;
	}

	template<bool simulate2d>
	//__launch_bounds__(SPHBSIZE, 3)
	__global__ void DSCalcEnergiesCauchyStressKernel(const int np, tfloat3* totEnergy, tsymatrix3f* cauchystress,
		const StDeformStrucIntData* __restrict__ DeformStrucDatag, const StDeformStrucIntArraysg arrays,
		float* __restrict__ DSKerSumVolg, tfloat3* particleenergy)
	{
		const int p1 = blockIdx.x * blockDim.x + threadIdx.x;
		if (p1 >= np) return;
		const unsigned bodyid = CODE_GetIbodyDeformStruc(arrays.DSCodeg[p1]);
		const StDeformStrucIntData body = DeformStrucDatag[bodyid];


		float4 pk2d = __ldg(&arrays.DSPiolKirg2D[p1]);
		matrix3f piolakp1 = make_matrix3f(pk2d.x, 0.0f, pk2d.y, 0.0f, 0.0f, 0.0f, pk2d.z, 0.0f, pk2d.w);

		pk2d = __ldg(&arrays.DSDefGradg2D[p1]);
		matrix3f defgrad = make_matrix3f(pk2d.x, 0.0f, pk2d.y, 0.0f, 1.0f, 0.0f, pk2d.z, 0.0f, pk2d.w);

		float2 defpk;
		//float jacob;
		if (simulate2d) {
			//jacob = cumath::Determinant2x2(defgrad);S
		}
		else {
			pk2d = __ldg(&arrays.DSPiolKirg3D[p1]);
			defpk = __ldg(&arrays.DSDefPk[p1]);

			piolakp1.a12 = pk2d.x; piolakp1.a21 = pk2d.y;
			piolakp1.a23 = pk2d.z; piolakp1.a32 = pk2d.w;
			piolakp1.a22 = defpk.y;

			pk2d = __ldg(&arrays.DSDefGradg3D[p1]);

			defgrad.a12 = pk2d.x; defgrad.a21 = pk2d.y;
			defgrad.a23 = pk2d.z; defgrad.a32 = pk2d.w;
			defgrad.a22 = defpk.x;

			//jacob = cumath::Determinant3x3(defgrad);
		}

		const float jac = cumath::Determinant3x3(defgrad);
		matrix3f cauchy = (1.0 / jac) * cumath::MulMatrix3x3(piolakp1, cumath::TrasMatrix3x3(defgrad));
		cauchystress[p1] = { cauchy.a11, cauchy.a12, cauchy.a13, cauchy.a22, cauchy.a23, cauchy.a33 };

		const float4 dispphi = __ldg(&arrays.DSDispPhig[p1]);
		const float4 vel = __ldg(&arrays.DSVelg[p1]);
		uint2 pairnsp1 = __ldg(&arrays.DSPairNSg[p1]);
		const float eqplastic = (arrays.DSEqPlasticg ? arrays.DSEqPlasticg[p1] : 0.0f);
		const tmatrix3f plastic_devp1 = (arrays.DSPlasticStraing ? arrays.DSPlasticStraing[p1] : make_tmatrix3f(0.0f));

		tfloat3 energies = DScalcenergiesp1<simulate2d>(defgrad, vel, dispphi.w, pairnsp1, arrays, eqplastic, plastic_devp1, body);
		float truncvol = DSKerSumVolg[p1];
		energies.x *= body.vol0 * body.vol0 / truncvol;
		energies.y *= truncvol;
		energies.z *= body.vol0 * body.vol0 / truncvol;

		if (particleenergy) particleenergy[p1] = energies;

		atomicAdd(&totEnergy[bodyid].x, energies.x);
		atomicAdd(&totEnergy[bodyid].y, energies.y);
		atomicAdd(&totEnergy[bodyid].z, energies.z);

	}

	__global__ void DSCauchyDiff(const int np, const tsymatrix3f* cauchystress, tsymatrix3f* cauchydiff,
		const StDeformStrucIntData* __restrict__ DeformStrucDatag, const StDeformStrucIntArraysg arrays)
	{
		const int p1 = blockIdx.x * blockDim.x + threadIdx.x;
		if (p1 >= np) return;

		uint2 pairnsp1 = __ldg(&arrays.DSPairNSg[p1]);
		tsymatrix3f cauchyp1 = cauchystress[p1];
		tsymatrix3f cauchydiffp1; cauchydiffp1.xx = cauchydiffp1.xy = cauchydiffp1.xz = cauchydiffp1.yy =
			cauchydiffp1.yz = cauchydiffp1.zz = 0.0f;
		float kersum = 0.0f;
#pragma unroll 8
		for (unsigned pair = 0; pair < pairnsp1.x; ++pair)
		{
			const unsigned p2 = __ldg(&arrays.DSPairJg[pairnsp1.y + pair]);
			const float kerd = __ldg(&arrays.DSKerg[pairnsp1.y + pair]);
			const tsymatrix3f cauchyp2 = cauchystress[p2];

			cauchydiffp1.xx = cauchydiffp1.xx - (cauchyp1.xx - cauchyp2.xx) * kerd;
			cauchydiffp1.yy = cauchydiffp1.yy - (cauchyp1.yy - cauchyp2.yy) * kerd;
			cauchydiffp1.zz = cauchydiffp1.zz - (cauchyp1.zz - cauchyp2.zz) * kerd;

			cauchydiffp1.xy = cauchydiffp1.xy - (cauchyp1.xy - cauchyp2.xy) * kerd;
			cauchydiffp1.yz = cauchydiffp1.yz - (cauchyp1.yz - cauchyp2.yz) * kerd;
			cauchydiffp1.xz = cauchydiffp1.xz - (cauchyp1.xz - cauchyp2.xz) * kerd;

			kersum = kersum + kerd;
		}
		if (kersum > ALMOSTZERO) {
			cauchydiffp1.xx = cauchydiffp1.xx / kersum;
			cauchydiffp1.yy = cauchydiffp1.yy / kersum;
			cauchydiffp1.zz = cauchydiffp1.zz / kersum;

			cauchydiffp1.xy = cauchydiffp1.xy / kersum;
			cauchydiffp1.yz = cauchydiffp1.yz / kersum;
			cauchydiffp1.xz = cauchydiffp1.xz / kersum;
		}

		cauchydiff[p1] = cauchydiffp1;
	}

	__global__ void DSSmoothCauchy(const int np, tsymatrix3f* cauchystress, const tsymatrix3f* cauchydiff,
		const StDeformStrucIntData* __restrict__ DeformStrucDatag, const StDeformStrucIntArraysg arrays)
	{
		const int p1 = blockIdx.x * blockDim.x + threadIdx.x;
		if (p1 >= np) return;
		tsymatrix3f cauchyp1 = cauchystress[p1];
		tsymatrix3f cauchydiffp1 = cauchydiff[p1];
		tsymatrix3f cauchyp1new;
		const float factor = 0.5f;

		cauchyp1new.xx = cauchyp1.xx + cauchydiffp1.xx * factor;
		cauchyp1new.yy = cauchyp1.yy + cauchydiffp1.yy * factor;
		cauchyp1new.zz = cauchyp1.zz + cauchydiffp1.zz * factor;

		cauchyp1new.xy = cauchyp1.xy + cauchydiffp1.xy * factor;
		cauchyp1new.yz = cauchyp1.yz + cauchydiffp1.yz * factor;
		cauchyp1new.xz = cauchyp1.xz + cauchydiffp1.xz * factor;

		cauchystress[p1] = cauchyp1new;
	}

	void DSCalcEnergiesCauchyStress(const int np, unsigned bodycnt, const StDeformStrucIntData* DeformStrucDatag,
		const StDeformStrucIntArraysg DSIntArraysg, const bool simulate2d, tfloat3* energies, tsymatrix3f* dscauchystress,
		float* __restrict__ DSKerSumVolg, tfloat3* particleenergies)
	{
		tfloat3* d_totEnergy;
		tsymatrix3f* d_cauchystress, * d_cauchdiff;
		cudaMalloc(&d_totEnergy, bodycnt * sizeof(tfloat3));
		cudaMemset(d_totEnergy, 0, bodycnt * sizeof(tfloat3));

		cudaMalloc(&d_cauchystress, np * sizeof(tsymatrix3f));
		cudaMemset(d_cauchystress, 0, np * sizeof(tsymatrix3f));

		cudaMalloc(&d_cauchdiff, np * sizeof(tsymatrix3f));
		cudaMemset(d_cauchdiff, 0, np * sizeof(tsymatrix3f));

		tfloat3* d_particleEnergy = nullptr;
		if (particleenergies) {
			cudaMalloc(&d_particleEnergy, np * sizeof(tfloat3));
			cudaMemset(d_particleEnergy, 0, np * sizeof(tfloat3));
		}

		dim3 sgridb = GetSimpleGridSize(np, SPHBSIZE);
		if (simulate2d)
			DSCalcEnergiesCauchyStressKernel<true> << <sgridb, SPHBSIZE >> > (np, d_totEnergy, d_cauchystress, DeformStrucDatag, DSIntArraysg, DSKerSumVolg, d_particleEnergy);
		else
			DSCalcEnergiesCauchyStressKernel<false> << <sgridb, SPHBSIZE >> > (np, d_totEnergy, d_cauchystress, DeformStrucDatag, DSIntArraysg, DSKerSumVolg, d_particleEnergy);
		//cudaDeviceSynchronize();
		/*DSCauchyDiff << <sgridb, SPHBSIZE >> > (np, d_cauchystress, d_cauchdiff, DeformStrucDatag, DSIntArraysg);
		cudaDeviceSynchronize();
		DSSmoothCauchy << <sgridb, SPHBSIZE >> > (np, d_cauchystress, d_cauchdiff, DeformStrucDatag, DSIntArraysg);
		cudaDeviceSynchronize();*/
		cudaMemcpy(energies, d_totEnergy, bodycnt * sizeof(tfloat3), cudaMemcpyDeviceToHost);
		cudaMemcpy(dscauchystress, d_cauchystress, np * sizeof(tsymatrix3f), cudaMemcpyDeviceToHost);
		if (particleenergies && d_particleEnergy) cudaMemcpy(particleenergies, d_particleEnergy, np * sizeof(tfloat3), cudaMemcpyDeviceToHost);

		cudaFree(d_totEnergy);
		cudaFree(d_cauchystress);
		cudaFree(d_cauchdiff);
		if (d_particleEnergy) cudaFree(d_particleEnergy);
	}







	__global__ void CheckExpressionValuesKernel(
		const JUserExpressionListGPU* expr_list, float* results, const unsigned expression_id, const int num_particles, const float time, const float dt, float dp)
	{
		const int idx = blockIdx.x * blockDim.x + threadIdx.x;

		if (idx < num_particles) {
			float result = NAN;

			if (expr_list) {
				JUserExpressionGPU* expr = expr_list->GetById(expression_id);

				if (expr) {
					float3 pos;
					pos.x = 0.5 / (num_particles)*idx;
					pos.y = 0.0;
					pos.z = 0.0;
					bool skip;
					result = expr->Evaluate(pos, pos, make_float3(0.0, 0.0, 0.0), pos.x, 0, dp, skip);
				}
			}

			if (results) results[idx] = result;
		}
	}


	void CheckExpressionValues(const JUserExpressionListGPU* d_expr_list, float* d_results, const int expression_id,
		const float time, const float dt, int num_particles, float dp)
	{
		const int block_size = 256;
		const int grid_size = (num_particles + block_size - 1) / block_size;

		CheckExpressionValuesKernel << <grid_size, block_size >> > (d_expr_list, d_results, expression_id, num_particles, time, dt, dp);

		cudaGetLastError();
		cudaDeviceSynchronize();
	}


	void ValidateExpressions(JUserExpressionListGPU* expr_list, JUserExpressionList* UserExpressions, const int expid, const float time, StDeformStrucData* defstrucdatac) {

		const StDeformStrucData body = defstrucdatac[0];
		float* d_results;
		const int num_particles = 100;
		cudaMalloc(&d_results, num_particles * sizeof(float));
		float dt = time / 100.0;
		CheckExpressionValues(expr_list, d_results, expid, time, dt, num_particles, body.dp);

		float* h_results = new float[num_particles];
		cudaMemcpy(h_results, d_results,
			num_particles * sizeof(float),
			cudaMemcpyDeviceToHost);

		for (int i = 0; i < num_particles; i++) {
			if (isnan(h_results[i])) {
				printf("Error in particle %d: Invalid result\n", i);
			}
			else {
				float x = 0.5 / (num_particles)*i;
				float z = 0.0;
				float y = 0.0;

				tfloat3 pos = TFloat3(x, y, z);
				tfloat3 disp = TFloat3(x, y, z);
				bool skip;
				printf("%d: %e \t %e\t %e\n", i, UserExpressions->GetById(1)->Evaluate(pos, pos, disp, x, 0, body.dp, skip), h_results[i], body.dp);
			}
		}

		for (float time_t = 0.0; time_t <= time; time_t = time_t + dt) {
			float x = 0.5 * time_t;
			float z = 0.0;
			float y = 0.0;
			tfloat3 pos = TFloat3(x, y, z);
			tfloat3 disp = TFloat3(x, y, z);
			bool skip;
			printf("%f:\t%e\n", time_t, UserExpressions->GetById(1)->Evaluate(pos, pos, disp, time_t, dt, 0.0, skip));
		}
		NAQIB_IS_HERE;
		delete[] h_results;
		cudaFree(d_results);
		NAQIB_IS_HERE;
		exit(0);
	}
}

