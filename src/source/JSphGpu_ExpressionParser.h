//HEAD_DSPH
/*
 <DUALSPHYSICS>  Copyright (c) 2021 by Dr Jose M. Dominguez et al. (see http://dual.sphysics.org/index.php/developers/).

 EPHYSLAB Environmental Physics Laboratory, Universidade de Vigo, Ourense, Spain.
 School of Mechanical, Aerospace and Civil Engineering, University of Manchester, Manchester, U.K.

 This file is part of DualSPHysics.

 DualSPHysics is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License
 as published by the Free Software Foundation; either version 2.1 of the License, or (at your option) any later version.

 DualSPHysics is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

 You should have received a copy of the GNU Lesser General Public License along with DualSPHysics. If not, see <http://www.gnu.org/licenses/>.
*/

/// \file JSphGpu_ExpressionParser.h \brief Declares the class \ref JUserExpression.

#ifndef _JSphGpuExpressionParser_
#define _JSphGpuExpressionParser_

#include "JSphCpu_ExpressionParser.h"
#include <cuda_runtime.h>

enum GPUOpType {
    OP_NUMBERg, OP_VAR_X0g, OP_VAR_Y0g, OP_VAR_Z0g, OP_VAR_Xg, OP_VAR_Yg, 
    OP_VAR_Zg, OP_VAR_UXg, OP_VAR_UYg, OP_VAR_UZg, OP_VAR_Tg, OP_VAR_DTg, OP_VAR_DXg,
    OP_LOCALg, OP_ADDg, OP_SUBg, OP_MULg, OP_DIVg, OP_POWg,
    OP_SINg, OP_COSg, OP_TANg, OP_SINHg, OP_COSHg, OP_TANHg, OP_COTg, OP_COTHg, OP_SQRTg, 
    OP_LOG10g, OP_LOGg, OP_COMP_LTg, OP_COMP_GTg, OP_COMP_LEg, OP_COMP_GEg, OP_COMP_EQg, OP_COMP_NEg,
    OP_LOGICAL_ANDg, OP_LOGICAL_ORg,
    OP_IFg, OP_ABSg, OP_UNARY_PLUSg, OP_UNARY_MINUSg
};

struct GPUOperation {
    GPUOpType type;
    union {
        float constant;
        int local_index;
        int jump_target;
    };
    __host__ __device__ static GPUOperation MakeIf() { return GPUOperation(OP_IFg); }
    __host__ __device__ GPUOperation(GPUOpType t) : type(t) {}
    __host__ __device__ static GPUOperation MakeNumber(float val) {
        GPUOperation op(OP_NUMBERg);
        op.constant = val;
        return op;
    }

    __host__ __device__ static GPUOperation MakeLocal(int idx) {
        GPUOperation op(OP_LOCALg);
        op.local_index = idx;
        return op;
    }
};

struct JUserExpressionGPU {
    GPUOperation* d_ops;
    float* d_locals;
    int num_ops;
    int num_locals;
    unsigned id;

    __device__ inline float Evaluate(float3 pos0, float3 pos, float3 disp,
        float t, float dt, float dx,
        bool& skip) const
    {
        const int MAX_STACK = 128;
        float stack_local[MAX_STACK];
        int   sp = -1;
        skip = false;

#define TRY_PUSH(v) do {                                  \
            if (sp + 1 >= MAX_STACK) {                            \
                sp = -1;                                          \
                return NAN;                                       \
            }                                                     \
            stack_local[++sp] = (v);                              \
        } while (0);

#define POP() (sp < 0 ? ([](){ union { unsigned i; float f; } u{0x7fffffff}; return u.f; }()) : stack_local[sp--])

        int pc = 0;
        while (pc < num_ops) {
            const GPUOperation& op = d_ops[pc++];

            switch (op.type) {
            case OP_NUMBERg: TRY_PUSH(op.constant); break;
            case OP_VAR_X0g: TRY_PUSH(pos0.x);      break;
            case OP_VAR_Y0g: TRY_PUSH(pos0.y);      break;
            case OP_VAR_Z0g: TRY_PUSH(pos0.z);      break;
            case OP_VAR_Xg:  TRY_PUSH(pos.x);       break;
            case OP_VAR_Yg:  TRY_PUSH(pos.y);       break;
            case OP_VAR_Zg:  TRY_PUSH(pos.z);       break;
            case OP_VAR_UXg: TRY_PUSH(disp.x);      break;
            case OP_VAR_UYg: TRY_PUSH(disp.y);      break;
            case OP_VAR_UZg: TRY_PUSH(disp.z);      break;
            case OP_VAR_Tg:  TRY_PUSH(t);           break;
            case OP_VAR_DTg: TRY_PUSH(dt);          break;
            case OP_VAR_DXg: TRY_PUSH(dx);          break;

            case OP_LOCALg:
                TRY_PUSH(d_locals ? d_locals[op.local_index] : 0.0f);
                break;
            case OP_ABSg: {
                float a = POP();
                TRY_PUSH(fabsf(a));
            } break;
            case OP_ADDg: { float b = POP(), a = POP(); TRY_PUSH(a + b); } break;
            case OP_SUBg: { float b = POP(), a = POP(); TRY_PUSH(a - b); } break;
            case OP_MULg: { float b = POP(), a = POP(); TRY_PUSH(a * b); } break;
            case OP_DIVg: { float b = POP(), a = POP(); TRY_PUSH(a / b); } break;
            case OP_POWg: { float b = POP(), a = POP(); TRY_PUSH(powf(a, b)); } break;

            case OP_SINg: { float a = POP(); TRY_PUSH(sinf(a)); } break;
            case OP_COSg: { float a = POP(); TRY_PUSH(cosf(a)); } break;
            case OP_TANg: { float a = POP(); TRY_PUSH(tanf(a)); } break;
            case OP_SINHg: { float a = POP(); TRY_PUSH(sinhf(a)); } break;
            case OP_COSHg: { float a = POP(); TRY_PUSH(coshf(a)); } break;
            case OP_TANHg: { float a = POP(); TRY_PUSH(tanhf(a)); } break;
            case OP_COTg: { float a = POP(); TRY_PUSH(1.0f / tanf(a)); } break;
            case OP_COTHg: { float a = POP(); TRY_PUSH(1.0f / tanhf(a)); } break;
            case OP_SQRTg: { float a = POP(); TRY_PUSH(sqrtf(a)); } break;
            case OP_LOG10g: { float a = POP(); TRY_PUSH(log10f(a)); } break;
            case OP_LOGg: { float a = POP(); TRY_PUSH(logf(a)); } break;

            case OP_COMP_LTg: {
                float b = POP(), a = POP();
                TRY_PUSH(a < b ? 1.0f : 0.0f);
            } break;
            case OP_COMP_GTg: {
                float b = POP(), a = POP();
                TRY_PUSH(a > b ? 1.0f : 0.0f);
            } break;
            case OP_COMP_LEg: {
                float b = POP(), a = POP();
                TRY_PUSH(a <= b ? 1.0f : 0.0f);
            } break;
            case OP_COMP_GEg: {
                float b = POP(), a = POP();
                TRY_PUSH(a >= b ? 1.0f : 0.0f);
            } break;

            case OP_COMP_EQg: {
                float b = POP(), a = POP();
                TRY_PUSH(fabsf(a - b) <= ALMOSTZERO ? 1.0f : 0.0f);
            } break;
            case OP_COMP_NEg: {
                float b = POP(), a = POP();
                TRY_PUSH(fabsf(a - b) > ALMOSTZERO ? 1.0f : 0.0f);
            } break;

            case OP_LOGICAL_ANDg: {
                float b = POP(), a = POP();
                TRY_PUSH((a != 0.0f && b != 0.0f) ? 1.0f : 0.0f);
            } break;
            case OP_LOGICAL_ORg: {
                float b = POP(), a = POP();
                TRY_PUSH((a != 0.0f || b != 0.0f) ? 1.0f : 0.0f);
            } break;

            case OP_IFg: {
                float fval = POP();
                float tval = POP();
                float cval = POP();
                TRY_PUSH((cval > 0.1f) ? tval : fval);
            } break;

            case OP_UNARY_PLUSg: {
                // Unary plus: do nothing, value already on stack
            } break;

            case OP_UNARY_MINUSg: {
                float a = POP();
                TRY_PUSH(-a);
            } break;

            default:
                return NAN;
            }
        }

        float val = (sp >= 0) ? stack_local[sp] : NAN;
        skip = (val == SKIP_SENTINELg);
        return val;

#undef TRY_PUSH
#undef POP
    }
};

class JUserExpressionListGPU {
private:
    JUserExpressionGPU** d_expressions;
    unsigned* d_ids;
    int count;

public:
    __host__ JUserExpressionListGPU(const JUserExpressionList& cpuList);
    __host__ ~JUserExpressionListGPU();

    __device__ JUserExpressionGPU* GetById(unsigned id) const {
        for (int i = 0; i < count; i++) {
            if (d_ids[i] == id) return d_expressions[i];
        }
        return nullptr;
    }

    __device__ int GetCount() const {
        return count;
    }

    JUserExpressionListGPU(const JUserExpressionListGPU&) = delete;
    JUserExpressionListGPU& operator=(const JUserExpressionListGPU&) = delete;
};
#endif


