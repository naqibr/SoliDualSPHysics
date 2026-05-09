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

/// \file JSphGpu_ExpressionParser.cu \brief implements the class \ref JUserExpressionGPU.

#include "JSphGpu_ExpressionParser.h"
#include <vector>
#include <unordered_map>
#include <stack>

#define CUDA_CHECK(err) do { \
    if(err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at %s:%d\n", \
        cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

static __host__ JUserExpressionGPU* CreateDeviceExpression(const JUserExpression& cpuExpr) {

    std::vector<GPUOperation> cpu_ops;
    std::vector<float> locals;
    std::unordered_map<std::string, int> local_indices;

    for (const auto& kv : cpuExpr.locals_) {
        local_indices[kv.first] = static_cast<int>(locals.size());
        locals.push_back(static_cast<float>(kv.second));
    }

    const auto& rpn = cpuExpr.getRPN();
    for (const auto& token : rpn) {
        switch (token.type) {
        case JUserExpression::NUMBER: {
            float num = static_cast<float>(std::stod(token.value));
            cpu_ops.push_back(GPUOperation::MakeNumber(num));
            break;
        }

        case JUserExpression::VARIABLE: {
            if (token.value == "x0") {
                cpu_ops.push_back({ OP_VAR_X0g });
            }
            else if (token.value == "y0") {
                cpu_ops.push_back({ OP_VAR_Y0g });
            }
            else if (token.value == "z0") {
                cpu_ops.push_back({ OP_VAR_Z0g });
            }
            else if (token.value == "x") {
                cpu_ops.push_back({ OP_VAR_Xg });
            }
            else if (token.value == "y") {
                cpu_ops.push_back({ OP_VAR_Yg });
            }
            else if (token.value == "z") {
                cpu_ops.push_back({ OP_VAR_Zg });
            }
            else if (token.value == "ux") {
                cpu_ops.push_back({ OP_VAR_UXg });
            }
            else if (token.value == "uy") {
                cpu_ops.push_back({ OP_VAR_UYg });
            }
            else if (token.value == "uz") {
                cpu_ops.push_back({ OP_VAR_UZg });
            }
            else if (token.value == "t") {
                cpu_ops.push_back({ OP_VAR_Tg });
            }
            else if (token.value == "dt") {
                cpu_ops.push_back({ OP_VAR_DTg });
            }
            else if (token.value == "dx") {
                cpu_ops.push_back({ OP_VAR_DXg });
            }
            else if (token.value == "skip" || token.value == "SKIP") {
                cpu_ops.push_back(GPUOperation::MakeNumber(SKIP_SENTINELg));
            }
            else {
                if (!local_indices.count(token.value)) {
                    throw std::runtime_error("Undefined variable: " + token.value);
                }
                cpu_ops.push_back(GPUOperation::MakeLocal(local_indices[token.value]));
            }
            break;
        }

        case JUserExpression::OPERATOR: {
            GPUOpType op;
            switch (token.value[0]) {
            case '+': op = OP_ADDg; break;
            case '-': op = OP_SUBg; break;
            case '*': op = OP_MULg; break;
            case '/': op = OP_DIVg; break;
            case '^': op = OP_POWg; break;
            default: throw std::runtime_error("Unknown operator: " + token.value);
            }
            cpu_ops.push_back({ op });
            break;
        }

        case JUserExpression::UNARY_OP: {
            GPUOpType op;
            if (token.value == "+") {
                op = OP_UNARY_PLUSg;
            }
            else if (token.value == "-") {
                op = OP_UNARY_MINUSg;
            }
            else {
                throw std::runtime_error("Unknown unary operator: " + token.value);
            }
            cpu_ops.push_back({ op });
            break;
        }

        case JUserExpression::FUNCTION: {
            if (token.value == "if") {
                cpu_ops.push_back(GPUOperation::MakeIf());
                break;
            }
            if (token.value == "pow") {
                cpu_ops.push_back({ OP_POWg });
                break;
            }
            GPUOpType op;
            if (token.value == "sin") op = OP_SINg;
            else if (token.value == "cos") op = OP_COSg;
            else if (token.value == "tan") op = OP_TANg;
            else if (token.value == "sinh") op = OP_SINHg;
            else if (token.value == "abs") op = OP_ABSg;
            else if (token.value == "cosh") op = OP_COSHg;
            else if (token.value == "tanh") op = OP_TANHg;
            else if (token.value == "cot") op = OP_COTg;
            else if (token.value == "coth") op = OP_COTHg;
            else if (token.value == "sqrt") op = OP_SQRTg;
            else if (token.value == "log") op = OP_LOG10g;
            else if (token.value == "ln") op = OP_LOGg;
            else throw std::runtime_error("Unsupported function: " + token.value);

            cpu_ops.push_back({ op });
            break;
        }

        case JUserExpression::COMP_OP: {
            GPUOpType op;
            if (token.value == "<") op = OP_COMP_LTg;
            else if (token.value == ">") op = OP_COMP_GTg;
            else if (token.value == "<=") op = OP_COMP_LEg;
            else if (token.value == ">=") op = OP_COMP_GEg;
            else if (token.value == "==") op = OP_COMP_EQg;
            else if (token.value == "!=") op = OP_COMP_NEg;
            else if (token.value == "and") op = OP_LOGICAL_ANDg;
            else if (token.value == "or") op = OP_LOGICAL_ORg;
            else throw std::runtime_error("Unsupported comparison: " + token.value);
            cpu_ops.push_back({ op });
            break;
        }

        default:
            throw std::runtime_error("Unsupported token type in RPN");
        }
    }

    JUserExpressionGPU* d_expr;
    CUDA_CHECK(cudaMalloc(&d_expr, sizeof(JUserExpressionGPU)));

    JUserExpressionGPU h_expr;
    h_expr.id = cpuExpr.id;
    h_expr.num_ops = static_cast<int>(cpu_ops.size());
    h_expr.num_locals = static_cast<int>(locals.size());

    CUDA_CHECK(cudaMalloc(&h_expr.d_ops, h_expr.num_ops * sizeof(GPUOperation)));
    CUDA_CHECK(cudaMemcpy(h_expr.d_ops, cpu_ops.data(),
        h_expr.num_ops * sizeof(GPUOperation),
        cudaMemcpyHostToDevice));

    if (h_expr.num_locals > 0) {
        CUDA_CHECK(cudaMalloc(&h_expr.d_locals, h_expr.num_locals * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(h_expr.d_locals, locals.data(),
            h_expr.num_locals * sizeof(float),
            cudaMemcpyHostToDevice));
    }
    else {
        h_expr.d_locals = nullptr;
    }

    CUDA_CHECK(cudaMemcpy(d_expr, &h_expr, sizeof(JUserExpressionGPU),
        cudaMemcpyHostToDevice));

    return d_expr;
}

JUserExpressionListGPU::JUserExpressionListGPU(const JUserExpressionList& cpuList) {
    count = cpuList.GetCount();

    JUserExpressionGPU** h_expressions = new JUserExpressionGPU * [count];
    unsigned* h_ids = new unsigned[count];

    for (int i = 0; i < count; i++) {
        h_expressions[i] = CreateDeviceExpression(*cpuList.UserExpList[i]);
        h_ids[i] = cpuList.UserExpList[i]->id;
    }

    CUDA_CHECK(cudaMalloc(&d_expressions, count * sizeof(JUserExpressionGPU*)));
    CUDA_CHECK(cudaMalloc(&d_ids, count * sizeof(unsigned)));

    CUDA_CHECK(cudaMemcpy(d_expressions, h_expressions,
        count * sizeof(JUserExpressionGPU*),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ids, h_ids, count * sizeof(unsigned),
        cudaMemcpyHostToDevice));

    delete[] h_expressions;
    delete[] h_ids;
}

JUserExpressionListGPU::~JUserExpressionListGPU() {
    if (d_expressions) {
        JUserExpressionGPU** h_expressions = new JUserExpressionGPU * [count];
        CUDA_CHECK(cudaMemcpy(h_expressions, d_expressions,
            count * sizeof(JUserExpressionGPU*),
            cudaMemcpyDeviceToHost));

        for (int i = 0; i < count; i++) {
            if (h_expressions[i]) {
                JUserExpressionGPU h_expr;
                CUDA_CHECK(cudaMemcpy(&h_expr, h_expressions[i],
                    sizeof(JUserExpressionGPU),
                    cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaFree(h_expr.d_ops));
                if (h_expr.d_locals) CUDA_CHECK(cudaFree(h_expr.d_locals));
                CUDA_CHECK(cudaFree(h_expressions[i]));
            }
        }
        delete[] h_expressions;
        CUDA_CHECK(cudaFree(d_expressions));
    }
    if (d_ids) CUDA_CHECK(cudaFree(d_ids));
}