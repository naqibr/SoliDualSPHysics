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

 ---

 <SOLIDUALSPHYSICS> SoliDualSPHysics Extensions:
 Copyright (c) 2024 by Dr. Naqib Rahimi and Dr. Georgios Moutsanidis.
 
 This file implements user-defined expression parsing for boundary conditions
 in deformable structures, enabling time and space-dependent forcing functions.
 
 Developed as part of the PhD thesis:
 "Computational Mechanics of Extreme Events: Advanced Multi-physics Modeling and 
 Simulations with Smoothed Particle Hydrodynamics, Isogeometric Analysis, and Phase Field"
 by Dr. Naqib Rahimi, supervised by Dr. Georgios Moutsanidis.
 
 Related publication:
 Rahimi, N., & Moutsanidis, G. (2026). "SoliDualSPHysics: An extension of DualSPHysics 
 to solid mechanics with hyperelasticity, plasticity, and fracture." [In preparation]
 
 These extensions are distributed under the same GNU LGPL v2.1+ license as DualSPHysics.
*/

/// \file JSphCpu_ExpressionParser.h \brief Declares the class \ref JUserExpression.
/// 
/// This file implements a flexible expression parser for user-defined boundary conditions
/// and forcing functions in solid mechanics simulations.
/// 
/// \authors Dr. Naqib Rahimi, Dr. Georgios Moutsanidis

#ifndef _JSphCpuExpressionParser_
#define _JSphCpuExpressionParser_

#include <functional>

#include "DualSphDef.h"
#include "JObject.h"
#include "JLog2.h"
#include "JXml.h"
#include "JSphMk.h"
#include "JAppInfo.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <stack>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <sstream>
#include <set>
#include <memory>
#include <cstring> 

class JUserExpression : protected JObject
{

private:
	const std::set<std::string> COMP_OPS = { "<=", ">=", "==", "!=", "<", ">", "and", "or" };
	
	const std::set<std::string> RESERVED_FUNCTIONS{
	"sin", "cos", "tan", "sinh", "cosh", "tanh",
	"cot", "coth", "sqrt", "log", "ln", "pow", "if","abs"
	};

	const std::unordered_map<std::string, int> OP_PRECEDENCE{
		{"if", 0}, {"(", 0}, {")", 0}, {"{", 0}, {"}", 0},
		{"or", 1}, {"and", 2},
		{"<", 3}, {">", 3}, {"<=", 3}, {">=", 3}, {"==", 3}, {"!=", 3},
		{"+", 4}, {"-", 4}, {"*", 5}, {"/", 5}, {"^", 6}
	};
	const std::unordered_map<std::string, bool> OP_ASSOCIATIVITY = {
		{"^",  false}, {"*",  true}, {"/",  true}, {"+",  true}, {"-",  true}, {"and", true}, {"or", true}
	};
	const std::unordered_map<std::string, int> FUNC_ARGS = {
		{"sin", 1}, {"cos", 1}, {"tan", 1}, {"sinh", 1}, {"cosh", 1},
		{"tanh", 1}, {"cot", 1}, {"coth", 1}, {"sqrt", 1}, {"log", 1},
		{"ln", 1}, {"pow", 2}, {"if", 3}, {"abs",1}
	};
	std::string expr_;
	static thread_local const double* var_ptr_;
	static const double SKIP_SENTINEL;
	enum VarIndex { IDX_X0 = 0, IDX_Y0, IDX_Z0, IDX_X , IDX_Y, IDX_Z , IDX_UX, IDX_UY, IDX_UZ, IDX_T, IDX_DT, IDX_DX};

	void ValidateReservedNames();

	void Tokenize();

	void ParseToRPN();

	//static void JumpTo(size_t new_pc) {
	//	execution_pc_ = new_pc;
	//}

	double EvaluateRPN();

	void CompileRPNToOps();
public:
	unsigned id;
	enum TokenType {
		NUMBER, VARIABLE, OPERATOR, FUNCTION,
		LEFT_PAREN, RIGHT_PAREN, COMP_OP, COMMA, UNARY_OP
	};

	struct Token {
		TokenType type;
		std::string value;
		size_t pos;
	};
	std::unordered_map<std::string, double> locals_;
	std::vector<Token> tokens_;
	std::vector<Token> rpn_;
	JUserExpression(const std::string& expr,
		const std::unordered_map<std::string, double>& locals, const unsigned id);

	double Evaluate(tdouble3 pos0, tdouble3 pos, tdouble3 disp, double t, double dt, double dx, bool& skip);
	float Evaluate(tfloat3 pos0, tfloat3 pos, tfloat3 disp, float t, float dt, float dx, bool& skip);
	std::vector<std::function<void(std::vector<double>&, size_t&)>> pre_compiled_ops_;
	const std::vector<Token>& getRPN() const { return rpn_; }
};

class JUserExpressionList : protected JObject
{
private:
	JLog2* Log;

	void LoadXml(const JXml* sxml, const std::string& place);
	void ReadXml(const JXml* sxml, TiXmlElement* lis);
	std::unordered_map<std::string, double> UserExpParseLocals(const std::string& str) const;
	bool isValidMathExpr(const std::string& in);
	void Reset();
	unsigned count;
public:
	std::unordered_map<unsigned, JUserExpression*> UserExpMap;					//Mapped list of user mathematical expressions
	std::vector<JUserExpression*> UserExpList;									///<List of user mathematical expressions
	JUserExpressionList(JXml* sxml, const std::string& place, const JSphMk* mkinfo);
	~JUserExpressionList();

	//Returns an expression by id
	JUserExpression* GetById(unsigned id) const {
		auto it = UserExpMap.find(id);
		return (it != UserExpMap.end()) ? it->second : nullptr;
	}
	unsigned GetCount()const { return(unsigned(UserExpList.size())); }

};

#endif


