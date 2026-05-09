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

/// \file JSphCpu_ExpressionParser.cpp \brief Declares the class \ref JUserExpression.
#define ABSF 
#include "JSphCpu_ExpressionParser.h"
thread_local const double* JUserExpression::var_ptr_ = nullptr;
const double JUserExpression::SKIP_SENTINEL = -DBL_MAX;
JUserExpression::JUserExpression(const std::string& expr,
	const std::unordered_map<std::string, double>& locals, const unsigned _id)
	: expr_(expr), locals_(locals), id(_id)
{
	ValidateReservedNames();
	Tokenize();
	ParseToRPN();
	CompileRPNToOps();
}
//Evaluate(pos0p1, posp1, TDouble3(0.0, 0.0, 0.0), currenttime, 0.0f, body.dp, skip)
double JUserExpression::Evaluate(tdouble3 pos0, tdouble3 pos, tdouble3 disp, double t, double dt, double dx, bool& skip) {
	double vars[12] = { pos0.x, pos0.y, pos0.z, pos.x, pos.y, pos.z,disp.x, disp.y, disp.z, t, dt,dx };
	var_ptr_ = vars;
	double result = EvaluateRPN();
	skip = (result == SKIP_SENTINEL);
	return result;
}

float JUserExpression::Evaluate(tfloat3 pos0, tfloat3 pos, tfloat3 disp, float t, float dt, float dx, bool& skip) {
	double vars[12] = { pos0.x, pos0.y, pos0.z, pos.x, pos.y, pos.z,disp.x, disp.y, disp.z, t, dt, dx };
	var_ptr_ = vars;
	double result = EvaluateRPN();
	skip = (result == SKIP_SENTINEL);
	return static_cast<float>(result);
}

void JUserExpression::ValidateReservedNames() {
	for (const auto& kv : locals_) {
		const std::string& key = kv.first;
		if (RESERVED_FUNCTIONS.count(key) ||
			key == "x0" || key == "y0" || key == "z0" || key == "x" || key == "y" || key == "z" || 
			key == "ux" || key == "uy" || key == "uz" || key == "t" || key == "dt" || key == "dx" ||
			key == "skip" || key == "SKIP" || key == "and" || key == "or") {
			throw std::runtime_error("Reserved name used in local variables: " + key);
		}
	}
}

void JUserExpression::Tokenize() {
	size_t i = 0;
	Token lastToken{ TokenType::COMMA, "", 0 }; // dummy "operand separator"

	while (i < expr_.size()) {
		if (isspace(expr_[i])) { i++; continue; }

		// --- detect signed numbers like -0.145 or +3.14e-5 ---
		// Only treat +/- as part of a number literal if it's followed immediately by a digit or dot
		// AND we're in a context where a unary operator would be valid.
		// If +/- is followed by a variable or function name (e.g., -Velmax), fall through to 
		// process it as a separate UNARY_OP token in the switch statement below.
		bool canBeUnary = (
			tokens_.empty() ||
			lastToken.type == OPERATOR ||
			lastToken.type == UNARY_OP ||
			lastToken.type == COMP_OP ||
			lastToken.type == LEFT_PAREN ||
			lastToken.type == COMMA ||
			lastToken.type == FUNCTION
			);

		if ((expr_[i] == '+' || expr_[i] == '-') && canBeUnary) {
			// Look ahead: sign followed IMMEDIATELY by digit or dot (no space)
			if (i + 1 < expr_.size() && (isdigit(expr_[i + 1]) || expr_[i + 1] == '.')) {
				size_t start = i;
				i++; // skip sign
				bool seenE = false;
				while (i < expr_.size()) {
					char c = expr_[i];
					if ((c == 'e' || c == 'E') && !seenE) {
						seenE = true;
						i++;
						if (i < expr_.size() && (expr_[i] == '+' || expr_[i] == '-')) i++;
					}
					else if (isdigit(c) || c == '.') {
						i++;
					}
					else break;
				}
				tokens_.push_back({ NUMBER, expr_.substr(start, i - start), start });
				lastToken = tokens_.back();
				continue;
			}
			// If not followed by digit/dot, fall through to process as unary operator below
		}

		// --- 2-character comparison ops (<=, >=, ==, !=, or, and) ---
		if (i + 1 < expr_.size()) {
			std::string two = expr_.substr(i, 2);
			if (COMP_OPS.count(two)) {
				tokens_.push_back({ COMP_OP, two, i });
				lastToken = tokens_.back();
				i += 2;
				continue;
			}
		}

		std::string one(1, expr_[i]);
		if (COMP_OPS.count(one)) {
			tokens_.push_back({ COMP_OP, one, i });
			lastToken = tokens_.back();
			i++;
			continue;
		}

		// --- numbers starting with digit or '.' ---
		if (isdigit(expr_[i]) || expr_[i] == '.') {
			size_t start = i;
			bool seenE = false;
			while (i < expr_.size()) {
				char c = expr_[i];
				if ((c == 'e' || c == 'E') && !seenE) {
					seenE = true;
					i++;
					if (i < expr_.size() && (expr_[i] == '+' || expr_[i] == '-')) i++;
				}
				else if (isdigit(c) || c == '.') {
					i++;
				}
				else break;
			}
			tokens_.push_back({ NUMBER, expr_.substr(start, i - start), start });
			lastToken = tokens_.back();
			continue;
		}

		// --- identifiers / functions ---
		if (isalpha(expr_[i])) {
			size_t start = i;
			while (i < expr_.size() && (isalnum(expr_[i]) || expr_[i] == '_')) i++;
			std::string ident = expr_.substr(start, i - start);

			if (ident == "and" || ident == "or") {
				tokens_.push_back({ COMP_OP, ident, start });
			}
			else if (RESERVED_FUNCTIONS.count(ident)) {
				tokens_.push_back({ FUNCTION, ident, start });
			}
			else {
				tokens_.push_back({ VARIABLE, ident, start });
			}
			lastToken = tokens_.back();
			continue;
		}

		// --- single char operators / parens / comma ---
		switch (expr_[i]) {
		case '+': case '-': case '*': case '/': case '^': {
			// Check if +/- should be unary
			bool isUnary = false;
			if (expr_[i] == '+' || expr_[i] == '-') {
				isUnary = (tokens_.empty() ||
					lastToken.type == OPERATOR ||
					lastToken.type == UNARY_OP ||
					lastToken.type == COMP_OP ||
					lastToken.type == LEFT_PAREN ||
					lastToken.type == COMMA ||
					lastToken.type == FUNCTION);
			}
			tokens_.push_back({ isUnary ? UNARY_OP : OPERATOR, std::string(1, expr_[i]), i });
			break;
		}
		case '(':
			tokens_.push_back({ LEFT_PAREN, "(", i });
			break;
		case ')':
			tokens_.push_back({ RIGHT_PAREN, ")", i });
			break;
		case ',':
			tokens_.push_back({ COMMA, ",", i });
			break;
		default:
			throw std::runtime_error(
				std::string("Invalid character in expression: '")
				+ expr_[i] + "' at position " + std::to_string(i)
			);
		}
		lastToken = tokens_.back();
		i++;
	}
}


void JUserExpression::ParseToRPN() {
	std::stack<Token> op_stack;
	std::stack<int> function_arg_counts;

	for (const auto& token : tokens_) {
		switch (token.type) {
		case NUMBER:
		case VARIABLE:
			rpn_.push_back(token);
			// After outputting an operand, pop any pending unary operators
			// (they have highest precedence and bind immediately to the operand)
			while (!op_stack.empty() && op_stack.top().type == UNARY_OP) {
				rpn_.push_back(op_stack.top());
				op_stack.pop();
			}
			break;

		case FUNCTION:
			op_stack.push(token);
			function_arg_counts.push(1);
			break;

		case COMMA:
			while (!op_stack.empty() && op_stack.top().type != LEFT_PAREN) {
				rpn_.push_back(op_stack.top());
				op_stack.pop();
			}
			if (!function_arg_counts.empty()) function_arg_counts.top()++;
			break;

		case LEFT_PAREN:
			op_stack.push(token);
			break;

		case RIGHT_PAREN: {
			while (!op_stack.empty() && op_stack.top().type != LEFT_PAREN) {
				rpn_.push_back(op_stack.top());
				op_stack.pop();
			}
			if (op_stack.empty()) throw std::runtime_error("Mismatched parentheses");
			op_stack.pop();
			if (!op_stack.empty() && op_stack.top().type == FUNCTION) {
				Token func = op_stack.top();
				op_stack.pop();
				rpn_.push_back(func);
			}
			break;
		}

		case UNARY_OP:
			// Unary operators have highest precedence, push directly
			op_stack.push(token);
			break;

		default: {
			// Default handles OPERATOR and COMP_OP
			// Pop operators from stack while they have higher (or equal with left-assoc) precedence
			// NOTE: UNARY_OP should never be popped here (they have highest precedence)
			while (!op_stack.empty()) {
				const Token& top = op_stack.top();
				// Don't pop unary operators, functions, or left parens
				if (top.type == UNARY_OP || top.type == FUNCTION || top.type == LEFT_PAREN)
					break;
				// Check precedence for binary operators and comparison operators
				if (OP_PRECEDENCE.at(top.value) > OP_PRECEDENCE.at(token.value) ||
					(OP_PRECEDENCE.at(top.value) == OP_PRECEDENCE.at(token.value) &&
						OP_ASSOCIATIVITY.at(top.value)))
				{
					rpn_.push_back(top);
					op_stack.pop();
				}
				else break;
			}
			op_stack.push(token);
			break;
		}
		}
	}

	while (!op_stack.empty()) {
		rpn_.push_back(op_stack.top());
		op_stack.pop();
	}
}

double JUserExpression::EvaluateRPN() {
	std::vector<double> stack;
	size_t pc = 0;
	while (pc < pre_compiled_ops_.size()) {
		pre_compiled_ops_[pc](stack, pc);
	}
	return stack.empty() ? 0.0 : stack.back();
}

void JUserExpression::CompileRPNToOps() {

	for (size_t rpn_idx = 0; rpn_idx < rpn_.size(); rpn_idx++) {
		const auto& token = rpn_[rpn_idx];
		switch (token.type) {
		case NUMBER: {
			double num = std::stod(token.value);
			pre_compiled_ops_.push_back(
				[num](std::vector<double>& stack, size_t& pc) {
					stack.push_back(num);
					pc++;
				}
			);
			break;
		}
		case VARIABLE: {
			auto it = locals_.find(token.value);
			if (token.value == "x0" || token.value == "y0" ||
				token.value == "z0" || token.value == "x" || 
				token.value == "y" || token.value == "z" || 
				token.value == "t" || token.value == "dt"||
				token.value == "dx" ||
				token.value == "ux" || token.value == "uy" || 
				token.value == "uz")
			{
				int idx = (token.value == "x0" ? IDX_X0
					: token.value == "y0" ? IDX_Y0
					: token.value == "z0" ? IDX_Z0
					: token.value == "x" ? IDX_X
					: token.value == "y" ? IDX_Y
					: token.value == "z" ? IDX_Z
					: token.value == "ux" ? IDX_UX
					: token.value == "uy" ? IDX_UY
					: token.value == "uz" ? IDX_UZ
					: token.value == "t" ? IDX_T
					: token.value == "dt" ? IDX_DT
					: token.value == "dx" ? IDX_DX
					: IDX_DX);

				pre_compiled_ops_.push_back(
					[idx](std::vector<double>& stack, size_t& pc) {
						stack.push_back(var_ptr_[idx]);
						pc++;
					}
				);
			}
			else if (token.value == "skip" || token.value == "SKIP") {
				pre_compiled_ops_.push_back(
					[](std::vector<double>& stack, size_t& pc) {
						stack.push_back(JUserExpression::SKIP_SENTINEL);
						pc++;
					}
				);
			}
			else if (it != locals_.end()) {
				double val = it->second;
				pre_compiled_ops_.push_back(
					[val](std::vector<double>& stack, size_t& pc) {
						stack.push_back(val);
						pc++;
					}
				);
			}
			else {
				throw std::runtime_error("Undefined variable: " + token.value);
			}
			break;
		}
		case OPERATOR: {
			pre_compiled_ops_.push_back(
				[token](std::vector<double>& stack, size_t& pc) {
					double b = stack.back(); stack.pop_back();
					double a = stack.back(); stack.pop_back();

					switch (token.value[0]) {
					case '+': stack.push_back(a + b); break;
					case '-': stack.push_back(a - b); break;
					case '*': stack.push_back(a * b); break;
					case '/': stack.push_back(a / b); break;
					case '^': stack.push_back(std::pow(a, b)); break;
					default: throw std::runtime_error("Unknown operator: " + token.value);
					}
					pc++;
				}
			);
			break;
		}
		case UNARY_OP: {
			pre_compiled_ops_.push_back(
				[token](std::vector<double>& stack, size_t& pc) {
					if (stack.empty()) {
						throw std::runtime_error("Invalid expression: insufficient operands for unary operator " + token.value);
					}
					double a = stack.back(); stack.pop_back();

					if (token.value == "+") {
						stack.push_back(a);  // Unary plus does nothing
					}
					else if (token.value == "-") {
						stack.push_back(-a);  // Unary minus negates
					}
					else {
						throw std::runtime_error("Unknown unary operator: " + token.value);
					}
					pc++;
				}
			);
			break;
		}
		case FUNCTION: {
			int num_args = FUNC_ARGS.at(token.value);
			pre_compiled_ops_.push_back(
				[token, num_args](std::vector<double>& stack, size_t& pc) {
					std::vector<double> args(num_args);
					for (int i = num_args - 1; i >= 0; --i) {
						args[i] = stack.back();
						stack.pop_back();
					}

					if (token.value == "pow") stack.push_back(pow(args[0], args[1]));
					else if (token.value == "abs") stack.push_back(fabs(args[0]));
					else if (token.value == "sin") stack.push_back(sin(args[0]));
					else if (token.value == "cos") stack.push_back(cos(args[0]));
					else if (token.value == "tan") stack.push_back(tan(args[0]));
					else if (token.value == "sinh") stack.push_back(sinh(args[0]));
					else if (token.value == "cosh") stack.push_back(cosh(args[0]));
					else if (token.value == "tanh") stack.push_back(tanh(args[0]));
					else if (token.value == "cot") stack.push_back(1.0 / tan(args[0]));
					else if (token.value == "coth") stack.push_back(1.0 / tanh(args[0]));
					else if (token.value == "sqrt") stack.push_back(sqrt(args[0]));
					else if (token.value == "log") stack.push_back(log10(args[0]));
					else if (token.value == "ln") stack.push_back(log(args[0]));
					else if (token.value == "if") stack.push_back(args[0]> 0.1 ? args[1] : args[2]);
					pc++;
				}
			);
			break;
		}
		case COMP_OP: {
			pre_compiled_ops_.push_back(
				[token](std::vector<double>& stack, size_t& pc) {
					double b = stack.back(); stack.pop_back();
					double a = stack.back(); stack.pop_back();
					bool result = false;

					if (token.value == "<") result = a < b;
					else if (token.value == ">") result = a > b;
					else if (token.value == "<=") result = a <= b;
					else if (token.value == ">=") result = a >= b;
					else if (token.value == "==") result = a == b;
					else if (token.value == "!=") result = a != b;
					else if (token.value == "and") result = a && b;
					else if (token.value == "or") result = a || b;

					stack.push_back(result ? 1.0 : 0.0);
					pc++;
				}
			);
			break;
		}
		default:
			throw std::runtime_error("Unsupported operation in user expression");
		}
	}
}


JUserExpressionList::JUserExpressionList(JXml* sxml, const std::string& place, const JSphMk* mkinfo)
	:Log(AppInfo.LogPtr())
{
	ClassName = "JUserExpressionList";
	Reset();
	LoadXml(sxml, place);
}

//======================================================================================= =
/// Destructor.
//========================================================================================
JUserExpressionList::~JUserExpressionList() {
	DestructorActive = true;
	Reset();
}

//========================================================================================
/// Initialisation of variables.
//========================================================================================
void JUserExpressionList::Reset() {
	for (auto exprPtr : UserExpList) {
		delete exprPtr;
	}
	UserExpList.clear();
	UserExpMap.clear();
}

std::unordered_map<std::string, double> JUserExpressionList::UserExpParseLocals(const std::string& str) const 
{
	std::unordered_map<std::string, double> locals;
	std::istringstream iss(str);
	std::string pair;

	auto trim = [](const std::string& s) {
		const std::string whitespace = " \t\n\r";
		size_t start = s.find_first_not_of(whitespace);
		if (start == std::string::npos) return std::string();
		size_t end = s.find_last_not_of(whitespace);
		return s.substr(start, end - start + 1);
		};

	while (std::getline(iss, pair, ';')) {
		pair = trim(pair);
		if (pair.empty()) continue;

		size_t eq = pair.find('=');
		if (eq == std::string::npos)
			throw std::runtime_error("Invalid local (missing '='): " + pair);

		std::string key = trim(pair.substr(0, eq));
		std::string val = trim(pair.substr(eq + 1));

		if (key.empty())
			throw std::runtime_error("Empty key in local: " + pair);
		if (locals.count(key))
			throw std::runtime_error("Duplicate local variable: " + key);

		try {
			locals[key] = std::stod(val);
		}
		catch (...) {
			throw std::runtime_error("Invalid numeric value for local '" + key + "': " + val);
		}
	}
	return locals;
}

static const std::string kWhitespace = " \t\r\n";

static auto TrimExpression = [](const std::string& str) {
	const auto begin = str.find_first_not_of(kWhitespace);
	if (begin == std::string::npos) return std::string();
	const auto end = str.find_last_not_of(kWhitespace);
	return str.substr(begin, end - begin + 1);
	};
//========================================================================================
/// Loads conditions of XML object.
//========================================================================================
void JUserExpressionList::LoadXml(const JXml* sxml, const std::string& place) 
{
	TiXmlNode* node = sxml->GetNodeSimple(place);
	if (!node)Run_Exceptioon(std::string("Cannot find the element \'") + place + "\'.");

	if (sxml->CheckNodeActive(node))ReadXml(sxml, node->ToElement());

}

// Returns true if `in` looks like a valid mathematical expression.
// Accepts: numbers (with optional exponent), identifiers, () , , and operators.
// Rejects: empty/whitespace-only, operator-only, dangling operator, bad parens/commas.
bool JUserExpressionList::isValidMathExpr(const std::string& in) {
	static auto IsIdentStart = [](char c) {
		return (std::isalpha(static_cast<unsigned char>(c)) || c == '_');
		};

	static auto IsIdentChar = [](char c) {
		return (std::isalnum(static_cast<unsigned char>(c)) || c == '_');
		};

	static auto IsDecDigit = [](char c) {
		return std::isdigit(static_cast<unsigned char>(c)) != 0;
		};

	static auto MatchMultiOp = [](const std::string& s, size_t i, std::string& op) {
		static const char* multi[] = { "<=", ">=", "==", "!=", "and", "or" };
		for (auto m : multi) {
			size_t L = std::strlen(m);
			if (i + L <= s.size() && s.compare(i, L, m) == 0) {
				op.assign(m);
				return true;
			}
		}
		return false;
		};

	static auto IsSingleOp = [](char c) {
		return c == '+' || c == '-' || c == '*' || c == '/' || c == '^' || c == '<' || c == '>';
		};

	std::string s = TrimExpression(in);
	if (s.empty()) return false;

	bool expectOperand = true;
	int parenDepth = 0;
	bool seenOperand = false;
	int parenDepthForCommas = 0;

	for (size_t i = 0; i < s.size();) {
		if (kWhitespace.find(s[i]) != std::string::npos) { ++i; continue; }

		if (expectOperand) {
			if (s[i] == '(') {
				++parenDepth;
				++parenDepthForCommas;
				++i;
				continue;
			}

			if (s[i] == '+' || s[i] == '-') {
				do {
					++i;
					while (i < s.size() && kWhitespace.find(s[i]) != std::string::npos) ++i;
				} while (i < s.size() && (s[i] == '+' || s[i] == '-'));
				continue;
			}

			if (IsDecDigit(s[i]) || (s[i] == '.' && i + 1 < s.size() && IsDecDigit(s[i + 1]))) {
				bool anyDigit = false;
				while (i < s.size() && IsDecDigit(s[i])) { anyDigit = true; ++i; }
				if (i < s.size() && s[i] == '.') {
					++i;
					while (i < s.size() && IsDecDigit(s[i])) { anyDigit = true; ++i; }
				}
				if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
					size_t j = i + 1;
					if (j < s.size() && (s[j] == '+' || s[j] == '-')) ++j;
					bool expDigits = false;
					while (j < s.size() && IsDecDigit(s[j])) { expDigits = true; ++j; }
					if (!anyDigit || !expDigits) return false;
					i = j;
				}
				seenOperand = true;
				expectOperand = false;
				continue;
			}

			if (IsIdentStart(s[i])) {
				size_t j = i + 1;
				while (j < s.size() && IsIdentChar(s[j])) ++j;
				size_t k = j;
				while (k < s.size() && kWhitespace.find(s[k]) != std::string::npos) ++k;
				if (k < s.size() && s[k] == '(') {
					++k;
					++parenDepth;
					++parenDepthForCommas;
					i = k;
					expectOperand = true;
				}
				else {
					i = j;
					seenOperand = true;
					expectOperand = false;
				}
				continue;
			}
			if (s[i] == ',') return false;

			return false;

		}
		else {
			std::string op;
			if (MatchMultiOp(s, i, op)) {
				i += op.size();
				expectOperand = true;
				continue;
			}

			if (i < s.size() && IsSingleOp(s[i])) {
				++i;
				expectOperand = true;
				continue;
			}

			if (s[i] == ')') {
				if (parenDepth == 0) return false;
				--parenDepth;
				if (parenDepthForCommas > 0) --parenDepthForCommas;
				++i;
				expectOperand = false;
				continue;
			}

			if (s[i] == ',') {
				if (parenDepthForCommas == 0) return false;
				++i;
				expectOperand = true;
				continue;
			}
			return false;
		}
	}

	return !expectOperand && parenDepth == 0 && seenOperand;
}

//========================================================================================
/// Reads list of configurations in the XML node.
//========================================================================================
void JUserExpressionList::ReadXml(const JXml* sxml, TiXmlElement* lis) {
	
	TiXmlElement* eleue = lis->FirstChildElement("userexpression");
	while (eleue) {
		if (sxml->CheckElementActive(eleue)) {
			const unsigned cnt = GetCount();
			if (cnt > MAX_DEFSTRUC_USREXP_NUM - 1)Run_Exceptioon("Maximum number of user expression has been reached.");
			tuserexpressiondata exp;
			exp.id = sxml->GetAttributeUnsigned(eleue, "id");
			exp.locals.clear();
			TiXmlElement* elmloc = sxml->GetFirstElement(eleue, "locals", true);
			if (elmloc) {
				std::string locals = sxml->GetAttributeStrSimple(elmloc, "value");
				exp.locals = UserExpParseLocals(locals);
			}
			TiXmlElement* elmexp = sxml->GetFirstElement(eleue, "expression");
			std::string expres = sxml->GetAttributeStrSimple(elmexp, "value");
			if (expres.empty() || !isValidMathExpr(expres)) {
				Log->PrintfWarning("Expression \'%s\' (ID:%u) ignored. Not a valid math expression", expres, exp.id);
			}
			else{
				exp.expression = TrimExpression(expres);
				JUserExpression* expres1 = new JUserExpression(exp.expression, exp.locals, exp.id);
				UserExpList.push_back(expres1);
				UserExpMap[expres1->id] = expres1;
			}
		}
		eleue = eleue->NextSiblingElement("userexpression");
	}
	//for (double t = 0.0; t < 25.0; t += 0.5) {
	//	tresultsuccess result = GetbyId(12)->Evaluate(0.2 + 300.5e-3, 0.0, 0.0, t);
	//	if (result.success) {
	//		std::cout << "t=" << t << " => " << result.value << "\n";
	//	}
	//	else Log->PrintfWarning("%s", result.error);
	//}
	//exit(0);
}


