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

/// \file JOutputCsv.cpp \brief Implements the class \ref JOutputCsv.

#include "JOutputCsv.h"
#include "JDataArrays.h"
#include "Functions.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cstdint>
#include <algorithm>
#include <cctype>

using namespace std;

namespace {

	std::string ExcelColumnName(unsigned index) {
		std::string name;
		++index;
		while (index) {
			unsigned rem = (index - 1) % 26;
			name.insert(name.begin(), char('A' + rem));
			index = (index - 1) / 26;
		}
		return name;
	}

	std::string EscapeXml(const std::string& value) {
		std::string result;
		result.reserve(value.size());
		for (char ch : value) {
			switch (ch) {
			case '&': result += "&amp;"; break;
			case '<': result += "&lt;"; break;
			case '>': result += "&gt;"; break;
			case '\"': result += "&quot;"; break;
			case '\'': result += "&apos;"; break;
			case '\r':
			case '\n': result += "&#10;"; break;
			case '\t': result += "&#9;"; break;
			default:
				if (static_cast<unsigned char>(ch) < 0x20 && ch != '\t') {
					result += ' ';
				}
				else result += ch;
			}
		}
		return result;
	}

	std::string Trim(const std::string& text) {
		size_t begin = 0;
		size_t end = text.size();
		while (begin < end && std::isspace(static_cast<unsigned char>(text[begin])))++begin;
		while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])))--end;
		return text.substr(begin, end - begin);
	}

	bool IsNumericString(const std::string& text) {
		const std::string trimmed = Trim(text);
		if (trimmed.empty()) return false;
		const char* begin = trimmed.c_str();
		char* endptr = nullptr;
		std::strtod(begin, &endptr);
		if (endptr == begin) return false;
		while (*endptr) {
			if (!std::isspace(static_cast<unsigned char>(*endptr))) return false;
			++endptr;
		}
		for (char ch : trimmed) {
			if (!(std::isdigit(static_cast<unsigned char>(ch)) || ch == '+' || ch == '-' || ch == '.' || ch == 'e' || ch == 'E')) return false;
		}
		return true;
	}

	class WorksheetWriter {
		unsigned RowIndex;
		std::ostringstream Sheet;
	public:
		WorksheetWriter() :RowIndex(1) {
			Sheet << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>";
			Sheet << "<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\"";
			Sheet << " xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">";
			Sheet << "<sheetData>";
		}

		void AddRow(const std::vector<std::string>& cells, const std::vector<bool>& numericFlags, bool forceText) {
			Sheet << "<row r=\"" << RowIndex << "\">";
			for (size_t c = 0; c < cells.size(); ++c) {
				const std::string cellRef = ExcelColumnName(static_cast<unsigned>(c)) + std::to_string(RowIndex);
				const std::string value = cells[c];
				const bool numeric = !forceText && c < numericFlags.size() && numericFlags[c] && !value.empty();
				if (numeric) {
					Sheet << "<c r=\"" << cellRef << "\"><v>" << EscapeXml(value) << "</v></c>";
				}
				else {
					Sheet << "<c r=\"" << cellRef << "\" t=\"inlineStr\"><is><t>" << EscapeXml(value) << "</t></is></c>";
				}
			}
			Sheet << "</row>";
			++RowIndex;
		}

		std::string Finish() {
			Sheet << "</sheetData></worksheet>";
			return Sheet.str();
		}
	};

	uint32_t ComputeCrc32(const unsigned char* data, size_t length) {
		static uint32_t table[256];
		static bool initialized = false;
		if (!initialized) {
			for (uint32_t i = 0; i < 256; ++i) {
				uint32_t c = i;
				for (int j = 0; j < 8; ++j) c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
				table[i] = c;
			}
			initialized = true;
		}
		uint32_t crc = 0xFFFFFFFFu;
		for (size_t i = 0; i < length; ++i) crc = table[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8);
		return crc ^ 0xFFFFFFFFu;
	}

	void WriteLe16(std::ofstream& out, uint16_t value) {
		out.put(static_cast<char>(value & 0xFF));
		out.put(static_cast<char>((value >> 8) & 0xFF));
	}

	void WriteLe32(std::ofstream& out, uint32_t value) {
		WriteLe16(out, static_cast<uint16_t>(value & 0xFFFF));
		WriteLe16(out, static_cast<uint16_t>((value >> 16) & 0xFFFF));
	}

	struct ZipEntryData {
		std::string name;
		std::string data;
	};

	struct ZipEntryMeta {
		std::string name;
		uint32_t crc = 0;
		uint32_t size = 0;
		uint32_t offset = 0;
	};

	bool WriteZipFile(const std::string& fname, const std::vector<ZipEntryData>& entries) {
		std::ofstream out(fname, std::ios::binary | std::ios::trunc);
		if (!out) return false;
		std::vector<ZipEntryMeta> metas;
		metas.reserve(entries.size());
		for (const auto& entry : entries) {
			ZipEntryMeta meta;
			meta.name = entry.name;
			meta.size = static_cast<uint32_t>(entry.data.size());
			meta.crc = ComputeCrc32(reinterpret_cast<const unsigned char*>(entry.data.data()), entry.data.size());
			meta.offset = static_cast<uint32_t>(out.tellp());

			WriteLe32(out, 0x04034b50u);
			WriteLe16(out, 20); // version needed to extract
			WriteLe16(out, 0);  // general purpose bit flag
			WriteLe16(out, 0);  // compression method (store)
			WriteLe16(out, 0);  // mod time
			WriteLe16(out, 0);  // mod date
			WriteLe32(out, meta.crc);
			WriteLe32(out, meta.size);
			WriteLe32(out, meta.size);
			WriteLe16(out, static_cast<uint16_t>(meta.name.size()));
			WriteLe16(out, 0);
			out.write(meta.name.data(), meta.name.size());
			out.write(entry.data.data(), entry.data.size());
			metas.push_back(meta);
		}

		const uint32_t centralDirOffset = static_cast<uint32_t>(out.tellp());
		for (const auto& meta : metas) {
			WriteLe32(out, 0x02014b50u);
			WriteLe16(out, 0x0314); // version made by (Unix, 2.0)
			WriteLe16(out, 20);
			WriteLe16(out, 0);
			WriteLe16(out, 0);
			WriteLe16(out, 0);
			WriteLe32(out, meta.crc);
			WriteLe32(out, meta.size);
			WriteLe32(out, meta.size);
			WriteLe16(out, static_cast<uint16_t>(meta.name.size()));
			WriteLe16(out, 0);
			WriteLe16(out, 0);
			WriteLe16(out, 0);
			WriteLe16(out, 0);
			WriteLe32(out, 0);
			WriteLe32(out, meta.offset);
			out.write(meta.name.data(), meta.name.size());
		}

		const uint32_t centralDirSize = static_cast<uint32_t>(out.tellp()) - centralDirOffset;
		WriteLe32(out, 0x06054b50u);
		WriteLe16(out, 0);
		WriteLe16(out, 0);
		WriteLe16(out, static_cast<uint16_t>(metas.size()));
		WriteLe16(out, static_cast<uint16_t>(metas.size()));
		WriteLe32(out, centralDirSize);
		WriteLe32(out, centralDirOffset);
		WriteLe16(out, 0);
		out.flush();
		return static_cast<bool>(out);
	}

	std::string CurrentIsoDate() {
		std::time_t now = std::time(nullptr);
		std::tm* gmt = std::gmtime(&now);
		char buffer[32];
		if (gmt && std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", gmt)) return buffer;
		return "1970-01-01T00:00:00Z";
	}

	std::string BuildContentTypes() {
		return "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\"><Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/><Default Extension=\"xml\" ContentType=\"application/xml\"/><Override PartName=\"/docProps/app.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.extended-properties+xml\"/><Override PartName=\"/docProps/core.xml\" ContentType=\"application/vnd.openxmlformats-package.core-properties+xml\"/><Override PartName=\"/xl/workbook.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml\"/><Override PartName=\"/xl/worksheets/sheet1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml\"/><Override PartName=\"/xl/styles.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml\"/></Types>";
	}

	std::string BuildRootRels() {
		return "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\"><Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"xl/workbook.xml\"/><Relationship Id=\"rId2\" Type=\"http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties\" Target=\"docProps/core.xml\"/><Relationship Id=\"rId3\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties\" Target=\"docProps/app.xml\"/></Relationships>";
	}

	std::string BuildWorkbookRels() {
		return "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\"><Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet\" Target=\"worksheets/sheet1.xml\"/><Relationship Id=\"rId2\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles\" Target=\"styles.xml\"/></Relationships>";
	}

	std::string BuildWorkbookXml() {
		return "<?xml version=\"1.0\" encoding=\"UTF-8\"?><workbook xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\"><sheets><sheet name=\"Sheet1\" sheetId=\"1\" r:id=\"rId1\"/></sheets></workbook>";
	}

	std::string BuildStylesXml() {
		return "<?xml version=\"1.0\" encoding=\"UTF-8\"?><styleSheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\"><fonts count=\"1\"><font><sz val=\"11\"/><color theme=\"1\"/><name val=\"Calibri\"/><family val=\"2\"/></font></fonts><fills count=\"1\"><fill><patternFill patternType=\"none\"/></fill></fills><borders count=\"1\"><border><left/><right/><top/><bottom/><diagonal/></border></borders><cellStyleXfs count=\"1\"><xf numFmtId=\"0\" fontId=\"0\" fillId=\"0\" borderId=\"0\"/></cellStyleXfs><cellXfs count=\"1\"><xf numFmtId=\"0\" fontId=\"0\" fillId=\"0\" borderId=\"0\" xfId=\"0\"/></cellXfs><cellStyles count=\"1\"><cellStyle name=\"Normal\" xfId=\"0\" builtinId=\"0\"/></cellStyles></styleSheet>";
	}

	std::string BuildAppXml() {
		return "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Properties xmlns=\"http://schemas.openxmlformats.org/officeDocument/2006/extended-properties\" xmlns:vt=\"http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes\"><Application>DualSPHysics</Application></Properties>";
	}

	std::string BuildCoreXml(const std::string& isoDate) {
		return "<?xml version=\"1.0\" encoding=\"UTF-8\"?><cp:coreProperties xmlns:cp=\"http://schemas.openxmlformats.org/package/2006/metadata/core-properties\" xmlns:dc=\"http://purl.org/dc/elements/1.1/\" xmlns:dcterms=\"http://purl.org/dc/terms/\" xmlns:dcmitype=\"http://purl.org/dc/dcmitype/\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"><dc:creator>DualSPHysics</dc:creator><cp:lastModifiedBy>DualSPHysics</cp:lastModifiedBy><dcterms:created xsi:type=\"dcterms:W3CDTF\">" + isoDate + "</dcterms:created><dcterms:modified xsi:type=\"dcterms:W3CDTF\">" + isoDate + "</dcterms:modified></cp:coreProperties>";
	}

	bool SaveWorksheetAsXlsx(const std::string& fname, const std::string& worksheetXml) {
		std::vector<ZipEntryData> entries;
		entries.push_back({ "[Content_Types].xml",BuildContentTypes() });
		entries.push_back({ "_rels/.rels",BuildRootRels() });
		entries.push_back({ "docProps/app.xml",BuildAppXml() });
		entries.push_back({ "docProps/core.xml",BuildCoreXml(CurrentIsoDate()) });
		entries.push_back({ "xl/workbook.xml",BuildWorkbookXml() });
		entries.push_back({ "xl/_rels/workbook.xml.rels",BuildWorkbookRels() });
		entries.push_back({ "xl/styles.xml",BuildStylesXml() });
		entries.push_back({ "xl/worksheets/sheet1.xml",worksheetXml });
		return WriteZipFile(fname, entries);
	}

} // namespace

//##############################################################################
//# JOutputCsv
//##############################################################################
//==============================================================================
/// Constructor.
//==============================================================================
JOutputCsv::JOutputCsv(bool csvsepcoma, bool createpath)
	:CsvSepComa(csvsepcoma), CreatPath(createpath)
{
	ClassName = "JOutputCsv";
	Reset();
}

//==============================================================================
/// Destructor.
//==============================================================================
JOutputCsv::~JOutputCsv() {
	DestructorActive = true;
	Reset();
}

//==============================================================================
/// Initialization of variables.
//==============================================================================
void JOutputCsv::Reset() {
	FileName = "";
}

//==============================================================================
/// Stores data as a tab-separated CSV file to minimize disk space and memory usage.
//==============================================================================
void JOutputCsv::SaveCsv(std::string fname, const JDataArrays& arrays, std::string head) {
	const char sep = '\t';
	std::string ext = fun::GetExtension(fname);
	if (ext.empty())fname = fun::AddExtension(fname, "csv");
	else if (fun::StrLower(ext) != "csv") {
		fname = fname.substr(0, fname.size() - ext.size() - 1);
		fname = fun::AddExtension(fname, "csv");
	}
	FileName = fname;
	const unsigned nf = arrays.Count();
	const unsigned nv = arrays.GetDataCount(true);
	if (nv != arrays.GetDataCount(false))Run_ExceptioonFile("The number of values in arrays is not the same.", fname);
	if (CreatPath)fun::MkdirPath(fun::GetDirParent(fname));
	struct ColumnInfo { unsigned arrayIndex; unsigned component; };
	std::vector<ColumnInfo> columns;
	std::vector<std::string> formats;
	std::vector<std::string> headers;
	columns.reserve(nf * 3);
	formats.reserve(nf * 3);

	std::ofstream file(fname.c_str(), std::ios::out | std::ios::binary);
	if (!file.is_open())Run_ExceptioonFile("Cannot open the file.", fname);

	if (!head.empty()) {
		while (!head.empty()) {
			file << fun::StrSplit("\n", head) << '\n';
		}
	}

	static const char* comps3[3] = { ".x",".y",".z" };
	static const char* comps6[6] = { ".xx",".xy",".xz",".yy",".yz",".zz" };

	for (unsigned cf = 0; cf < nf; cf++) {
		const JDataArrays::StDataArray& ar = arrays.GetArrayCte(cf);
		const std::string keyname = ar.keyname;
		const std::string units = arrays.GetArrayUnits(cf);
		const int dim = arrays.GetArrayDim(cf);
		const std::string fmt = arrays.GetArrayFmt(cf);
		if (dim == 1) {
			headers.push_back(keyname + units);
			columns.push_back({ cf,0 });
			formats.push_back(fmt);
		}
		else if (dim == 3) {
			for (unsigned c = 0; c < 3; c++) {
				headers.push_back(keyname + comps3[c] + units);
				columns.push_back({ cf,c });
				formats.push_back(fmt);
			}
		}
		else if (dim == 6) {
			for (unsigned c = 0; c < 6; c++) {
				headers.push_back(keyname + comps6[c] + units);
				columns.push_back({ cf,c });
				formats.push_back(fmt);
			}
		}
		else Run_ExceptioonFile(fun::PrintStr("Dimension %d of array \'%s\' is invalid.", dim, keyname.c_str()), fname);
	}

	if (!headers.empty()) {
		for (size_t i = 0; i < headers.size(); ++i) {
			if (i) file << sep;
			file << headers[i];
		}
		file << '\n';
	}

	std::vector<std::string> row(columns.size());
	for (unsigned cv = 0; cv < nv; cv++) {
		for (size_t cc = 0; cc < columns.size(); ++cc) {
			const ColumnInfo info = columns[cc];
			const JDataArrays::StDataArray& ar = arrays.GetArrayCte(info.arrayIndex);
			const std::string& fmt = formats[cc];
			std::string value;
			switch (ar.type) {
			case TypeUchar: { const byte* v = (byte*)ar.ptr;  value = fun::PrintStr(fmt.c_str(), v[cv]); }break;
			case TypeUshort: { const word* v = (word*)ar.ptr;  value = fun::PrintStr(fmt.c_str(), v[cv]); }break;
			case TypeUint: { const unsigned* v = (unsigned*)ar.ptr;  value = fun::PrintStr(fmt.c_str(), v[cv]); }break;
			case TypeInt: { const int* v = (int*)ar.ptr;  value = fun::PrintStr(fmt.c_str(), v[cv]); }break;
			case TypeFloat: { const float* v = (float*)ar.ptr;  value = fun::PrintStr(fmt.c_str(), v[cv]); }break;
			case TypeDouble: { const double* v = (double*)ar.ptr;  value = fun::PrintStr(fmt.c_str(), v[cv]); }break;
			case TypeUint3: { const tuint3* v = (tuint3*)ar.ptr;  const unsigned comp[3] = { v[cv].x,v[cv].y,v[cv].z }; value = fun::PrintStr(fmt.c_str(), comp[info.component]); }break;
			case TypeInt3: { const tint3* v = (tint3*)ar.ptr;  const int comp[3] = { v[cv].x,v[cv].y,v[cv].z }; value = fun::PrintStr(fmt.c_str(), comp[info.component]); }break;
			case TypeFloat3: { const tfloat3* v = (tfloat3*)ar.ptr;  const float comp[3] = { v[cv].x,v[cv].y,v[cv].z }; value = fun::PrintStr(fmt.c_str(), comp[info.component]); }break;
			case TypeDouble3: { const tdouble3* v = (tdouble3*)ar.ptr;  const double comp[3] = { v[cv].x,v[cv].y,v[cv].z }; value = fun::PrintStr(fmt.c_str(), comp[info.component]); }break;
			case TypeSyMatrix3f: { const tsymatrix3f* v = (tsymatrix3f*)ar.ptr; const float comp[6] = { v[cv].xx,v[cv].xy,v[cv].xz,v[cv].yy,v[cv].yz,v[cv].zz }; value = fun::PrintStr(fmt.c_str(), comp[info.component]); }break;
			case TypeSyMatrix3d: { const tsymatrix3d* v = (tsymatrix3d*)ar.ptr; const double comp[6] = { v[cv].xx,v[cv].xy,v[cv].xz,v[cv].yy,v[cv].yz,v[cv].zz }; value = fun::PrintStr(fmt.c_str(), comp[info.component]); }break;
			default: Run_ExceptioonFile(fun::PrintStr("Type of array \'%s\' is invalid.", TypeToStr(ar.type)), fname);
			}
			row[cc] = value;
		}
		for (size_t cc = 0; cc < row.size(); ++cc) {
			if (cc) file << sep;
			file << row[cc];
		}
		file << '\n';
	}
}

//==============================================================================
/// Stores a generic table in CSV format (tab-separated).
//==============================================================================
void JOutputCsv::SaveCsvTable(const std::string& fname, const std::vector<std::string>& headers, const std::vector<std::vector<std::string>>& rows) {
	const char sep = '\t';
	const std::string ext = fun::GetExtension(fname);
	std::string outname = fname;
	if (ext.empty())outname = fun::AddExtension(fname, "csv");
	else if (fun::StrLower(ext) != "csv") {
		outname = fname.substr(0, fname.size() - ext.size() - 1);
		outname = fun::AddExtension(outname, "csv");
	}
	fun::MkdirPath(fun::GetDirParent(outname));

	std::ofstream file(outname.c_str(), std::ios::out | std::ios::binary);
	if (!file.is_open())fun::RunExceptioonFun(__FILE__, __LINE__, __func__, "Cannot open the file.", outname);

	if (!headers.empty()) {
		for (size_t c = 0; c < headers.size(); ++c) {
			if (c) file << sep;
			file << headers[c];
		}
		file << '\n';
	}

	for (const auto& row : rows) {
		for (size_t c = 0; c < row.size(); ++c) {
			if (c) file << sep;
			file << row[c];
		}
		file << '\n';
	}
}

//==============================================================================
/// Calculates statistic information of unitary arrays.
//==============================================================================
template<typename T> void JOutputCsv::CalculateStatsArray1(unsigned ndata, T* data
	, double& valmin, double& valmax, double& valmean)const
{
	double vmin = 0, vmax = 0, vmea = 0;
	if (ndata)vmin = vmax = double(data[0]);
	for (unsigned p = 0; p < ndata; p++) {
		const double v = double(data[p]);
		vmin = (vmin < v ? vmin : v);
		vmax = (vmax > v ? vmax : v);
		vmea = vmea + v;
	}
	vmea = vmea / double(ndata);
	//-Saves results.
	valmin = vmin; valmax = vmax; valmean = vmea;
}

//==============================================================================
/// Calculates statistic information of triple arrays.
//==============================================================================
template<typename T> void JOutputCsv::CalculateStatsArray3(unsigned ndata, T* data
	, tdouble4& valmin, tdouble4& valmax, tdouble4& valmean)const
{
	tdouble4 vmin = TDouble4(0);
	tdouble4 vmax = TDouble4(0);
	tdouble4 vmea = TDouble4(0);
	if (ndata) {
		const T vv = data[0];
		const double vx = double(vv.x);
		const double vy = double(vv.y);
		const double vz = double(vv.z);
		const tdouble4 v = TDouble4(vx, vy, vz, sqrt(vx * vx + vy * vy + vz * vz));
		vmin = vmax = v;
	}
	for (unsigned p = 0; p < ndata; p++) {
		const T vv = data[p];
		const double vx = double(vv.x);
		const double vy = double(vv.y);
		const double vz = double(vv.z);
		const tdouble4 v = TDouble4(vx, vy, vz, sqrt(vx * vx + vy * vy + vz * vz));
		vmin = MinValues(vmin, v);
		vmax = MaxValues(vmax, v);
		vmea = vmea + v;
	}
	vmea = vmea / TDouble4(ndata);
	//-Saves results.
	valmin = vmin; valmax = vmax; valmean = vmea;
}

////==============================================================================
///// Compute and stores statistic information in CSV format.
////==============================================================================
//void JOutputCsv::SaveStatsCsv(std::string fname,bool create,int part,double timestep,const JDataArrays &arrays,std::string head){
//  if(fun::GetExtension(fname).empty())FileName=fname=fun::AddExtension(fname,".csv");
//  const char csvsep=(CsvSepComa? ',': ';');
//  const unsigned nf=arrays.GetCount();
//  const unsigned nv=arrays.GetDataCount();
//  if(CreateDirs)fun::MkdirPath(fun::GetDirParent(fname));
//  ofstream pf;
//  if(create)pf.open(fname.c_str());
//  else pf.open(fname.c_str(),ios_base::app);
//  if(pf){
//    if(create){
//      //-Saves lines of head.
//      while(!head.empty())pf << fun::StrSplit("\n",head) << endl;
//      //-Saves head.
//      pf << fun::StrCsvSep(CsvSepComa,"Part;Time [s];Count;");
//      for(unsigned cf=0;cf<nf;cf++){
//        const JDataArray* ar=arrays.GetArray(cf);
//        if(ar->GetPointer() && !ar->GetHidden()){
//          const string units=ar->GetUnits();
//          const string name=ar->GetName();
//          if(fun::StrLower(name)!="pos"){
//            pf << name << "_min"  << units << csvsep;
//            pf << name << "_max"  << units << csvsep;
//            pf << name << "_mean" << units << csvsep;
//          }
//          if(ar->IsTriple()){
//            pf << name << "_Xmin"  << units << csvsep;
//            pf << name << "_Xmax"  << units << csvsep;
//            pf << name << "_Xmean" << units << csvsep;
//            pf << name << "_Ymin"  << units << csvsep;
//            pf << name << "_Ymax"  << units << csvsep;
//            pf << name << "_Ymean" << units << csvsep;
//            pf << name << "_Zmin"  << units << csvsep;
//            pf << name << "_Zmax"  << units << csvsep;
//            pf << name << "_Zmean" << units << csvsep;
//          }
//        }
//      }
//      pf << endl;
//    }
//    //-Saves data.
//    pf << fun::PrintStr("%u",part)+csvsep;
//    pf << fun::PrintStr("%20.12E",timestep)+csvsep;
//    pf << fun::PrintStr("%u",nv)+csvsep;
//    for(unsigned cf=0;cf<nf;cf++){
//      const JDataArray* ar=arrays.GetArray(cf);
//      if(ar->GetPointer() && !ar->GetHidden()){
//        string fmt=JDataArraysDef::GetFmt(JDataArraysDef::TpDouble);
//        fmt=fmt+csvsep+fmt+csvsep+fmt;
//        double vmin,vmax,vmea;
//        tdouble4 vmin3,vmax3,vmea3;
//        switch(ar->GetType()){
//          case JDataArraysDef::TpChar:   { const char     *v=ar->GetPointerChar();     CalculateStatsArray1(nv,v,vmin,vmax,vmea); }break;
//          case JDataArraysDef::TpUchar:  { const byte     *v=ar->GetPointerUchar();    CalculateStatsArray1(nv,v,vmin,vmax,vmea); }break;
//          case JDataArraysDef::TpShort:  { const short    *v=ar->GetPointerShort();    CalculateStatsArray1(nv,v,vmin,vmax,vmea); }break;
//          case JDataArraysDef::TpUshort: { const word     *v=ar->GetPointerUshort();   CalculateStatsArray1(nv,v,vmin,vmax,vmea); }break;
//          case JDataArraysDef::TpInt:    { const int      *v=ar->GetPointerInt();      CalculateStatsArray1(nv,v,vmin,vmax,vmea); }break;
//          case JDataArraysDef::TpUint:   { const unsigned *v=ar->GetPointerUint();     CalculateStatsArray1(nv,v,vmin,vmax,vmea); }break;
//          case JDataArraysDef::TpLlong:  { const llong    *v=ar->GetPointerLlong();    CalculateStatsArray1(nv,v,vmin,vmax,vmea); }break;
//          case JDataArraysDef::TpUllong: { const ullong   *v=ar->GetPointerUllong();   CalculateStatsArray1(nv,v,vmin,vmax,vmea); }break;
//          case JDataArraysDef::TpFloat:  { const float    *v=ar->GetPointerFloat();    CalculateStatsArray1(nv,v,vmin,vmax,vmea); }break;
//          case JDataArraysDef::TpDouble: { const double   *v=ar->GetPointerDouble();   CalculateStatsArray1(nv,v,vmin,vmax,vmea); }break;
//          case JDataArraysDef::TpInt3:   { const tint3    *v=ar->GetPointerInt3();     CalculateStatsArray3(nv,v,vmin3,vmax3,vmea3); }break;
//          case JDataArraysDef::TpUint3:  { const tuint3   *v=ar->GetPointerUint3();    CalculateStatsArray3(nv,v,vmin3,vmax3,vmea3); }break;
//          case JDataArraysDef::TpFloat3: { const tfloat3  *v=ar->GetPointerFloat3();   CalculateStatsArray3(nv,v,vmin3,vmax3,vmea3); }break;
//          case JDataArraysDef::TpDouble3:{ const tdouble3 *v=ar->GetPointerDouble3();  CalculateStatsArray3(nv,v,vmin3,vmax3,vmea3); }break;
//          default: Run_Exceptioon("Type of array is invalid.");
//        }
//        if(!ar->IsTriple())pf << fun::PrintStr(fmt.c_str(),vmin,vmax,vmea)+csvsep;
//        else{
//          if(fun::StrLower(ar->GetName())!="pos"){
//            pf << fun::PrintStr(fmt.c_str(),vmin3.w,vmax3.w,vmea3.w)+csvsep;
//          }
//          pf << fun::PrintStr(fmt.c_str(),vmin3.x,vmax3.x,vmea3.x)+csvsep;
//          pf << fun::PrintStr(fmt.c_str(),vmin3.y,vmax3.y,vmea3.y)+csvsep;
//          pf << fun::PrintStr(fmt.c_str(),vmin3.z,vmax3.z,vmea3.z)+csvsep;
//        }
//      }
//    }
//    pf << endl;
//    if(pf.fail())Run_ExceptioonFile("File writing failure.",fname);
//    pf.close();
//  }
//  else Run_ExceptioonFile("Cannot open the file.",fname);
//}




