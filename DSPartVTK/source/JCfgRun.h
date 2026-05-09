/*
 <DUALSPHYSICS>  Copyright (c) 2020, Dr Jose M. Dominguez et al. (see http://dual.sphysics.org/index.php/developers/). 

 EPHYSLAB Environmental Physics Laboratory, Universidade de Vigo, Ourense, Spain.
 School of Mechanical, Aerospace and Civil Engineering, University of Manchester, Manchester, U.K.

 This file is part of DualSPHysics. 

 DualSPHysics is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or (at your option) any later version. 

 DualSPHysics is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details. 

 You should have received a copy of the GNU General Public License, along with DualSPHysics. If not, see <http://www.gnu.org/licenses/>. 
*/

/// \file JCfgRun.h \brief Declares the class \ref JCfgRun.

#ifndef _JCfgRun_
#define _JCfgRun_

#pragma warning(disable : 4996) //Anula sprintf() deprecated.

#include "TypesDef.h"
#include "Functions.h"
#include "JObject.h"

#include <string>
#include <cstring>
#include <sstream>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>

//##############################################################################
//# JCfgRun
//##############################################################################
/// \brief Defines the class responsible of collecting the execution parameters by command line.

class JCfgRun : protected JObject
{
protected:
  int ParmDef;

public:
  bool PrintInfo;

  std::string DirIn;
  std::string FileIn;
  std::string FileXml;

  int First,Last;

  std::string DsSaveVtk;
  std::string DsSaveCsv;
  std::string DsOnlyMk;
  std::vector<std::string> DsFieldSpecs;
  
public:
  void ClearFilesIn(){ DirIn=""; FileIn=""; }

public:
  JCfgRun();
  void Reset();
  void VisuInfo()const;
  void VisuConfig()const;
  void PrintConfig(const std::string& tx)const;
  void LoadArgv(int argc,char** argv);
  void LoadFile(std::string fname,int lv);
  void LoadOpts(std::string *optlis,int optn,int lv,std::string file);
  void ErrorParm(const std::string &opt,int optc,int lv,const std::string &file)const;
  bool VtuOptGet(std::string name);
  void ValidaCfg();
};

#endif

