/*
 <TOVTK4_DEFSTRUCT>  Copyright (c) 2020, Dr Jose M. Dominguez et al. (see http://dual.sphysics.org/index.php/developers/). 

 EPHYSLAB Environmental Physics Laboratory, Universidade de Vigo, Ourense, Spain.
 School of Mechanical, Aerospace and Civil Engineering, University of Manchester, Manchester, U.K.

 This file is part of DualSPHysics. 

 DualSPHysics is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or (at your option) any later version. 

 DualSPHysics is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details. 

 You should have received a copy of the GNU General Public License, along with DualSPHysics. If not, see <http://www.gnu.org/licenses/>. 
*/

/// \file JCfgRun.cpp \brief Implements the class \ref JCfgRun.

#include "JCfgRun.h"

using namespace std;
using namespace fun;

//==============================================================================
/// Constructor.
//==============================================================================
JCfgRun::JCfgRun(){
  ClassName="JCfgRun";
  Reset();
}

//==============================================================================
/// Initialisation of variables.
//==============================================================================
void JCfgRun::Reset(){
  PrintInfo=false; ParmDef=0;
  ClearFilesIn();
  FileXml="";
  First=-1;  Last=-1;
  DsSaveVtk = ""; DsSaveCsv = "";
  DsOnlyMk = "";
  DsFieldSpecs.clear();
}

//==============================================================================
/// Shows information about execution parameters.
//==============================================================================
void JCfgRun::VisuInfo()const{
  printf("Information about parameters of execution:\n\n");
  printf("  DSPartVTK <options>\n\n");
  printf("  Basic options:\n");
  printf("    -h                 Shows information about parameters\n");
  printf("    -opt <file>        Loads configuration of a file\n\n");
  printf("  Define input files :\n");
  printf("    -dirin <dir>       Directory with particle data\n");
  printf("    -filein <file>     File with particle data\n");
  printf("    -filexml file.xml  Loads xml file with information of particles\n\n");
  printf("    -first:<int>       Indicates the first file to be computed\n");
  printf("    -last:<int>        Indicates the last file to be computed\n\n");
  printf("  Define output files for deformable structures:\n");
  printf("    -savevtk <file>  Template for deformable-structure VTK output (use %%mk%% placeholder)\n");
  printf("    -savecsv <file>  Template for deformable-structure CSV output (use %%mk%% placeholder)\n");
  printf("    -onlymk:<ranges> Restrict deformable export to matching Mk identifiers\n");
  printf("    -fields:<spec>   Per-body optional fields, e.g. 1=Disp,Cauchy,Phasefield or *=Disp,Cauchy,Plasticstrain,Energies,Contactforce, default is Disp+Phasefield/PlasticStrain\n\n"); 
  printf("  Examples:\n");
  printf("     DSPartVTK -dirin . -filexml case.xml -savevtk DefStruc_%%mk%% -onlymk:11\n");
  printf("\n");
}

//==============================================================================
/// Shows current configuration.
//==============================================================================
void JCfgRun::VisuConfig()const{
  printf("\nConfiguration of execution:\n");
  string ln="\n";
  PrintVar("  DirIn",DirIn,ln);
  PrintVar("  FileIn",FileIn,ln);
  PrintVar("  FileXml",FileXml,ln);
  if(First>=0)PrintVar("  First",First,ln);
  if(Last>=0)PrintVar("  Last",Last,ln);
  PrintVar("  DsSaveVtk", DsSaveVtk, ln);
  PrintVar("  DsSaveCsv", DsSaveCsv, ln);
  PrintVar("  DsOnlyMk", DsOnlyMk, ln);
  if (!DsFieldSpecs.empty()) {
      std::string specs;
      for (size_t i = 0; i < DsFieldSpecs.size(); i++) {
          if (i)specs += ";";
          specs += DsFieldSpecs[i];
      }
      PrintVar("  DsFields", specs, ln);
  }
  printf("\n");
}

//==============================================================================
/// Loads execution parameters from the command line.
//==============================================================================
void JCfgRun::LoadArgv(int argc,char** argv){
  const char met[]="LoadArgv";
  Reset();
  const int MAXOPTS=100;
  string *optlis=new string[MAXOPTS];
  int optn=0;
  for(int c=0;c<argc-1;c++){
    string tex=StrTrim(argv[c+1]);
    int pos=int(tex.find(" "));
    if(pos>0){
      while(pos>0){
        bool divide=((tex[0]=='-' || tex[0]=='#') || (pos+2<tex.size() && ((tex[pos+1]=='-' && tex[pos+2]!=' ') || tex[pos+1]=='#')));
        //printf("  tex[%s]  pos:%d  divide=%d\n",tex.c_str(),pos,(divide? 1: 0));
        if(divide){
          if(optn>=MAXOPTS)RunException(met,"Has exceeded the maximum configuration options.");
          optlis[optn]=tex.substr(0,pos); optn++;
          tex=tex.substr(pos+1);
          pos=int(tex.find(" "));
        }
        else pos=int(tex.find(" ",pos+1));
      }
    }
    if(optn>=MAXOPTS)RunException(met,"Has exceeded the maximum configuration options.");
    optlis[optn]=tex; optn++;
  }
  //for(int c=0;c<optn;c++)printf("[%d]=[%s]\n",c,optlis[c].c_str());
  if(optn)LoadOpts(optlis,optn,0,"");
  delete[] optlis;
  if(!optn)PrintInfo=true;
  if(!PrintInfo){ 
    VisuConfig(); 
  }
  else VisuInfo();
}

//==============================================================================
/// Loads execution parameters from a text file.
//==============================================================================
void JCfgRun::LoadFile(std::string fname,int lv){
  const char met[]="LoadFile";
  //printf("\nFile:[%s] lv:%d\n",fname.c_str(),lv);
  const int MAXOPTS=50;
  int optn=0;
  std::string *optlis=new std::string[MAXOPTS];
  std::ifstream pf;
  pf.open(fname.c_str());
  if(pf){
    while(!pf.eof()&&optn<MAXOPTS){
      std::string tex;  pf >> tex;
      if(tex!=""){
        if(optn<MAXOPTS)optlis[optn]=tex;
        optn++;
      }
      //printf("FF[%s]\n",tex.c_str());
    } 
    if(!pf.eof()&&pf.fail())RunException(met,"Error reading data from the file.",fname);
    pf.close();
  }
  else RunException(met,"The file can not be opened.",fname);
  if(optn>=MAXOPTS){
    char cad[128];
    sprintf(cad,"File with too many lines (Maximum=%d)",MAXOPTS);
    RunException(met,cad,fname);
  }
  if(optn>0)LoadOpts(optlis,optn,lv,fname);
  delete[] optlis;
}

//==============================================================================
/// Generates error of unknown parameter.
//==============================================================================
void JCfgRun::ErrorParm(const std::string &opt,int optc,int lv,const std::string &file)const{
  const char met[]="ErrorParm";
  char cad[256];
  std::string tx;
  sprintf(cad,"Parameter \"%s\" unrecognized or invalid. ",opt.c_str());  tx=cad;
  sprintf(cad,"(Level cfg:%d, Parameter:%d)",lv,optc);  tx=tx+cad;
  if(file!="")RunException(met,tx,file); else RunException(met,tx);
}

//==============================================================================
/// Loads execution parameters.
//==============================================================================
void JCfgRun::LoadOpts(std::string *optlis,int optn,int lv,std::string file){
  const char met[]="LoadOpts";
  if(lv>=10)RunException(met,"No more than 10 levels of recursive configuration are allowed.");
  for(int c=0;c<optn;c++){
    std::string opt=optlis[c];
    //printf("[%d]=[%s]\n",c,optlis[c].c_str());
    if(opt[0]=='-'){
      std::string tx=opt.substr(1);
      int pos=int(tx.find("#"));
      if(pos>0)tx=tx.substr(0,pos);
      pos=int(tx.find(":"));
      std::string txopt="",txopt2="",txword=StrUpper(pos>0? tx.substr(0,pos): tx);
      if(pos>0)txopt=tx.substr(pos+1);
      tx=txopt;
      pos=int(tx.find(":"));
      txopt=(pos>0? tx.substr(0,pos): tx);
      if(pos>0)txopt2=tx.substr(pos+1);
      if(txword=="DIRIN"&&c+1<optn){ ClearFilesIn(); DirIn=optlis[c+1]; c++; }
      else if(txword=="FILEIN"&&c+1<optn){ ClearFilesIn(); FileIn=optlis[c+1]; c++; }
      else if(txword=="FILEXML"&&c+1<optn){ FileXml=optlis[c+1]; c++; }
      else if(txword=="FIRST"){ First=atoi(txopt.c_str()); if(First<0)First=0; } 

      else if (txword == "LAST") { Last = atoi(txopt.c_str()); if (Last < 0)Last = -1; }
      else if (txword == "SAVEVTK" && c + 1 < optn) { DsSaveVtk = optlis[c + 1]; c++; }
      else if (txword == "SAVECSV" && c + 1 < optn) { DsSaveCsv = optlis[c + 1]; c++; }
      else if (txword == "ONLYMK") { DsOnlyMk = txopt; }
      else if (txword == "FIELDS") { if (!txopt.empty()) DsFieldSpecs.push_back(txopt); }
      else if(txword=="OPT"&&c+1<optn){ LoadFile(optlis[c+1],lv+1); c++; }
      else if(txword=="H"||txword=="HELP"||txword=="?")PrintInfo=true;
      else ErrorParm(opt,c,lv,file);
    }
  }
}

//==============================================================================
// Checks configuration.
//==============================================================================
void JCfgRun::ValidaCfg(){
  const char met[]="ValidaCfg";
  
  // Check for input directory or file
  if (DirIn.empty() && FileIn.empty())
      RunException(met, "Input directory or file was not defined (use -dirin <dir> or -filein <file>).");
  
  // Check for output files
  if (DsSaveVtk.empty() && DsSaveCsv.empty())
      RunException(met, "Output files were not defined (use -savevtk <file> or -savecsv <file>).");
}

