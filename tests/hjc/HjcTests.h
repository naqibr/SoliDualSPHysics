// SPDX-License-Identifier: LGPL-2.1-or-later
#ifndef HJC_TESTS_H
#define HJC_TESTS_H
#include "HjcMaterial.h"
#include <limits>
#include <stdexcept>

namespace hjc_tests {
inline hjc::Parameters parameters() {
  hjc::Parameters p={2.417e10,.29,2.06,.0013,.866,1.19e8,8.2e6,1.,.01,5.,
    4.e7,.00124,1.2e9,.011,.04,1.,1.287e10,1.631e10,6.495e10};
  return p;
}
inline void check(bool ok,const char* message) {
  if(!ok) throw std::runtime_error(message);
}
inline void close(double a,double b,double atol,const char* message) {
  check(std::isfinite(a) && std::fabs(a-b)<=atol,message);
}

inline void runAnalytic() {
  hjc::Parameters p=parameters();
  check(hjc::valid(p),"valid parameters rejected");
  hjc::Parameters invalid=p; invalid.muc=0;
  check(!hjc::valid(invalid),"zero crushing strain accepted");
  invalid=p; invalid.fc=std::numeric_limits<double>::quiet_NaN();
  check(!hjc::valid(invalid),"NaN accepted");
  hjc::State s=hjc::initialize(p);
  const double hydro[6]={-1.e-6,-1.e-6,-1.e-6,0,0,0};
  hjc::Result r=hjc::update(p,hydro,1.e-6,s);
  close(r.pressure,p.pc/p.muc*std::expm1(3.e-6),1.e-4,"elastic hydrostatic pressure");
  close(s.plastic_volume,0.,0.,"elastic volume must be reversible");
  close(s.plastic_strain,0.,1.e-14,"hydrostatic plastic shear strain");
  for(int axis=3;axis<6;++axis) {
    s=hjc::initialize(p); double e[6]={}; e[axis]=1.e-6;
    r=hjc::update(p,e,1.e-6,s);
    close(s.stress[axis],p.g*e[axis],1.e-6,"engineering shear convention");
    close(r.equivalent_stress,std::sqrt(3.)*p.g*e[axis],1.e-6,"von Mises shear invariant");
    e[axis]*=-1; hjc::update(p,e,1.e-6,s);
    close(s.stress[axis],0.,1.e-6,"elastic shear unloading");
  }
  // The R13 EFMIN gate is observable: shear alone with these parameters
  // accumulates plastic strain but does not activate damage.
  s=hjc::initialize(p); const double shear[6]={0,0,0,.01,0,0};
  r=hjc::update(p,shear,1.e-4,s);
  check(s.plastic_strain>0,"plastic return inactive");
  close(s.damage,0.,0.,"EFMIN gate");
  p.d1=4.; s=hjc::initialize(p);
  const double initial=s.damage;
  check(initial>0 && initial<1,"R13 initial damage");
  for(int i=0;i<120;++i) {
    const double oldD=s.damage,oldEp=s.plastic_strain;
    r=hjc::update(p,shear,1.e-4,s);
    check(s.damage>=oldD && s.damage<=1 && s.plastic_strain>=oldEp,"irreversible history");
    check(std::isfinite(r.pressure) && r.acoustic_modulus>0,"finite result");
  }
  close(s.damage,1.,1.e-12,"damage saturation");
  const double compression[6]={-.01,-.01,-.01,.01,0,0};
  r=hjc::update(p,compression,1.e-4,s);
  check(r.pressure>0 && r.equivalent_stress>0,"damaged concrete retains confined strength");
  const double oldMup=s.plastic_volume;
  const double release[6]={.001,.001,.001,0,0,0};
  hjc::update(p,release,1.e-4,s);
  close(s.plastic_volume,oldMup,1.e-14,"plastic volume healed during unloading");
  // Very low rates cannot produce a negative radial scaling factor.
  p.c=1.; s=hjc::initialize(p);
  r=hjc::update(p,shear,1.e8,s);
  check(r.equivalent_stress>=0 && std::isfinite(r.equivalent_stress),"negative rate strength");
}
}
#endif
