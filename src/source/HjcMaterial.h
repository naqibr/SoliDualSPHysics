// SPDX-License-Identifier: LGPL-2.1-or-later
// Holmquist-Johnson-Cook concrete, with the documented R13-compatible update.
#ifndef HJC_MATERIAL_H
#define HJC_MATERIAL_H

#include <cmath>
#ifdef __CUDACC__
#define HJC_HD __host__ __device__
#else
#define HJC_HD
#endif

namespace hjc {

struct Parameters {
  double g, a, b, c, n, fc, tension, rate0, efmin, sfmax;
  double pc, muc, pl, mul, d1, d2, k1, k2, k3;
};

// Stress ordering: xx, yy, zz, xy, yz, zx. All stresses are Cauchy stresses.
// Compression is positive for pressure and plastic_volume, negative for strain.
struct State {
  double stress[6];
  double volume_strain, plastic_volume, max_compression, density_measure;
  double damage, plastic_strain;
};

struct Result {
  double pressure, equivalent_stress, fracture_strain, acoustic_modulus;
};

struct Particle {
  State state;
  double deformation[9];
  double acoustic_modulus, internal_work, previous_dt;
};

inline bool valid(const Parameters& p) {
  const double values[] = {p.g,p.a,p.b,p.c,p.n,p.fc,p.tension,p.rate0,
    p.efmin,p.sfmax,p.pc,p.muc,p.pl,p.mul,p.d1,p.d2,p.k1,p.k2,p.k3};
  for(unsigned i=0;i<sizeof(values)/sizeof(values[0]);++i)
    if(!std::isfinite(values[i])) return false;
  return p.g>0 && p.a>=0 && p.b>=0 && p.c>=0 && p.n>0 && p.fc>0 &&
    p.tension>0 && p.rate0>0 && p.efmin>=0 && p.sfmax>0 &&
    p.pc>0 && p.muc>0 && p.pl>p.pc && p.mul>0 && p.d1>0 && p.d2>=0 &&
    p.k1>0 && p.mul+p.pl/p.k1>p.muc;
}

HJC_HD inline State initialize(const Parameters& p) {
  State s = {};
  const double ef = p.d1*pow(p.tension/p.fc,p.d2);
  // R13's cold-start convention; this is not an imposed physical prestrain.
  if(ef>p.efmin) s.damage=fmin(1.0,1.e-4/ef);
  return s;
}

HJC_HD inline Particle initializeParticle(const Parameters& p) {
  Particle point={};
  point.state=initialize(p);
  point.deformation[0]=point.deformation[4]=point.deformation[8]=1.0;
  point.acoustic_modulus=p.pc/p.muc+4.0*p.g/3.0;
  return point;
}

// deps contains engineering shear increments. The caller supplies an objective
// frame and manages initialization, rotations, timestep acceptance and erosion.
HJC_HD inline Result update(const Parameters& p, const double deps[6],
                            double dt, State& s) {
  const double third=(deps[0]+deps[1]+deps[2])/3.0;
  double dev[6];
  for(int i=0;i<3;++i) dev[i]=deps[i]-third;
  for(int i=3;i<6;++i) dev[i]=0.5*deps[i];
  const double norm=dev[0]*dev[0]+dev[1]*dev[1]+dev[2]*dev[2]+
    2.0*(dev[3]*dev[3]+dev[4]*dev[4]+dev[5]*dev[5]);
  const double rate=dt>0 ? sqrt(2.0/3.0)*sqrt(norm)/dt : 0.0;
  const double oldp=-(s.stress[0]+s.stress[1]+s.stress[2])/3.0;
  for(int i=0;i<3;++i) s.stress[i]+=oldp+2.0*p.g*dev[i];
  for(int i=3;i<6;++i) s.stress[i]+=p.g*deps[i];
  const double trial=sqrt(1.5*(s.stress[0]*s.stress[0]+
    s.stress[1]*s.stress[1]+s.stress[2]*s.stress[2])+
    3.0*(s.stress[3]*s.stress[3]+s.stress[4]*s.stress[4]+
    s.stress[5]*s.stress[5]));

  s.volume_strain+=deps[0]+deps[1]+deps[2];
  const double mu=expm1(-s.volume_strain);
  const double k0=p.pc/p.muc;
  double pressure, dpdmu, dmup=0.0;
  // The previous density measure selects the EOS branch and unloading modulus.
  if(s.density_measure<1.0) {
    const double k=k0+(p.k1-k0)*s.density_measure;
    const double h=(p.pl-p.pc)/p.mul;
    const double trialp=k*(mu-s.plastic_volume);
    dmup=fmax(0.0,(trialp-p.pc-h*s.plastic_volume)/(k+h));
    s.plastic_volume+=dmup;
    pressure=k*(mu-s.plastic_volume);
    dpdmu=k;
  } else {
    const double m=(mu-p.mul)/(1.0+p.mul);
    pressure=p.k1*m+p.k2*m*m+p.k3*m*m*m;
    dpdmu=(p.k1+2.0*p.k2*m+3.0*p.k3*m*m)/(1.0+p.mul);
  }
  s.max_compression=fmax(s.max_compression,mu);
  s.density_measure=fmin(1.0,fmax(0.0,
    (s.max_compression-p.muc)/(p.mul+p.pl/p.k1-p.muc)));
  pressure=fmax(pressure,-p.tension*(1.0-s.damage));

  double strength=pressure<0.0 ?
    fmax(0.0,p.a*(1.0-s.damage-pressure/p.tension)) :
    p.a*(1.0-s.damage)+p.b*pow(pressure/p.fc,p.n);
  // Preserve the sub-reference logarithm; zero strength is bounded for
  // parameters/rates outside the finite positive-strength reference paths.
  strength=fmax(0.0,fmin(p.sfmax,
    strength*(1.0+p.c*log(fmax(rate,1.e-8)/p.rate0))));
  const double normalized_trial=trial/p.fc;
  const double scale=normalized_trial<strength ? 1.0 :
    strength/fmax(normalized_trial,1.e-12);
  for(int i=0;i<6;++i) s.stress[i]*=scale;
  const double dep=strength>0.0 ? (1.0-scale)*trial/(3.0*p.g) : 0.0;
  s.plastic_strain+=dep;
  const double ef=p.d1*pow(fmax(pressure/p.fc+p.tension/p.fc,0.0),p.d2);
  // R13 EFMIN is an activation threshold, not a denominator floor.
  if(ef>p.efmin) s.damage=fmin(1.0,s.damage+(dep+dmup)/ef);
  for(int i=0;i<3;++i) s.stress[i]-=pressure;
  Result r={pressure,trial*scale,ef,
    fmax(0.0,1.0+mu)*fmax(0.0,dpdmu)+4.0*p.g/3.0};
  return r;
}

} // namespace hjc
#undef HJC_HD
#endif
