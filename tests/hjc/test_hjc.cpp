// SPDX-License-Identifier: LGPL-2.1-or-later
#include "HjcTests.h"
#include "HjcReference.h"
#include "HjcKinematics.h"
#include <iostream>

static void kinematics() {
  const hjc::Parameters p=hjc_tests::parameters();
  hjc::Particle point=hjc::initializeParticle(p);
  double f[9]={1,0,0,0,1,0,0,0,1},pk[9];
  hjc_tests::check(hjc::advance(p,f,1.e-6,point,pk),"identity deformation");
  for(double x:point.state.stress) hjc_tests::close(x,0.,1.e-6,"stress-free identity");

  // Finite rigid rotation of a nonzero, elastic stress tensor.
  point.state.stress[0]=1.e6; point.state.stress[1]=-2.e6;
  point.state.stress[2]=1.e6; point.state.stress[3]=.4e6;
  const double a=.73,c=std::cos(a),s=std::sin(a);
  const double rotation[9]={c,-s,0,s,c,0,0,0,1};
  const double initial[9]={1.e6,.4e6,0,.4e6,-2.e6,0,0,0,1.e6};
  double temp[9],expected[9];
  hjc::multiply(rotation,initial,temp);hjc::multiply(temp,rotation,expected,true);
  hjc_tests::check(hjc::advance(p,rotation,1.e-6,point,pk),"rigid rotation rejected");
  const int map[6]={0,4,8,1,5,2};
  for(int i=0;i<6;++i) hjc_tests::close(point.state.stress[i],expected[map[i]],.001,"rigid rotation objectivity");
  hjc_tests::close(point.internal_work,0.,1.e-5,"rigid rotation did work");

  // Noncommuting stretch/shear path, with a changing superposed rotation.
  hjc::Particle plain=hjc::initializeParticle(p),turned=plain;
  for(int k=1;k<=500;++k) {
    const double t=k/500.;
    const double F[9]={std::exp(-.02*t),.025*t,.005*t,0,1,.015*t,0,0,1};
    const double angle=1.7*t,ca=std::cos(angle),sa=std::sin(angle);
    const double Q[9]={ca,0,sa,0,1,0,-sa,0,ca};
    double rotatedF[9],pkA[9],pkB[9],expectedPk[9];
    hjc::multiply(Q,F,rotatedF);
    hjc_tests::check(hjc::advance(p,F,2.e-7,plain,pkA),"plain mixed path");
    hjc_tests::check(hjc::advance(p,rotatedF,2.e-7,turned,pkB),"rotated mixed path");
    hjc::multiply(Q,pkA,expectedPk);
    for(int i=0;i<9;++i) hjc_tests::close(pkB[i],expectedPk[i],.1,"PK1 frame invariance");
    hjc_tests::close(plain.state.damage,turned.state.damage,1.e-8,"rotated damage");
    hjc_tests::close(plain.state.plastic_volume,turned.state.plastic_volume,1.e-10,"rotated plastic volume");
    hjc_tests::close(plain.internal_work,turned.internal_work,.01,"rotated work");
  }
  const hjc::Particle saved=point;
  f[0]=-1.;
  hjc_tests::check(!hjc::advance(p,f,1.e-6,point,pk),"negative Jacobian accepted");
  hjc_tests::close(point.state.damage,saved.state.damage,0.,"rejected step changed history");
  for(int i=0;i<9;++i) hjc_tests::close(point.deformation[i],saved.deformation[i],0.,"rejected step changed F");
}
int main() {
  try { hjc_tests::runAnalytic(); hjc_tests::runReference(); kinematics(); }
  catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
  std::cout<<"HJC material, R13 transitions and finite-rotation tests passed\n";
}
