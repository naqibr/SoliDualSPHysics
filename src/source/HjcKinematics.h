// SPDX-License-Identifier: LGPL-2.1-or-later
#ifndef HJC_KINEMATICS_H
#define HJC_KINEMATICS_H
#include "HjcMaterial.h"
#ifdef __CUDACC__
#define HJC_HD __host__ __device__
#else
#define HJC_HD
#endif

namespace hjc {

HJC_HD inline double determinant(const double* f) {
  return f[0]*(f[4]*f[8]-f[5]*f[7])-f[1]*(f[3]*f[8]-f[5]*f[6])+
         f[2]*(f[3]*f[7]-f[4]*f[6]);
}

HJC_HD inline void inverse(const double* f,double* a,double j) {
  a[0]=(f[4]*f[8]-f[5]*f[7])/j; a[1]=(f[2]*f[7]-f[1]*f[8])/j;
  a[2]=(f[1]*f[5]-f[2]*f[4])/j; a[3]=(f[5]*f[6]-f[3]*f[8])/j;
  a[4]=(f[0]*f[8]-f[2]*f[6])/j; a[5]=(f[2]*f[3]-f[0]*f[5])/j;
  a[6]=(f[3]*f[7]-f[4]*f[6])/j; a[7]=(f[1]*f[6]-f[0]*f[7])/j;
  a[8]=(f[0]*f[4]-f[1]*f[3])/j;
}

HJC_HD inline void multiply(const double* a,const double* b,double* c,bool bt=false) {
  for(int i=0;i<3;++i) for(int j=0;j<3;++j) {
    c[3*i+j]=0.0;
    for(int k=0;k<3;++k) c[3*i+j]+=a[3*i+k]*b[bt?3*j+k:3*k+j];
  }
}

// Symmetric eigensystem by bounded Jacobi sweeps, including repeated eigenvalues.
HJC_HD inline bool eigensystem(double* a,double* v) {
  for(int i=0;i<9;++i) v[i]=(i==0 || i==4 || i==8)?1.0:0.0;
  for(int sweep=0;sweep<32;++sweep) {
    int p=0,q=1;
    if(fabs(a[2])>fabs(a[1])) q=2;
    if(fabs(a[5])>fabs(a[3*p+q])) {p=1;q=2;}
    const double apq=a[3*p+q];
    const double size=fmax(fabs(a[0]),fmax(fabs(a[4]),fabs(a[8])));
    if(fabs(apq)<=1.e-14*size) return a[0]>0 && a[4]>0 && a[8]>0;
    const double tau=(a[3*q+q]-a[3*p+p])/(2.0*apq);
    const double t=(tau>=0?1.0:-1.0)/(fabs(tau)+sqrt(1.0+tau*tau));
    const double c=1.0/sqrt(1.0+t*t),s=t*c;
    a[3*p+p]-=t*apq; a[3*q+q]+=t*apq;
    a[3*p+q]=a[3*q+p]=0.0;
    for(int k=0;k<3;++k) {
      if(k!=p && k!=q) {
        const double x=a[3*k+p],y=a[3*k+q];
        a[3*k+p]=a[3*p+k]=c*x-s*y;
        a[3*k+q]=a[3*q+k]=s*x+c*y;
      }
      const double x=v[3*k+p],y=v[3*k+q];
      v[3*k+p]=c*x-s*y; v[3*k+q]=s*x+c*y;
    }
  }
  return false;
}

// Incremental polar rotation and logarithmic strain preserve finite rigid-body
// rotations. Input/output matrices use row-major order. A failed update leaves
// the material history unchanged; the host reports the invalid deformation.
HJC_HD inline bool advance(const Parameters& p,const double* f,double dt,
                           Particle& point,double* pk1) {
  const double j=determinant(f),oldj=determinant(point.deformation);
  if(!(j>1.e-12 && oldj>1.e-12 && dt>0.0)) return false;
  double invold[9],inc[9],left[9],v[9],vinv[9]={},logv[9]={},rotation[9];
  inverse(point.deformation,invold,oldj); multiply(f,invold,inc);
  multiply(inc,inc,left,true);
  if(!eigensystem(left,v)) return false;
  for(int i=0;i<3;++i) for(int k=0;k<3;++k) for(int l=0;l<3;++l) {
    const double weight=v[3*i+l]*v[3*k+l];
    vinv[3*i+k]+=weight/sqrt(left[4*l]);
    logv[3*i+k]+=0.5*weight*log(left[4*l]);
  }
  multiply(vinv,inc,rotation);
  State next=point.state;
  const double* s=next.stress;
  double stress[9]={s[0],s[3],s[5],s[3],s[1],s[4],s[5],s[4],s[2]};
  double temp[9],rotated[9];
  multiply(rotation,stress,temp); multiply(temp,rotation,rotated,true);
  next.stress[0]=rotated[0]; next.stress[1]=rotated[4]; next.stress[2]=rotated[8];
  next.stress[3]=rotated[1]; next.stress[4]=rotated[5]; next.stress[5]=rotated[2];
  double deps[6]={logv[0],logv[4],logv[8],2*logv[1],2*logv[5],2*logv[2]};
  double oldstress[6]; for(int i=0;i<6;++i) oldstress[i]=next.stress[i];
  const Result result=update(p,deps,dt,next);
  for(int i=0;i<6;++i) if(!(fabs(next.stress[i])<1.e300)) return false;
  if(!(result.acoustic_modulus>0 && result.acoustic_modulus<1.e300)) return false;
  const double current[9]={next.stress[0],next.stress[3],next.stress[5],
    next.stress[3],next.stress[1],next.stress[4],next.stress[5],next.stress[4],next.stress[2]};
  double invf[9]; inverse(f,invf,j); multiply(current,invf,pk1,true);
  for(int i=0;i<9;++i) {pk1[i]*=j;point.deformation[i]=f[i];}
  double work=0;
  for(int i=0;i<6;++i) work+=0.5*(oldstress[i]+next.stress[i])*deps[i];
  point.internal_work+=sqrt(j*oldj)*work;
  point.state=next; point.acoustic_modulus=result.acoustic_modulus;
  return true;
}

}
#undef HJC_HD
#endif
