import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title='2D Truss Damage Lab', layout='wide')
st.title('2D Truss Damage & Redistribution Lab')
st.caption('2D linear-elastic truss FEM • damage members • watch force paths migrate • monitor stiffness loss')

# ---------------- FEM core ----------------
def element_data(nodes, e, E, A, damage):
    i,j=e; xi,yi=nodes[i]; xj,yj=nodes[j]
    dx=xj-xi; dy=yj-yi; L=np.hypot(dx,dy); c=dx/L; s=dy/L
    scale=max(1.0-damage, 1e-8)
    k=(E*A*scale/L)*np.array([
        [c*c,c*s,-c*c,-c*s], [c*s,s*s,-c*s,-s*s],
        [-c*c,-c*s,c*c,c*s], [-c*s,-s*s,c*s,s*s]
    ])
    return L,c,s,k,scale

def assemble(nodes, elems, E, A, damage):
    nd=2*len(nodes); K=np.zeros((nd,nd))
    for m,e in enumerate(elems):
        L,c,s,k,_=element_data(nodes,e,E[m],A[m],damage[m])
        dof=[2*e[0],2*e[0]+1,2*e[1],2*e[1]+1]
        K[np.ix_(dof,dof)] += k
    return K

def solve(nodes, elems, E, A, damage, F, fixed):
    K=assemble(nodes,elems,E,A,damage); all_d=np.arange(len(F)); free=np.array([d for d in all_d if d not in fixed])
    Kff=K[np.ix_(free,free)]; Ff=F[free]
    eig=np.linalg.eigvalsh(Kff); tol=max(np.max(np.abs(eig))*1e-10 if len(eig) else 1e-12,1e-12)
    stable=np.min(eig)>tol
    if stable:
        uf=np.linalg.solve(Kff,Ff)
    else:
        uf=np.linalg.pinv(Kff,rcond=1e-10)@Ff
    U=np.zeros_like(F); U[free]=uf
    R=K@U-F
    return U,R,K,eig,stable

def member_results(nodes, elems, E, A, damage, U):
    rows=[]
    for m,e in enumerate(elems):
        L,c,s,_,scale=element_data(nodes,e,E[m],A[m],damage[m])
        dof=[2*e[0],2*e[0]+1,2*e[1],2*e[1]+1]; ue=U[dof]
        ext=np.dot(np.array([-c,-s,c,s]),ue)
        strain=ext/L; stress=E[m]*scale*strain; force=stress*A[m]
        rows.append((m,e[0],e[1],L,damage[m],strain,stress,force))
    return pd.DataFrame(rows,columns=['member','i','j','L','damage','strain','stress','axial_force'])

# Warren-ish redundant truss
nodes=np.array([[0.,0.],[2.,0.],[4.,0.],[6.,0.],[1.,2.],[3.,2.],[5.,2.]])
elems=[(0,1),(1,2),(2,3),(4,5),(5,6),(0,4),(4,1),(1,5),(5,2),(2,6),(6,3),(4,2),(1,6)]
nm=len(elems); E=np.full(nm,200e9); A=np.full(nm,3e-4)
labels=[f'M{m}: {i}–{j}' for m,(i,j) in enumerate(elems)]

with st.sidebar:
    st.header('Experiment')
    P=st.slider('Downward load at node 5 (kN)',5.0,100.0,40.0,5.0)*1000
    primary=st.selectbox('Primary damaged member',range(nm),format_func=lambda m:labels[m],index=7)
    d1=st.slider('Primary damage',0,100,55,1)/100
    second_on=st.checkbox('Add second damaged member',False)
    secondary=st.selectbox('Second member',range(nm),format_func=lambda m:labels[m],index=9,disabled=not second_on)
    d2=st.slider('Second damage',0,100,0,1,disabled=not second_on)/100
    amp=st.slider('Deformation amplification',1,500,100,5)
    st.caption('100% damage is represented by a tiny residual stiffness for numerical diagnostics.')

damage=np.zeros(nm); damage[primary]=d1
if second_on: damage[secondary]=max(damage[secondary],d2)
F=np.zeros(2*len(nodes)); F[2*5+1]=-P
# node 0 pin: ux,uy; node 3 roller: uy
fixed=[0,1,7]
U,R,K,eig,stable=solve(nodes,elems,E,A,damage,F,fixed)
res=member_results(nodes,elems,E,A,damage,U)
positive=eig[eig>0]; lam_min=float(np.min(positive)) if len(positive) else 0
cond=float(np.max(positive)/np.min(positive)) if len(positive) else np.inf

c1,c2,c3,c4=st.columns(4)
c1.metric('Status','Stable' if stable else 'Mechanism / near-singular')
c2.metric('Max |u|',f'{np.max(np.abs(U))*1000:.3f} mm')
c3.metric('Smallest K eigenvalue',f'{lam_min:.3e}')
c4.metric('K condition estimate',f'{cond:.2e}' if np.isfinite(cond) else '∞')

st.subheader('Structure, deformation and force path')
fig,ax=plt.subplots(figsize=(10,5))
# undeformed
for i,j in elems: ax.plot([nodes[i,0],nodes[j,0]],[nodes[i,1],nodes[j,1]],'--',linewidth=1,alpha=.35)
deformed=nodes+amp*U.reshape(-1,2)
maxf=max(res.axial_force.abs().max(),1.0)
for _,r in res.iterrows():
    m=int(r.member); i,j=elems[m]; f=r.axial_force; lw=1.5+5*abs(f)/maxf
    # default matplotlib colors: C3 tension, C0 compression, C7 near-zero
    color='C3' if f>1e-6 else ('C0' if f< -1e-6 else 'C7')
    ax.plot([deformed[i,0],deformed[j,0]],[deformed[i,1],deformed[j,1]],color=color,linewidth=lw,alpha=max(.2,1-damage[m]*.75))
    mid=(deformed[i]+deformed[j])/2; ax.text(mid[0],mid[1],str(m),fontsize=8)
ax.scatter(deformed[:,0],deformed[:,1],s=35,zorder=3)
for n,p in enumerate(deformed): ax.text(p[0],p[1]+.12,f'N{n}',ha='center',fontsize=9)
ax.set_aspect('equal'); ax.grid(alpha=.2); ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
ax.set_title(f'Deformed ×{amp} — red: tension, blue: compression; thickness ∝ |axial force|')
st.pyplot(fig,use_container_width=True); plt.close(fig)

left,right=st.columns([1.15,.85])
with left:
    st.subheader('Member mechanics')
    show=res.copy(); show['damage']*=100; show['stress']/=1e6; show['axial_force']/=1000
    show=show.rename(columns={'damage':'damage_%','stress':'stress_MPa','axial_force':'force_kN'})
    st.dataframe(show.style.format({'L':'{:.3f}','damage_%':'{:.0f}','strain':'{:.3e}','stress_MPa':'{:.3f}','force_kN':'{:.3f}'}),use_container_width=True)
with right:
    st.subheader('Support reactions')
    rdf=pd.DataFrame({'DOF':['N0 x','N0 y','N3 y'],'reaction_kN':[R[0]/1000,R[1]/1000,R[7]/1000]})
    st.dataframe(rdf.style.format({'reaction_kN':'{:.3f}'}),hide_index=True,use_container_width=True)
    st.subheader('Lowest stiffness modes')
    edf=pd.DataFrame({'mode':np.arange(1,min(7,len(eig))+1),'eigenvalue':eig[:6]})
    st.dataframe(edf.style.format({'eigenvalue':'{:.4e}'}),hide_index=True,use_container_width=True)

st.divider(); st.header('Progressive failure experiment')
scan_member=st.selectbox('Member to progressively remove',range(nm),format_func=lambda m:labels[m],index=primary,key='scan')
levels=np.linspace(0,.999,60); records=[]
base_damage=np.zeros(nm)
if second_on and secondary!=scan_member: base_damage[secondary]=d2
for d in levels:
    dd=base_damage.copy(); dd[scan_member]=d
    uu,_,_,ee,ss=solve(nodes,elems,E,A,dd,F,fixed)
    pos=ee[ee>0]
    records.append((d*100,np.max(np.abs(uu))*1000,np.min(pos) if len(pos) else 0,ss))
prog=pd.DataFrame(records,columns=['damage_%','max_disp_mm','lambda_min','stable'])
a,b=st.columns(2)
with a:
    fig,ax=plt.subplots(); ax.plot(prog['damage_%'],prog.max_disp_mm); ax.set_xlabel('Damage (%)'); ax.set_ylabel('Max displacement (mm)'); ax.grid(alpha=.25); st.pyplot(fig,use_container_width=True); plt.close(fig)
with b:
    fig,ax=plt.subplots(); ax.semilogy(prog['damage_%'],np.maximum(prog.lambda_min,1e-12)); ax.set_xlabel('Damage (%)'); ax.set_ylabel('Smallest stiffness eigenvalue'); ax.grid(alpha=.25); st.pyplot(fig,use_container_width=True); plt.close(fig)

st.header('Force redistribution: healthy vs damaged')
Uh,*_=solve(nodes,elems,E,A,np.zeros(nm),F,fixed); healthy=member_results(nodes,elems,E,A,np.zeros(nm),Uh)
comp=pd.DataFrame({'member':np.arange(nm),'healthy_kN':healthy.axial_force/1000,'damaged_kN':res.axial_force/1000})
comp['change_kN']=comp.damaged_kN-comp.healthy_kN
st.bar_chart(comp.set_index('member')[['healthy_kN','damaged_kN']])
st.dataframe(comp.style.format({'healthy_kN':'{:.3f}','damaged_kN':'{:.3f}','change_kN':'{:+.3f}'}),hide_index=True,use_container_width=True)

st.info('Interpretation: this is a small-displacement, pin-jointed, linear-elastic truss model. A falling reduced-stiffness eigenvalue is a useful mechanism/stiffness-loss diagnostic here, but it is not a general nonlinear buckling or collapse criterion.')
