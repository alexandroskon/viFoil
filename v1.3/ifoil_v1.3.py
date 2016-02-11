# IFOIL v. 1.2

# A Linear-Strength Vortex Panel Method for the prediction of
# pressure distribution over a multi-element airfoil or biplane arrangement.
# The numerical scheme is described by Katz and Plotkin in "Low Speed Aerodynamics" 
# [Section 11.4.2] 

# Programmed by Alexandros Kontogiannis
# November 2015

#-----------------------------------------------------------------------
import numpy as np

# IFOIL Modules
from singulib import *
from gtrans import *
from airfoilmanip import *
from aeroplot import *

#-----------------------------------------------------------------------
# Input airfoil coordinates (in Xfoil format)
# XF,ZF are airfoil coordinates and N the number of panels
XF,ZF,N = INPUT_AIRFOIL('NLR7301')
XF2,ZF2,N2 = INPUT_AIRFOIL('FLAP')

# Set location of secondary airfoil
#XF2 = XF2 + 0.3
#ZF2 = ZF2 - 10000.0

# Airfoil paneling
#-----------------------------------------------------------------------
# --------Panel Endpoints--------Colloc. Points-------------------------
# PT1[i,0] -> Xj  PT2[i,0] -> Xj+1  XC -> Xi  n[i,0] -> nx  t[i,0] -> tx   
# PT1[i,1] -> Zj  PT2[i,1] -> Zj+1  ZC -> Zi  n[i,1] -> nz  t[i,1] -> tz
# TH is the panel angle and AL the AoA of the airfoil (freestream)

PT11,PT21,XC1,ZC1,TH1,n1,t1,AL1 = FOIL_PANELING(XF,ZF,N)
PT12,PT22,XC2,ZC2,TH2,n2,t2,AL2 = FOIL_PANELING(XF2,ZF2,N2)

# Merge airfoil matrices for a more compact algorithm
PT1 = np.append(PT11,PT12,axis=0)
PT2 = np.append(PT21,PT22,axis=0)
XC  = np.append(XC1,XC2)
ZC  = np.append(ZC1,ZC2)
TH  = np.append(TH1,TH2)
n   = np.append(n1,n2,axis=0)
t   = np.append(t1,t2,axis=0)
AL  = np.append(AL1,AL2)

# Initialize matrices
a     = np.zeros((N+N2+2,N+N2+2))
b     = np.zeros((N+N2+2,N+N2+2))
DL    = np.zeros(N+N2+1)
RHS_F = np.zeros(N+N2+2)
RHS_S = np.zeros(N+N2+2)
RHS   = np.zeros(N+N2+2)

# Setup aerodynamic influence coefficient (AIC) matrix 
#-----------------------------------------------------------------------
# i-th panel collocation point
for I in range(0,N+N2+1):
    
    # Change of indexing due to Kutta Condition
    # Indeces of airfoil coordinates lag one step in comparison with the 
    # indeces of influence coefficients
    if I < N:
        i = I
    elif I == N:
        continue
    else:
        i = I-1
    
    # j-th corner-point panel index (where singularities are placed)       
    for J in range(0,N+N2+1):
        # Calculate velocities at i-th colloc. point due to singularities on 
        # the j-th and (j+1)-th corner points (linear singularity)
        
        # Indeces of airfoil coordinates lag one step in comparison with the 
        # indeces of influence coefficients
        if J < N:
            j = J
        elif J == N:
            continue
        else:
            j = J-1     
        
        # Self-induced effect for i=j
        if i == j:
            U1,W1,U2,W2 = VOR2DL(1,1,XC[i],ZC[i],PT1[j,0],PT1[j,1],PT2[j,0],PT2[j,1],TH[j],True)
            # Save panel length for lift coefficient calc.
            DL[j] = PAN_LEN(PT1[j,0],PT1[j,1],PT2[j,0],PT2[j,1],TH[j])
        else:
            U1,W1,U2,W2 = VOR2DL(1,1,XC[i],ZC[i],PT1[j,0],PT1[j,1],PT2[j,0],PT2[j,1],TH[j],False)
        
        
        # Compute c[i,j] influences for unit-vorticity
        if (J == 0 or J == N+1):
            a[I,J] =  U1*n[i,0] + W1*n[i,1]
            HOLDA  =  U2*n[i,0] + W2*n[i,1]
            b[I,J] =  U1*t[i,0] + W1*t[i,1]
            HOLDB  =  U2*t[i,0] + W2*t[i,1]
        elif (J == N-1 or J == N+N2):
            a[I,J]   =  U1*n[i,0] + W1*n[i,1] + HOLDA
            a[I,J+1] =  U2*n[i,0] + W2*n[i,1]
            b[I,J]   =  U1*t[i,0] + W1*t[i,1] + HOLDB
            b[I,J+1] =  U2*t[i,0] + W2*t[i,1]
        else:
            a[I,J]   =  U1*n[i,0] + W1*n[i,1] + HOLDA
            HOLDA    =  U2*n[i,0] + W2*n[i,1]
            b[I,J]   =  U1*t[i,0] + W1*t[i,1] + HOLDB
            HOLDB    =  U2*t[i,0] + W2*t[i,1]
          
# Setup RHS vector
#-----------------------------------------------------------------------
sig = np.zeros(N+N2+2)

for I in range(0,N+N2+1):
    
    # Change of indexing due to Kutta Condition
    # Indeces of airfoil coordinates lag one step in comparison with the 
    # indeces of influence coefficients
    if I < N:
        i = I
    elif I == N:
        continue
    else:
        i = I-1
    
    # RHS FREESTREAM terms computed at each collocation point 
    RHS_F[I] = -np.cos(AL[0])*n[i,0] - np.sin(AL[0])*n[i,1]
    
    # RHS SOURCE terms computed at each collocation point
    # Source terms are employed for the WALL-TRANSPIRATION MODEL
    # modeling the BL's dispacement thickness 
    SUM = 0
    for J in range(0,N+N2+1):
        
        # Indeces of airfoil coordinates lag one step in comparison with the 
        # indeces of influence coefficients
        if J < N:
            j = J
        elif J == N:
            continue
        else:
            j = J-1 
        
        if (i == j):
            U1,W1,U2,W2 = SOR2DL(sig[j],sig[j+1],XC[i],ZC[i],PT1[j,0],PT1[j,1],PT2[j,0],PT2[j,1],TH[j],True)
        else:
            U1,W1,U2,W2 = SOR2DL(sig[j],sig[j+1],XC[i],ZC[i],PT1[j,0],PT1[j,1],PT2[j,0],PT2[j,1],TH[j],False)
        SUM = SUM - ((U1+U2)*n[i,0] + (W1+W2)*n[i,1])    
    RHS_S[I] = SUM

# Total RHS vector is due to freestream and sources
RHS = RHS_F + RHS_S 

# Kutta - Condition
#-----------------------------------------------------------------------
# For airfoils with SHARP TE nodes i=1 and i=N+1 coincide. To circumvent
# this problem the i=N equation is discarded and replaced by an interpolation
# of the mean gamma to the TE. [p.3 in Ref. "XFOIL: A Design and Analysis System 
# for Low Reynolds number Airfoils", Mark Drela]
# (g_3-2*g_2+g_1) - (g_n-1 - 2*g_n + g_n+1) = 0, the interpolation relation
#a[N,0]   =  1
#a[N,1]   = -2
#a[N,2]   =  1
#a[N,N-2] = -1
#a[N,N-1] =  2
#a[N,N]   = -1

# Impose Kutta Condition
# Primary Airfoil
a[N,0] = 1
a[N,N] = 1
# Secondary Airfoil
a[N+N2+1,N+1]  = 1
a[N+N2+1,N+N2+1] = 1

print((a))
print((RHS))
# Solve system to obtain unit-vorticity
#-----------------------------------------------------------------------
g = np.linalg.solve(a,RHS)

# Post-processing data
#-----------------------------------------------------------------------

# Aerodynamic computations
#-----------------------------------------------------------------------
# Primary airfoil
CP1  = np.zeros(N)
V1   = np.zeros(N)
CL1  = 0
for i in range(0,N):
    VEL = 0
    for j in range(0,N+N2+2):
        VEL = VEL + b[i,j]*g[j]
        
    V1[i] = VEL + np.cos(AL[0])*t[i,0]+np.sin(AL[0])*t[i,1]
    CP1[i] = 1-V1[i]**2
    CL1 = CL1 + (-1*CP1[i])*(np.cos(AL[0])*np.cos(TH[i]) + np.sin(AL[0])*np.sin(TH[i]))*DL[i]

# Secondary airfoil
CP2  = np.zeros(N2)
V2   = np.zeros(N2)
CL2  = 0
for i in range(N+1,N+N2+1):
    I = i - (N+1)
    VEL = 0
    for j in range(0,N+N2+2):
        VEL = VEL + b[i,j]*g[j]
        
    V2[I] = VEL + np.cos(AL[1])*t[i-1,0]+np.sin(AL[1])*t[i-1,1]
    CP2[I] = 1-V2[I]**2
    CL2 = CL2 + (-1*CP2[I])*(np.cos(AL[1])*np.cos(TH[i-1]) + np.sin(AL[1])*np.sin(TH[i-1]))*DL[i-1]
  
# Plot results
#-----------------------------------------------------------------------
aeroplot(RHS,CL1,CL2,CP1,CP2,g,XF,ZF,XF2,ZF2,XC1,ZC1,XC2,ZC2,AL)

