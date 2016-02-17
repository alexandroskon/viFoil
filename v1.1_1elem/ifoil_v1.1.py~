# IFOIL v1.1 - 1 ELEMENT CODE FOR TESTING OF THE METHOD

# A Linear-Strength Vortex/Source Panel Method for the prediction of airfoil performance. 
# The numerical scheme is described by Katz and Plotkin in "Low Speed Aerodynamics" 
# [Section 11.4.2] 

# Programmed by Alexandros Kontogiannis
# Feb 2016

#-----------------------------------------------------------------------
import numpy as np

# iFOIL Modules
from singulib import *
from gtrans import *
from airfoilmanip import *
from aeroplot import *

#-----------------------------------------------------------------------
# Input airfoil coordinates (in Xfoil format)
# XF,ZF are airfoil coordinates and N the number of panels
XF,ZF,N = INPUT_AIRFOIL('s1223')
PT1,PT2,XC,ZC,TH,n,t = FOIL_PANELING(XF,ZF,N)

# Initialize matrices
a     = np.zeros((N+1,N+1))
b     = np.zeros((N+1,N+1))
DL    = np.zeros(N)
RHS_F = np.zeros(N+1)
RHS_S = np.zeros(N+1)
RHS   = np.zeros(N+1)

# Setup aerodynamic influence coefficient (AIC) matrix 
#-----------------------------------------------------------------------
# i-th panel collocation point
for i in range(0,N):
    # j-th corner-point panel index (where singularities are placed)       
    for j in range(0,N):
        # Calculate velocities at i-th colloc. point due to singularities on 
        # the j-th and (j+1)-th corner points (linear singularity)   
        
        # Self-induced effect for i=j
        if i == j:
            U1,W1,U2,W2 = VOR2DL(1,1,XC[i],ZC[i],PT1[j,0],PT1[j,1],PT2[j,0],PT2[j,1],TH[j],True)
            # Save panel length for lift coefficient calc.
            DL[j] = PAN_LEN(PT1[j,0],PT1[j,1],PT2[j,0],PT2[j,1],TH[j])
        else:
            U1,W1,U2,W2 = VOR2DL(1,1,XC[i],ZC[i],PT1[j,0],PT1[j,1],PT2[j,0],PT2[j,1],TH[j],False)
        
        # Compute c[i,j] influences for unit-vorticity
        if (j == 0):
            a[i,0] =  U1*n[i,0] + W1*n[i,1]
            HOLDA  =  U2*n[i,0] + W2*n[i,1]
            b[i,0] =  U1*t[i,0] + W1*t[i,1]
            HOLDB  =  U2*t[i,0] + W2*t[i,1]
        elif (j == N-1):
            a[i,N-1] =  U1*n[i,0] + W1*n[i,1] + HOLDA
            a[i,N]   =  U2*n[i,0] + W2*n[i,1]
            b[i,N-1] =  U1*t[i,0] + W1*t[i,1] + HOLDB
            b[i,N]   =  U2*t[i,0] + W2*t[i,1]
        else:
            a[i,j]   =  U1*n[i,0] + W1*n[i,1] + HOLDA
            HOLDA    =  U2*n[i,0] + W2*n[i,1]
            b[i,j]   =  U1*t[i,0] + W1*t[i,1] + HOLDB
            HOLDB    =  U2*t[i,0] + W2*t[i,1]
          
          
# Setup RHS vector
#-----------------------------------------------------------------------
sig = np.zeros(N+1)

# DEFINE FREESTREAM AOA
ALPHA = input('Enter angle of attack (deg):\t')
AL=ALPHA/(180/np.pi)

for i in range(0,N):
    
    # RHS FREESTREAM terms computed at each collocation point 
    RHS_F[i] = -np.cos(AL)*n[i,0] - np.sin(AL)*n[i,1]
    
    # RHS SOURCE terms computed at each collocation point
    # Source terms are employed for the WALL-TRANSPIRATION MODEL
    # modeling the BL's dispacement thickness 
    SUM = 0
    for j in range(0,N):
        
        #if (i == j or (i == 0 and j == N-1) or (i == N-1 and j == 0)):
        if (i == j):
            U1,W1,U2,W2 = SOR2DL(sig[j],sig[j+1],XC[i],ZC[i],PT1[j,0],PT1[j,1],PT2[j,0],PT2[j,1],TH[j],True)
        else:
            U1,W1,U2,W2 = SOR2DL(sig[j],sig[j+1],XC[i],ZC[i],PT1[j,0],PT1[j,1],PT2[j,0],PT2[j,1],TH[j],False)
        SUM = SUM - ((U1+U2)*n[i,0] + (W1+W2)*n[i,1])    
    RHS_S[i] = SUM

# Total RHS vector is due to freestream and sources
RHS = RHS_F + RHS_S

print('RHS_S')
print(RHS_S)
print('RHS_F')
print(RHS_F)
# Kutta - Condition
#-----------------------------------------------------------------------

# Proper treatment of SHARP and BLUNT trailing edge
SHARP_TE = True

# For BLUNT trailing edges M. Drela provides a correction formula
# [ref. XFOIL paper]

# For airfoils with SHARP TE nodes i=1 and i=N+1 coincide. To circumvent
# this problem the i=N equation is discarded and replaced by an interpolation
# of the mean gamma to the TE. [p.3 in Ref. "XFOIL: A Design and Analysis System 
# for Low Reynolds number Airfoils", Mark Drela]
# (g_3-2*g_2+g_1) - (g_n-1 - 2*g_n + g_n+1) = 0 is the interpolation relation

if SHARP_TE:
    # Impose Interpolated Kutta Condition
    a[N,0]   =  1
    a[N,1]   = -2
    a[N,2]   =  1
    a[N,N-2] =  1
    a[N,N-1] = -2
    a[N,N]   =  1
else:
    # Impose Standard Kutta Condition
    a[N,0] = 1
    a[N,N] = 1


# Solve system to obtain unit-vorticity
#-----------------------------------------------------------------------
g = np.linalg.solve(a,RHS)

# Compute wake trajectory
#-----------------------------------------------------------------------

# Number of wake points
# XFOIL -> NW = N/8 + 2
NW = N/4+2

# Update matrix size to include wake
XF = np.append(XF,np.zeros(NW),axis=0)
ZF = np.append(ZF,np.zeros(NW),axis=0)
TH = np.append(TH,np.zeros(NW),axis=0)
DL = np.append(DL,np.zeros(NW),axis=0)
n  = np.append(n,np.zeros([NW,2]),axis=0)

# Set first wake point a tiny distance behind TE
# Treat SHARP and BLUNT trailing edge in separate
if SHARP_TE == False:
    XTE = 0.5*(XF[0]+XF[N])
    ZTE = 0.5*(ZF[0]+ZF[N])
    SX = 0.5*(ZF[N] - ZF[0])
    SZ = 0.5*(XF[0] - XF[N])
    SMOD = np.sqrt(SX**2 + SZ**2)
    n[N,0]  = SX/SMOD
    n[N,1]  = SZ/SMOD
    XF[N+1] = XTE - 0.0001*n[N,1]
    ZF[N+1] = ZTE + 0.0001*n[N,0]
else:
    XTE = XF[N]
    ZTE = ZF[N]
    n[N,0] = 0.5*(n[0,0]+n[N-1,0])   
    n[N,1] = 0.5*(n[0,1]+n[N-1,1])
    XF[N+1] = XTE - 0.0001*n[N,1]
    ZF[N+1] = ZTE + 0.0001*n[N,0]
    # Singularity issue?
    #XF[N+1] = XF[N]
    #ZF[N+1] = ZF[N]
    
#DL[N]   = S(N)

# Calculate velocity components at first point
U = 0; W = 0
for j in range(0,N):
    U1,W1,U2,W2 = VOR2DL(g[j],g[j+1],XF[N+1],ZF[N+1],PT1[j,0],PT1[j,1],PT2[j,0],PT2[j,1],TH[j],False)
    U = U + (U1+U2)
    W = W + (W1+W2)      

# Add freestream contribution
U = U + np.cos(AL)
W = W + np.sin(AL)

# Set unit vector normal to wake at first point
n[N+1,0] = W / np.sqrt(W**2 + U**2)
n[N+1,1] = U / np.sqrt(W**2 + U**2)
    
# Set rest of wake points
for i in range(N+2,N+NW):
    #DS = SNEW(I) - SNEW(I-1)
    DLW = 0.2*DL.max()
    
    # Set new point DL downstream of last point
    XF[i] = XF[i-1] + DLW*n[i-1,1]
    ZF[i] = ZF[i-1] + DLW*n[i-1,0]
    #S(I) = S(I-1) + DL
    
    # Set angle of wake panel normal
    TH[i] = np.arctan2((ZF[i]-ZF[i-1]),(XF[i]-XF[i-1]))
    
    if i == N+NW-1:
        continue
    
    # Calculate velocity components for next point
    U = 0; W = 0
    for j in range(0,N):
        U1,W1,U2,W2 = VOR2DL(g[j],g[j+1],XF[i],ZF[i],PT1[j,0],PT1[j,1],PT2[j,0],PT2[j,1],TH[j],False)
        U = U + (U1+U2)
        W = W + (W1+W2) 
    
    U = U + np.cos(AL)
    W = W + np.sin(AL)
    
    # Calculate normal vector for next point
    n[i,0] = W / np.sqrt(W**2 + U**2)
    n[i,1] = U / np.sqrt(W**2 + U**2)
    
    # set angle of wake panel normal
    # APANEL(I) = ATAN2( PSI_Y , PSI_X )
    
# Post-processing data
#-----------------------------------------------------------------------

# Aerodynamic computations
#-----------------------------------------------------------------------
# Primary airfoil
CP  = np.zeros(N)
V   = np.zeros(N)
CL  = 0

for i in range(0,N):
    VEL = 0
    for j in range(0,N+1):
        VEL = VEL + b[i,j]*g[j]
        
    V[i] = VEL + np.cos(AL)*t[i,0]+np.sin(AL)*t[i,1]
    CP[i] = 1-V[i]**2
    CL = CL + (-1*CP[i])*(np.cos(AL)*np.cos(TH[i]) + np.sin(AL)*np.sin(TH[i]))*DL[i]
    
    #REPORT BUG: CL != 0 FOR SYMMETRIC AIRFOILS
# Plot results
#-----------------------------------------------------------------------
aeroplot(RHS,CL,CP,g,XF,ZF,XC,ZC,AL,N,NW)

