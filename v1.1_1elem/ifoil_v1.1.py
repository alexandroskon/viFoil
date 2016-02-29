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
XF,ZF,N = INPUT_AIRFOIL('sd7037')
PT1,PT2,XC,ZC,TH,n,t = FOIL_PANELING(XF,ZF,N)

# Initialize matrices
a     = np.zeros((N+1,N+1))
b     = np.zeros((N+1,N+1))
DL    = np.zeros(N)
RHS_F = np.zeros(N+1)
RHS_S = np.zeros(N+1)
RHS   = np.zeros(N+1)

# Setup aerodynamic influence coefficient (AIC) matrix
# or unit-vorticity matrix 
#-----------------------------------------------------------------------
# i-th panel collocation point
for i in range(0,N):
    # j-th corner-point panel index (where singularities are placed)       
    for j in range(0,N):
        # Calculate velocities at i-th colloc. point due to singularities on 
        # the j-th and (j+1)-th corner points (linear singularity)   
        
        # Self-induced effect for i=j
        if i == j:
            U1,W1,U2,W2 = VOR2DL(1,1,XC[i],ZC[i],PT1[j,0],PT1[j,1],PT2[j,0], \
            PT2[j,1],TH[j],True)
            # Save panel length for lift coefficient calc.
            DL[j] = PAN_LEN(PT1[j,0],PT1[j,1],PT2[j,0],PT2[j,1],TH[j])
        else:
            U1,W1,U2,W2 = VOR2DL(1,1,XC[i],ZC[i],PT1[j,0],PT1[j,1],PT2[j,0],PT2[j,1],\
            TH[j],False)
        
        # Compute influences for unit-vorticity
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

# Define freestream angle of attack
ALPHA = input('Enter angle of attack (deg):\t')
AL=ALPHA/(180/np.pi)

for i in range(0,N):
    # RHS FREESTREAM terms computed at each collocation point 
    RHS_F[i] = -np.cos(AL)*n[i,0] - np.sin(AL)*n[i,1]    

# Total RHS vector is due to freestream and sources
#RHS = RHS_F + RHS_S
RHS = RHS_F


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

# FUTURE CORRECTION for blunt trailing edges: need to close i=0 and i=N 
# panel with a i=N+1 panel to properly calculate Cp. 
    
# Solve system to obtain unit-vorticity (dense matrix)
#-----------------------------------------------------------------------
g = np.linalg.solve(a,RHS)


# Compute wake trajectory
#-----------------------------------------------------------------------
XF,ZF,PT1,PT2,XC,ZC,TH,DL,n,t,NW = XZWAKE(g,PT1,PT2,XF,ZF,XC,ZC,TH,DL,n,t,N,AL,SHARP_TE)


# Compute source influence coefficients - Wall-transpiration model
#-----------------------------------------------------------------------
c   = np.zeros((N+NW+1,N+NW+1))
d   = np.zeros((N+NW+1,N+NW+1))
sig = np.zeros(N+NW+1)

# Source terms are employed for the WALL-TRANSPIRATION MODEL
# modeling the BL's dispacement thickness 

# Setup aerodynamic influence coefficient (AIC) matrix (Source Influences)
#-------------------------------------------------------------------------
# i-th panel collocation point
for i in range(0,N+NW):
    # i = N corresponds to an imaginary connecting panel, nothing to compute
    if i == N:
        continue
    # j-th corner-point panel index (where singularities are placed)       
    for j in range(0,N+NW):
        # Calculate velocities at i-th colloc. point due to singularities on 
        # the j-th and (j+1)-th corner points (linear singularity)   

        
        # j = N corresponds to an imaginary connecting panel, nothing to compute
        if j == N:
            continue
        
        # Self-induced effect for i=j
        if i == j:
            U1,W1,U2,W2 = SOR2DL(1,1,XC[i],ZC[i],PT1[j,0],PT1[j,1],PT2[j,0], \
            PT2[j,1],TH[j],True)
        else:
            U1,W1,U2,W2 = SOR2DL(1,1,XC[i],ZC[i],PT1[j,0],PT1[j,1],PT2[j,0],PT2[j,1],\
            TH[j],False)
        
        # Compute c[i,j] influences for unit-vorticity
        if (j == 0 or j == N+1):
            c[i,j] =  U1*n[i,0] + W1*n[i,1]
            HOLDC  =  U2*n[i,0] + W2*n[i,1]
            d[i,j] =  U1*t[i,0] + W1*t[i,1]
            HOLDD  =  U2*t[i,0] + W2*t[i,1]
        elif (j == N-1 or j == N+NW-1):
            c[i,j]   =  U1*n[i,0] + W1*n[i,1] + HOLDC
            c[i,j+1] =  U2*n[i,0] + W2*n[i,1]
            d[i,j]   =  U1*t[i,0] + W1*t[i,1] + HOLDD
            d[i,j+1] =  U2*t[i,0] + W2*t[i,1]
        else:
            c[i,j]   =  U1*n[i,0] + W1*n[i,1] + HOLDC
            HOLDC    =  U2*n[i,0] + W2*n[i,1]
            d[i,j]   =  U1*t[i,0] + W1*t[i,1] + HOLDD
            HOLDD    =  U2*t[i,0] + W2*t[i,1]
'''
print('N')
print(N)
print('NW')
print(NW)
print('XC')
print(len(XC))
print('PT')
print(len(PT1))
print('XF')
print(len(XF))
print('TH')
print(len(TH))
print('DL')
print(len(DL))
print('n')
print(len(n))
print(TH)
print(XC)
print(PT1)
print(DL)
print(c)'''

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
    for j in range(0,N+NW+1):
        VEL = VEL + c[i,j]*sig[j]
        
    V[i] = VEL + np.cos(AL)*t[i,0]+np.sin(AL)*t[i,1]
    #V[i] = (g[i]+g[i+1])/4 + np.cos(AL)*t[i,0]+np.sin(AL)*t[i,1]
    CP[i] = 1-V[i]**2
    CL = CL + (-1*CP[i])*(np.cos(AL)*np.cos(TH[i]) + np.sin(AL)*np.sin(TH[i]))*DL[i]
    #CL = CL + (g[i]+g[i+1])*DL[i]
    #REPORT BUG: CL != 0 FOR SYMMETRIC AIRFOILS
# Plot results
#-----------------------------------------------------------------------
aeroplot(RHS,CL,CP,g,XF,ZF,XC,ZC,AL,N,NW,n,t)
