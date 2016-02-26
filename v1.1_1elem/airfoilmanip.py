import numpy as np
from singulib import *

# INPUT AIRFOIL COORDINATES (XFOIL FORMAT)
#-----------------------------------------------------------------------
def INPUT_AIRFOIL(AIRFOIL):
    #FOILCOORD = raw_input('Input airfoil coordinates file:\t')
    FOILCOORD = AIRFOIL
    # !! PANELS NUMBER IS CONSTRAINED TO THE INPUT FILE  !!
    # !! TO MODIFY PANELING OF THE UPPER AND LOWER SURF. !!
    # !! EDIT THE COORDINATES WITH XFOIL AND INPUT THE   !!
    # !! MODIFIED FILE IN THIS SCRIPT.                   !!

    # SAVE COORDINATES TO BUFFER AND CLOSE FILE
    with open(FOILCOORD, 'r') as f:
        data = f.readlines()
        X  = [] # X AIRFOIL COORDINATES
        Z  = [] # Z AIRFOIL COORDINATES
        #for i, line in enumerate(data,0):
        for line in data:
            #try to read name of the airfoil - bugged
            #if i == 0:
            #    FOIL_NAME = data[0] 
            #else:
            p = line.split()
            X.append(float(p[0]))
            Z.append(float(p[1]))

    # IDENTIFY NUMBER OF PANELS
    N = len(X)-1

    # CONVERT TO NUMPY ARRAYS
    XF = np.array(X)
    ZF = np.array(Z)

    # CONVERT TO CLOCKWISE COORDINATE ORDER
    XF = XF[::-1]
    ZF = ZF[::-1]
    
    return XF,ZF,N
    
#-----------------------------------------------------------------------


# AIRFOIL PANELING
#-----------------------------------------------------------------------
def FOIL_PANELING(XF,ZF,N):
    
#-------------------------AIRFOIL PANELING------------------------------
#-----------------------------------------------------------------------
# --------Panel Endpoints--------Colloc. Points-------------------------
# PT1[i,0] -> Xj  PT2[i,0] -> Xj+1  XC -> Xi  n[i,0] -> nx  t[i,0] -> tx   
# PT1[i,1] -> Zj  PT2[i,1] -> Zj+1  ZC -> Zi  n[i,1] -> nz  t[i,1] -> tz 
#-----------------------------------------------------------------------  
    
    # COMPUTE PANEL ANGLE
    TH = np.zeros(N)
    for i in range(0,N):
        TH[i] = np.arctan2((ZF[i+1]-ZF[i]),(XF[i+1]-XF[i]))
      
    # PANEL NORMAL AND TANGENT VECTORS
    n = np.zeros((N,2))
    t = np.zeros((N,2))
    n[:,0] = -np.sin(TH)
    n[:,1] =  np.cos(TH)   
    t[:,0] =  np.cos(TH)
    t[:,1] =  np.sin(TH)

    # COLLOCATION POINTS ARE PANEL MIDPOINTS
    XC = np.zeros(N)
    ZC = np.zeros(N)
    for i in range(0,N):
        XC[i] = (XF[i]+XF[i+1])/2
        ZC[i] = (ZF[i]+ZF[i+1])/2
      
    # ESTABLISH COORDINATES OF PANEL END POINTS
    PT1 = np.zeros((N,2))
    PT2 = np.zeros((N,2))
    for i in range(0,N):
        PT1[i,0]=XF[i]
        PT2[i,0]=XF[i+1]
        PT1[i,1]=ZF[i]
        PT2[i,1]=ZF[i+1]
        
    return PT1,PT2,XC,ZC,TH,n,t

#-----------------------------------------------------------------------


# Compute wake trajectory
#-----------------------------------------------------------------------
def XZWAKE(g,PT1,PT2,XF,ZF,XC,ZC,TH,DL,n,t,N,AL,SHARP_TE):
    # Number of wake points
    # XFOIL -> NW = N/8 + 2
    NW = N/4+2
    
    # Exponential wake panel spacing
    DLW = np.zeros(NW-1)
    xx = -1
    xxstep = 0.04
    for i in range(0,NW-1):
        # Exponential spacing with base the minimum panel length of the airfoil
        DLW[i]=DL.min()**(-xx)
        xx = xx + xxstep
        
        # Max wake length constrained to max panel length
        if DLW[i] > DL.max():
            DLW[i] = DL.max()
          
    # Update matrix size to include wake
    XF  = np.append(XF,np.zeros(NW),axis=0)
    ZF  = np.append(ZF,np.zeros(NW),axis=0)
    PT1 = np.append(PT1,np.zeros([NW,2]),axis=0)
    PT2 = np.append(PT2,np.zeros([NW,2]),axis=0)
    XC  = np.append(XC,np.zeros(NW),axis=0)
    ZC  = np.append(ZC,np.zeros(NW),axis=0)
    TH  = np.append(TH,np.zeros(NW),axis=0)
    DL  = np.append(DL,np.zeros(NW),axis=0)
    n   = np.append(n,np.zeros([NW,2]),axis=0)
    t  = np.append(t,np.zeros([NW,2]),axis=0)

    # Set first wake point a tiny distance behind TE
    # Treat SHARP and BLUNT trailing edge in separate
    if SHARP_TE == False:
        XTE = 0.5*(XF[0]+XF[N])
        ZTE = 0.5*(ZF[0]+ZF[N])
        #SX = 0.5*(ZF[N] - ZF[0])
        #SZ = 0.5*(XF[0] - XF[N])
        #SMOD = np.sqrt(SX**2 + SZ**2)
        #n[N,0]  = SX/SMOD
        #n[N,1]  = SZ/SMOD
        XF[N+1] = XTE 
        ZF[N+1] = ZTE
    else:
        XTE = XF[N]
        ZTE = ZF[N]
        #n[N,0] = 0.5*(n[0,0]+n[N-1,0])   
        #n[N,1] = 0.5*(n[0,1]+n[N-1,1])
        #XF[N+1] = XTE - 0.0001*n[N,1]
        #ZF[N+1] = ZTE + 0.0001*n[N,0]
        XF[N+1] = XTE - 0.00001
        ZF[N+1] = ZTE + 0.00001
    
    #DL[N]   = S(N)

    # Calculate velocity components at first point
    U = 0; W = 0
    for j in range(0,N):
        U1,W1,U2,W2 = VOR2DL(g[j],g[j+1],XF[N+1],ZF[N+1],PT1[j,0],PT1[j,1],PT2[j,0]\
        ,PT2[j,1],TH[j],False)
        U = U + (U1+U2)
        W = W + (W1+W2)      

    # Velocity field as superposition of vortex strenghts and freestream
    U = U + np.cos(AL)
    W = W + np.sin(AL)
        
    # Set rest of wake points
    for i in range(N+2,N+NW+1):
        #DS = SNEW(I) - SNEW(I-1)
        #DLW = 0.2*DL.max()
        
        # Set new point DL downstream of last point
        XF[i] = XF[i-1] + DLW[i-N-2]*U
        ZF[i] = ZF[i-1] + DLW[i-N-2]*W
        #S(I) = S(I-1) + DL
        
        # Set angle of wake panel normal
        TH[i-1] = np.arctan2((ZF[i]-ZF[i-1]),(XF[i]-XF[i-1]))
        
        # Set wake panel normal and tangential vector
        n[i-1,0] = -np.sin(TH[i-1])
        n[i-1,1] =  np.cos(TH[i-1])
        t[i-1,0] =  np.cos(TH[i-1])
        t[i-1,1] =  np.sin(TH[i-1])
        
        if i == N+NW:
            continue
        
        # Calculate velocity components for next point
        U = 0; W = 0
        for j in range(0,N):
            U1,W1,U2,W2 = VOR2DL(g[j],g[j+1],XF[i],ZF[i],PT1[j,0],PT1[j,1],PT2[j,0],\
            PT2[j,1],TH[j],False)
            U = U + (U1+U2)
            W = W + (W1+W2) 
        
        # Velocity field as superposition of vortex strenghts and freestream
        U = U + np.cos(AL)
        W = W + np.sin(AL)

    
    # Store coordinates of wake panel end points
    for i in range(N+2,N+NW+1):
        PT1[i-1,0]=XF[i-1]
        PT2[i-1,0]=XF[i]
        PT1[i-1,1]=ZF[i-1]
        PT2[i-1,1]=ZF[i]
    
    # Store coordinates of wake panel collocation points
    for i in range(N+2,N+NW+1):
        XC[i-1] = (XF[i-1]+XF[i])/2
        ZC[i-1] = (ZF[i-1]+ZF[i])/2
        
    # Store wake panel lengths
    for i in range(N+1,N+NW):
        DL[i] = DLW[i-(N+1)] 
              
    return XF,ZF,PT1,PT2,XC,ZC,TH,DL,n,t,NW
