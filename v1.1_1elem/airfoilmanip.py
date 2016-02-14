import numpy as np

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
