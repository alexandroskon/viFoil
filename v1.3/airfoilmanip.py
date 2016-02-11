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

    # Save coordinates to buffer and close
    with open(FOILCOORD, 'r') as f:
        data = f.readlines()
        X  = [] # X coordinate of airfoil
        Z  = [] # Z coordinate of airfoil
        #for i, line in enumerate(data,0):
        for line in data:
            #if i == 0:
            #    FOIL_NAME = data[0] 
            #else:
            p = line.split()
            X.append(float(p[0]))
            Z.append(float(p[1]))

    # Identify number of panels
    N = len(X)-1

    # Convert to numpy arrays  
    XF = np.array(X)
    ZF = np.array(Z)

    # Convert to ClockWise coordinate order
    #XF = XF[::-1]
    #ZF = ZF[::-1]
    
    return XF,ZF,N
    
#-----------------------------------------------------------------------


# AIRFOIL PANELING
#-----------------------------------------------------------------------
def FOIL_PANELING(XF,ZF,N):
    # Find panel angle
    TH = np.zeros(N)

    for i in range(0,N):
        TH[i] = np.arctan2((ZF[i+1]-ZF[i]),(XF[i+1]-XF[i]))
      
    # Define angle of attack of airfoil
    ALPHA = input('Enter angle of attack of the airfoil (deg):\t')
    AL=ALPHA/(180/np.pi)
     
    # Compute panel normal and tangent vectors
    n = np.zeros((N,2))
    t = np.zeros((N,2))
    # Normal Vector 
    n[:,0] = -np.sin(TH)
    n[:,1] =  np.cos(TH)
    # Tangent Vector    
    t[:,0] =  np.cos(TH)
    t[:,1] =  np.sin(TH)

    XC = np.zeros(N)
    ZC = np.zeros(N)
    # Collocation points are panel midpoints
    for i in range(0,N):
        XC[i] = (XF[i]+XF[i+1])/2
        ZC[i] = (ZF[i]+ZF[i+1])/2

    PT1 = np.zeros((N,2))
    PT2 = np.zeros((N,2))
      
    # Establish coordinates of panel end points
    for i in range(0,N):
        PT1[i,0]=XF[i]
        PT2[i,0]=XF[i+1]
        PT1[i,1]=ZF[i]
        PT2[i,1]=ZF[i+1]
        
    return PT1,PT2,XC,ZC,TH,n,t,AL
