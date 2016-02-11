import numpy as np
from gtrans import *

# INFLUENCE SUBROUTINE (VELOCITIES) FOR LINEARLY VARYING VORTEX STRENGTH
#-----------------------------------------------------------------------
def VOR2DL(g1,g2,XC,ZC,X1,Z1,X2,Z2,TH,SELF_IND):
    
    pi = np.pi
    
    # TRANSFORM COLLOCATION POINT COORDS TO LOCAL PANEL C.S.
    XT    = XC-X1
    ZT    = ZC-Z1
    X2T   = X2-X1
    Z2T   = Z2-Z1
    
    X,Z   = ROT(XT,ZT,TH)
    X2,Z2 = ROT(X2T,Z2T,TH)
    Z2    =  0
    
    # COMPUTE R1, R2, TH1, TH2
    R1  = np.sqrt(X**2+Z**2)
    R2  = np.sqrt((X-X2)**2+Z**2)
    TH1 = np.arctan2(Z,X)
    TH2 = np.arctan2(Z,X-X2)
    
    # COMPUTE VELOCITY COMPONENTS (U,W)_A, (U,W)_B AT POINT P(X,Z) AND 
    # ACCOUNT FOR THE SELF-INDUCED EFFECT ON PANEL.THESE VELOCITIES ARE 
    # IN THE JTH REFERENCE FRAME.
    if SELF_IND:
        ua = -0.5*(X-X2)/(X2)
        ub =  0.5*(X)/(X2)
        wa = -0.15916
        wb =  0.15916
    else:
        ua = ((-Z*np.log(R2/R1)+(X2-X)*(TH2-TH1)))/(2*np.pi*X2)
        ub = (Z*np.log(R2/R1)+X*(TH2-TH1))/(2*np.pi*X2)
        wa =-((X2-Z*(TH2-TH1))-X*np.log(R1/R2)+X2*np.log(R1/R2))/(2*np.pi*X2)
        wb = ((X2-Z*(TH2-TH1))-X*np.log(R1/R2))/(2*np.pi*X2)
        
    #TRANSFORM THE LOCAL VELOCITIES INTO THE GLOBAL REFERENCE FRAME.
    ua,wa = ROT(ua,wa,-TH)
    ub,wb = ROT(ub,wb,-TH)

    return ua,wa,ub,wb

#-----------------------------------------------------------------------

# INFLUENCE SUBROUTINE (VELOCITIES) FOR LINEARLY VARYING SOURCE STRENGTH
#-----------------------------------------------------------------------
def SOR2DL(s1,s2,XC,ZC,X1,Z1,X2,Z2,TH,SELF_IND):
    
    pi = np.pi
    
    # TRANSFORM COLLOCATION POINT COORDS TO LOCAL PANEL C.S.
    XT    = XC-X1
    ZT    = ZC-Z1
    X2T   = X2-X1
    Z2T   = Z2-Z1
    
    X,Z   = ROT(XT,ZT,TH)
    X2,Z2 = ROT(X2T,Z2T,TH)
    Z2    =  0
    
    # COMPUTE R1, R2, TH1, TH2
    R1  = np.sqrt(X**2+Z**2)
    R2  = np.sqrt((X-X2)**2+Z**2)
    TH1 = np.arctan2(Z,X)
    TH2 = np.arctan2(Z,X-X2)
    
    # COMPUTE VELOCITY COMPONENTS (U,W)_A, (U,W)_B AT POINT P(X,Z) AND 
    # ACCOUNT FOR THE SELF-INDUCED EFFECT ON PANEL.THESE VELOCITIES ARE 
    # IN THE JTH REFERENCE FRAME.
    if SELF_IND:
        ua =  0.15916*s1
        ub = -0.15916*s2
        wa = -0.5*s1*(X-X2)/X2
        wb =  0.5*s2*(X)/X2
    else:
        ua = s1*(X2-X)/(2*pi*(X2-X1))*np.log(R1/R2)+Z*s1/(2*pi*(X2-X1))*\
              ((X2-X1)/Z+(TH2-TH1))
        ub = s2*(X-X1)/(2*pi*(X2-X1))*np.log(R1/R2)-Z*s2/(2*pi*(X2-X1))*\
        ((X2-X1)/Z+(TH2-TH1))
        wa = Z*s1/(2*pi*(X2-X1))*np.log(R2/R1)+s1*(X2-X)/(2*pi*(X2-X1))*\
        (TH2-TH1)
        wb = -Z*s2/(2*pi*(X2-X1))*np.log(R2/R1)+s2*(X-X1)/(2*pi*(X2-X1))*\
        (TH2-TH1)
    
    #TRANSFORM THE LOCAL VELOCITIES INTO THE GLOBAL REFERENCE FRAME.
    ua,wa = ROT(ua,wa,-TH)
    ub,wb = ROT(ub,wb,-TH)

    return ua,wa,ub,wb

#-----------------------------------------------------------------------

# INFLUENCE SUBROUTINE (VELOCITIES) FOR LINEARLY VARYING SOURCE STRENGTH
#-----------------------------------------------------------------------
def SOR2DL2(g1,g2,XC,ZC,X1,Z1,X2,Z2,TH,SELF_IND):
    
    pi = np.pi
    
    # TRANSFORM COLLOCATION POINT COORDS TO LOCAL PANEL C.S.
    XT    = XC-X1
    ZT    = ZC-Z1
    X2T   = X2-X1
    Z2T   = Z2-Z1
    
    X,Z   = ROT(XT,ZT,TH)
    X2,Z2 = ROT(X2T,Z2T,TH)
    Z2    =  0
    
    # COMPUTE R1, R2, TH1, TH2
    R1  = np.sqrt(X**2+Z**2)
    R2  = np.sqrt((X-X2)**2+Z**2)
    TH1 = np.arctan2(Z,X)
    TH2 = np.arctan2(Z,X-X2)
    
    # COMPUTE VELOCITY COMPONENTS (U,W)_A, (U,W)_B AT POINT P(X,Z) AND 
    # ACCOUNT FOR THE SELF-INDUCED EFFECT ON PANEL.THESE VELOCITIES ARE 
    # IN THE JTH REFERENCE FRAME.
    if SELF_IND:
        ua =  0.15916
        ub = -0.15916
        wa = -0.5*(X-X2)/X2
        wb =  0.5*(X)/X2
    else:
        ua =  ((X2-Z*(TH2-TH1))-X*np.log(R1/R2)+ \
              X2*np.log(R1/R2))/(6.28319*X2)
        ub = -((X2-Z*(TH2-TH1))-X*np.log(R1/R2))/(6.28319*X2)
        wa = -(Z*np.log(R2/R1)+X*(TH2-TH1)-X2*(TH2-TH1))/ \
              (6.28319*X2)
        wb =  (Z*np.log(R2/R1)+X*(TH2-TH1))/(6.28319*X2)   
    
    #TRANSFORM THE LOCAL VELOCITIES INTO THE GLOBAL REFERENCE FRAME.
    ua,wa = ROT(ua,wa,-TH)
    ub,wb = ROT(ub,wb,-TH)

    return ua,wa,ub,wb

#-----------------------------------------------------------------------


# INFLUENCE SUBROUTINE (POTENTIAL) FOR LINEARLY VARYING VORTEX STRENGTH
#-----------------------------------------------------------------------
def PHILV(g1,g2,XC,ZC,X1,Z1,X2,Z2,TH,SELF_IND):
    
    pi = np.pi
    
    # TRANSFORM COLLOCATION POINT COORDS TO LOCAL PANEL C.S.
    XT    = XC-X1
    ZT    = ZC-Z1
    X2T   = X2-X1
    Z2T   = Z2-Z1
    
    X,Z   = ROT(XT,ZT,TH)
    X2,Z2 = ROT(X2T,Z2T,TH)
    Z2    =  0
    
    # COMPUTE R1, R2, TH1, TH2
    R1  = np.sqrt(X**2+Z**2)
    R2  = np.sqrt((X-X2)**2+Z**2)
    TH1 = np.arctan2(Z,X)
    TH2 = np.arctan2(Z,X-X2)
    
    # COMPUTE POTENTIAL PHI AT POINT P(X,Z) AND ACCOUNT FOR
    # THE SELF-INDUCED EFFECT ON PANEL. BY DOING THIS THE
    # SMALL INWARD DISPLACEMENT OF THE COLLOCATION POINT DUE
    # TO THE DIRICHLET B.C. IS AVOIDED.
    if SELF_IND:
        # + OR - ?
        PHI = -g1/2*(X-X2)-g2/16*(X1**2+2*X1*X2-3*X2**2)
        
        PHA = -g1/2*((X-X1)+1/(8*(X2-X1))*(X1**2+2*X1*X2-3*X2**2))
        
        PHB = -g2/(16*(X2-X1))*(X1**2+2*X1*X2-3*X2**2)
        
    else:
        #PHI = -g1/(2*pi)*((X-X1)*TH1-(X-X2)*TH2+Z/2*np.log(R1**2/R2**2)) - \
        #       g2/(2*pi)*(X*Z/2*np.log(R1**2/R2**2)+Z/2*(X1-X2)+(X**2-X1**2-Z**2)/2*TH1 - \
        #       (X**2-X2**2-Z**2)/2*TH2)
               
        PHA = -g1/(2*pi)*((X-X1)*TH1-(X-X2)*TH2+Z/2*np.log(R1**2/R2**2) +
              1/(X2-X1)*(X*Z/2*np.log(R1**2/R2**2)+Z/2*(X1-X2)+(X**2-X1**2-Z**2)/2*TH1 - \
              (X**2-X2**2-Z**2)/2*TH2))
              
        PHB = -g2/(2*pi*(X2-X1))*(X*Z/2*np.log(R1**2/R2**2)+Z/2*(X1-X2)+(X**2-X1**2-Z**2)/2*TH1 - \
              (X**2-X2**2-Z**2)/2*TH2)
              
        PHI = PHA + PHB

    return PHI, PHA, PHB

#-----------------------------------------------------------------------


# INFLUENCE SUBROUTINE (POTENTIAL) FOR LINEARLY VARYING DOUBLET STRENGTH
#-----------------------------------------------------------------------
def PHILD(g1,g2,XC,ZC,X1,Z1,X2,Z2,TH,SELF_IND):
    
    pi = np.pi
    
    # TRANSFORM COLLOCATION POINT COORDS TO LOCAL PANEL C.S.
    XT    = XC-X1
    ZT    = ZC-Z1
    X2T   = X2-X1
    Z2T   = Z2-Z1
    
    X,Z   = ROT(XT,ZT,TH)
    X2,Z2 = ROT(X2T,Z2T,TH)
    Z2    =  0
    
    # COMPUTE R1, R2, TH1, TH2
    R1  = np.sqrt(X**2+Z**2)
    R2  = np.sqrt((X-X2)**2+Z**2)
    TH1 = np.arctan2(Z,X)
    TH2 = np.arctan2(Z,X-X2)
    
    # COMPUTE POTENTIAL PHI AT POINT P(X,Z) AND ACCOUNT FOR
    # THE SELF-INDUCED EFFECT ON PANEL. BY DOING THIS THE
    # SMALL INWARD DISPLACEMENT OF THE COLLOCATION POINT DUE
    # TO THE DIRICHLET B.C. IS AVOIDED.
    if SELF_IND:
        PHA = -0.5*(X/X2-1)
        PHB =  0.5*(X/X2)
    else:
        PHA =  0.15916*(X/X2*(TH2-TH1)+Z/X2* \
               np.log(R2/R1)-(TH2-TH1))
        PHB = -0.15916*(X/X2*(TH2-TH1)+Z/X2*np.log(R2/R1))

    PHI = PHA + PHB    

    return PHI, PHA, PHB

#-----------------------------------------------------------------------
    

# INFLUENCE SUBROUTINE (POTENTIAL) FOR LINEARLY VARYING SOURCE STRENGTH
#-----------------------------------------------------------------------
def PHILS(s1,s2,XC,ZC,X1,Z1,X2,Z2,TH,SELF_IND):
    
    pi = np.pi
    
    # TRANSFORM COLLOCATION POINT COORDS TO LOCAL PANEL C.S.
    XT    = XC-X1
    ZT    = ZC-Z1
    X2T   = X2-X1
    Z2T   = Z2-Z1
    
    X,Z   = ROT(XT,ZT,TH)
    X2,Z2 = ROT(X2T,Z2T,TH)
    Z2    =  0
    
    # COMPUTE R1, R2, TH1, TH2
    R1  = np.sqrt(X**2+Z**2)
    R2  = np.sqrt((X-X2)**2+Z**2)
    TH1 = np.arctan2(Z,X)
    TH2 = np.arctan2(Z,X-X2)
    
    # COMPUTE POTENTIAL PHI AT POINT P(X,Z) AND ACCOUNT FOR
    # THE SELF-INDUCED EFFECT ON PANEL. BY DOING THIS THE
    # SMALL INWARD DISPLACEMENT OF THE COLLOCATION POINT DUE
    # TO THE DIRICHLET B.C. IS AVOIDED.
    if SELF_IND:
        #MUST BE CHECKED FOR VALIDITY
        if (X2-X1>0):
            PHI = s1/(2*pi)*(X2-X1)*np.log(((X2-X1)/2)**2)+s2/(4*pi)*(X2**2-X1**2)*(np.log((X2-X1)/2)-1/2)
        else:
            PHI = s1/(2*pi)*(X1-X2)*np.log(((X1-X2)/2)**2)+s2/(4*pi)*(X1**2-X2**2)*(np.log((X1-X2)/2)-1/2)
    else:
        PHI = s1/(4*pi)*((X-X1)*np.log(R1**2)-(X-X2)*np.log(R2**2)+2*Z*(TH2-TH1)) + \
              s2/(4*pi)*((X**2-X1**2-Z**2)/2*np.log(R1**2)-(X**2-X2**2-Z**2)/2*np.log(R2**2) + \
              2*X*Z*(TH2-TH1)-X*(X2-X1))
        
    return PHI

#-----------------------------------------------------------------------

