import numpy as np

# CLOCKWISE ROTATION MATRIX
#-----------------------------------------------------------------------
def ROT(U,V,THETA):
    RMAT= np.matrix([[ np.cos(THETA) , np.sin(THETA)],
                     [-np.sin(THETA) , np.cos(THETA)]])
    U2= RMAT[0,0]*U+RMAT[0,1]*V
    V2= RMAT[1,0]*U+RMAT[1,1]*V
        
    return U2,V2
    

# PANEL LENGTH
#-----------------------------------------------------------------------
def PAN_LEN(X1,Z1,X2,Z2,TH):
    # Transforming coordinates to local panel CS
    X2T   = X2-X1
    Z2T   = Z2-Z1
    X2    = ROT(X2T,Z2T,TH)[0]
    
    return X2



