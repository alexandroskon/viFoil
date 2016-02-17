import matplotlib.pyplot as plt
from matplotlib.pyplot import *
rcParams['figure.figsize'] = 15, 10
np.set_printoptions(threshold=np.nan)


def aeroplot(RHS,CL,CP,g,XF,ZF,XC,ZC,AL,N,NW):
    #print(FOIL_NAME)
    #print('RHS')
    #print(RHS)
    print('CL')
    print(CL)

    #np.savetxt('CPDATA_8DEG',CP)
    #np.set_printoptions(threshold='nan')

    print('Vortex-Strengths')
    print(g)
    
    plt.plot(XF[0:N+1],ZF[0:N+1], color='k', linestyle='-', marker='o')
    plt.plot(XF[N+1:N+NW],ZF[N+1:N+NW], color='r', linestyle='-', marker='o')
    plt.plot(XC,ZC, color='k', linestyle='', marker='x')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.rcParams.update({'font.size': 15})
    plt.xlabel("x/c")
    plt.ylabel("z/c")
    #plt.xlim(-0.1, 1.1)
    #plt.ylim(-0.2, 0.2)
    plt.grid()

    #Show Normal Panel Vectors
    #plt.quiver(XC,ZC,n1[:,0],n1[:,1])
    #plt.quiver(XC,ZC,t1[:,0],t1[:,1])
    
    fig, ax1 = plt.subplots()
    ax1.plot(XC,CP, marker='o', color='k', linestyle='-')
    ax1.set_xlabel('x/c')
    # Make the y-axis label and tick labels match the line color.
    ax1.set_ylabel(r'$C_p$', color='k', fontsize=20)
    for tl in ax1.get_yticklabels():
        tl.set_color('k')
    ax1 = plt.gca()
    plt.grid()
    ax1.set_frame_on(False)
    ax1.set_xlim([-0.1, 1.1])
    ax1.set_ylim([1.2*np.amin(CP), 1.2*np.amax(CP)])
    ax1.set_ylim(ax1.get_ylim()[::-1])
    #ax1.xaxis.set_ticks(np.arange(-0.1, 1.1, 0.1))
    #ax1.yaxis.set_ticks(np.arange(1, np.amin(CP), -1.0))

    # Hack airfoil plot to make it smaller?
    
    ax2 = ax1.twinx()
    ax2.plot(XF[0:N+1],ZF[0:N+1], 'k', linewidth=2.5)
    ax2.plot(XF[N+1:N+NW],ZF[N+1:N+NW], 'k', linewidth=2.5)
    ax2.set_ylabel('Airfoil Thickness', color='k')
    ax2.set_xlim([-0.1, 1.1])
    #ax2.set_ylim([1.2*np.amin(CP), 1.2*np.amax(CP)])
    #ax2.set_ylim(ax2.get_ylim()[::-1])
    ax2 = plt.gca()
    plt.gca().set_aspect('equal', adjustable='box-forced')
    for tl in ax2.get_yticklabels():
        tl.set_color('k')
    ax2.axes.get_yaxis().set_visible(False)
    ax2.set_frame_on(False)


    #Plot Coefficients in Cp plot
    txtboxsize = 0.25*abs(np.amin(CP)-np.amax(CP));
    ax1.text(0.8,np.amin(CP), r'$\alpha $', fontsize=25)
    ax1.text(0.8,np.amin(CP)+0.3*txtboxsize, r'$C_l $', fontsize=20)
    ax1.text(0.8,np.amin(CP)+0.6*txtboxsize, r'$C_m $', fontsize=20)
    ax1.text(0.8,np.amin(CP)+0.9*txtboxsize, r'$C_d $', fontsize=20)
    ALs = str(AL*180/np.pi)
    CLs = str(CL)
    #Cms = str(Cm)
    #Cds = str(Cd)
    ax1.text(0.87,np.amin(CP),'%.6s' % ALs, fontsize=18)
    ax1.text(0.87,np.amin(CP)+0.3*txtboxsize,'%.6s' % CLs, fontsize=18)
    ax1.text(0.87,np.amin(CP)+0.6*txtboxsize,'-', fontsize=20)
    ax1.text(0.87,np.amin(CP)+0.9*txtboxsize,'-', fontsize=20)
    #ax1.text(0.85,np.amin(CP)+0.20,'%.6s' % Cms, fontsize=20)
    #ax1.text(0.85,np.amin(CP)+0.35,'%.6s' % Cds, fontsize=20)
    

    plt.show()
