import matplotlib.pyplot as plt
from matplotlib.pyplot import *
rcParams['figure.figsize'] = 15, 10
np.set_printoptions(threshold=np.nan)


def aeroplot(RHS,CL1,CL2,CP1,CP2,g,XF,ZF,XF2,ZF2,XC1,ZC1,XC2,ZC2,AL):
    #print(FOIL_NAME)
    #print('RHS')
    #print(RHS)
    print('CL')
    print(CL1)
    print(CL2)
    print(CP1)
    print(CP2)

    #np.savetxt('CPDATA_8DEG',CP)
    #np.set_printoptions(threshold='nan')

    print('Vortex-Strengths')
    print(g)

    plt.plot(XF,ZF, color='k', linestyle='-', marker='o')
    plt.plot(XC1,ZC1, color='k', linestyle='', marker='x')
    plt.plot(XF2,ZF2, color='k', linestyle='-', marker='o')
    plt.plot(XC2,ZC2, color='k', linestyle='', marker='x')
    plt.xlim(-0.1, 1.5)
    plt.ylim(-1.2, 0.2)
    plt.grid()

    #Show Normal Panel Vectors
    #plt.quiver(XC1,ZC1,n1[:,0],n1[:,1])
    #plt.quiver(XC1,ZC1,t1[:,0],t1[:,1])
    #plt.quiver(XC2,ZC2,n2[:,0],n2[:,1])
    #plt.quiver(XC2,ZC2,t2[:,0],t2[:,1])

    plt.gca().set_aspect('equal', adjustable='box')
    plt.rcParams.update({'font.size': 15})
    plt.xlabel("x/c")
    plt.ylabel("z/c")
    #plt.xlim(-0.1, 1.1)
    #plt.ylim(-0.2, 0.2)


    fig, ax1 = plt.subplots()
    ax1.plot(XC1,CP1, marker='o', color='k', linestyle='-')
    ax1.plot(XC2,CP2, marker='o', color='k', linestyle='-')
    ax1.set_xlabel('x/c')
    # Make the y-axis label and tick labels match the line color.
    ax1.set_ylabel(r'$C_p$', color='k', fontsize=20)
    for tl in ax1.get_yticklabels():
        tl.set_color('k')
    ax1.set_xlim([0, 1.3])
    ax1.set_ylim([np.amin(CP1)-1, np.amax(CP1)+1])
    ax1 = plt.gca()
    ax1.set_ylim(ax1.get_ylim()[::-1])
    plt.grid()
    ax1.set_frame_on(False)
    ax1.xaxis.set_ticks(np.arange(0, 1.4, 0.1))
    ax1.yaxis.set_ticks(np.arange(1, -13, -1.0))

    ax2 = ax1.twinx()
    ax2.plot(XF,ZF, 'k', linewidth=2.5)
    ax2.plot(XF2,ZF2, 'k', linewidth=2.5)
    ax2.set_ylabel('Airfoil Thickness', color='k')
    ax2.set_xlim([0, 1.3])
    #ax2.set_ylim([1.0, 1.5])
    ax2 = plt.gca()
    plt.gca().set_aspect('equal', adjustable='box-forced')
    for tl in ax2.get_yticklabels():
        tl.set_color('k')
    ax2.axes.get_yaxis().set_visible(False)
    ax2.set_frame_on(False)


    #Plot Coefficients in Cp plot
    ax1.text(0.75,np.amin(CP1), r'$\alpha $', fontsize=25)
    ax1.text(0.75,np.amin(CP1)+0.5, r'$C_l $', fontsize=25)
    ax1.text(0.75,np.amin(CP1)+1.0, r'$C_m $', fontsize=25)
    ax1.text(0.75,np.amin(CP1)+1.5, r'$C_d $', fontsize=25)
    ALs = str(AL[0]*180/np.pi)
    CLs = str((CL1+CL2)/1.32)
    #Cms = str(Cm)
    #Cds = str(Cd)
    ax1.text(0.82,np.amin(CP1),'%.6s' % ALs, fontsize=18)
    ax1.text(0.82,np.amin(CP1)+0.5,'%.6s' % CLs, fontsize=18)
    ax1.text(0.82,np.amin(CP1)+1.0,'-', fontsize=20)
    ax1.text(0.82,np.amin(CP1)+1.5,'-', fontsize=20)
    #ax1.text(0.85,np.amin(CP1)+0.20,'%.6s' % Cms, fontsize=20)
    #ax1.text(0.85,np.amin(CP1)+0.35,'%.6s' % Cds, fontsize=20)

    plt.show()
