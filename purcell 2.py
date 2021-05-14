import numpy as np
import scipy as sp
import stokeslets as slts
import matplotlib.pyplot as plt
from scipy import interpolate

def alternating_stroke(tim,shift,power=0.4,strokes=4):
    T = np.max(tim) #maximum time
    stroke = np.zeros(np.size(tim))
    step_1 = (tim<0)
    step_2 = (tim<0)
    shift = shift*T/strokes #shift is measured in stroke fractions
    for i in np.arange(shift, T, T/strokes):
        step_1 = step_1|((tim>=i)&(tim<i+0.25*T/strokes))
        step_2 = step_2|((tim>=i+0.5*T/strokes)&(tim<i+0.75*T/strokes))
    stroke[step_1] = -power
    stroke[step_2] = power
    #each stroke for an arm is composed of an upward torque, a gap, a downward torque and another gap (equally divided in time)
    return stroke

def stroke_input(P,tim,r,strokes=4,f=0):
    tau = np.array([np.zeros(np.shape(tim)+(3,))]*np.size(r))
    tau[1,:,-1] = alternating_stroke(tim,power=P,strokes=strokes,
                                     shift=0)-alternating_stroke(tim,power=f*P,strokes=strokes,shift=0.25)
    tau[0,:,-1] = -alternating_stroke(tim,power=P,strokes=strokes,
                                      shift=0.25)+alternating_stroke(tim,power=f*P,strokes=strokes,shift=0) 
    return tau

def delphinought(tau,pswimmer,ph):
    arm=[1,0] #second arm
    shifts = np.where(np.diff(tau[0,:,-1])!=0)[0]+1 #every instant the motors switch
    #the second shift is when the second arm finishes its first stroke
    x = pswimmer[shifts[1]+1].T[0][arm[0]]-pswimmer[shifts[1]+1].T[0][arm[1]]
    y = pswimmer[shifts[1]+1].T[1][arm[0]]-pswimmer[shifts[1]+1].T[1][arm[1]]
    angle = ph-np.arctan(y/x)
    return angle

def init(et,ph,size=1,dt=0.01,T=16):
    s = np.zeros([4,3]);
    a = size/(2+et) #one-arm length
    b = a*et #body length
    s[1,0:2] = np.array([-b/2,0]) #motor 1
    s[2,0:2] = np.array([b/2,0]) #motor 2
    s[0,0:2] = np.array([-a*np.cos(ph)-b/2,a*np.sin(ph)]) #arm 1
    s[3,0:2] = np.array([a*np.cos(ph)+b/2,a*np.sin(ph)]) #arm 2
    r = np.array([1,2]) #motor specifications
    tim = np.arange(0,T,dt)
    return s,r,tim

def stroke_function(omega,delphinought):
    s = omega/np.max(omega)+delphinought/np.max(delphinought)
    return s

def purcell_plot(swimmer,tim,ph,et,plot=False,save=False,figfile=""):
    arm1 = swimmer[:,0]-swimmer[:,1] #arm 1 vector 
    body = swimmer[:,1]-swimmer[:,2] #body vector
    arm2 = swimmer[:,3]-swimmer[:,2] #arm 2 vector
    t1 = np.array([np.arctan2(np.array([0,0,-1]).dot(np.cross(arm2[i],-body[i])),arm2[i].dot(-body[i])) 
                   for i in range(np.size(tim))])*180/np.pi #theta1
    t2 = np.array([np.arctan2(np.array([0,0,-1]).dot(np.cross(arm1[i],body[i])),arm1[i].dot(body[i])) 
                   for i in range(np.size(tim))])*180/np.pi #theta2
    
    if plot:
        plt.figure(figsize=(7,6))
        plt.xlim([-ph*180/np.pi*1.5,1.5*ph*180/np.pi])
        plt.ylim([-ph*180/np.pi*1.5,1.5*ph*180/np.pi])
        plt.xlabel(r'$\theta_1 \ (^{\circ})$')
        plt.ylabel(r'$\theta_2 \ (^{\circ})$')
        plt.axvline(0,color='grey'); plt.axhline(0,color='grey')
        plt.scatter(t1,t2,c=tim,s=1.5,cmap='viridis')
        plt.colorbar()
        plt.title(r"$\phi = "+str(int(round(ph*180/np.pi,0)))+"^{\circ},\ \eta = "+str(round(et,2))+"$")
        if save:
            plt.savefig(figfile+"purcell_plot_"+str(round(ph,2))+"_"+str(round(et,2))+".png",dpi=300,bbox_inches="tight")
        plt.show()
    
    return t1,t2
    

def P0(et,ph,lim=[0.5,1.5],res=0.1,plot=False,save=False,figfile="",T=6,strokes=0.75,k=100,e=0.3,c=0.6,gridsize=1):
    
    P_vec = np.arange(lim[0],lim[1]+res,res) #range of P values
    delphi0 = np.zeros(np.shape(P_vec))
    
    for i,P in enumerate(P_vec):
        s,r,tim = init(et,ph,T=T)
        tau = stroke_input(P,tim,r,strokes=strokes)
        R = slts.mesher(np.arange(-gridsize, gridsize, 0.1))
        _,pswimmer = slts.evolve(tau,tim,R,r,s,k=k,e=e,c=c)
        delphi0[i] = delphinought(tau,pswimmer,ph)
        print("Progress:",np.round((i+1)/np.size(P_vec)*100,2),"%")
        
    #modifying the function:
    if np.any(np.diff(delphi0)>0): 
        jump = np.where(np.diff(delphi0)>0)[0][0]+1
        delphi0[jump:] = delphi0[jump:]-np.pi
    
    func = sp.interpolate.interp1d(delphi0, P_vec)
    P0 = func([0])[0]
    
    if plot:
        _,ax = plt.subplots()
        ax.plot(P_vec,delphi0,'o',color='seagreen')
        ax.plot(func(delphi0),delphi0,alpha=0.5)
        plt.title("$\phi = "+str(round(ph,2))+",\ \eta = "+str(round(et,2))+"$")
        ax.legend(["simulated values","interpolated values"],loc='lower left')
        ax.set_ylabel(r"$\Delta\phi_0$")
        ax.set_xlabel(r"$P$")
        plt.text(0.8,0.9,"$P_0 = "+str(round(P0,3))+"$",transform=ax.transAxes)
        if save:
            plt.savefig(figfile+"parameter_space_"+str(round(ph,2))+"_"+str(round(et,2))+".png",dpi=300,bbox_inches="tight")
        plt.show()
    
    return P0

def fit():
    M = np.array([[0.37471675, 0.95780107, 0.05813255],
                    [0.26757945, 1.15469392, 0.01554458],[8.09012094e-02, 1.21398877e+00, 6.42651528e-04]])
    return M