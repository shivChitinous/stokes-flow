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

def delphihalf(tau,pswimmer,ph):
    shifts = np.where(np.diff(tau[0,:,-1])!=0)[0]+1 #every instant the motors switch
    #the second shift is when the second arm finishes its first stroke
    t1,t2 = thetas(pswimmer,shifts[1])
    angle = (ph-t2)+(ph+t1)
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

def thetas(swimmer,step):
    arm1 = swimmer[step,0]-swimmer[step,1] #arm 1 vector 
    body = swimmer[step,1]-swimmer[step,2] #body vector
    arm2 = swimmer[step,3]-swimmer[step,2] #arm 2 vector
    t1 = np.arctan2(np.array([0,0,-1]).dot(np.cross(arm2,-body)),arm2.dot(-body)) #theta1
    t2 = np.arctan2(np.array([0,0,-1]).dot(np.cross(arm1,body)),arm1.dot(body)) #theta2
    return t1,t2

def purcell_plot(swimmer,tim,ph,et,plot=False,save=False,figfile=""):   
    t1 = np.array([thetas(swimmer,i)[0] for i in range(np.size(tim))])*180/np.pi #theta1
    t2 = np.array([thetas(swimmer,i)[1] for i in range(np.size(tim))])*180/np.pi #theta2
    
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

def P_minimizer(P,et,ph,T=5,strokes=0.625,k=100,e=0.3,c=0.6,gridsize=1,res=0.1):
    s,r,tim = init(et,ph,T=T)
    tau = stroke_input(P,tim,r,strokes=strokes)
    R = slts.mesher(np.arange(-gridsize, gridsize, res))
    _,pswimmer = slts.evolve(tau,tim,R,r,s,k=k,e=e,c=c)
    delphiH = delphihalf(tau,pswimmer,ph)
    return delphiH

def fit():
    M = np.array([[0.33373467,1.02850784,0.03312612],
                    [0.46860363,0.9070047,0.06290173],[0.26994491,1.04919476,0.0259014]])
    return M