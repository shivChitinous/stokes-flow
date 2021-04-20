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

def init(et,ph,size=0.5,dt=0.01,T=16):
    s = np.zeros([4,3]);
    s[1,0:2] = np.array([-size/2,0]) #motor 1
    s[2,0:2] = np.array([size/2,0]) #motor 2
    s[0,0:2] = np.array([-size*et*np.cos(ph)-size/2,size*et*np.sin(ph)]) #arm 1
    s[3,0:2] = np.array([size*et*np.cos(ph)+size/2,size*et*np.sin(ph)]) #arm 2
    r = np.array([1,2]) #motor specifications
    tim = np.arange(0,T,dt)
    return s,r,tim

def stroke_function(omega,delphinought):
    s = omega/np.max(omega)+delphinought/np.max(delphinought)
    return s

def P0(et,ph,lim=[0.5,1.5],res=0.1,plot=False,figfile="",T=8,strokes=1,k=100,e=0.3,c=0.6):
    
    P_vec = np.arange(lim[0],lim[1]+res,res) #range of P values
    delphi0 = np.zeros(np.shape(P_vec))
    
    for i,P in enumerate(P_vec):
        s,r,tim = init(et,ph,T=T)
        tau = stroke_input(P,tim,r,strokes=strokes)
        R = slts.mesher()
        try:
            _,pswimmer = slts.evolve(tau,tim,R,r,s,k=k,e=e,c=c)
            delphi0[i] = delphinought(tau,pswimmer,ph)
        except ValueError: #sometimes the swimmer can swim out of frame
            delphi0[i] = None
        print("Progress:",np.round((i+1)/np.size(P_vec)*100,2),"%")
        
    #modifying the function:
    jump = np.where(np.diff(delphi0)>0)[0][0]+1
    delphi0[jump:] = delphi0[jump:]-np.pi
    
    func = sp.interpolate.interp1d(delphi0, P_vec)
    P0 = func([0])[0]
    
    if plot:
        plt.plot(P_vec,delphi0,'o',color='seagreen')
        plt.plot(func(delphi0),delphi0,alpha=0.5)
        plt.legend(["simulated values","interpolated values"])
        plt.ylabel(r"$\Delta\phi_0$")
        plt.xlabel(r"$P$")
        plt.annotate("$P_0 = "+str(round(P0,3))+"$",(0.7*np.max(P_vec),np.mean(delphi0)))
        plt.savefig(figfile+"parameter_space_"+str(round(ph,2))+"_"+str(round(et,2))+".png",dpi=300,bbox_inches="tight")
    
    return P0