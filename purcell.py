import numpy as np
import scipy as sp
from itertools import product
import stokeslets as slts

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

def stroke_input(P,f,tim,r,strokes=4):
    tau = np.array([np.zeros(np.shape(tim)+(3,))]*np.size(r))
    tau[1,:,-1] = alternating_stroke(tim,power=P,strokes=strokes,
                                     shift=0)-alternating_stroke(tim,power=f*P,strokes=strokes,shift=0.25)
    tau[0,:,-1] = -alternating_stroke(tim,power=P,strokes=strokes,
                                      shift=0.25)+alternating_stroke(tim,power=f*P,strokes=strokes,shift=0) 
    return tau

def omega(tau,pswimmer,arm=[3,2]):
    #angular velocity difference between arm end points
    #syntax: arm = [outer_point,inner_point]
    diff = np.linalg.norm([np.diff(pswimmer[:].T[0][arm[0]]),np.diff(pswimmer[:].T[1][arm[0]])
            ],axis=0)-np.linalg.norm([np.diff(pswimmer[:].T[0][arm[1]]),np.diff(pswimmer[:].T[1][arm[1]])],axis=0)
    rms_diff = np.sqrt(np.mean((diff[(np.abs(tau[np.mod(arm[1],2),:,-1])-np.abs(tau[np.mod(arm[0],2),:,-1]))[1:]>0])**2))
    return rms_diff

def delphinought(tau,pswimmer,ph):
    arm=[3,2]  #syntax: arm = [right_point,left_point]
    first_shift = np.where((np.abs(tau[np.mod(arm[1],2),:,-1])-np.abs(tau[np.mod(arm[0],2),:,-1]))>0)[0][0]
    x = pswimmer[first_shift].T[0][arm[0]]-pswimmer[first_shift].T[0][arm[1]]
    y = pswimmer[first_shift].T[1][arm[0]]-pswimmer[first_shift].T[1][arm[1]]
    angle = np.abs(ph-np.abs(np.arctan(y/x)))
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