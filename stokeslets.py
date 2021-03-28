import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from itertools import product

def mesher(scale = np.arange(-1, 1, 0.1)):
    x = y = z = scale
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    R = np.array([X,Y,Z])
    return R

def initialize(Ns=100,L=0.1):
    #number of elements and length of rod
    s = np.zeros([Ns,3]); s[:,0] = np.arange(0,L+L/Ns,L/(Ns-1)) #rod elements
    r = np.array([0]) #which rod elements have motors
    return s,r

def rotlet(R,tau,p,e):
    #x-x0
    R = np.array([R[i]-p[i] for i in range(3)]) 
    
    #distance squared
    r2 = np.sum(R**2,axis=0) 
    
    #scaling
    const = ((2*r2)+(5*(e**2)))/(16*np.pi*((r2+(e**2))**(5/2)))  
    
    #vector field
    U = const*np.cross(tau,R,axis=0) 
    return U

def stokeslet(R,f,p,e):
    #x-x0
    R = np.array([R[i]-p[i] for i in range(3)])
    
    #distance squared
    r2 = np.sum(R**2,axis=0)
    
    #forces at each point in the grid
    F = np.array([(0*R[i])+f[i] for i in range(3)])
    
    #constants for scaling
    const_1 = ((r2)+(2*(e**2)))
    const_2 = (8*np.pi*((r2+(e**2))**(3/2)))
    
    #dot product
    dot = np.sum([f[i]*R[i] for i in range(3)],axis=0)
    
    #velocity field
    U = ((dot*R) + (F*const_1))/const_2
    return U

def Up(U,p,R):
    V = np.array(np.shape(U))
    V = [sp.interpolate.RegularGridInterpolator((R[0][:,0,0], R[1][0,:,0], R[2][0,0,:]), U[i]) for i in range(3)]
    Vp = np.array([V[i](p)[0] for i in range(3)])
    return Vp

def hookes(s,i,j,k,l0):
    #force acting on j due to i
    f = (k*(np.linalg.norm(s[j]-s[i])-l0)/(l0*(np.linalg.norm(s[j]-s[i]))))*(-(s[j]-s[i]))
    return f

def evolve (tau,tim,R,r,s,k,e):
    dt = tim[1]-tim[0]
    l0 = np.zeros(np.shape(s)[0]-1)
    for joint,_ in enumerate(l0):
        l0[joint] = np.linalg.norm(s[joint+1]-s[joint])
    
    #initialize
    Ust = np.zeros(np.shape(R))
    
    #velocity field for each instant in time
    U = np.zeros((tim.size,)+np.shape(R))
    
    #position of rod elements for each instant in time
    rod = np.zeros((tim.size,)+np.shape(s))
    
    for i,t in enumerate(tim):
        #reset
        Urot = np.zeros(np.shape(R))
        
        #get velocities due to torques
        for m,mot in enumerate(r):
            Urot += rotlet(R,tau[m][i],s[mot],e)
        
        #get velocity field for time instant
        U[i] = Urot+Ust

        #get positions of rod elements due to flow
        for si in range(s.shape[0]):
            s[si] += (Up(U[i],s[si],R)*dt)
            
        rod[i] = s
        
        #reset
        Ust = np.zeros(np.shape(R))
        
        #get forces due to elements on their neighbours
        for si in range(s.shape[0]-1):
            f = hookes(s,si,si+1,k,l0[si])
            Ust += stokeslet(R,f,s[si+1],e)+stokeslet(R,-f,s[si],e)
        
    return U,rod

