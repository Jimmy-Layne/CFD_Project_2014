import numpy as np
from matplotlib.pyplot import *
import time, sys
from IPython import display
import pdb


def linearConvection(nx=41,nt=25,c=1):
	#Set the spacing of the x descritization
	dx=2.0/(nx-1)
	dt=nt/1000
	
	#initial condiditions on u are given as x=2:.5<=x<=1 and x=1 else
	u=np.ones(nx)
	u[.5/dx:1/dx+1]=2
	#plot the initial conditions and set up the figure and axis'
	fig=figure()
	ax=fig.add_subplot(111)
	indvar=np.linspace(0,2,nx)
	func=ax.plot(indvar,u,'k--')
	#for each time step, loop through the positions, setting each un to un+1
	for t in range(nt):
		#copy the old u to the new one
		un=u.copy()
		#Now loop through each position, beginning at 1, since we want
		#to maintain the boundary condidtions.
		for x in range(1,nx):
			u[x]=un[x]-c*dt/dx*(un[x]-un[x-1])
			
		func[0].remove()
		func=ax.plot(indvar,u,'k')
		display.clear_output()
		display.display(fig)
		draw()
		time.sleep(0.1)
def oneDconvection(nx=41,nt=25,sigma=0.2):
	#this function is basically exactly the same as the one before,
	#except the PDE we are solving hs a non-linear term, so this is
	#gonna look really similar
	dx=2.0/(nx-1)
	dt=sigma*dx/2
	
	#initial condiditions on u are given as x=2:0.5<=x<=1 and x=1 else
	u=np.ones(nx)
	u[.5/dx:1/dx+1]=2
	#plot the initial conditions and set up the figure and axis'
	fig=figure()
	ax=gca()
	indvar=np.linspace(0,2,nx)
	func=plot(indvar,u,'k--')
	
	for t in range(nt):
		#copy the old u to the new one
		un=u.copy()
		#Now loop through each position, beginning at 1, since we want
		#to maintain the boundary condidtions.
		for x in range(nx):
			
			#behold the non-linear term:
			u[x]=un[x]-un[x]*dt/dx*(un[x]-un[x-1])
			#pdb.set_trace()
		func[0].remove()
		func=ax.plot(indvar,u,'k')
		display.clear_output()
		display.display(fig)
		draw()
		time.sleep(0.05)



def diffusion(nx=41,nt=20,nu=0.3,sigma=.2):
	#Calculate the time step based on the input parameters
	#Why do we do it this way?
	dx=2./(nx-1)
	dt=sigma*dx**2/nu
	#create the arrays and set the initial condidions
	u=np.ones(nx)
	u[.5/dx:1/dx+1]=2
	indvar=np.linspace(0,2,nx)
	#plot the initial conditions and set up the figure and axis'
	fig=figure()
	ax=gca()
	indvar=np.linspace(0,2,nx)
	func=plot(indvar,u,'k--')
	#initialize the time advancing loop
	for t in range(nt):
		un=u.copy()
		for x in range(1,nx-1):
			u[x]=un[x]+nu*dt/dx**2*(un[x+1]+un[x-1]-2*un[x])
		func[0].remove()
		func=ax.plot(indvar,u,'k')
		display.clear_output(wait=True)
		display.display(fig)
		draw()
		time.sleep(0.1)


def oneDBurgers(nx=101,nt=100,nu=.07):
	#define the space and time steps:
	dx=2*np.pi/(nx-1)
	dt=dx*nu
	
	#now we'll create the initial conditions for the function
	xax=np.linspace(0,2*np.pi,nx)
	t=0
	u=np.empty(nx)
	uanalytical=np.empty(nx)
	for i in range(nx):
		
		u[i]=uinit(xax[i],t,nu)
		
		
	fig=figure(figsize=(11,7),dpi=100)
	ax=gca()
	indvar=np.linspace(0,2*np.pi,nx)
	num=plot(xax,u,'r--',lw=2,label='initial')
	ana=plot(xax,u)
	xlim([0,2*np.pi])
	ylim([0,10])
	#create the time advancing loop
	for t in range(nt):
		un=u.copy()
		for x in range(nx-1):
			u[x]=un[x]-un[x]*dt/dx*(un[x]-un[x-1])+dt/dx**2*nu*(un[x+1]+un[x-1]-2*un[x])
		#we want the boundary conditions to be periodic, so we need to make the array values wrap around when they reach the end.
		un[-1]=un[-1]+un[-1]*dt/dx*(un[-1]-un[-2])+nu*dt/dx**2*(un[0]+un[-2]+2*un[-1])
	
	#create the analytical solution to plot against the numerical
		for i in range(nx):
			uanalytical[i]=uinit(xax[i],t*dt,nu)
		num[0].remove()
		ana[0].remove()
		num=plot(xax,u,'bo',lw=2,label='Numerical solution')
		ana=plot(xax,uanalytical,color='g',label='Analytical solution')
		xlim([0,2*np.pi])
		ylim([0,10])
		display.clear_output(wait=True)
		display.display(fig)
		draw()
		time.sleep(0.1)
def uinit(x,t,nu):
	return(-2*nu*(phiprime(x,t,nu)/phi(x,t,nu))+4)


def phi(x,t,nu):
	return(np.exp(-(x-4*t)**2/(4*nu*(t+1)))+np.exp(-(x-4*t-2*np.pi)**2/(4*nu*(t+1))))


def phiprime(x,t,nu):
	return(-1/(2*nu*(t+1))*((x-4*t)*np.exp(-(x-4*t)**2/(4*nu*(t+1)))+(x-4*t-2*np.pi)*np.exp(-(x-4*t-2*np.pi)**2/(4*nu*(t+1)))))
