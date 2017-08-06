import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from IPython import display
import time,math,sys

#main function for advection algorithm

def main(nx=101,ny=101,nt=80,sigma=0.1,rho=1):

	dx=2.0/(nx-1)
	dy=2.0/(ny-1)
	#cfl condition
	dt=sigma*dx

	#set the x and y axis spacing, and use them to create
	# a mesh for 3d plotting
	xax=np.linspace(0,2,nx)
	yax=np.linspace(0,2,ny)
	xx,yy=np.meshgrid(xax,yax)
	
	Q=1.0/(2.0*np.pi*sigma**2)*np.exp(-(xx**2+yy**2)/(2.0*sigma**2)+1)
	
	#plot initial conditions
	fig=plt.figure()
	ax=plt.gcf().add_subplot(111,projection='3d')
	surf=ax.plot_surface(xx,yy,Q,cmap=cm.coolwarm)
	
	#for t in range(nt)
	
def setFlux(u,v,Q,nx,ny):
	#check which direction flow field is going, multiply by appropriate upwind (against the flow) cell
	#The phi array is ny by nx by 4. For x3=0,1,2,3 the fluxes are left, right, bottom, and top respectively
	phi=np.zeros((ny,nx,4))
	flux = np.zeros((ny,nx,4))
	#set the appropriate flux at each position
	for y in range(ny):
		for x in range(nx):
			#if positive, return u. if negative, return 0.
			flux[y,x,0] = max(u[y,x],0)
			#if negative, return u. if positive, return 0.
			flux[y,x,1] = min(u[y,x],0)
			#and similarly for v
			flux[y,x,2] = max(v[y,x],0)
			flux[y,x,3] = min(v[y,x],0)
		#bulk of env.
	phi[1:-1,1:-1,0]=flux[1:-1,1:-1,0]*Q[1:-1,:-2]+flux[1:-1,1:-1,1]*Q[1:-1,1:-1]
	phi[1:-1,1:-1,1]=flux[1:-1,2:,0]*Q[1:-1,1:-1]+flux[1:-1,2:,1]*Q[1:-1,2:]
	phi[1:-1,1:-1,2]=flux[1:-1,1:-1,2]*Q[:-2,1:-1]+flux[1:-1,1:-1,3]*Q[1:-1,1:-1]
	phi[1:-1,1:-1,3]=flux[2:,1:-1,2]*Q[1:-1,1:-1]+flux[2:,1:-1,3]*Q[2:,1:-1]
	#x-boundary excluding corners
	phi[1:-1,-1,0]=flux[1:-1,-1,0]*Q[1:-1,-2]+flux[1:-1,-1,1]*Q[1:-1,-1]
	phi[1:-1,-1,1]=flux[1:-1,0,0]*Q[1:-1,-1]+flux[1:-1,0,1]*Q[1:-1,0]
	phi[1:-1,-1,2]=flux[1:-1,-1,2]*Q[:-2,-1]+flux[1:-1,-1,3]*Q[1:-1,-1]
	phi[1:-1,-1,3]=flux[2:,-1,2]*Q[1:-1,-1]+flux[2:,-1,3]*Q[2:,-1]
	phi[1:-1,0,0]=flux[1:-1,0,0]*Q[1:-1,-1]+flux[1:-1,0,1]*Q[1:-1,0]
	phi[1:-1,0,1]=flux[1:-1,1,0]*Q[1:-1,0]+flux[1:-1,1,1]*Q[1:-1,1]
	phi[1:-1,0,2]=flux[1:-1,0,2]*Q[:-2,0]+flux[1:-1,0,3]*Q[1:-1,0]
	phi[1:-1,0,3]=flux[2:,0,2]*Q[1:-1,0]+flux[2:,0,3]*Q[2:,0]
	#y-boundary excluding corners
	phi[-1,1:-1,0]=flux[-1,1:-1,0]*Q[-1,:-2]+flux[-1,1:-1,1]*Q[-1,1:-1]
	phi[-1,1:-1,1]=flux[-1,2:,0]*Q[-1,1:-1]+flux[-1,2:,1]*Q[-1,2:]
	phi[-1,1:-1,2]=flux[-1,1:-1,2]*Q[-2,1:-1]+flux[-1,1:-1,3]*Q[-1,1:-1]
	phi[-1,1:-1,3]=0 #zero flux through the face on top
	phi[0,1:-1,0]=flux[0,1:-1,0]*Q[0,:-2]+flux[0,1:-1,1]*Q[0,1:-1]
	phi[0,1:-1,1]=flux[0,2:,0]*Q[0,1:-1]+flux[0,2:,1]*Q[0,2:]
	phi[0,1:-1,2]=0 #zero flux through the face on bottom
	phi[0,1:-1,3]=flux[1,1:-1,2]*Q[0,1:-1]+flux[1,1:-1,3]*Q[1,1:-1]
	#corners
	phi[-1,-1,0]=flux[-1,-1,0]*Q[-1,-2]+flux[-1,-1,1]*Q[-1,-1]
	phi[-1,-1,1]=flux[-1,0,0]*Q[-1,-1]+flux[-1,0,1]*Q[-1,0]
	phi[-1,-1,2]=flux[-1,-1,2]*Q[-2,-1]+flux[-1,-1,3]*Q[-1,-1]
	phi[-1,-1,3]=0 #and similar
	phi[0,-1,0]=flux[0,-1,0]*Q[0,-2]+flux[0,-1,1]*Q[0,-1]
	phi[0,-1,1]=flux[0,0,0]*Q[0,-1]+flux[0,0,1]*Q[0,0]
	phi[0,-1,2]=0
	phi[0,-1,3]=flux[1,-1,2]*Q[0,-1]+flux[1,-1,3]*Q[1,-1]
	phi[-1,0,0]=flux[-1,0,0]*Q[-1,-1]+flux[-1,0,1]*Q[-1,0]
	phi[-1,0,1]=flux[-1,1,0]*Q[-1,0]+flux[-1,1,1]*Q[-1,1]
	phi[-1,0,2]=flux[-1,0,2]*Q[-2,0]+flux[-1,0,3]*Q[-1,0]
	phi[-1,0,3]=0
	phi[0,0,0]=flux[0,0,0]*Q[0,-1]+flux[0,0,1]*Q[0,0]
	phi[0,0,1]=flux[0,1,0]*Q[0,0]+flux[0,1,1]*Q[0,1]
	phi[0,0,2]=0
	phi[0,0,3]=flux[1,0,2]*Q[0,0]+flux[1,0,3]*Q[1,0]
	return(phi)

	#UPWIND SCHEME: ACTIVATE
def upWind(Q,S,u,v,dx,dy,dt,nx,ny):
	#dQ/dt + d(phi)/dx + d(phi)/dy = S
	Qn=Q.copy()
	delx=1/dx; dely=1/dy
	phi=setFlux(u,v,Qn,nx,ny)
	Q=Qn-dt*((phi[:,:,1]-phi[:,:,0])*delx+(phi[:,:,3]-phi[:,:,2])*dely-S)
	return(Q)
