import numpy as np
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D
import time, sys
from IPython import display
import pdb
from matplotlib import cm

def TDLConvect(nx=81,ny=81,nt=100,c=1,sigma=0.2):
	#Same deal as phase 1, this time with 2 vars
	dx=2.0/(nx-1)
	dy=2.0/(ny-1)
	dt=sigma*dx
	#axis variables
	xax=np.linspace(0,2,nx)
	yax=np.linspace(0,2,ny)
	#flow definition
	u=np.ones((nx,ny))
	#placeholder
	un=np.ones((nx,ny))
	#setup the initial conditions
	u[.5/dx:1/dx+1,.5/dy:1/dy+1]=2
	xx,yy=np.meshgrid(xax,yax)
	fig=figure()
	ax=fig.gca(projection='3d')
	surf=ax.plot_surface(xx,yy,u,cmap=cm.coolwarm)
	title('Behold the moving lump')
	for t in range(nt+1):
		un=u.copy()
		#for x in range(1,len(u[0])):
		#	for y in range(1,len(u[1])):
		#		u[x,y]=un[x,y]-c*dt/dx*(un[x,y]-un[x-1,y])-c*dt/dy*(un[x,y]-un[x,y-1])
		u[1:,1:]=un[1:,1:]-un[1:,1:]*dt/dx*(un[1:,1:]-un[0:-1,1:])

		#explicitly set boundaries to 1
		u[0,:]=1
		u[-1,:]=1
		u[:,0]=1
		u[:,-1]=1

		#replot the function each time the timestep iterates.
		surf.remove()
		surf=ax.plot_surface(xx,yy,u,cmap=cm.coolwarm)
		display.clear_output(wait=True)
		display.display(fig)
		ax.set_zlim([0.5,2.5])
		draw()
		time.sleep(0.1)


def TDConvect(nx=101,ny=101,nt=80,sigma=0.2,rho=1):
	#set the width of the temporal and spacial steps
	dx=2.0/(nx-1)
	dy=2.0/(ny-1)
	#cfl condition
	dt=sigma*dx
	
	#set the x and y axis spacing, and use them to create
	# a mesh for 3d plotting
	xax=np.linspace(0,2,nx)
	yax=np.linspace(0,2,ny)
	xx,yy=np.meshgrid(xax,yax)
	
	#Initialize the velocity functions and set the initial conditions:
	#note that, since this is non linear, the flow function has 2 
	#components which make up the velocity so we need o approximate
	#both to know how the flow goes.
	u=np.ones((ny,nx))
	v=np.ones((ny,nx))
	#placeholder arrays
	un=np.ones((ny,nx))
	vn=np.ones((ny,nx))
	#set box initial conditions
	#u[.5/nx:1/nx+1,.5/ny:1/ny+1]=2
	#v[.5/nx:1/nx+1,.5/ny:1/ny+1]=2
	
	u[.5/dx:1/dx+1,.5/dy:1/dy+1]=2
	v[.5/dx:1/dx+1,.5/dy:1/dy+1]=2
	
	

	#Plot the u and v function separately
	fig=figure(figsize=(11,7), dpi=100)
	axu=gcf().add_subplot(121,projection='3d')
	surfu=axu.plot_surface(xx,yy,u,cmap =cm.coolwarm)
	title('u component')
	ylabel('y-axis')
	xlabel('x-axis')
	
	axv=subplot(122,projection='3d')
	surfv=axv.plot_surface(xx,yy,v,cmap=cm.coolwarm)
	title('v component')
	ylabel('y-axis')
	xlabel('x-axis')

	
	for t in range(nt+1):
		un=u.copy()
		vn=v.copy()
		#instead of using nested loops, we will use array 
		#operations to assign the new values all at once.
		#setting u component
		u[1:,1:]=un[1:,1:]-(un[1:,1:]*dt/dx*(un[1:,1:]-un[0:-1,1:]))-(vn[1:,1:]*dt/dy*(un[1:,1:]-un[1:,0:-1]))
		#setting v component
		v[1:,1:]=vn[1:,1:]-(un[1:,1:]*dt/dx*(vn[1:,1:]-vn[0:-1,1:]))-(vn[1:,1:]*dt/dy*(vn[1:,1:]-vn[1:,0:-1]))
		



		#setting u component

	#			u[x,y]=un[x,y]-un[x,y]*dt/dx*(un[x,y]-un[x-1,y])+un[x,y]*dt/dy*(un[x,y]-un[x,y-1])
		#setting v component
	#			v[x,y]=vn[x,y]-un[x,y]*dt/dx*(vn[x,y]-vn[x-1,y])+vn[x,y]*dt/dy*(vn[x,y]-vn[x,y-1])
	



	#now set all of the boundaries to 1
		u[0,:]=1
		u[-1,:]=1
		v[0,:]=1
		v[-1,:]=1
		u[:,0]=1
		u[:,-1]=1
		v[:,0]=1
		v[:,-1]=1
	
		#plot the array at each time step
		surfu.remove()
		surfu=axu.plot_surface(xx,yy,u,cmap=cm.coolwarm)
		surfv.remove()
		surfv=axv.plot_surface(xx,yy,v,cmap=cm.coolwarm)
		display.clear_output(wait=True)
		display.display(fig)
		axu.set_zlim([0.5,2.5])
		axv.set_zlim([0.5,2.5])
		draw()
		time.sleep(0.01)
		



def TDDiffusion(nx=31,ny=31,nt=17,nu=0.5,sigma=.25):
	dx=2.0/(nx-1)
	dy=2.0/(ny-1)
	dt=sigma*dx*dy/nu

	x=np.linspace(0,2,nx)
	y=np.linspace(0,2,ny)
	
	u=np.ones((nx,ny))
	un=np.ones((nx,ny))

	u[.5/dx:1/dx+1,.5/dy:1/dy+1]=2

	fig=figure()
	ax=gca(projection='3d')
	xx,yy=np.meshgrid(x,y)
	surf=ax.plot_surface(xx,yy,u,rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0)
	#ax.set_xlim(1,2)
	#ax.set_ylim(1,2)
	#ax.set_zlim(1,2.5)
	title('Behold the melting lump')
	for t in range(nt+1):
		un=u.copy()
		u[1:-1,1:-1]=un[1:-1,1:-1]+nu*dt/dx**2*(un[2:,1:-1]+un[0:-2,1:-1]-2*un[1:-1,1:-1])+nu*dt/dy**2*(un[1:-1,2:]+un[1:-1,0:-2]-2*un[1:-1,1:-1])
		u[0,:]=1
		u[-1,:]=1
		u[:,0]=1
		u[:,-1]=1
	
		surf.remove()
		surf=ax.plot_surface(xx,yy,u,rstride=1,cstride=1,linewidth=0,cmap=cm.coolwarm)
		ax.set_zlim(1,2.5)
		display.clear_output(wait=True)
		display.display(fig)
		draw()
		time.sleep(0.01)
	
	
