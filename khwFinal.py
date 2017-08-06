import matplotlib.pyplot as plt
import numpy as np
from IPython import display
import time,math,sys
#meat-n-potatoes at the bottom. Happy St. Paddy's Day!

	#CONTINUOUSLY VARYING PROFILE
def initContVar(Q,top,bottom,ny):
	m=(top-bottom)/ny
	for i in range(ny):
		#first-order approximation to density variation in gravity
		Q[i,:]=i
	return(m*Q+bottom)

	#EXPONENTIALLY VARYING PROFILE
def initExpVar(Q,top,bottom,ny,H):
	A=(top-bottom)
	for i in range(ny):
		#realistic density profile in gravity
		Q[i,:]=np.expm1(-(i-ny)/H)
	return(A*Q+bottom)

	#HYPERBOLICALLY VARYING PROFILE
def initHypVar(Q,top,bottom,ny,k):
	A=0.5*(top-bottom)/np.tanh(k*ny/2)
	for i in range(ny):
		#analogous to gradfree but more 'realistic', i.e. continuous
		#scales gradient sharpness with k
		Q[i,:]=np.tanh(k*(i-ny/2))
	return(A*Q+0.5*(bottom+top))

	#INITIAL PERTURBATION
def initVPerturb(v,nx,ny,k,Ri,F):
	for x in range(nx):
		for y in range(ny):
			#up on the left, down on the right, biggest in the middle, zero at the walls
			v[y,x]=np.sin(2*np.pi*x/nx)*np.sin(np.pi*y/ny)
	return(F*Ri*v) #scale so it actually has an effect but doesn't dominate/create weird regions of super high density

def initVLine(v,nx,ny,k,Ri,F):
	for x in range(nx):
		v[ny/2+1,x]=np.sin(2*np.pi*x/nx)
	return(F*Ri*v)

#ABSOLUTE MAGNITUDE OF VELOCITY
def magni(u,v):
	#absolute magnitude
	uv=u**2+v**2
	return(np.sqrt(uv))

#DENSITY UPDATE
def updateDens(rho,u,v,dx,dy,dt,nx,ny):
	#vector notation
	#d(rho)/dt + div(rho*u) = 0
	#source term is zero everywhere
	m1 = np.amax(rho)
	m2 = np.amin(rho)
	nit=int(((m1**2-m2**2)/(m1*m2))**2) #scaling so that we update more times for higher gradients to give it a chance to smooth out and quell the rippling we observe
	for t in range(nit):
		rho=upWind(rho,np.zeros_like(rho),u,v,dx,dy,dt,nx,ny)
	return(rho)

#VELOCITY (MOMENTUM) UPDATE
def updateVels(u,v,rho,dx,dy,dt,nx,ny,g,T):
	#vector notation
	#d(u)/dt + dotProduct(u,grad(u)) = S
	un=u.copy()
	Su=setSrcU(rho,dx,dy,T)
	u=upWind(u,Su,u,v,dx,dy,dt,nx,ny)
	Sv=setSrcV(rho,dx,dy,g,T)
	v=upWind(v,Sv,un,v,dx,dy,dt,nx,ny)
	return(u,v)

	#SOURCE TERM, U
def setSrcU(rho,dx,dy,T):
	#vector notation
	#S_u = -(T/rho)*drho/dx
	S=np.zeros_like(rho)
	#bulk
	S[1:-1,1:-1]=rho[1:-1,1:-1]-rho[1:-1,:-2]
	#x-bc
	S[1:-1,-1]=rho[1:-1,-1]-rho[1:-1,-2]
	S[1:-1,0]=rho[1:-1,0]-rho[1:-1,-1]
	#y-bc
	S[-1,1:-1]=rho[-1,1:-1]-rho[-1,:-2]
	S[0,1:-1]=rho[0,1:-1]-rho[0,:-2]
	#corners
	S[-1,-1]=rho[-1,-1]-rho[-1,-2]
	S[-1,0]=rho[-1,0]-rho[-1,-1]
	S[0,-1]=rho[0,-1]-rho[0,-2]
	S[0,0]=rho[0,0]-rho[0,-1]
	return(-T*S/(dx*rho))

#SOURCE TERM, V
def setSrcV(rho,dx,dy,g,T):
	#vector notation
	#S_v = -(T/rho)*drho/dy + g
	S=np.zeros_like(rho)
	#bulk
	S[1:-1,1:-1]=rho[1:-1,1:-1]-rho[:-2,1:-1]
	#x-bc
	S[1:-1,-1]=rho[1:-1,-1]-rho[:-2,-1]
	S[1:-1,0]=rho[1:-1,0]-rho[:-2,0]
	#y-bc
	S[-1,1:-1]=S[-2,1:-1] #dS/dy==0 at boundary
	S[0,1:-1]=S[1,1:-1]
	#corners
	S[-1,-1]=S[-2,-1]
	S[-1,0]=S[-2,0]
	S[0,-1]=S[1,-1]
	S[0,0]=S[1,0]
	return(-T*S/(dy*rho)+g)

##FLUXES FOR UPWINDING SCHEME
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

#Richardson number is a ratio of bouyancy (pot. energy) to advection (kin. energy). smaller = more turbulent, in general
#width describes how wide the graph is, nt is the total sim time, sigma enforces the CFL condition as long as it's below 1, T scales the temperature for use in the Equation of State substitution, Hscale changes the exponential distribution's scale height, Kscale changes the hyperbolic tangent's scale height, and the rest are self-explanatory
def run(width=15,nt=10000,sigma=0.2,rho1=.1,rho2=.5,Ri=0.15,T=273.15,Hscale=10,Kscale=50,initU='tanh',initRho='tanh',graph='density',cmap='jet'):

	#MESH SCALE
	#grid elements per 'unit' length/width
	sclFac=20
	#MESH PROPERTIES
	ny=sclFac*width+1; nx=ny

	#CONSTANTS
	#gravity accel.
	g=-9.81
	#scale height for exp. distribution (smaller H => more drastic gradient)
	H=Hscale*ny
	#scale height for hyp. distribution (larger  k => more drastic gradient)
	k=float(Kscale)/ny
	#square mesh
	dy=1./sclFac
	dx=dy

	#MESH INITIALIZATION
	u=np.zeros((ny,nx)); v=np.zeros((ny,nx)); rho=np.zeros((ny,nx))
	#total time elapsed
	timeElapsed=0; timeCheck=0; printCount=1

	#FIELD INITIALIZATION
	#calculating velocities given the Richardson number and densities
	diffsq=-g*nx*(rho2**2-rho1**2)/(8*np.pi*rho1*rho2*Ri)
	#escape if the sqrt will end up with an imaginary component
	if diffsq<0:
		print("\n\nPlease enter positive values for the initial physical parameters.")
		exit()
	#equal and opposite directions: note the extra factor of 1/4 in the equation above used to preserve appropriate velocity *difference*
	F2=np.sqrt(diffsq)
	F1=-F2
	#initialize the velocities and density using opt. arg.
	v=initVPerturb(v,nx,ny,k,Ri,F2)

	if initRho=='tanh':
		rho=initHypVar(rho,rho1,rho2,ny,k)
	elif initRho=='cont':
		rho=initContVar(rho,rho1,rho2,ny)
	elif initRho=='nograd':
		rho[int(ny/2):ny,:]=rho1
		rho[:int(ny/2),:]=rho2
	else:
		print("\n\nError: misunderstood initialization conditions for density. Defaulting to exponential profile.")
		rho=initExpVar(rho,rho1,rho2,ny,H)

	if initU=='tanh':
		u=initHypVar(u,F1,F2,ny,k)
	elif initU=='cont':
		u=initContVar(u,F1,F2,ny)
	elif initU=='nograd':
		u[int(ny/2):ny,:]=F1
		u[:int(ny/2),:]=F2
	else:
		print("\n\nError: misunderstood initialization conditions for horizontal velocity. Defaulting to exponential profile.")
		u=initExpVar(u,F1,F2,ny,H)

	#calculate the total mass lost/gained during simulation against this
	rho0=np.sum(rho)

	#FIGURE INITIALIZATION
	fig=plt.figure(figsize=(10,10)) #force size
	ax=fig.add_subplot(111)
	ax.tick_params(direction='out',top='off',right='off')
	#for the quiver
	x=np.arange(nx)
	y=np.arange(ny)
	X,Y=np.meshgrid(x,y)

	#there's a lot more lines of code here than there *could be* but its for the sake of less if statements run each cycle. not sure how important it is really but anything we can do to reduce computation time is preferable
	if graph=='density':
		#PLOT
		#density heatmap using optional parameter cmap
		dens=ax.imshow(rho,cmap=cmap,origin='lower',aspect='equal')
		#quiv=plt.quiver(X[::nx/25,::nx/25],Y[::nx/25,::nx/25],u[::nx/25,::nx/25],v[::nx/25,::nx/25],pivot='mid',headwidth=2.618)
		plt.title("Density, Ri={:.3f}, Density Ratio={:.1f}, t={:.5f}s".format(Ri,rho2/rho1,timeElapsed))
		#flipped so the right densities are above/below (drho/dy is negative)
		cbar = plt.colorbar(dens,shrink=-0.618,ticks=[rho1,(rho1+rho2)/2,rho2])
		#ticks for the initial max, min, and mean densities
		cbar.ax.set_yticklabels(['    {:.2f}'.format(rho1),'    {:.2f}'.format((rho1+rho2)/2),'    {:.2f}'.format(rho2)])
		plt.draw()
		plt.savefig("{:05d}.png".format(printCount))
		#EVOLVE TIME
		for t in range(nt):
			#update values
			dt=sigma*dx/max(np.amax(u),np.amax(v))
			rho=updateDens(rho,u,v,dx,dy,dt,nx,ny)
			print("conserved? {:e}".format(np.sum(rho)-rho0))
			u,v=updateVels(u,v,rho,dx,dy,dt,nx,ny,g,T)
			#counters for display/picture saving for gifs
			timeElapsed+=dt; timeCheck+=dt
			#keeping track of the simulation progression
			print('dt={:.4e}\t\tt={:.4e}'.format(dt,timeElapsed))

			#update figure: it's ugly but it's the only way that updates the colorbar that I have patience to code in
			if timeCheck>0.0001:
				plt.clf()
				ax=fig.add_subplot(111)
				dens=ax.imshow(rho,cmap=cmap,origin='lower',aspect='equal')
				#quiv=plt.quiver(X[::nx/25,::nx/25],Y[::nx/25,::nx/25],u[::nx/25,::nx/25],v[::nx/25,::nx/25],pivot='mid',headwidth=2.618)
				ax.tick_params(direction='out',top='off',right='off')
				plt.title("Density, Ri={:.3f}, Density Ratio={:.1f}, t={:.5f}s".format(Ri,rho2/rho1,timeElapsed))
				#flipped colorbar with custom ticks
				cbar = plt.colorbar(dens,shrink=-0.618,ticks=[rho1,(rho1+rho2)/2,rho2])
				#tick labels
				cbar.ax.set_yticklabels(['    {:.2f}'.format(rho1),'    {:.2f}'.format((rho1+rho2)/2),'    {:.2f}'.format(rho2)])
				display.clear_output(wait=True)
				display.display(fig)
				plt.draw()
				#save fig for later converting to gif
				plt.savefig("{:05d}.png".format(printCount))
				#keeps track of the image number
				printCount+=1
				#keeps track of the time elapsed from the last savefig()
				timeCheck-=0.0001

	elif graph=='horizvel':
		#PLOT
		#horiz. velocity heatmap using optional parameter cmap
		xvel=ax.imshow(u,cmap=cmap,origin='lower',aspect='equal')
		#quiv=plt.quiver(X[::nx/25,::nx/25],Y[::nx/25,::nx/25],u[::nx/25,::nx/25],v[::nx/25,::nx/25],pivot='mid',headwidth=2.618)
		plt.title("Horizontal Velocity, Ri={:.3f}, Density Ratio={:.1f}, t={:.5f}s".format(Ri,rho2/rho1,timeElapsed))
		#du/dy>0
		cbar = plt.colorbar(xvel,shrink=0.618,ticks=[F1,(F1+F2)/2,F2])
		#ticks for the initial max, min, and mean horiz. vels
		cbar.ax.set_yticklabels(['    {:.2f}'.format(F1),'    {:.2f}'.format((F1+F2)/2),'    {:.2f}'.format(F2)])
		plt.draw()

		#EVOLVE TIME
		for t in range(nt):
			#update values
			dt=sigma*dx/max(np.amax(u),np.amax(v))
			rho=updateDens(rho,u,v,dx,dy,dt,nx,ny)
			print("conserved? {:e}".format(np.sum(rho)-rho0))
			u,v=updateVels(u,v,rho,dx,dy,dt,nx,ny,g,T)
			#keeping track of the simulation progression
			print('dt={:.4e}\t\tt={:.4e}'.format(dt,timeElapsed))
			#counters for display/picture saving for gifs
			timeElapsed+=dt; timeCheck+=dt
			#update figure: it's ugly but it's the only way that updates the colorbar that I have patience to code in
			if timeCheck>0.0001:
				plt.clf()
				ax=fig.add_subplot(111)
				xvel=ax.imshow(u,cmap=cmap,origin='lower',aspect='equal')
				#quiv=plt.quiver(X[::nx/25,::nx/25],Y[::nx/25,::nx/25],u[::nx/25,::nx/25],v[::nx/25,::nx/25],pivot='mid',headwidth=2.618)
				ax.tick_params(direction='out',top='off',right='off')
				plt.title("Horizontal Velocity, Ri={:.3f}, Density Ratio={:.1f}, t={:.5f}s".format(Ri,rho2/rho1,timeElapsed))
				#colorbar with custom ticks
				cbar = plt.colorbar(xvel,shrink=0.618,ticks=[F1,(F1+F2)/2,F2])
				#tick labels
				cbar.ax.set_yticklabels(['    {:.2f}'.format(F1),'    {:.2f}'.format((F1+F2)/2),'    {:.2f}'.format(F2)])
				display.display(fig)
				plt.draw()
				#save fig for later converting to gif
				plt.savefig("{:05d}.png".format(printCount))
				#keeps track of the image number
				printCount+=1
				#keeps track of the time elapsed from the last savefig()
				timeCheck-=0.0001
	
	elif graph=='vertivel':
		#PLOT
		#vert. velocity heatmap using optional parameter cmap
		yvel=ax.imshow(v,cmap=cmap,origin='lower',aspect='equal')
		#quiv=plt.quiver(X[::nx/25,::nx/25],Y[::nx/25,::nx/25],u[::nx/25,::nx/25],v[::nx/25,::nx/25],pivot='mid',headwidth=2.618)
		plt.title("Vertical Velocity, Ri={:.3f}, Density Ratio={:.1f}, t={:.5f}s".format(Ri,rho2/rho1,timeElapsed))
		cbar = plt.colorbar(yvel,shrink=0.618,ticks=[F1*Ri,(F1+F2)*Ri/2,F2*Ri])
		#ticks for the initial max, min, and mean vert. vels
		cbar.ax.set_yticklabels(['    {:.2f}'.format(F1*Ri),'    {:.2f}'.format((F1+F2)*Ri/2),'    {:.2f}'.format(F2*Ri)])
		plt.draw()

	#EVOLVE TIME
		for t in range(nt):
			#update values
			dt=sigma*dx/max(np.amax(u),np.amax(v))
			rho=updateDens(rho,u,v,dx,dy,dt,nx,ny)
			print("conserved? {:e}".format(np.sum(rho)-rho0))
			u,v=updateVels(u,v,rho,dx,dy,dt,nx,ny,g,T)
			#keeping track of the simulation progression
			print('dt={:.4e}\t\tt={:.4e}'.format(dt,timeElapsed))
			#counters for display/picture saving for gifs
			timeElapsed+=dt; timeCheck+=dt
			#update figure: it's ugly but it's the only way that updates the colorbar that I have patience to code in
			if timeCheck>0.0001:
				plt.clf()
				ax=fig.add_subplot(111)
				yvel=ax.imshow(v,cmap=cmap,origin='lower',aspect='equal')
				#quiv=plt.quiver(X[::nx/25,::nx/25],Y[::nx/25,::nx/25],u[::nx/25,::nx/25],v[::nx/25,::nx/25],pivot='mid',headwidth=2.618)
				ax.tick_params(direction='out',top='off',right='off')
				plt.title("Vertical Velocity, Ri={:.3f}, Density Ratio={:.1f}, t={:.5f}s".format(Ri,rho2/rho1,timeElapsed))
				#colorbar with custom ticks
				cbar = plt.colorbar(yvel,shrink=0.618,ticks=[F1*Ri,(F1+F2)*Ri/2,F2*Ri])
				#tick labels
				cbar.ax.set_yticklabels(['    {:.2f}'.format(F1*Ri),'    {:.2f}'.format((F1+F2)*Ri/2),'    {:.2f}'.format(F2*Ri)])
				display.display(fig)
				plt.draw()
				#save fig for later converting to gif
				plt.savefig("{:05d}.png".format(printCount))
				#keeps track of the image number
				printCount+=1
				#keeps track of the time elapsed from the last savefig()
				timeCheck-=0.0001

	elif graph=='magnivel':
		#PLOT
		#vel. magnitude heatmap using optional parameter cmap
		mag = magni(u,v)
		mvel=ax.imshow(mag,cmap=cmap,origin='lower',aspect='equal')
		#quiv=plt.quiver(X[::nx/25,::nx/25],Y[::nx/25,::nx/25],u[::nx/25,::nx/25],v[::nx/25,::nx/25],pivot='mid',headwidth=2.618)
		plt.title("Velocity Magnitude, Ri={:.3f}, Density Ratio={:.1f}, t={:.5f}s".format(Ri,rho2/rho1,timeElapsed))
		m1 = np.amin(mag); m2 = (np.amax(mag)+np.amin(mag))/2; m3 = np.amax(mag)
		cbar = plt.colorbar(mvel,shrink=0.618,ticks=[m1,m2,m3])
		#ticks for the initial max, min, and mean vel. mags
		cbar.ax.set_yticklabels(['    {:.2f}'.format(m1),'    {:.2f}'.format(m2),'    {:.2f}'.format(m3)])
		plt.draw()

		#EVOLVE TIME
		for t in range(nt):
			#update values
			dt=sigma*dx/max(np.amax(u),np.amax(v))
			rho=updateDens(rho,u,v,dx,dy,dt,nx,ny)
			print("conserved? {:e}".format(np.sum(rho)-rho0))
			u,v=updateVels(u,v,rho,dx,dy,dt,nx,ny,g,T)
			#keeping track of the simulation progression
			print('dt={:.4e}\t\tt={:.4e}'.format(dt,timeElapsed))
			#counters for display/picture saving for gifs
			timeElapsed+=dt; timeCheck+=dt
			#update figure: it's ugly but it's the only way that updates the colorbar that I have patience to code in
			if timeCheck>0.0001:
				plt.clf()
				ax=fig.add_subplot(111)
				mvel=ax.imshow(magni(u,v),cmap=cmap,origin='lower',aspect='equal')
				#quiv=plt.quiver(X[::nx/25,::nx/25],Y[::nx/25,::nx/25],u[::nx/25,::nx/25],v[::nx/25,::nx/25],pivot='mid',headwidth=2.618)
				ax.tick_params(direction='out',top='off',right='off')
				plt.title("Velocity Magnitude, Ri={:.3f}, Density Ratio={:.1f}, t={:.5f}s".format(Ri,rho2/rho1,timeElapsed))
				#colorbar with custom ticks
				cbar = plt.colorbar(mvel,shrink=0.618,ticks=[m1,m2,m3])
				#tick labels
				cbar.ax.set_yticklabels(['    {:.2f}'.format(m1),'    {:.2f}'.format(m2),'    {:.2f}'.format(m3)])
				display.display(fig)
				plt.draw()
				#save fig for later converting to gif
				plt.savefig("{:05d}.png".format(printCount))
				#keeps track of the image number
				printCount+=1
				#keeps track of the time elapsed from the last savefig()
				timeCheck-=0.0001
	else:
		print("Sorry, didn't understand the graphing option. Please choose a parameter from the options below:\n'density' (default)\t'magnivel'\n'horizvel'\t\t'vertivel'")