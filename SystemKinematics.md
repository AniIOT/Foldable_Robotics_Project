---
title: System Kinematics 
---

# Code:
```python
%matplotlib inline
```


```python
#Set model to freely swim
use_free_move = False
```


```python
import pynamics
from pynamics.frame import Frame
from pynamics.variable_types import Differentiable,Constant
from pynamics.system import System
from pynamics.body import Body
from pynamics.dyadic import Dyadic
from pynamics.output import Output,PointsOutput
from pynamics.output_points_3d import PointsOutput3D
from pynamics.constraint import AccelerationConstraint,KinematicConstraint
from pynamics.particle import Particle
import pynamics.integration
import numpy
import matplotlib.pyplot as plt
plt.ion()
from math import pi,sin
import sympy
from sympy import sqrt
import math
import scipy.optimize
from matplotlib import animation, rc
from IPython.display import HTML


system = System()
pynamics.set_system(__name__,system)
```


```python
#Defining Constants of System
seg_len = .005
seg_mass = 1
seg_h = 1
seg_th = 1

lA = Constant(seg_len,'lA',system)
lB = Constant(seg_len,'lB',system)
lC = Constant(seg_len,'lC',system)
lD = Constant(seg_len,'lD',system)
lE = Constant(seg_len,'lE',system)
lF = Constant(seg_len,'lF',system)
lG = Constant(seg_len,'lG',system) #Tail segment
lP = Constant(seg_len*5,'lP',system) #Constrained length

mA = Constant(seg_mass,'mA',system)
mB = Constant(seg_mass,'mB',system)
mC = Constant(seg_mass,'mC',system)
mD = Constant(seg_mass,'mD',system)
mE = Constant(seg_mass,'mE',system)
mF = Constant(seg_mass,'mF',system)
mG = Constant(seg_mass,'mG',system)

g = Constant(9.81,'g',system)
b = Constant(1e-1,'b',system)
k = Constant(1e0,'k',system)

freq = Constant(.1,'freq',system)
amp = Constant(55*pi/180,'torque',system)
torque = Constant(1e0,'torque',system)

Ixx_A = Constant(1/12*seg_mass*(seg_h*2 + seg_th**2),'Ixx_A',system)
Iyy_A = Constant(1/12*seg_mass*(seg_h**2 + seg_len**2),'Iyy_A',system)
Izz_A = Constant(1/12*seg_mass*(seg_len**2 + seg_th**2),'Izz_A',system)
Ixx_B = Constant(1/12*seg_mass*(seg_h*2 + seg_th**2),'Ixx_B',system)
Iyy_B = Constant(1/12*seg_mass*(seg_h**2 + seg_len**2),'Iyy_B',system)
Izz_B = Constant(1/12*seg_mass*(seg_len**2 + seg_th**2),'Izz_B',system)
Ixx_C = Constant(1/12*seg_mass*(seg_h*2 + seg_th**2),'Ixx_C',system)
Iyy_C = Constant(1/12*seg_mass*(seg_h**2 + seg_len**2),'Iyy_C',system)
Izz_C = Constant(1/12*seg_mass*(seg_len**2 + seg_th**2),'Izz_C',system)
Ixx_D = Constant(1/12*seg_mass*(seg_h*2 + seg_th**2),'Ixx_D',system)
Iyy_D = Constant(1/12*seg_mass*(seg_h**2 + seg_len**2),'Iyy_D',system)
Izz_D = Constant(1/12*seg_mass*(seg_len**2 + seg_th**2),'Izz_D',system)
Ixx_E = Constant(1/12*seg_mass*(seg_h*2 + seg_th**2),'Ixx_E',system)
Iyy_E = Constant(1/12*seg_mass*(seg_h**2 + seg_len**2),'Iyy_E',system)
Izz_E = Constant(1/12*seg_mass*(seg_len**2 + seg_th**2),'Izz_E',system)
Ixx_F = Constant(1/12*seg_mass*(seg_h*2 + seg_th**2),'Ixx_F',system)
Iyy_F = Constant(1/12*seg_mass*(seg_h**2 + seg_len**2),'Iyy_F',system)
Izz_F = Constant(1/12*seg_mass*(seg_len**2 + seg_th**2),'Izz_F',system)
Ixx_G = Constant(1/12*seg_mass*(seg_h*2 + seg_th**2),'Ixx_G',system)
Iyy_G = Constant(1/12*seg_mass*(seg_h**2 + seg_len**2),'Iyy_G',system)
Izz_G = Constant(1/12*seg_mass*(seg_len**2 + seg_th**2),'Izz_G',system)
```


```python
#Set integration tolerance
tol = 1e-3
```


```python
#Set simulation run time
tinitial = 0
tfinal = 10
fps = 30
tstep = 1/fps
t = numpy.r_[tinitial:tfinal:tstep]
```


```python
#Define derivatives of frames
qA,qA_d,qA_dd = Differentiable('qA',system)
qB,qB_d,qB_dd = Differentiable('qB',system)
qC,qC_d,qC_dd = Differentiable('qC',system)
qD,qD_d,qD_dd = Differentiable('qD',system)
qE,qE_d,qE_dd = Differentiable('qE',system)
qF,qF_d,qF_dd = Differentiable('qF',system)
qG,qG_d,qG_dd = Differentiable('qG',system)

if use_free_move:
    x,x_d,x_dd = Differentiable('x',system)
    #y,y_d,y_dd = Differentiable('y',system)
```


```python
#set initial conditions
initialvalues = {}
initialvalues[qA]=70*pi/180
initialvalues[qA_d]=0*pi/180
initialvalues[qB]=30*pi/180
initialvalues[qB_d]=0*pi/180
initialvalues[qC]=0*pi/180
initialvalues[qC_d]=0*pi/180
initialvalues[qD]=0*pi/180
initialvalues[qD_d]=0*pi/180
initialvalues[qE]=-10*pi/180
initialvalues[qE_d]=0*pi/180
initialvalues[qF]=-40*pi/180
initialvalues[qF_d]=0*pi/180
initialvalues[qG]=0*pi/180
initialvalues[qG_d]=0*pi/180

if use_free_move:
    initialvalues[x]=0*pi/180
    initialvalues[x_d]=0*pi/180
    #initialvalues[y]=0*pi/180
    #initialvalues[y_d]=0*pi/180

statevariables = system.get_state_variables()
ini0 = [initialvalues[item] for item in statevariables]
```


```python
#Frames
N = Frame('N',system)
A = Frame('A',system)
B = Frame('B',system)
C = Frame('C',system)
D = Frame('D',system)
E = Frame('E',system)
F = Frame('F',system)
G = Frame('G',system)

system.set_newtonian(N)

A.rotate_fixed_axis(N,[0,0,1],qA,system)
B.rotate_fixed_axis(N,[0,0,1],qB,system)
C.rotate_fixed_axis(N,[0,0,1],qC,system)
D.rotate_fixed_axis(N,[0,0,1],qD,system)
E.rotate_fixed_axis(N,[0,0,1],qE,system)
F.rotate_fixed_axis(N,[0,0,1],qF,system)
G.rotate_fixed_axis(N,[0,0,1],qG,system)
```


```python
#Vectors

if use_free_move:
    pNA=x*N.x + 0*N.y + 0*N.z
    pP = lP*N.x + x*N.x
else:
    pNA=0*N.x + 0*N.y + 0*N.z
    pP = lP*N.x
    
pAB= pNA + lA*A.x
pBC = pAB + lB*B.x
pCD = pBC + lC*C.x
pDE = pCD + lD*D.x
pEF = pDE + lE*E.x
pFG = pEF + lF*F.x
pGtip = pFG + lG*G.x

#Center of Mass
pAcm=pNA+lA/2*A.x
pBcm=pAB+lB/2*B.x
pCcm=pBC+lC/2*C.x
pDcm=pCD+lD/2*D.x
pEcm=pDE+lE/2*E.x
pFcm=pEF+lF/2*F.x
pGcm=pFG+lG/2*G.x

#Angular Velocity
wNA = N.get_w_to(A)
wAB = A.get_w_to(B) 
wBC = B.get_w_to(C)
wCD = C.get_w_to(D) 
wDE = D.get_w_to(E)
wEF = E.get_w_to(F)
wFG = F.get_w_to(G)

#Velocities 
vGtip=pGtip.time_derivative()

#Interia and Bodys
IA = Dyadic.build(A,Ixx_A,Iyy_A,Izz_A)
IB = Dyadic.build(B,Ixx_B,Iyy_B,Izz_B)
IC = Dyadic.build(C,Ixx_C,Iyy_C,Izz_C)
ID = Dyadic.build(D,Ixx_D,Iyy_D,Izz_D)
IE = Dyadic.build(E,Ixx_E,Iyy_E,Izz_E)
IF = Dyadic.build(F,Ixx_F,Iyy_F,Izz_F)
IG = Dyadic.build(G,Ixx_G,Iyy_G,Izz_G)

BodyA = Body('BodyA',A,pAcm,mA,IA,system)
BodyB = Body('BodyB',B,pBcm,mB,IB,system)
BodyC = Body('BodyC',C,pCcm,mC,IC,system)
BodyD = Body('BodyD',D,pDcm,mD,ID,system)
BodyE = Body('BodyE',E,pEcm,mE,IE,system)
BodyF = Body('BodyF',F,pFcm,mF,IF,system)
BodyG = Body('BodyG',G,pGcm,mG,IG,system)
```


```python
#Forces
#system.addforce(torque*sympy.cos(freq*2*pi*system.t)*A.z,wNA)

#Damping Forces
system.addforce(-b*wNA,wNA)
system.addforce(-b*wAB,wAB)
system.addforce(-b*wBC,wBC)
system.addforce(-b*wCD,wCD)
system.addforce(-b*wDE,wDE)
system.addforce(-b*wEF,wEF)
system.addforce(-b*wFG,wFG)

#Spring Force (Torsion)
system.add_spring_force1(k,(qA)*N.z,wNA) 
system.add_spring_force1(k,(qB-qA)*N.z,wAB)
system.add_spring_force1(k,(qC-qB)*N.z,wBC)
system.add_spring_force1(k,(qD-qC)*N.z,wCD) 
system.add_spring_force1(k,(qE-qD)*N.z,wDE)
system.add_spring_force1(k,(qF-qE)*N.z,wEF)
system.add_spring_force1(k,(qG-qF)*N.z,wFG)
```




    (<pynamics.force.Force at 0x1e3874a5760>,
     <pynamics.spring.Spring at 0x1e3875af7c0>)




```python
#Constraints for initial condition

eq = []

eq.append(pFG-pP)
    
eq_scalar = []
eq_scalar.append(eq[0].dot(N.x))
eq_scalar.append(eq[0].dot(N.y))
eq_scalar
```




    [lA*cos(qA) + lB*cos(qB) + lC*cos(qC) + lD*cos(qD) + lE*cos(qE) + lF*cos(qF) - lP, lA*sin(qA) + lB*sin(qB) + lC*sin(qC) + lD*sin(qD) + lE*sin(qE) + lF*sin(qF)]




```python
#Solve for Intial Conditions
if use_free_move:
    qi = [qA,x]
else:  
    qi = [qA]
    
qd = [qB,qC,qD,qE,qF,qG]

eq_scalar_c = [item.subs(system.constant_values) for item in eq_scalar]
defined = dict([(item,initialvalues[item]) for item in qi])
eq_scalar_c = [item.subs(defined) for item in eq_scalar_c]

error = (numpy.array(eq_scalar_c)**2).sum()

f = sympy.lambdify(qd,error)

def function(args):
    return f(*args)

guess = [initialvalues[item] for item in qd]

result = scipy.optimize.minimize(function,guess)
if result.fun>1e-6:
    raise(Exception("out of tolerance"))
    
ini = []
for item in system.get_state_variables():
    if item in qd:
        ini.append(result.x[qd.index(item)])
    else:
        ini.append(initialvalues[item])
```


```python
#Plot Initial State
points = [pNA,pAB,pBC,pCD,pDE,pEF,pFG,pGtip]
points_output = PointsOutput(points, constant_values=system.constant_values)
points_output.calc(numpy.array([ini0,ini]),numpy.array([0,1]))
points_output.plot_time()
```

    2022-04-08 17:12:12,727 - pynamics.output - INFO - calculating outputs
    2022-04-08 17:12:13,015 - pynamics.output - INFO - done calculating outputs
    




    <AxesSubplot:>




    
![png](/01_Documents/04_Presentations/System_Kinematics/output_13_2.png)
    


Plot of Device: Points are as follow from left to right. (pNA,pAB,pBC,pCD,pDE,pEF,pFG,pGtip). All the frames are relative to the newtonian frame. See last page of document to see diagram of device.


```python
#Solve for Torque at end effector (qG_tip)

#Derivative of constraints
eq_d = [(system.derivative(item)) for item in eq_scalar]
eq_d = sympy.Matrix(eq_d)
eq_d = eq_d.subs(system.constant_values)

qi = sympy.Matrix([qA_d])
qd = sympy.Matrix([qB_d,qC_d,qD_d,qE_d,qF_d,qG_d])

AA = eq_d.jacobian(qi)

BB = eq_d.jacobian(qd)

J_int = -BB.transpose()*AA
J_int

vout = pGtip.time_derivative()
vout = sympy.Matrix([vout.dot(N.x), vout.dot(N.y)])

Ji = vout.jacobian(qi)
Jd = vout.jacobian(qd)

J = Ji + Jd*J_int

J = [item.subs(system.constant_values).subs({qA:ini[0], qB:ini[1], qC:ini[2], qD:ini[3], qE:ini[4], qF:ini[5], qG:ini[6]}) for item in J]

print('Velocity at pGtip')
J
```

    Velocity at pGtip
    




    [-0.00469839990567111, 0.00171000544586941]




```python
#Adding Dynamic Constraints

#Position of motor limits
pos = amp*sympy.cos(freq*2*pi*system.t)

eq = []

eq.append(pFG-pP)
#eq.append(pos*N.z-qA*N.z)

eq_scalar = []
eq_scalar.append(eq[0].dot(N.x))
eq_scalar.append(eq[0].dot(N.y))
#eq_scalar.append(eq[1].dot(N.z))
eq_scalar_d = [system.derivative(item) for item in eq_scalar]
eq_scalar_dd = [system.derivative(item) for item in eq_scalar_d]

system.add_constraint(AccelerationConstraint(eq_scalar_dd))
```


```python
#Solve model and plot angles

f,ma = system.getdynamics()

func1,lambda1 = system.state_space_post_invert(f,ma,return_lambda = True)

states=pynamics.integration.integrate(func1,ini,t,rtol=tol,atol=tol,hmin=tol, args=({'constants':system.constant_values},))


plt.figure()
artists = plt.plot(t,states[:,:7])
plt.legend(artists,['qA','qB','qC','qD','qE','qF','qG'])
```

    2022-04-08 17:12:14,427 - pynamics.system - INFO - getting dynamic equations
    2022-04-08 17:12:18,296 - pynamics.system - INFO - solving a = f/m and creating function
    2022-04-08 17:12:21,534 - pynamics.system - INFO - substituting constrained in Ma-f.
    2022-04-08 17:12:22,178 - pynamics.system - INFO - done solving a = f/m and creating function
    2022-04-08 17:12:22,178 - pynamics.system - INFO - calculating function for lambdas
    2022-04-08 17:12:22,380 - pynamics.integration - INFO - beginning integration
    2022-04-08 17:12:22,380 - pynamics.system - INFO - integration at time 0000.00
    2022-04-08 17:12:23,125 - pynamics.integration - INFO - finished integration
    




    <matplotlib.legend.Legend at 0x1e38999c0d0>




    
![png](/01_Documents/04_Presentations/System_Kinematics/output_17_2.png)
    



```python
points_output = PointsOutput(points,system)
y = points_output.calc(states,t)
points_output.plot_time(20)
```

    2022-04-08 17:12:23,720 - pynamics.output - INFO - calculating outputs
    2022-04-08 17:12:23,769 - pynamics.output - INFO - done calculating outputs
    




    <AxesSubplot:>




    
![png](/01_Documents/04_Presentations/System_Kinematics/output_18_2.png)
    



```python
#Constraint Forces

lambda2 = numpy.array([lambda1(item1,item2,system.constant_values) for item1,item2 in zip(t,states)])
plt.figure()
plt.plot(t, lambda2)
```




    [<matplotlib.lines.Line2D at 0x1e38bb7ae80>,
     <matplotlib.lines.Line2D at 0x1e38bb7aee0>]




    
![png](/01_Documents/04_Presentations/System_Kinematics/output_19_1.png)
    



```python
#Energy Plot

KE = system.get_KE()
PE = system.getPESprings()
energy_output = Output([KE-PE],system)
energy_output.calc(states,t)
energy_output.plot_time(t)
```

    2022-04-08 17:12:25,509 - pynamics.output - INFO - calculating outputs
    2022-04-08 17:12:25,662 - pynamics.output - INFO - done calculating outputs
    


    
![png](/01_Documents/04_Presentations/System_Kinematics/output_20_1.png)
    



```python
#Plotting Motion of Tail

plt.plot(y[:,7,0],y[:,7,1])
plt.axis('equal')
plt.title('Position of Tail')
plt.xlabel('Position X (m)')
plt.ylabel('Position Y (m)')
```




    Text(0, 0.5, 'Position Y (m)')




    
![png](/01_Documents/04_Presentations/System_Kinematics/output_21_1.png)
    



```python
points_output.animate(fps = fps,movie_name = 'render.mp4',lw=2,marker='o',color=(1,0,0,1),linestyle='-')
HTML(points_output.anim.to_html5_video())
```




<video width="432" height="288" controls autoplay loop>
  <source type="video/mp4" src="data:video/mp4;base64,AAAAIGZ0eXBNNFYgAAACAE00ViBpc29taXNvMmF2YzEAAAAIZnJlZQABHrhtZGF0AAACoAYF//+c
3EXpvebZSLeWLNgg2SPu73gyNjQgLSBjb3JlIDE2MSAtIEguMjY0L01QRUctNCBBVkMgY29kZWMg
LSBDb3B5bGVmdCAyMDAzLTIwMjAgLSBodHRwOi8vd3d3LnZpZGVvbGFuLm9yZy94MjY0Lmh0bWwg
LSBvcHRpb25zOiBjYWJhYz0xIHJlZj0zIGRlYmxvY2s9MTowOjAgYW5hbHlzZT0weDM6MHgxMTMg
bWU9aGV4IHN1Ym1lPTcgcHN5PTEgcHN5X3JkPTEuMDA6MC4wMCBtaXhlZF9yZWY9MSBtZV9yYW5n
ZT0xNiBjaHJvbWFfbWU9MSB0cmVsbGlzPTEgOHg4ZGN0PTEgY3FtPTAgZGVhZHpvbmU9MjEsMTEg
ZmFzdF9wc2tpcD0xIGNocm9tYV9xcF9vZmZzZXQ9LTIgdGhyZWFkcz05IGxvb2thaGVhZF90aHJl
YWRzPTEgc2xpY2VkX3RocmVhZHM9MCBucj0wIGRlY2ltYXRlPTEgaW50ZXJsYWNlZD0wIGJsdXJh
eV9jb21wYXQ9MCBjb25zdHJhaW5lZF9pbnRyYT0wIGJmcmFtZXM9MyBiX3B5cmFtaWQ9MiBiX2Fk
YXB0PTEgYl9iaWFzPTAgZGlyZWN0PTEgd2VpZ2h0Yj0xIG9wZW5fZ29wPTAgd2VpZ2h0cD0yIGtl
eWludD0yNTAga2V5aW50X21pbj0yNSBzY2VuZWN1dD00MCBpbnRyYV9yZWZyZXNoPTAgcmNfbG9v
a2FoZWFkPTQwIHJjPWNyZiBtYnRyZWU9MSBjcmY9MjMuMCBxY29tcD0wLjYwIHFwbWluPTAgcXBt
YXg9NjkgcXBzdGVwPTQgaXBfcmF0aW89MS40MCBhcT0xOjEuMDAAgAAAEpNliIQAM//+9uy+BTX9
n9CXESzF2kpwPiqkgIB3NMAAAAMAAHgdKF7oOEqr0MAAG3AAb80UfrEXym4ZgCKFvRvywAGLCH55
JXmheJKwhqGj/ybr7HUYVcx5zd9XVy3mRQJ9t+hyB/6IjiRfZ78sfJo/h7zAmSQnejMG8nSKVu+t
V4OAuF8JZgPgWe6wIxHhavTDeeZBNZF2vbGiR0nhWoKRA0QUH19UA0enfK8k+Z84rUoKnkkbClBC
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
MQAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAGwASAASAAAAEgAAAAAAAAAAQAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAABj//wAAADVhdmNDAWQAFf/hABhnZAAVrNlBsJaEAAADAAQAAAMA
8DxYtlgBAAZo6+PLIsD9+PgAAAAAHHV1aWRraEDyXyRPxbo5pRvPAyPzAAAAAAAAABhzdHRzAAAA
AAAAAAEAAAEsAAACAAAAABhzdHNzAAAAAAAAAAIAAAABAAAA+wAACOBjdHRzAAAAAAAAARoAAAAB
AAAEAAAAAAEAAAgAAAAAAgAAAgAAAAABAAAGAAAAAAEAAAIAAAAABQAABAAAAAABAAAGAAAAAAEA
AAIAAAAAAQAABAAAAAABAAAGAAAAAAEAAAIAAAAAAQAACAAAAAACAAACAAAAAAEAAAYAAAAAAQAA
AgAAAAABAAAEAAAAAAEAAAYAAAAAAQAAAgAAAAABAAAIAAAAAAIAAAIAAAAAAQAABgAAAAABAAAC
AAAAAAMAAAQAAAAAAQAACAAAAAACAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIA
AAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAA
AAABAAACAAAAAAEAAAgAAAAAAgAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAA
AAEAAAYAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAA
AQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAABgAAAAABAAACAAAAAAEAAAgAAAAAAgAAAgAAAAAB
AAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAgAAAAAAgAAAgAAAAABAAAKAAAAAAEA
AAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAA
CgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAAC
AAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAA
AAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAA
AAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAA
AAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAA
AQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAAB
AAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEA
AAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAA
BAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAK
AAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIA
AAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAA
AAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAA
AAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAA
AQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAAB
AAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEA
AAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAA
AAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAE
AAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoA
AAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAA
AAABAAAGAAAAAAEAAAIAAAAAAgAABAAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAA
AAEAAAgAAAAAAgAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAA
AQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAAB
AAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAgAAAAAAgAAAgAAAAACAAAEAAAAAAEA
AAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAA
AgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAA
AAAAAAEAAAIAAAAAAQAABgAAAAABAAACAAAAAAEAAAQAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAA
AAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAA
AAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAEAAAA
AAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAABgAAAAABAAACAAAAAAEAAAYAAAAA
AQAAAgAAAAABAAAIAAAAAAIAAAIAAAAAAQAABAAAAAAcc3RzYwAAAAAAAAABAAAAAQAAASwAAAAB
AAAExHN0c3oAAAAAAAAAAAAAASwAABU7AAACjwAAAJwAAACEAAACPQAAALoAAAHDAAACDgAAAk8A
AAJ/AAACkAAAA3QAAAD7AAACRQAAAqoAAACiAAADLAAAAOcAAACZAAAC1AAAALgAAAIvAAAC3gAA
AL4AAAO7AAABEwAAAN8AAALAAAAAtQAAAhsAAAHXAAAB/wAAA14AAACxAAAAqgAAApMAAADBAAAA
fgAAAFIAAAGmAAAAmAAAAFAAAABbAAACugAAAMUAAAB8AAAAmgAAAtoAAADCAAAAkwAAAzMAAAEB
AAAApwAAAIgAAALcAAAAlAAABBoAAADjAAAAowAAAJIAAAN8AAAAzgAAAIIAAACGAAACvAAAAGsA
AALxAAAAjAAAAF4AAAIpAAAAfwAAADkAAAA4AAAB2wAAADMAAAAeAAAApAAAAEMAAAAaAAAAHwAA
AhgAAABlAAAAPgAAADAAAAI0AAAAkgAAADUAAABDAAACkgAAAMAAAABYAAAAWwAAAq8AAACyAAAA
YAAAAHkAAAL3AAAAzgAAAGQAAABsAAAC2QAAAMsAAABZAAAAWAAAArsAAADDAAAAXgAAAEoAAAJ5
AAAAgAAAADQAAAAnAAABRwAAAFwAAAAkAAAAGgAAAK4AAABBAAAAFwAAACcAAAH8AAAAbgAAACgA
AAArAAACOgAAAIIAAAA7AAAAQAAAAoQAAACfAAAAPwAAAD4AAALvAAAArAAAAFIAAABHAAACegAA
AJMAAAA5AAAARAAAAmsAAACHAAAANAAAADIAAAJDAAAAawAAACEAAAAjAAAB6QAAAEQAAAAxAAAA
IQAAAHQAAAAoAAAAGAAAABsAAAH0AAAAPwAAABkAAAAmAAAB8QAAAEwAAAAbAAAAJwAAAlsAAABl
AAAAKQAAACsAAAJeAAAAegAAAC0AAAAtAAACcwAAAGAAAAAjAAAAHAAAAkcAAABYAAAAJQAAABsA
AAHzAAAALwAAABwAAAATAAABrQAAACMAAAAeAAAAFwAAAIUAAAAkAAAAGAAAABUAAAEAAAAAJgAA
ABUAAAAYAAABswAAADIAAAAhAAAAFwAAAhUAAABHAAAAHQAAACIAAAHgAAAANwAAABwAAAAdAAAC
XwAAAEcAAAAeAAAAIgAAAfIAAABDAAAAIQAAABoAAAH1AAAAKAAAABoAAAAdAAABPwAAACoAAAC+
AAAAywAAATsAAAAoAAAAHgAAABUAAACQAAAAJAAAABcAAAFhAAAAIgAAABUAAAAaAAABmgAAACcA
AAAYAAAAHgAAAd4AAAAnAAAAHgAAAB4AAAIzAAAAJAAAAB4AAAAkAAAARgAAACAAAAAgAAAY4gAA
ASgAAAFoAAAAGwAAABUAAAAYAAABKgAAACMAAAAXAAAAIAAAASYAAAAtAAAAJgAAABsAAACDAAAA
JwAAABoAAAAbAAAARgAAABsAAAA+AAAA5AAAACkAAAAcAAAAIAAAAQIAAAAhAAAAGwAAABUAAAEZ
AAAALwAAACIAAAApAAABRAAAACgAAAAXAAAAGAAAAFcAAAC0AAAAJQAAAB0AAAAYAAAAQQAAAB4A
AABPAAAAGAAAAC4AAAAkAAAAGgAAABgAAAAUc3RjbwAAAAAAAAABAAAAMAAAAGJ1ZHRhAAAAWm1l
dGEAAAAAAAAAIWhkbHIAAAAAAAAAAG1kaXJhcHBsAAAAAAAAAAAAAAAALWlsc3QAAAAlqXRvbwAA
AB1kYXRhAAAAAQAAAABMYXZmNTguNDUuMTAw
">
  Your browser does not support the video tag.
</video>




    
![png](/01_Documents/04_Presentations/System_Kinematics/output_22_1.png)
    


# Discussion

1. How many degrees of freedom does your device have? How many motors? If the answer is not the same, what determines the state of the remaining degrees of freedom? How did you arrive at that number?
 
Our device has five degrees of freedom, and the team plans to actuate it using a single servo motor. The rest of the segments will be constrained except for an unconstrained tail. We arrived at this conclusion by constraining multiple joints to see at what point we lose all foldable motion.
 
2. If your mechanism has more than one degree of freedom, please describe how those multiple degrees of freedom will work together to create a locomotory gait or useful motion. What is your plan for synchronizing, especially if passive energy storage?
 
In order to achieve the desired undulatory motion, the plan is to tune the stiffness of each joint. Through simulation, the team will determine the necessary stiffness of the joints in order to achieve the sinusoidal movement.
 
3. How did you estimate your expected end-effector forces
 
The forces that would act upon were calculated using the pynamics package in Python while running the simulation of a given force. It is almost impossible determine the expected forces at the tail of eels because they are unstable systems when held stationary. Additionally for this device, the force of the output is not critical and it is more crucial to tune the spine to imitate the motion of eels.
 
4. How did you estimate your expected end-effector speeds
 
Similarly to force estimations, the speeds were also estimated using pynamics. These estimations allowed the team to estimate the torque of the end efector to be [0.003, -0.003]. These values have been validated through the Biomechanics Background assignment. 



### I. Discussion

1. How many degrees of freedom does your device have? How many motors? If the answer is not the same, what determines the state of the remaining degrees of freedom? How did you arrive at that number?
Ans. Our device has five degrees of freedom, and the team plans to actuate it using a single servo motor. The rest of the segments will be constrained except for an unconstrained tail. We arrived at this conclusion by constraining multiple joints to see at what point we lose all foldable motion.

2. If your mechanism has more than one degree of freedom, please describe how those multiple degrees of freedom will work together to create a locomotory gait or useful motion. What is your plan for synchronizing, especially if passive energy storage?
Ans. In order to achieve the desired undulatory motion, the plan is to tune the stiffness of each joint. Through simulation, the team will determine the necessary stiffness of the joints in order to achieve the sinusoidal movement.

3. How did you estimate your expected end-effector forces?
Ans. The forces that would act upon were calculated using the pynamics package in Python while running the simulation of a given force. It is almost impossible determine the expected forces at the tail of eels because they are unstable systems when held stationary. Additionally for this device, the force of the output is not critical and it is more crucial to tune the spine to imitate the motion of eels

4. How did you estimate your expected end-effector speeds?
Ans. Similar to force estimations, the speeds were also estimated using pynamics. These estimations allowed the team to estimate the torque of the end efector to be [0.003, -0.003]. These values have been validated through the Biomechanics Background assignment.

### II. Figures

![jpg](/01_Documents/04_Presentations/System_Kinematics/Paper_Model_top.jpg)

Fig. 1: Paper Model Top View

![jpg](/01_Documents/04_Presentations/System_Kinematics/Paper_Model_side.jpg)

Fig. 2: Paper Model Side View
 
![jpg](/01_Documents/04_Presentations/System_Kinematics/CAD.jpg)
 
Fig. 3: Annotated CAD Model 
