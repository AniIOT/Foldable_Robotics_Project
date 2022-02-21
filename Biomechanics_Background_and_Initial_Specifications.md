## Biomechanics Background and Initial Specifications
### Bryan Carlton1, Aniruddha Anand Damle1, Anson Kwan1, Jacob Yoshitake1
### Team 2: Swimming
### 1Ira A. Fulton Schools of Engineering

### I. Bio Inspiration
The organism that we will draw inspiration from for our research question and system are eels.
Eels exhibit a form of undulatory motion called 
anguilliform locomotion which propagates
waves along their spines [1]. The five most
relevant sources about eels and fish based
undulatory motion are as follows:

I. Hydrodynamics of undulatory
propulsion [1]*
II. Muscles, elastic energy, and the
dynamics of body stiffness in swimming
eels [2]*
III. The hydrodynamics of Eel Swimming
[3]
IV. The hydrodynamics of eel swimming II.
effect of swimming speed [4]*
V. Interactions between internal forces,
body stiffness, and fluid environment in
a neuromechanical model of lamprey
swimming [5]

Of these sources, we will discuss I, II, and IV
since they are the most relevant to our research
question.

Hydrodynamics of undulatory propulsion
discusses the history of existing work conducted
on different forms of undulating locomotion and
propulsion [1]. This is particularly helpful for
our research question because it provides
context and basis of knowledge as we start to
explore anguilliform locomotion. Additionally,
from this paper, we are able to extract critical
information about the gait cycle and time
duration of a single oscillation for eels.
Muscles, elastic energy, and the dynamics of
body stiffness in swimming eels analyzes the
kinematic and dynamic properties of eels to
derive generalized values regarding muscle
activation, work, and stiffness along the body of
eels [2]. This research is pertinent to our
research discussion because it gives us
information regarding how to link kinematic
properties of eels to the foldable system that we
are interested in developing. The most relevant
values include the weight, length, and body
cross sectional area. These values are important
to deriving the linear velocity of eels as well as
provide a method of comparing these elements
of our device against an eel.
The hydrodynamics of eel swimming II. Effect of
swimming speed discusses the kinematics and
hydrodynamics of eels but goes into further
detail regarding dynamic conditions that eels
swim under [4]. Through the method of using
two synchronized high speed cameras, the
researchers were able to calculate frequency,
wave speed, velocities of the system, and jet
speeds produced by the eels [2]. This
information is valuable to us because it allows
us to approximate the linear velocity of eels as
well as the power needed to sustain this form of
locomotion. We are then able to specify the
requirements needed for a servo motor to power
our system.

### II. Other bio-inspired robots
Although the team has already decided on an eel
inspired design, it is also important to look at
other similar designs and other sources of
biological sources. Below are five different
sources that detail different bio-inspired robots.
I. Kinematic Evaluation of a Series of Soft
Actuators in Designing and Eel-inspired
Robot [6]*
II. How the Body Contributes to The Wake
in Undulatory Fish Swimming: Flow
Fields of a Swimming Eel [7]*
III. Thrust generation during Steady
Swimming and Acceleration from Rest
in Anguilliform Swimmers [8]
IV. Development and Motion Control of
Biomimetic Underwater Robots: A
Survey [9]
V. Flow patterns of Larval fish: Undulatory
Swimming in the Intermediate Flow
Regime [10]*
Kinematic Evaluation of a Series of Soft
Actuators in Designing and Eel-inspired Robot
This paper discusses the motion generated
through soft actuators in an eel inspired robot
[6]. Despite this being a soft robot, it provides an
in depth analysis and data on the locomotion of
an eel like robot design. As well this paper
provides information on the manufacturing of
their actuator which can guide the team's design.
How the Body Contributes to The Wake in
Undulatory Fish Swimming: Flow Fields of a
Swimming Eel
This paper discusses the body mechanics, and
how that affects the thrust of the robot [7]. This
paper provides the kinematics of an eel like
robot. It also provides helpful analysis of the
undulatory movement that eels and similar fish
use to move through water
Flow patterns of Larval fish: Undulatory
Swimming in the Intermediate Flow Regime
This paper focuses on fish larvae designs, and
the forces between the body of the larvae and the
water [10]. This paper provides helpful force
analysis between water and submerged
undulatory robots. Having these force
interactions helps the team understand how the
water affects movement and stabilization of the
robot system.

### III. Table

| Parameter | Unit | Value Range | Reference |
| Length | m | .024-.033| [2] |
| Weight | kg | 0.037-0.057 | [2] |
| Frequency | Hz | 1.3 +/- 0.10 | [4] |
| Wave Speed | L/s | 0.39 +/- 0.02 | [4] | 
| Slip | L/L | 0.784 +/-0.002 | [4] | 
| Max Area | 𝑚2 | 0.0001018 | [2] | 
| Gait Time | s | 0.26 | [1] | 
