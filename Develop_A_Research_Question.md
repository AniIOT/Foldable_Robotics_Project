# EGR 557 Foldable Robotics
## Develop A Research Question
## Team 2
### Bryan Carlton1, Aniruddha Anand Damle1, Anson Kwan1, Jacob Yoshitake1
### 1Ira A. Fulton Schools of Engineering

### I. Question
How can a length-constrained segmented beam with a compliant tail propagate a sinusoidal wave to produce anguilliform locomotion?

### II. Tractability
In order to make this project tractable within the time frame of 15 weeks, we will be focusing on tuning our segmented beam to produce motion inspired by anguilliform locomotion or ‘eel-like’ swimming. Additionally, we will be focusing on only trying to tune one position for the segmented beam. This means that we will be only focusing on optimizing the joint stiffness for the mechanism for only one configuration of the system. For the purposes of this project, we will be starting in an ‘S’ configuration and try to flip the peaks of the stable position through a simple input of a servo motor. This will greatly improve the scope of this research because it defines the goal of the research as producing this configuration as opposed to finding the optimal configuration to produce this locomotion. By focusing our research on anguilliform locomotion and defining the configuration that we will use, the scope of the research will produce successful results for this semester.

### III. Novelty
For the purposes of understanding the various research spaces that anguilliform locomotion-based robots exist, we used keywords such as ‘anguilliform locomotion’, ‘undulatory motion’, ‘soft robotics’, ‘laminate’, and ‘eel inspired’. From these keywords, we were able to get an understanding of the existing research being conducted on anguilliform-inspired robots within the foldable robotics field.
From these keywords, the most cited source was “A Geometric Approach to Anguilliform Locomotion: Modeling of an Underwater Eel Robot” [1]. This paper discusses the kinematics and simplification of anguilliform locomotion into rigid body segments and defined points of rotation [1]. This paper primarily focuses on using rigid bodies and actuation at every link. Unlike this paper, our research question will be focusing on creating undulations in the body of our system through only a single actuation and tuning our system to propagate the motion. The next most cited source was “Bio-inspired aquatic robotics by untethered piezohydroelastic actuation’ [2]. This journal entry discussed the tuning and utilization of piezoelectric laminates to produce the sinusoidal locomotion that characterizes many underwater movements [2]. Both this paper and our research questions discuss the concepts of single actuated, tunable laminate structures, but this paper focuses on carangiform locomotion which is exhibited in fish. Additionally, this paper uses
the laminate structure as the form of actuation, while our question focuses on a method of
propagating a tuned laminate structure that produces anguilliform-like locomotion. The third most
relevant and cited source was “AmphiBot I: An amphibious snake-like robot” [3]. The core
concept of this paper revolves around a wheeled multi-actuated spine that can either traverse land
or water through snake-like or eel-like locomotion [3]. This paper represents how anguilliform
locomotion can be accomplished using traditional actuators and linkages. Our question differs
entirely from this paper because our objective is to produce this kind of movement using only a
singular actuator and a precisely tuned mechanical system. Lastly, the other paper that was most
cited for these keywords were “Body wave generation for anguilliform locomotion using a fiberreinforced
soft fluidic elastomer actuator array toward the development of the EEL-inspired
Underwater Soft Robot” [4]. This journal article focused on utilizing bellows-based actuators to
inflate faces of a soft spine in order to create the curvatures present in eels to create a simplified
model and robot [4]. While both this research paper and our question revolve around simplifying
the traditionally complex systems that generate anguilliform locomotion, this is achieved in
different ways. The goal of our research is to use a single traditional actuator, such as a servo, to
produce this locomotion, while this paper focuses on using multiple pneumatic bellows to deform
the spine.

### IV. Interesting
With the development of newer and faster manufacturing methods, as well as various materials
such as ionic polymer-metal composites (IPMCs), Shape memory alloys (SMAs), and
magnetostrictive thin films, the development of bioinspired soft-robot locomotion has received a
lot of attention and has recently been the topic of discussion between the scientific communities.
The scientific and engineering communities have expressed a strong interest in applying biological
underwater locomotion to marine robots in order to improve performance in various marine
missions such as search and rescue, naval intelligence, and surveillance and reconnaissance. The
motivation for eel-like biometric locomotion ranges from underwater sensing and exploration for
long-term ecological sustainability to drug delivery and disease screening in medicine.

### V. Open-Ended
In addition to our research question, there are a slew of other questions that could be asked. From
the way that we defined and structured our research question, our primary focus is anguiliform
locomotion and tuning for a single starting configuration. This means that there is an opportunity
to conduct more research on optimizing the initial configuration for anguilliform locomotion or
adapting the mechanism for different forms of aquatic locomotion as well as terrestrial locomotion.
Additionally, even within the scope of our question, there is potential to do a deeper look into
optimizing a tail or compliant attachment to propagate the motion produced from the mechanism.
Our research question's goal is to start looking into the possibilities for this mechanism, which is
just a starting point for a variety of applications and dynamics that can be used with this system.

### VI. Modularity
Our question can be applied to a wide range of research topics such as biomimicry, laminate
robotics, aquatic propulsion, and unmanned surface vehicles (USV). Because we are focusing on
inspired locomotion from eels, biologists can compare locomotion behaviors of eels with the
device we develop. Through that research, it is possible for new or additional correlations could
be drawn from this comparison. Research in laminate robotics can further develop this mechanism
to be applied to multiple different environments or be use the data from tuning joint stiffness with
the experimental material for future mechanisms. For researchers interested in aquatic propulsion,
studies could be conducted in simulating and testing the fluid dynamics of the mechanism in
relation to correlating fluid environments that produce the best results. USV are an up-and-coming
field of research that focus mainly on producing devices and mechanisms that can be controlled
remotely in variable viscous fluids and land. Through our research, future work can be done on
developing and controlling this mechanism so that it can be applied to this field. There may be
other applications of our research to other topics, but these three are the most prominent that we
could recognize.

### VII. Team Fit
Our team members have a variety of interests and abilities that will be necessary to this research.
Anson Kwan is currently conducting research on simulating and analyzing soft beams to produce
anguilliform locomotion and has an interest in crossing over the concept of singular actuation to
create complex motion into laminate systems. He has experience with python-based simulation as
well as using optimization algorithms such as SciPy and CMA-ES to solve and fit complex
dynamic systems. Through this research question, he hopes to learn more about laminate structures
and how they apply to anguilliform locomotion and how joint design in laminate design affects
stiffness and compliance
Jacob Yoshitake has a variety of skills based in 3D modeling and manufacturing. His extensive
experience with SolidWorks will allow the team to be able to create 3D models of the robot at a
rapid pace, as well as simulate it though either SolidWorks or ANSYS. He also understands a
wide variety of manufacturing techniques (3D printing, laser cutting, welding, milling, lathe, etc.)
as well as a wealth of experience manufacturing mechanisms that will be paramount to being able
to produce a working robot. Though his interests are mostly in aerospace, the research into
propulsion of an aquatic robot also interests him.
Bryan Carlton is currently working on a research project in which he is working on another
bioinspired design. With this research question, he hopes to learn how to apply biological concepts
to robotics in order to provide simpler, effective solutions whether it be in exoskeleton
development or an underwater vehicle. This project allows Bryan to research simpler solutions for
underwater locomotion while learning new robotic methods. He currently has experience with
CAD softwares such as Fusion 360 and Solidworks. He has also worked with electrical design
software Cadence. Bryan also has experience in programming languages such as C/C++,
MATLAB/Simulink, and a little bit of Python.
Aniruddha Anand Damle is interested in creating a low-cost mechanism that can be used for
terrestrial robots. He has a strong background in embedded systems, machine learning, and
communication protocols, allowing us to quickly develop functional prototypes. He's also worked
with 3D modeling and production techniques like 3D printing, laser cutting, and lathes. He has
previously worked on multi-layered PCB design and development, as well as various ARM-based
platforms. Although his interest is in Swarm Robotics, he hopes to develop a low-cost muscle
design that can be used to build an origami-inspired quadruped robot with this research question.

### VIII. Topic Fit
For this research to be successful, it will be utilizing the techniques of analysis of geometric
kinematics, dynamics, and laminate manufacturing processes for our device that will be taught
over this course. In order to determine the joint stiffnesses of the segmented beam, we will need
to conduct experimental data and fit the simulated data to it. To achieve this, we will have to solve
for the kinematics of the system as well as simulate the dynamics of the model to find the joint
characteristics. Additionally, in order to find the optimal joint stiffness to match our desired
dynamics and starting configuration, we will need to solve for the dynamics of the system which
is a major topic of this course. Finally, the segmented beam will be manufactured using foldable
robotics processes. Foldable techniques are critical to answering this question because it allows
for the flexibility of easily experimenting with a number of joints and links using inexpensive
materials and low manufacturing time. Since varying the stiffness of joints is just a relationship to
how much material is removed between the joints and there is a multitude of configurations for
how to design the joints, foldable techniques are an ideal solution to prototype our device.

### IX. Bibliography
[1] K. A. Melsaac and J. P. Ostrowski, “A geometric approach to anguilliform locomotion:
Modelling of an underwater Eel Robot,” Proceedings 1999 IEEE International Conference on
Robotics and Automation (Cat. No.99CH36288C), pp. 2843–2848, May 1999.
[2] L. Cen and A. Erturk, “Bio-inspired Aquatic Robotics by untethered piezohydroelastic
actuation,” Bioinspiration & Biomimetics, vol. 8, no. 1, p. 016006, 2013.
[3] A. Crespi, A. Badertscher, A. Guignard, and A. J. Ijspeert, “Amphibot I: An amphibious
snake-like robot,” Robotics and Autonomous Systems, vol. 50, no. 4, pp. 163–175, 2005.
[4] H. Feng, Y. Sun, P. A. Todd, and H. P. Lee, “Body wave generation for anguilliform
locomotion using a fiber-reinforced soft fluidic elastomer actuator array toward the development
of the EEL-inspired Underwater Soft Robot,” Soft Robotics, vol. 7, no. 2, pp. 233–250, 2020.
