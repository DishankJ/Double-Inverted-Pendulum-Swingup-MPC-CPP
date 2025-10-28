import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

df = pd.read_csv("build/state_trajectory.csv", header=None)

time = df.iloc[:, 0]
states = df.iloc[:, 1:4]

plt.figure()
for i in range(states.shape[1]):
    plt.plot(time, states.iloc[:, i], label=f"State {i+1}")

plt.xlabel("Time")
plt.ylabel("State")
plt.title("State trajectories")
plt.legend(['x', 'theta1', 'theta2'])
plt.grid(True)
plt.show()

dt = 0.03;
m0 = 0.6;
m1 = 0.2;
m2 = 0.2;
L1 = 0.5;
L2 = 0.5;
l1 = L1/2;
l2 = L2/2;
g = 9.8;
I1 = (1.0/12.0)*m1*pow(L1,2);
I2 = (1.0/12.0)*m2*pow(L2,2);

# animation
fig, ax = plt.subplots()
ax.set_xlim(-4, 6)
ax.set_ylim(-1.5, 1.5)
plt.grid()

x1_mass = states.iloc[:,0]
x1_pendulum = states.iloc[:,0] + L1 * np.sin(states.iloc[:,1])
y1_pendulum = L1 * np.cos(states.iloc[:,1])
x2_pendulum = states.iloc[:,0] + L1 * np.sin(states.iloc[:,1]) + L2 * np.sin(states.iloc[:,2])
y2_pendulum = L1 * np.cos(states.iloc[:,1]) + L2 * np.cos(states.iloc[:,2])

def update(frame):
    pendulum1.set_data([x1_mass[frame], x1_pendulum[frame]], [0, y1_pendulum[frame]])
    pendulum2.set_data([x1_pendulum[frame], x2_pendulum[frame]], [y1_pendulum[frame], y2_pendulum[frame]])
    mass1.set_data([x1_mass[frame]], [0])
    mass2.set_data([x1_pendulum[frame]], [y1_pendulum[frame]])
    mass3.set_data([x2_pendulum[frame]], [y2_pendulum[frame]])
    return mass1, pendulum1, mass2, pendulum2, mass3

mass1, = ax.plot([x1_mass[0]], [0], 'o', markersize=4*int(m0)+1, color='red')
pendulum1, = ax.plot([x1_mass[0], x1_pendulum[0]], [0, y1_pendulum[0]], lw=2, color='blue')
mass2, = ax.plot([x1_pendulum[0]], [y1_pendulum[0]], 'o', markersize=4*int(m1)+1, color='red')
pendulum2, = ax.plot([x1_pendulum[0], x2_pendulum[0]], [y1_pendulum[0], y2_pendulum[0]], lw=2, color='orange')
mass3, = ax.plot([x2_pendulum[0]], [y2_pendulum[0]], 'o', markersize=4*int(m2)+1, color='red')

animation = FuncAnimation(fig, update, frames=len(states.iloc[:,0]), interval=int(1/dt), blit=True)
# animation.save('dip_anim.gif', writer='pillow')
plt.show()