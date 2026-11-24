import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d



x0 = 0
y0 = 0
z0 = 0
vx0 = 1
vy0 = 0
vz0 = 0.3
q = 1
m = 1
B = 1

N = 200

omega = q*B/m
dt = 0.1 / omega
trange = np.arange(N) * dt

vx_an = np.cos(omega*trange)
vy_an = -np.sin(omega*trange)
x_an = np.sin(omega*trange)/omega
y_an = np.cos(omega*trange)/omega - 1/omega
z_an = vz0 * trange


x_sym = np.zeros(N)
y_sym = np.zeros(N)
z_sym = np.zeros(N)
vx_sym = np.zeros(N)
vy_sym = np.zeros(N)
vz_sym = np.zeros(N)
vx_sym[0] = vx0
vy_sym[0] = vy0
vz_sym[0] = vz0


def sym_push(r1,r2,r3,v1,v2,v3, a,dt):
    D = 1+a**2
    r1new = r1 + dt/2*(v1 + (1-a**2)/D*v1 + 2*a/D*v2)
    r2new = r2 + dt/2*(v2 + -a*2/D*v1 + (1-a**2)/D*v2)
    r3new = r3 + dt/2*(v3 + v3)
    v1new = (1-a**2)/D * v1 + 2*a/D * v2
    v2new = -2*a/D * v1 + (1-a**2)/D * v2
    v3new = v3
    return r1new, r2new, r3new, v1new, v2new, v3new 


a = dt/2*omega
for it in range(1, N):
    x_sym[it], y_sym[it], z_sym[it], vx_sym[it], vy_sym[it], vz_sym[it] = sym_push(
        x_sym[it-1], y_sym[it-1],z_sym[it-1], vx_sym[it-1], vy_sym[it-1], vz_sym[it-1],
        a, dt
    )


import time
from matplotlib import animation


FRAMES = N
FPS = 24
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

ax.set_xlim([-1.1,1.1])
ax.set_ylim([-2.1,0.1])
ax.set_zlim([-0.5,6.5])
ax.set_aspect("equal")

line_an = None
line_sym = None
h_an = None
h_sym = None
tstart = time.time()
ax.scatter(x_sym[0],  y_sym[0],  z_sym[0],  color="red",  s=60, marker="o")
i = 0
h_an = ax.scatter(x_an[i],  y_an[i],  z_an[i],  color="C0",  s=120, marker="x",zorder=7, label="analytic")
h_sym = ax.scatter(x_sym[i],  y_sym[i],  z_sym[i],  color="C3",  s=60, marker="s", facecolor="none",zorder=6, label="symplectic")
plt.legend()
def animate(i):
    global line_an,line_sym, h_an, h_sym
    for artist in (line_an, line_sym, h_an, h_sym):
        if artist:
            artist.remove()

    line_an = ax.plot(x_an[0:i],y_an[0:i],z_an[0:i], color="C0",linewidth=5)[0]
    line_sym = ax.plot(x_sym[0:i],y_sym[0:i],z_sym[0:i],color="C3")[0]
    h_an = ax.scatter(x_an[i],  y_an[i],  z_an[i],  color="C0",  s=120, marker="x",zorder=7)
    h_sym = ax.scatter(x_sym[i],  y_sym[i],  z_sym[i],  color="C3",  s=60, marker="s", facecolor="none",zorder=6)


    if i == FRAMES - 1:
        global tstart
        fps = FRAMES / (time.time() - tstart)
        print(f'Expected FPS: {FPS}; Average FPS: {fps}')
        tstart = time.time()

ani = animation.FuncAnimation(fig, animate, interval=1000 / FPS, frames=FRAMES)
# plt.show()

ani.save("sym.gif")