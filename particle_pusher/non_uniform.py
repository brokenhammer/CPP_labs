"""Implicit mid-point rule in a non-uniform magnetic field.

Vector potential   A(r) = (0, B0*(A2*x^2 + A1*x), 0)
Magnetic field     B(r) = curl A = (0, 0, B0*(2*A2*x + A1))
                   (uniform part B0*A1 plus a linear gradient 2*B0*A2*x)

Two formulations of the implicit (Gauss-Legendre / mid-point) rule are compared:

  canonical       (r, p) with p = m*v + q*A(r),  H = |p - q*A(r)|^2 / (2m)
      r_{n+1} = r_n + dt * v_mid
      p_{n+1} = p_n + dt * q * G(r_mid) @ v_mid        with G[i,j] = dA_j/dr_i
      v_mid   = (p_mid - q*A(r_mid)) / m

  non-canonical   (r, v)
      r_{n+1} = r_n + dt * v_mid
      v_{n+1} = v_n + dt * (q/m) * v_mid x B(r_mid)

with r_mid = (r_n + r_{n+1})/2, p_mid = (p_n + p_{n+1})/2, v_mid = (v_n + v_{n+1})/2.

Both rules are second order and symmetric in time, and both are implicit.  They
are NOT equivalent for a non-uniform field: since p = m*v + q*A(r) is a
nonlinear, position dependent change of variables, the mid-point rule does not
commute with it.  What each discrete map preserves exactly also differs:

  * canonical map   p_y = m*v_y + q*A_y(x) is conserved to round-off (y is
                    ignorable), but the energy only to O(dt^2), because
                    H = |p - qA|^2/(2m) is quartic in x for this field.
  * (r, v) map      |v|, hence the energy, is conserved to round-off since it is
                    a quadratic invariant of the (r, v) system.  Because A_y is
                    at most quadratic in x the trapezoidal flux term is exact, so
                    this map keeps p_y to round-off as well.

Both maps converge to the same orbit at O(dt^2) but differ from each other at
O(dt^2) for any finite dt (see the refinement table printed at the end).

Test case (bouncing in x, drift in y):
    x0 = 0, vx0 != 0, vy0 = 0   ->   p_y = m*v_y + q*A_y(x) = 0 at all times,
so the exact solution obeys
    v_y = -(q/m)*A_y(x) ,   v_x^2 = vx0^2 - ((q/m)*A_y(x))^2
and the orbit bounces between the two roots of A_y(x) = +-m*vx0/q.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.integrate import solve_ivp
from scipy.optimize import root

# ----------------------------------------------------------------- parameters
q = 1.0
m = 1.0
B0 = 1.0                      # A_y = B0*(A2*x^2 + A1*x), see "fields" below

x0, y0, z0 = 0.0, 0.0, 0.0
vx0, vy0, vz0 = 1.0, 0.0, 0.3

omega0 = q * B0 / m           # reference gyro-frequency scale
dt = 0.1 / omega0
N = 1000
trange = np.arange(N) * dt


# --------------------------------------------------------------------- fields
# Single source of truth: every other field quantity is derived from A_y(x).
A1 = 1.0                      # A_y = B0*(A2*x^2 + A1*x)
A2 = 0.1                      #   ->  B_z = dA_y/dx = B0*(2*A2*x + A1)


def A_y(x):
    """Non-zero component of the vector potential, A = (0, A_y(x), 0)."""
    return B0 * (A2 * x ** 2 + A1 * x)


def dA_y(x):
    """dA_y/dx = B_z(x)."""
    return B0 * (2.0 * A2 * x + A1)


def A_vec(r):
    """Vector potential A = (0, A_y(x), 0)."""
    return np.array([0.0, A_y(r[0]), 0.0])


def B_vec(r):
    """Magnetic field B = curl A = (0, 0, dA_y/dx)."""
    return np.array([0.0, 0.0, dA_y(r[0])])


def grad_A(r):
    """Matrix G with G[i, j] = dA_j/dr_i (only A_y depends on x)."""
    G = np.zeros((3, 3))
    G[0, 1] = dA_y(r[0])                   # dA_y/dx
    return G


# ------------------------------------------------------------------ integrators
def can_push(r, p, dt):
    """One implicit mid-point step in canonical coordinates (r, p).

    Solves the 6x6 non-linear system
        r_new = r + dt * (p_mid - q*A(r_mid)) / m
        p_new = p + dt * q * G(r_mid) @ v_mid
    """
    def residual(u):
        r_new, p_new = u[:3], u[3:]
        r_mid = 0.5 * (r + r_new)
        p_mid = 0.5 * (p + p_new)
        v_mid = (p_mid - q * A_vec(r_mid)) / m
        res = np.empty(6)
        res[:3] = r_new - r - dt * v_mid
        res[3:] = p_new - p - dt * q * (grad_A(r_mid) @ v_mid)
        return res

    u0 = np.concatenate([r + dt * (p - q * A_vec(r)) / m, p])
    sol = root(residual, u0, method="hybr", tol=1e-12)
    return sol.x[:3], sol.x[3:]


def noncan_push(r, v, dt):
    """One implicit mid-point step in non-canonical coordinates (r, v).

    Solves the 6x6 non-linear system
        r_new = r + dt * v_mid
        v_new = v + dt * (q/m) * v_mid x B(r_mid)

    For a uniform field this system has a closed-form solution, namely the
    rotation matrix used in symplectic_pusher.py (`sym_push`, Cayley transform
    with a = dt*omega/2).  Here B depends on r_mid, so it must be solved
    iteratively.
    """
    def residual(u):
        r_new, v_new = u[:3], u[3:]
        r_mid = 0.5 * (r + r_new)
        v_mid = 0.5 * (v + v_new)
        res = np.empty(6)
        res[:3] = r_new - r - dt * v_mid
        res[3:] = v_new - v - dt * (q / m) * np.cross(v_mid, B_vec(r_mid))
        return res

    u0 = np.concatenate([r + dt * v, v])
    sol = root(residual, u0, method="hybr", tol=1e-12)
    return sol.x[:3], sol.x[3:]


# ------------------------------------------------------- momentum <-> velocity
def v_from_p(r, p):
    """v = (p - q*A(r))/m for an array of states, shape (N, 3)."""
    v = p / m
    v[:, 1] -= q * A_y(r[:, 0]) / m
    return v


def p_from_v(r, v):
    """p = m*v + q*A(r) for an array of states, shape (N, 3)."""
    p = m * v
    p[:, 1] += q * A_y(r[:, 0])
    return p


# ------------------------------------------------------------------- main runs
def run_canonical(dt, N):
    """Integrate the canonical (r, p) formulation."""
    r = np.zeros((N, 3))
    p = np.zeros((N, 3))
    r[0] = (x0, y0, z0)
    p[0] = m * np.array([vx0, vy0, vz0]) + q * A_vec(r[0])
    for it in range(1, N):
        r[it], p[it] = can_push(r[it - 1], p[it - 1], dt)
    return r, p


def run_noncanonical(dt, N):
    """Integrate the non-canonical (r, v) formulation."""
    r = np.zeros((N, 3))
    v = np.zeros((N, 3))
    r[0] = (x0, y0, z0)
    v[0] = (vx0, vy0, vz0)
    for it in range(1, N):
        r[it], v[it] = noncan_push(r[it - 1], v[it - 1], dt)
    return r, v


r_can, p_can = run_canonical(dt, N)
v_can = v_from_p(r_can, p_can)        # v is recovered, not integrated
H_can = 0.5 * m * np.sum(v_can ** 2, axis=1)

r_non, v_non = run_noncanonical(dt, N)
p_non = p_from_v(r_non, v_non)
H_non = 0.5 * m * np.sum(v_non ** 2, axis=1)


# ------------------------------------------------------------ reference solution
def rhs(t, s):
    r, v = s[:3], s[3:]
    return np.concatenate([v, (q / m) * np.cross(v, B_vec(r))])


s0 = np.concatenate([r_can[0], v_can[0]])
ref = solve_ivp(rhs, (0.0, trange[-1]), s0, t_eval=trange,
                method="DOP853", rtol=1e-13, atol=1e-13)
r_ref = ref.y[:3].T


# ------------------------------------------------------------------ diagnostics
err_can = np.linalg.norm(r_can - r_ref, axis=1)
err_non = np.linalg.norm(r_non - r_ref, axis=1)
diff = np.linalg.norm(r_can - r_non, axis=1)

print(f"A_y = B0*({A2}*x^2 + {A1}*x),  B_z = B0*({2 * A2}*x + {A1}),  "
      f"B0 = {B0},  q = {q},  m = {m}")
print(f"dt = {dt:.4g},  N = {N},  T = {trange[-1]:.4g}")
print("max |x| (canonical / non-canonical / reference) : "
      f"{np.abs(r_can[:, 0]).max():.6f} / {np.abs(r_non[:, 0]).max():.6f} / "
      f"{np.abs(r_ref[:, 0]).max():.6f}")
print(f"max |r - r_ref| : canonical {err_can.max():.3e}   "
      f"non-canonical {err_non.max():.3e}")
print(f"max |r_can - r_non| (the two schemes differ) : {diff.max():.3e}")
print("max |dE/E|      : canonical "
      f"{np.abs(H_can / H_can[0] - 1).max():.3e}   "
      f"non-canonical {np.abs(H_non / H_non[0] - 1).max():.3e}")
print(f"max |p_y|       : canonical {np.abs(p_can[:, 1]).max():.3e}   "
      f"non-canonical {np.abs(p_non[:, 1]).max():.3e}")
print("max |v| drift   : canonical "
      f"{np.abs(np.linalg.norm(v_can, axis=1) / np.linalg.norm(v_can[0]) - 1).max():.3e}   "
      "non-canonical "
      f"{np.abs(np.linalg.norm(v_non, axis=1) / np.linalg.norm(v_non[0]) - 1).max():.3e}")


# ------------------------------------------------------------------------ plots
n_plot = min(N, int(round(20.0 / dt)))          # first few bounce periods
sl = slice(0, n_plot)

fig = plt.figure(figsize=(12, 5))

ax = fig.add_subplot(1, 2, 1, projection="3d")
ax.plot(*r_ref[sl].T, color="C0", linewidth=3, alpha=0.5, label="reference")
ax.plot(*r_can[sl].T, color="C3", linestyle="--", label="canonical (r,p)")
ax.plot(*r_non[sl].T, color="C2", linestyle=":", linewidth=2,
        label="non-canonical (r,v)")
ax.set_aspect("equal")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.xaxis.set_major_locator(MaxNLocator(3))
# ax.set_title()
ax.legend()

ax = fig.add_subplot(1, 2, 2)
ax.plot(r_ref[sl, 0], r_ref[sl, 1], color="C0", linewidth=3, alpha=0.5,
        label="reference")
ax.plot(r_can[sl, 0], r_can[sl, 1], color="C3", linestyle="--",
        label="canonical (r,p)")
ax.plot(r_non[sl, 0], r_non[sl, 1], color="C2", linestyle=":", linewidth=2,
        label="non-canonical (r,v)")
ax.set_aspect("equal")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Orbit in x-y plane")
ax.legend()

fig.tight_layout()

fig, axs = plt.subplots(2, 2, figsize=(12, 7))

zoom = trange <= 40.0                       # a few bounce periods only
axs[0, 0].plot(trange[zoom], r_ref[zoom, 0], color="C0", linewidth=3, alpha=0.5,
               label="reference")
axs[0, 0].plot(trange[zoom], r_can[zoom, 0], color="C3", linestyle="--",
               label="canonical")
axs[0, 0].plot(trange[zoom], r_non[zoom, 0], color="C2", linestyle=":",
               label="non-canonical")
axs[0, 0].set_xlabel("t")
axs[0, 0].set_ylabel("x")
axs[0, 0].set_title(r"$x(t)$")
axs[0, 0].legend()

axs[0, 1].semilogy(trange[1:], err_can[1:], color="C3", linestyle="--",
                   label="canonical vs reference")
axs[0, 1].semilogy(trange[1:], err_non[1:], color="C2", linestyle=":",
                   label="non-canonical vs reference")
axs[0, 1].semilogy(trange[1:], diff[1:], color="C4", linewidth=1,
                   label="canonical vs non-canonical")
axs[0, 1].set_xlabel("t")
axs[0, 1].set_ylabel(r"$|\Delta\mathbf{r}|$")
# axs[0, 1].set_title("O(dt^2) errors and the difference between the maps")
axs[0, 1].legend()

axs[1, 0].semilogy(trange, np.abs(H_can / H_can[0] - 1) + 1e-17, color="C3",
                   linestyle="--", label="canonical: |H/H0 - 1| = O(dt^2)")
axs[1, 0].semilogy(trange, np.abs(H_non / H_non[0] - 1) + 1e-17, color="C2",
                   linestyle=":", label="non-canonical: |Ek/Ek0 - 1| (round-off)")
axs[1, 0].set_xlabel("t")
axs[1, 0].set_ylabel("relative error")
# axs[1, 0].set_title("energy: exact for (r,v), O(dt^2) oscillation for (r,p)")
axs[1, 0].legend()

axs[1, 1].plot(trange, p_can[:, 1], color="C3", linestyle="--",
               label="canonical")
axs[1, 1].plot(trange, p_non[:, 1], color="C2", linestyle=":",
               label="non-canonical")
axs[1, 1].set_xlabel("t")
# axs[1, 1].set_ylabel(r"$p_y \times 10^{15}$")
axs[1,1].set_ylabel(r"$p_y$")
axs[1, 1].set_ylim([-1, 1])
axs[1, 1].set_title(r"$p_y = m v_y + q A_y(x)$")
axs[1, 1].legend()

fig.tight_layout()
plt.show()


# ------------------------------------------------------ convergence (refinement)
print("\nrefinement study: both maps are 2nd order and differ at O(dt^2)")
T = trange[-1]
ref_dense = solve_ivp(rhs, (0.0, T), s0, method="DOP853",
                      rtol=1e-13, atol=1e-13, dense_output=True)

print(f"{'dt':>9} {'err can':>10} {'order':>6} {'err non':>10} {'order':>6} "
      f"{'|can-non|':>10} {'order':>6}")
prev = None
for fac in (1, 2, 4, 8):
    dt_i = dt / fac
    N_i = int(round(T / dt_i)) + 1
    ti = np.arange(N_i) * dt_i
    ri_can, _ = run_canonical(dt_i, N_i)
    ri_non, _ = run_noncanonical(dt_i, N_i)
    ri_ref = ref_dense.sol(ti)[:3].T
    e_can = np.linalg.norm(ri_can - ri_ref, axis=1).max()
    e_non = np.linalg.norm(ri_non - ri_ref, axis=1).max()
    e_dif = np.linalg.norm(ri_can - ri_non, axis=1).max()
    if prev is None:
        print(f"{dt_i:9.5f} {e_can:10.3e} {'-':>6} {e_non:10.3e} {'-':>6} "
              f"{e_dif:10.3e} {'-':>6}")
    else:
        print(f"{dt_i:9.5f} {e_can:10.3e} {np.log2(prev[0] / e_can):6.2f} "
              f"{e_non:10.3e} {np.log2(prev[1] / e_non):6.2f} "
              f"{e_dif:10.3e} {np.log2(prev[2] / e_dif):6.2f}")
    prev = (e_can, e_non, e_dif)