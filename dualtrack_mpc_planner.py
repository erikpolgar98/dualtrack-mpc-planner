
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import math
import time

import numpy as np
import scipy.sparse as sp
import osqp


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------

@dataclass
class VehicleParams: #Vehilce parameters

    m: float = 1500.0          # total vehicle mass [kg]
    ms: float = 1320.0         # sprung mass [kg]
    lf: float = 1.15           # CG to front axle [m]
    lr: float = 1.586          # CG to rear axle [m]
    h: float = 0.43            # CG to roll axis distance [m]
    d: float = 0.77            # half track width [m]
    Iz: float = 2400.0         # yaw inertia [kg m^2]
    g: float = 9.81            # gravity [m/s^2]
    C1: float = 13.098         # tire empirical constant [rad^-1]
    C2: float = -0.001045      # tire empirical constant [N^-1 rad^-1]

    length: float = 4.698      # vehicle length for collision/visualization [m]
    width: float = 1.829       # vehicle width for collision/visualization [m]

    @property
    def wheelbase(self) -> float:
        return self.lf + self.lr        #calculated wheelbase


@dataclass
class MPCParams:

    dt: float = 0.10            #sampling
    N: int = 50                 #prediction horizon     

    
    #Vehicle speed
    vx_ref_kmh: float = 50.0 #speed in km/h
    
    @property
    def vx_ref(self) -> float:
        return self.vx_ref_kmh / 3.6        #convert to m/s

    #Weights
    QX: float = 35.0
    QY: float = 45.0
    Qphi: float = 12.0
    QX_terminal: float = 80.0
    QY_terminal: float = 110.0
    Qphi_terminal: float = 30.0


    Qvy: float = 3.0
    Qr: float = 8.0
    Qay_soft: float = 1.40


    R_delta: float = 3.0
    R_d_delta: float = 100.0

    # Constraints
    delta_max: float = math.radians(18.0)
    phi_min: float = -math.pi
    phi_max: float = math.pi
    Y_min: float = -300.0
    Y_max: float = 300.0
    ay_max: float = 6.0

    # Reference extraction
    reference_ds: Optional[float] = None
    reference_ds_min: float = 0.05

    # Obstacle avoidance/corridor
    obstacle_clearance: float = 0.75
    obstacle_reference_margin_min: float = 38.0
    obstacle_reference_margin_max: float = 130.0
    obstacle_reference_margin_time: float = 6.0

    obstacle_constraint_margin_min: float = 6.0
    obstacle_constraint_margin_max: float = 18.0
    obstacle_constraint_margin_time: float = 0.8
    obstacle_soft_extra_margin: float = 0.40

    obstacle_lateral_activation_margin: float = 2.7
    obstacle_reference_weight: float = 0.90
    obstacle_preferred_side: float = 1.0       # +1 local-left, -1 local-right, 0 auto
    obstacle_slack_weight: float = 2.0e3

    # Solver settings.
    osqp_verbose: bool = False
    osqp_polish: bool = False
    osqp_max_iter: int = 2500
    osqp_eps_abs: float = 1e-3
    osqp_eps_rel: float = 1e-3

    # Real-time settings. One SQP/QP solve per control step is much faster and
    # works well with warm-starting from the previous solution.
    sqp_iterations: int = 1
    sqp_alpha: float = 1.0


@dataclass
class Obstacle:
    X: float
    Y: float
    radius: float = 1.0


@dataclass
class ReferencePath:
    X: np.ndarray
    Y: np.ndarray
    phi: np.ndarray

    @staticmethod
    def from_xy(X: np.ndarray, Y: np.ndarray) -> "ReferencePath":
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if len(X) < 2:
            phi = np.zeros_like(X)
        else:
            phi = np.unwrap(np.arctan2(np.gradient(Y), np.gradient(X)))
        return ReferencePath(X=X, Y=Y, phi=phi)


@dataclass
class ObstacleCorridorConstraint:
    k: int
    side: float
    normal_x: float
    normal_y: float
    rhs: float
    obs_index: int


@dataclass
class MPCResult:
    status: str
    solve_time_ms: float
    cost: float
    x_pred: np.ndarray          # (N+1, 5): [vy, phi, r, X, Y]
    delta_seq: np.ndarray       # (N,)
    delta_apply: float
    local_ref: ReferencePath
    obstacle_constraints: List[ObstacleCorridorConstraint] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def nearest_index(X_ref: np.ndarray, Y_ref: np.ndarray, X: float, Y: float) -> int:
    d = np.hypot(X_ref - X, Y_ref - Y)
    return int(np.argmin(d))


def cumulative_arclength(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if len(X) <= 1:
        return np.zeros_like(X)
    ds = np.hypot(np.diff(X), np.diff(Y))
    return np.concatenate([[0.0], np.cumsum(ds)])


def nearest_arclength(X_ref: np.ndarray, Y_ref: np.ndarray, s_ref: np.ndarray, X: float, Y: float) -> float:
    """Nearest arclength coordinate on a polyline using segment projection."""
    X_ref = np.asarray(X_ref, dtype=float)
    Y_ref = np.asarray(Y_ref, dtype=float)
    s_ref = np.asarray(s_ref, dtype=float)

    if len(X_ref) < 2:
        return 0.0

    best_d2 = float("inf")
    best_s = float(s_ref[0])

    for i in range(len(X_ref) - 1):
        ax, ay = X_ref[i], Y_ref[i]
        bx, by = X_ref[i + 1], Y_ref[i + 1]
        vx_seg = bx - ax
        vy_seg = by - ay
        seg_len2 = vx_seg * vx_seg + vy_seg * vy_seg
        if seg_len2 <= 1e-12:
            continue
        t = ((X - ax) * vx_seg + (Y - ay) * vy_seg) / seg_len2
        t = float(np.clip(t, 0.0, 1.0))
        px = ax + t * vx_seg
        py = ay + t * vy_seg
        d2 = (X - px) ** 2 + (Y - py) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_s = float(s_ref[i] + t * (s_ref[i + 1] - s_ref[i]))

    return best_s


def align_angle_sequence(phi_ref: np.ndarray, phi_current: float) -> np.ndarray:
    """Unwrap and shift a heading sequence near the current heading."""
    phi = np.unwrap(np.asarray(phi_ref, dtype=float))
    if len(phi) == 0:
        return phi
    shift = 2.0 * np.pi * round((phi_current - phi[0]) / (2.0 * np.pi))
    return np.unwrap(phi + shift)


def path_normal(phi: float) -> Tuple[float, float]:
    """Left-hand normal of the local path heading."""
    return -math.sin(phi), math.cos(phi)


def smoothstep(z: float) -> float:
    """Smooth 0..1 transition used for low-speed model blending."""
    z = float(np.clip(z, 0.0, 1.0))
    return z * z * (3.0 - 2.0 * z)


def smooth_compact_bell(z: float) -> float:
    """C2-smooth compact bell: 1 at z=0, 0 at |z|>=1."""
    z = abs(float(z))
    if z >= 1.0:
        return 0.0
    a = 1.0 - z * z
    return a * a * a


def smooth_lateral_activation(abs_lat: float, full_lat: float, fade_width: float) -> float:
    """Smoothly disable obstacle shaping far away from the obstacle laterally."""
    if fade_width <= 1e-9:
        return 1.0 if abs_lat <= full_lat else 0.0
    z = (abs_lat - full_lat) / fade_width
    if z <= 0.0:
        return 1.0
    if z >= 1.0:
        return 0.0
    return 1.0 - smoothstep(z)


# -----------------------------------------------------------------------------
# Dual-track 2-DOF nonlinear vehicle model
# -----------------------------------------------------------------------------

class DualTrack2DOFModel:

    nx = 5
    nu = 1

    def __init__(self, vp: VehicleParams):
        self.vp = vp

    def slip_angles(self, x: np.ndarray, delta_f: float, vx: float) -> Tuple[float, float, float, float]:
        vy, _phi, r, _X, _Y = np.asarray(x, dtype=float)

        # Regularize the slip-speed denominator. At parking/creep speeds the
        # classical slip-angle formula contains r/vx terms that become
        # numerically ill-conditioned and physically less meaningful.
        # The tire model is still kept, but its low-speed singularity is removed.
        vx_slip_floor = 4.0  # [m/s] ≈ 14.4 km/h
        vx_eff = math.sqrt(vx * vx + vx_slip_floor * vx_slip_floor)

        beta = math.atan2(vy, vx_eff)
        alpha_f = beta + self.vp.lf * r / vx_eff - delta_f
        alpha_r = beta - self.vp.lr * r / vx_eff
        return alpha_f, alpha_f, alpha_r, alpha_r

    def vertical_loads(self, ay: float) -> Tuple[float, float, float, float]:
        vp = self.vp
        L = vp.wheelbase

        transfer_front = vp.lr / (2.0 * vp.d * L) * (vp.ms * ay * vp.h)
        transfer_rear = vp.lf / (2.0 * vp.d * L) * (vp.ms * ay * vp.h)

        N1 = vp.m * vp.g * vp.lr / (2.0 * L) - transfer_front
        N2 = vp.m * vp.g * vp.lr / (2.0 * L) + transfer_front
        N3 = vp.m * vp.g * vp.lf / (2.0 * L) - transfer_rear
        N4 = vp.m * vp.g * vp.lf / (2.0 * L) + transfer_rear

        return tuple(float(max(N, 1.0)) for N in (N1, N2, N3, N4))

    def tire_lateral_force(self, alpha: float, N: float) -> float:
        # With the slip-angle convention used in this code, the tire lateral
        # force must oppose the slip angle for passive stability.
        return -(self.vp.C1 * alpha * N + self.vp.C2 * alpha * N * N)

    def tire_forces(self, x: np.ndarray, delta_f: float, vx: float) -> np.ndarray:
        r = float(np.asarray(x)[2])
        ay = vx * r
        alphas = self.slip_angles(x, delta_f, vx)
        Ns = self.vertical_loads(ay)
        return np.array([self.tire_lateral_force(a, N) for a, N in zip(alphas, Ns)], dtype=float)

    def f_dynamic_continuous(self, x: np.ndarray, u: np.ndarray | float, vx: float) -> np.ndarray:
        delta_f = float(np.asarray(u).reshape(-1)[0])
        vy, phi, r, X, Y = np.asarray(x, dtype=float)

        Fys = self.tire_forces(x, delta_f, vx)

        vy_dot = np.sum(Fys) / self.vp.m - vx * r
        r_dot = (self.vp.lf * (Fys[0] + Fys[1]) - self.vp.lr * (Fys[2] + Fys[3])) / self.vp.Iz
        phi_dot = r
        X_dot = vx * math.cos(phi) - vy * math.sin(phi)
        Y_dot = vx * math.sin(phi) + vy * math.cos(phi)

        return np.array([vy_dot, phi_dot, r_dot, X_dot, Y_dot], dtype=float)

    def f_kinematic_continuous(self, x: np.ndarray, u: np.ndarray | float, vx: float) -> np.ndarray:
        delta_f = float(np.asarray(u).reshape(-1)[0])
        vy, phi, r, X, Y = np.asarray(x, dtype=float)

        L = self.vp.wheelbase
        beta = math.atan((self.vp.lr / L) * math.tan(delta_f))
        phi_dot = vx / L * math.tan(delta_f) * math.cos(beta)
        X_dot = vx * math.cos(phi + beta)
        Y_dot = vx * math.sin(phi + beta)

        # Keep the existing state layout. At very low speed, vy and r are
        # auxiliary states, so they are relaxed toward kinematic consistency.
        tau_vy = 0.40
        tau_r = 0.40
        vy_dot = -vy / tau_vy
        r_dot = (phi_dot - r) / tau_r

        return np.array([vy_dot, phi_dot, r_dot, X_dot, Y_dot], dtype=float)

    def f_continuous(self, x: np.ndarray, u: np.ndarray | float, vx: float) -> np.ndarray:
        # Below ~7 km/h the kinematic model dominates. Above ~22 km/h the
        # original dual-track tire model dominates. Between them the transition
        # is smooth, so the linearized QP remains well behaved.
        v_abs = abs(vx)
        v_low = 2.0
        v_high = 6.0
        w_dyn = smoothstep((v_abs - v_low) / (v_high - v_low))

        f_kin = self.f_kinematic_continuous(x, u, vx)
        f_dyn = self.f_dynamic_continuous(x, u, vx)
        return (1.0 - w_dyn) * f_kin + w_dyn * f_dyn

    def f_discrete(self, x: np.ndarray, u: np.ndarray | float, vx: float, dt: float) -> np.ndarray:
        return np.asarray(x, dtype=float) + dt * self.f_continuous(x, u, vx)

    def linearize_discrete(self, x: np.ndarray, u: float, vx: float, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Numerical Jacobian linearization of the nonlinear discrete model.

        Returns A, B, c:
            x_{k+1} ≈ A x_k + B u_k + c
        """
        x = np.asarray(x, dtype=float)
        u_arr = np.array([float(u)], dtype=float)
        f0 = self.f_discrete(x, u_arr, vx, dt)

        A = np.zeros((self.nx, self.nx))
        B = np.zeros((self.nx, self.nu))
        eps_x = 1e-5
        eps_u = 1e-5


        for i in range(self.nx):
            dx = np.zeros(self.nx)
            dx[i] = eps_x
            fp = self.f_discrete(x + dx, u_arr, vx, dt)
            A[:, i] = (fp - f0) / eps_x

        fp = self.f_discrete(x, u_arr + eps_u, vx, dt)
        B[:, 0] = (fp - f0) / eps_u

        c = f0 - A @ x - B @ u_arr
        return A, B, c


# -----------------------------------------------------------------------------
# Robust MPC planner
# -----------------------------------------------------------------------------
#1. lokális referenciaútvonal kivágása,
#2. akadályfigyelő referencia módosítása,
#3. MPC optimalizálási probléma felépítése,
#4. QP megoldása OSQP solverrel,
#5. első kormányparancs kiválasztása,
#6. predikált állapottrajektória visszaadása,
#7. szimulációban a jármű állapotának léptetése.


class RobustDualTrackMPCPlanner:
    def __init__(self, vp: Optional[VehicleParams] = None, mp: Optional[MPCParams] = None):
        self.vp = vp or VehicleParams()
        self.mp = mp or MPCParams()
        self.model = DualTrack2DOFModel(self.vp)
        self._warm_X: Optional[np.ndarray] = None
        self._warm_U: Optional[np.ndarray] = None

    def extract_reference_window(
        self,
        full_ref: ReferencePath,
        X: float,
        Y: float,
        vx: float = 0.0,
        phi_current: float = 0.0,
    ) -> ReferencePath:

        N = self.mp.N
        X_full = np.asarray(full_ref.X, dtype=float)
        Y_full = np.asarray(full_ref.Y, dtype=float)
        phi_full = np.unwrap(np.asarray(full_ref.phi, dtype=float))

        if len(X_full) == 0:
            zeros = np.zeros(N + 1)
            return ReferencePath(X=zeros, Y=zeros, phi=zeros)

        if len(X_full) == 1:
            Xw = np.full(N + 1, X_full[0])
            Yw = np.full(N + 1, Y_full[0])
            phiw = np.full(N + 1, phi_full[0] if len(phi_full) else 0.0)
            return ReferencePath(X=Xw, Y=Yw, phi=align_angle_sequence(phiw, phi_current))

        s_full = cumulative_arclength(X_full, Y_full)
        s0 = nearest_arclength(X_full, Y_full, s_full, X, Y)

        if self.mp.reference_ds is None:
            ds_step = max(abs(vx) * self.mp.dt, self.mp.reference_ds_min)
        else:
            ds_step = max(float(self.mp.reference_ds), self.mp.reference_ds_min)

        s_query = s0 + ds_step * np.arange(N + 1)
        s_query = np.clip(s_query, s_full[0], s_full[-1])

        Xw = np.interp(s_query, s_full, X_full)
        Yw = np.interp(s_query, s_full, Y_full)
        phiw = np.interp(s_query, s_full, phi_full)
        phiw = align_angle_sequence(phiw, phi_current)
        return ReferencePath(X=Xw, Y=Yw, phi=phiw)

    def obstacle_reference_margin(self, vx: float) -> float:
        return float(np.clip(
            self.mp.obstacle_reference_margin_time * abs(vx),
            self.mp.obstacle_reference_margin_min,
            self.mp.obstacle_reference_margin_max,
        ))

    def obstacle_constraint_margin(self, vx: float) -> float:
        return float(np.clip(
            self.mp.obstacle_constraint_margin_time * abs(vx),
            self.mp.obstacle_constraint_margin_min,
            self.mp.obstacle_constraint_margin_max,
        ))

    def speed_adaptive_state_weights(self, vx: float) -> Tuple[float, float]:
        v = abs(vx)
        v_kmh = 3.6 * v
        qvy = self.mp.Qvy * (1.0 + 0.0005 * v_kmh * v_kmh)

        qr = self.mp.Qr + self.mp.Qay_soft * v * v
        return float(qvy), float(qr)

    def speed_adaptive_steering_weights(self, vx: float) -> Tuple[float, float]:
        v_kmh = 3.6 * abs(vx)
        r_delta = self.mp.R_delta * (1.0 + 0.0008 * v_kmh * v_kmh)
        r_d_delta = self.mp.R_d_delta * (1.0 + 0.0030 * v_kmh * v_kmh)
        return float(np.clip(r_delta, self.mp.R_delta, 25.0)), float(np.clip(r_d_delta, self.mp.R_d_delta, 1400.0))

    def choose_obstacle_side(self, obs: Obstacle, ref: ReferencePath) -> float:
        pref = self.mp.obstacle_preferred_side
        if pref > 0:
            return 1.0
        if pref < 0:
            return -1.0


        if len(ref.X) == 0:
            return 1.0

        d2 = (ref.X - obs.X) ** 2 + (ref.Y - obs.Y) ** 2
        k = int(np.argmin(d2))
        nx, ny = path_normal(float(ref.phi[k]))
        lateral = nx * (obs.X - ref.X[k]) + ny * (obs.Y - ref.Y[k])
        if abs(lateral) < 1e-6:
            return 1.0
        return -math.copysign(1.0, lateral)

    def build_obstacle_aware_reference(
        self,
        ref: ReferencePath,
        obstacles: Optional[List[Obstacle]],
        vx: float,
    ) -> Tuple[ReferencePath, List[ObstacleCorridorConstraint]]:
        if not obstacles:
            return ref, []


        X0 = ref.X.copy()
        Y0 = ref.Y.copy()
        offset = np.zeros_like(X0)
        constraints: List[ObstacleCorridorConstraint] = []

        reference_margin = self.obstacle_reference_margin(vx)
        constraint_margin = self.obstacle_constraint_margin(vx)

        for obs_idx, obs in enumerate(obstacles):
            side = self.choose_obstacle_side(obs, ref)
            hard_safe_offset = obs.radius + 0.5 * self.vp.width + self.mp.obstacle_clearance
            soft_safe_offset = hard_safe_offset + self.mp.obstacle_soft_extra_margin

            for k in range(len(X0)):
                phi_k = float(ref.phi[k])
                tx = math.cos(phi_k)
                ty = math.sin(phi_k)
                nx, ny = path_normal(phi_k)

                dx_ref = ref.X[k] - obs.X
                dy_ref = ref.Y[k] - obs.Y
                local_long = tx * dx_ref + ty * dy_ref
                local_lat = nx * dx_ref + ny * dy_ref


                z_ref = abs(local_long) / max(reference_margin, 1e-6)
                bump = smooth_compact_bell(z_ref)
                if bump > 0.0:
                    lat_gain = smooth_lateral_activation(
                        abs(local_lat),
                        soft_safe_offset,
                        self.mp.obstacle_lateral_activation_margin,
                    )
                    if lat_gain > 0.0:
                        target_offset = side * soft_safe_offset - local_lat
                        offset[k] += self.mp.obstacle_reference_weight * bump * lat_gain * target_offset

                z_hard = abs(local_long) / max(constraint_margin, 1e-6)
                hard_gain = smooth_compact_bell(z_hard)
                if hard_gain > 0.0:
                    lat_gain_hard = smooth_lateral_activation(
                        abs(local_lat),
                        hard_safe_offset,
                        self.mp.obstacle_lateral_activation_margin,
                    )
                    required_offset = hard_safe_offset * hard_gain * lat_gain_hard
                    if required_offset > 1e-6:
                        rhs = required_offset + side * (nx * obs.X + ny * obs.Y)
                        constraints.append(
                            ObstacleCorridorConstraint(
                                k=k,
                                side=side,
                                normal_x=nx,
                                normal_y=ny,
                                rhs=rhs,
                                obs_index=obs_idx,
                            )
                        )

        if len(offset) >= 5:
            ds_local = np.hypot(np.diff(X0), np.diff(Y0))
            ds_mean = float(np.mean(ds_local)) if len(ds_local) else 1.0
            smooth_distance = min(8.0, max(2.0, 0.12 * reference_margin))
            radius = int(round(smooth_distance / max(ds_mean, 1e-3)))
            radius = int(np.clip(radius, 1, max(1, len(offset) // 4)))
            if radius >= 1:
                xker = np.arange(-radius, radius + 1, dtype=float)
                sigma = max(radius / 2.0, 1.0)
                kernel = np.exp(-0.5 * (xker / sigma) ** 2)
                kernel /= np.sum(kernel)
                pad = np.pad(offset, (radius, radius), mode="edge")
                offset = np.convolve(pad, kernel, mode="valid")

        X = X0.copy()
        Y = Y0.copy()
        for k in range(len(X)):
            nx, ny = path_normal(float(ref.phi[k]))
            X[k] = X0[k] + offset[k] * nx
            Y[k] = Y0[k] + offset[k] * ny

        Y = np.clip(Y, self.mp.Y_min + 0.10, self.mp.Y_max - 0.10)

        phi = np.unwrap(np.arctan2(np.gradient(Y), np.gradient(X)))
        phi = align_angle_sequence(phi, ref.phi[0] if len(ref.phi) else 0.0)
        return ReferencePath(X=X, Y=Y, phi=phi), constraints

    def initial_guess(self, x0: np.ndarray, ref: ReferencePath, delta_prev: float) -> Tuple[np.ndarray, np.ndarray]:
        N = self.mp.N
        Xbar = np.zeros((N + 1, self.model.nx))
        Ubar = np.full(N, delta_prev, dtype=float)

        # Shift the previous solution by one sample as a warm initial guess.
        # This improves solve time and also reduces command jitter.
        if self._warm_X is not None and self._warm_U is not None:
            if self._warm_X.shape == (N + 1, self.model.nx) and self._warm_U.shape == (N,):
                Xbar[:-1] = self._warm_X[1:]
                Xbar[-1] = self._warm_X[-1]
                Xbar[0] = x0
                Ubar[:-1] = self._warm_U[1:]
                Ubar[-1] = self._warm_U[-1]
                return Xbar, Ubar

        Xbar[0] = x0
        for k in range(1, N + 1):
            Xbar[k, 0] = 0.0
            Xbar[k, 1] = ref.phi[k]
            Xbar[k, 2] = 0.0
            Xbar[k, 3] = ref.X[k]
            Xbar[k, 4] = ref.Y[k]

        return Xbar, Ubar

    def steering_rate_limit(self, vx: float) -> float:
        """Speed-adaptive steering-rate limit per MPC sample.

        Creep speeds need a permissive limit for manoeuvrability. At road speeds
        the limit is reduced to prevent sharp, oscillatory avoidance motions.
        The returned value is the allowed steering change over one dt step.
        """
        v_kmh = 3.6 * abs(vx)
        if v_kmh <= 5.0:
            limit_deg = 8.0
        elif v_kmh <= 20.0:
            a = (v_kmh - 5.0) / 15.0
            limit_deg = (1.0 - a) * 8.0 + a * 2.0
        elif v_kmh <= 50.0:
            a = (v_kmh - 20.0) / 30.0
            limit_deg = (1.0 - a) * 2.0 + a * 0.8
        elif v_kmh <= 150.0:
            a = (v_kmh - 50.0) / 100.0
            limit_deg = (1.0 - a) * 0.8 + a * 0.45
        else:
            limit_deg = 0.45
        return math.radians(limit_deg)

    def build_qp(
        self,
        x0: np.ndarray,
        ref: ReferencePath,
        vx: float,
        delta_prev: float,
        Xbar: np.ndarray,
        Ubar: np.ndarray,
        obstacle_constraints: List[ObstacleCorridorConstraint],
    ):
        mp = self.mp
        nx, N = self.model.nx, mp.N

        nX = (N + 1) * nx
        nU = N
        nSoc = len(obstacle_constraints)
        nz = nX + nU + nSoc
        soc_offset = nX + nU

        # Objective
        rowsP, colsP, dataP = [], [], []
        q = np.zeros(nz)
        qvy_eff, qr_eff = self.speed_adaptive_state_weights(vx)
        r_delta_eff, r_d_delta_eff = self.speed_adaptive_steering_weights(vx)

        def add_quad(idx: int, weight: float, ref_value: float):
            if weight <= 0.0:
                return
            rowsP.append(idx)
            colsP.append(idx)
            dataP.append(2.0 * weight)
            q[idx] += -2.0 * weight * ref_value

        for k in range(N + 1):
            base = k * nx
            if k < N:
                wX, wY, wphi = mp.QX, mp.QY, mp.Qphi
            else:
                wX, wY, wphi = mp.QX_terminal, mp.QY_terminal, mp.Qphi_terminal

            # State layout: [vy, phi, r, X, Y]
            add_quad(base + 3, wX, ref.X[k])
            add_quad(base + 4, wY, ref.Y[k])
            add_quad(base + 1, wphi, ref.phi[k])
            add_quad(base + 0, qvy_eff, 0.0)
            add_quad(base + 2, qr_eff, 0.0)

        for k in range(N):
            ui = nX + k
            add_quad(ui, r_delta_eff, 0.0)

            # steering-rate cost: (u_k - u_{k-1})^2
            rowsP.append(ui)
            colsP.append(ui)
            dataP.append(2.0 * r_d_delta_eff)
            if k == 0:
                q[ui] += -2.0 * r_d_delta_eff * delta_prev
            else:
                uj = nX + k - 1
                rowsP.extend([uj, ui, uj])
                colsP.extend([uj, uj, ui])
                dataP.extend([2.0 * r_d_delta_eff, -2.0 * r_d_delta_eff, -2.0 * r_d_delta_eff])

        for i in range(nSoc):
            idx = soc_offset + i
            rowsP.append(idx)
            colsP.append(idx)
            dataP.append(2.0 * mp.obstacle_slack_weight)

        P = sp.csc_matrix((dataP, (rowsP, colsP)), shape=(nz, nz))

        # Constraints
        rows, cols, data = [], [], []
        l, u = [], []
        row = 0

        def add(r, c, v):
            rows.append(r)
            cols.append(c)
            data.append(float(v))

        # initial state equality
        for i in range(nx):
            add(row, i, 1.0)
            l.append(float(x0[i]))
            u.append(float(x0[i]))
            row += 1

        # dynamics equality
        for k in range(N):
            A, B, c = self.model.linearize_discrete(Xbar[k], Ubar[k], vx, mp.dt)
            xk = k * nx
            xkp1 = (k + 1) * nx
            uk = nX + k

            for i in range(nx):
                add(row + i, xkp1 + i, 1.0)
                for j in range(nx):
                    if abs(A[i, j]) > 1e-12:
                        add(row + i, xk + j, -A[i, j])
                add(row + i, uk, -B[i, 0])
                l.append(float(c[i]))
                u.append(float(c[i]))
            row += nx

        # state bounds: phi, Y, lateral acceleration via r
        for k in range(N + 1):
            base = k * nx

            add(row, base + 1, 1.0)
            l.append(mp.phi_min)
            u.append(mp.phi_max)
            row += 1

            add(row, base + 4, 1.0)
            l.append(mp.Y_min)
            u.append(mp.Y_max)
            row += 1

            # |vx*r| <= ay_max
            add(row, base + 2, vx)
            l.append(-mp.ay_max)
            u.append(mp.ay_max)
            row += 1

        # steering bounds
        for k in range(N):
            uk = nX + k
            add(row, uk, 1.0)
            l.append(-mp.delta_max)
            u.append(mp.delta_max)
            row += 1

        # steering rate bounds
        d_delta_max = self.steering_rate_limit(vx)
        for k in range(N):
            uk = nX + k
            add(row, uk, 1.0)
            if k == 0:
                l.append(delta_prev - d_delta_max)
                u.append(delta_prev + d_delta_max)
            else:
                uj = nX + k - 1
                add(row, uj, -1.0)
                l.append(-d_delta_max)
                u.append(d_delta_max)
            row += 1

        # obstacle slack nonnegativity
        for i in range(nSoc):
            add(row, soc_offset + i, 1.0)
            l.append(0.0)
            u.append(np.inf)
            row += 1

        # Local obstacle corridor constraints:
        # side * n_k dot ([X_k,Y_k] - obs) + slack >= safe_offset
        for i, oc in enumerate(obstacle_constraints):
            k = int(np.clip(oc.k, 0, N))
            base = k * nx
            add(row, base + 3, oc.side * oc.normal_x)
            add(row, base + 4, oc.side * oc.normal_y)
            add(row, soc_offset + i, 1.0)
            l.append(float(oc.rhs))
            u.append(np.inf)
            row += 1

        A_mat = sp.csc_matrix((data, (rows, cols)), shape=(row, nz))
        return P, q, A_mat, np.asarray(l), np.asarray(u), (nX, nU, nSoc)

    def solve_mpc(
        self,
        x0: np.ndarray,
        ref: ReferencePath,
        vx: float,
        delta_prev: float,
        obstacle_constraints: List[ObstacleCorridorConstraint],
    ) -> MPCResult:
        t0 = time.perf_counter()

        Xbar, Ubar = self.initial_guess(x0, ref, delta_prev)

        best = None
        # One warm-started SQP/QP iteration is used for real-time operation.
        for _ in range(max(1, int(self.mp.sqp_iterations))):
            P, q, A, l, u, shape = self.build_qp(x0, ref, vx, delta_prev, Xbar, Ubar, obstacle_constraints)

            solver = osqp.OSQP()
            solver.setup(
                P=P,
                q=q,
                A=A,
                l=l,
                u=u,
                verbose=self.mp.osqp_verbose,
                polish=self.mp.osqp_polish,
                max_iter=self.mp.osqp_max_iter,
                eps_abs=self.mp.osqp_eps_abs,
                eps_rel=self.mp.osqp_eps_rel,
            )
            nX, nU, nSoc = shape
            z0 = np.zeros(nX + nU + nSoc)
            z0[:nX] = Xbar.reshape(-1)
            z0[nX:nX + nU] = Ubar
            solver.warm_start(x=z0)

            res = solver.solve()
            if res.x is None or res.info.status_val not in (1, 2):
                break

            nX, nU, _ = shape
            Xnew = res.x[:nX].reshape(self.mp.N + 1, self.model.nx)
            Unew = res.x[nX:nX+nU].copy()

            # Damped update. With one SQP step alpha=1 is normally best.
            alpha = float(self.mp.sqp_alpha)
            Xbar = alpha * Xnew + (1.0 - alpha) * Xbar
            Ubar = alpha * Unew + (1.0 - alpha) * Ubar
            best = (res.info.status, float(res.info.obj_val), Xbar.copy(), Ubar.copy())

        solve_ms = (time.perf_counter() - t0) * 1e3

        if best is None:
            self._warm_X = None
            self._warm_U = None
            # Safe fallback: gradually return steering to zero.
            dmax = self.steering_rate_limit(vx)
            if abs(delta_prev) <= dmax:
                delta = 0.0
            else:
                delta = delta_prev - math.copysign(dmax, delta_prev)

            return MPCResult(
                status="failed",
                solve_time_ms=solve_ms,
                cost=np.inf,
                x_pred=np.tile(x0, (self.mp.N + 1, 1)),
                delta_seq=np.full(self.mp.N, delta),
                delta_apply=float(delta),
                local_ref=ref,
                obstacle_constraints=obstacle_constraints,
            )

        status, cost, Xpred, Useq = best
        self._warm_X = Xpred.copy()
        self._warm_U = Useq.copy()

        return MPCResult(
            status=status,
            solve_time_ms=solve_ms,
            cost=cost,
            x_pred=Xpred,
            delta_seq=Useq,
            delta_apply=float(Useq[0]),
            local_ref=ref,
            obstacle_constraints=obstacle_constraints,
        )

    def plan(
        self,
        state: np.ndarray,
        full_ref: ReferencePath,
        vx: float,
        delta_prev: float,
        obstacles: Optional[List[Obstacle]] = None,
    ) -> MPCResult:
        state = np.asarray(state, dtype=float)
        ref_window = self.extract_reference_window(full_ref, state[3], state[4], vx=vx, phi_current=state[1])
        local_ref, obstacle_constraints = self.build_obstacle_aware_reference(ref_window, obstacles, vx)
        local_ref.phi = align_angle_sequence(local_ref.phi, state[1])
        return self.solve_mpc(state, local_ref, vx, delta_prev, obstacle_constraints)

    def step_vehicle(self, state: np.ndarray, delta_f: float, vx: float) -> np.ndarray:
        return self.model.f_discrete(state, np.array([delta_f]), vx, self.mp.dt)


# -----------------------------------------------------------------------------
# Demo
# -----------------------------------------------------------------------------

def make_straight_reference(vx: float, total_time: float, dt: float) -> ReferencePath:
    t = np.arange(0.0, total_time + dt, dt)
    X = vx * t
    Y = np.zeros_like(X)
    return ReferencePath.from_xy(X, Y)


def demo_one_step() -> None:
    vp = VehicleParams()
    mp = MPCParams()
    planner = RobustDualTrackMPCPlanner(vp, mp)
    vx = mp.vx_ref
    ref = make_straight_reference(vx, 8.0, mp.dt)
    obstacles = [Obstacle(14.0, 0.0, 0.8)]
    state = np.zeros(5)
    res = planner.plan(state, ref, vx, 0.0, obstacles)
    print("Status:", res.status)
    print("Solve time [ms]:", round(res.solve_time_ms, 2))
    print("First predicted states [vy, phi, r, X, Y]:")
    print(res.x_pred[:5])
    print("Obstacle constraints:", len(res.obstacle_constraints))


if __name__ == "__main__":
    demo_one_step()
