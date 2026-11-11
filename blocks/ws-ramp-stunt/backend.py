# ── EsaSim block ──────────────────────────────────────────────────────
# Ramp stunt (ws-ramp-stunt) · version 1 · machine block, Python: def run(ctx)
# A hypercar drives up a 21 m crescent ramp, hangs upside down on its suction fans, drives off the lip, and ten cold-gas nozzles land it: learn the landing by PPO on the GPU, or run the stunt live.
# Workspace: Gianluca De Novi's workspace · Author: Gianluca De Novi
# Saved: 2026-11-11
#
# Inputs   ctx.inputs["id"] is a list of files: .path .name .size .read_text() .read_json()
#   policy      Files (pt)                Policies · optional · The stunt-policy dataset (stunt_policy_*.pt): learning goes on from the furthest one; the stunt flies the newest that lands
# Outputs  ctx.output("id", "name.ext") is the path to write; ctx.write_json / ctx.write_text
#   policy      Files (pt)                Policy · Learn the landing: stunt_policy_<date>_<NN>pct.pt (network, input scaling, lesson reached, share of attempts off the lip that land)
#   curve       Table (csv)               Learning curve · Learn the landing: training.csv, minute by minute
#   runs        Table (csv)               Runs · Run the stunt: runs.csv, each run's outcome, touchdown speed, lean and air used
#   telemetry   Live telemetry            ctx.telemetry("channel", data) while it runs
# Settings ctx.params["key"]
#   mode        select    What to do = "Learn the landing" of Learn the landing | Run the stunt
#   accelerator select    Accelerator = "auto" of auto | NVIDIA GPU | CPU
#   cars        number    Cars at once = 4096
#   minutes     number    Minutes at most = 60
#   target      number    Stop at landing rate = 0.9
#   replay_every number    Replay every (s) = 12
#   from_scratch bool      Start from scratch = false
#   seed        number    Seed = 1
#   hold        number    Hold (s) = 3
#   exit        number    Exit speed (m/s) = 7
#   wind        number    Wind (m/s) = 2
#   speed       number    Run-in speed (m/s) = 8
#   hold_back   number    Hold before the lip (m) = 7
#   runup       number    Run-up (m) = 60
#   policy_file text      Policy file = ""
#   runs_per_run number    Stunts per run = 1
#   auto_start  bool      Start by itself = false
#   channel     text      Training channel = "train"
#   car_channel text      Car channel = "car"
#   events_channel text      Events channel = "event"
#   control_channel text      Controls channel = "control"
# Runs on  Any machine, NVIDIA GPU. Return a dict to fill the "result" output.
# Packages torch[cuda] (installed by the worker with pip when missing)
# EsaSim rewrites these lines from the block's definition each time it is saved: change the definition, not them.
# ── end of EsaSim block header ────────────────────────────────────────

"""Ramp stunt. A two-seat electric hypercar drives up a crescent ramp about 21 m high that bends over until it is
nearly a ceiling, stays on it held only by suction fans, stops there upside down, drives off the lip, and ten
compressed-air nozzles roll it upright, slow its fall and set it down on the tarmac. Everything is computed by the
physics.

What to do: Learn the landing: the controller of the nozzles learns it by reinforcement learning (PPO), thousands of
cars at once on the GPU, lesson by lesson (a short drop nearly level; on its side or nose down; upside down; off the
lip after the scripted drive up the ramp); a lesson opens when half the attempts of the one before land. Landed: on its
four wheels on the tarmac, under 3 m/s down and 15 degrees of lean at touchdown, nothing but the tyres touching,
settled level a second later. One attempt is replayed in the 3D view every few seconds; the policy
(stunt_policy_<date>_<NN>pct.pt) is written when learning ends, with the best one seen off the lip.
Run the stunt: the whole stunt played in real time: the car waits on the start line; press Start and it drives up the
ramp with its fans on, holds upside down, drives off the lip, and the newest policy that lands flies it down."""
import math

try:
    import torch
except ImportError:  # the block says how to install it
    torch = None



def pick_device(want):
    """(device, label, kind): an NVIDIA GPU or the CPU. auto takes the GPU when this PyTorch can use one."""
    if torch is None:
        raise RuntimeError("This block needs PyTorch on the worker. With an NVIDIA GPU: pip install torch --index-url "
                           "https://download.pytorch.org/whl/cu124 (CPU only: pip install torch).")
    w = str(want or "auto").strip().lower()
    if w.startswith("nvidia") or w == "auto":
        if torch.cuda.is_available():
            n = torch.cuda.get_device_name(0)
            return torch.device("cuda"), (n if n.upper().startswith("NVIDIA") else "NVIDIA " + n), "nvidia"
        if w.startswith("nvidia"):
            raise RuntimeError("NVIDIA GPU asked, but this PyTorch (%s) cannot use one: install the CUDA build of PyTorch "
                               "(pip install torch --index-url https://download.pytorch.org/whl/cu124)." % torch.__version__)
    if w.startswith("tenstorrent"):
        raise RuntimeError("Tenstorrent: this block does not run on the Tenstorrent card yet. Choose auto, NVIDIA GPU or CPU.")
    import shutil
    note = "" if not shutil.which("nvidia-smi") else (" - this machine has an NVIDIA GPU but this PyTorch (%s) cannot use it: install the CUDA build "
                                                      "(pip install torch --index-url https://download.pytorch.org/whl/cu124)" % torch.__version__)
    return torch.device("cpu"), "CPU (%d threads)%s" % (torch.get_num_threads(), note), "cpu"


# ── ramp stunt physics ─────────────────────────────────────────────────────────────────────────────────────────
# The stunt as shown at a rocket test site in central Texas, at night: a two-seat electric hypercar drives up a
# crescent-shaped ramp about 21 m high that bends over until it is nearly a ceiling, stays on it held by suction fans
# (up to 2 g), stops there almost upside down, drives off the lip, and its compressed-air thrusters (ten nozzles fed by
# about 250 kg of air in two carbon-overwrapped tanks) roll it upright, slow its fall and set it down on the tarmac.
# Everything is computed by the physics, many cars at once (PyTorch, on the GPU or the CPU):
#   * the car: a rigid body (6 degrees of freedom), 1,950 kg empty plus its air, 4.31 x 1.90 x 1.15 m, inertia from its shape;
#   * four wheels: suspension (spring, damper, bump stop) and tyres (grip that saturates, rolling resistance), all-wheel
#     drive and brakes, steering on the front axle;
#   * the body: ten points around it (bumpers, sills, roof) that touch the ground or the ramp: a crash;
#   * two suction fans under the floor: they pull the car toward the surface below the floor only while the skirts seal
#     (the gap is small and the surface faces the floor), up to 2 g together, spinning up and down in a quarter second;
#   * ten cold-gas nozzles: four at the corners blowing down (lift), four blowing up (to roll and pitch with couples),
#     two at the nose blowing sideways (yaw); valves open in 25 ms; thrust held by a regulator until the tanks run low;
#     the air they blow leaves the car (it gets lighter);
#   * the ramp: arcs and a straight in the x-z plane, from the road up to its lip at 168 deg, 5 m wide (RAMP below);
#     the ground: flat tarmac; air drag and wind.
# World frame: metres, x along the run-up (the car drives toward +x into the ramp, and leaves the lip going -x), y to the
# left, z up. Body frame: x forward, y left, z up (roof). Quaternions are (x, y, z, w).

G0 = 9.81
DRY = 1950.0                      # kg, with its driver dummy
AIR0 = 250.0                      # kg of air in the two tanks (about 600 litres)
INERTIA = (650.0, 3000.0, 3300.0) # roll, pitch, yaw (kg m2), with full tanks low in the floor
LENGTH, WIDTH, HEIGHT = 4.31, 1.90, 1.15
WHEEL_R = 0.33
RIDE = 0.50                       # the centre of mass above the tarmac, standing
# wheels: suspension hard points (body), rest length to the tyre's contact patch
# the stunt car rides 5 cm higher than a road car would (its suspension raised for the ramp): the floor 17-20 cm up
WHEELS = [(1.50, 0.837, -0.145), (1.50, -0.837, -0.145), (-1.316, 0.837, -0.145), (-1.316, -0.837, -0.145)]   # 2.82 m wheelbase, 1.67 m track
REST = 0.40
K_SUS, C_SUS, TRAVEL = 120000.0, 9000.0, 0.12        # stiff, as a stunt car's suspension under 2 g of suction
K_BUMP, C_BUMP = 1.2e6, 3.0e4
MU, ROLL_RES, SLIP_V = 1.10, 0.012, 0.35
DRIVE_MAX = 26000.0               # N at the contact patches (all four wheels), what the tyres can use
# body points that must never touch anything (bumpers, sills, roof)
BODY = [(2.22, 0.62, -0.29), (2.22, -0.62, -0.29), (-2.05, 0.80, -0.25), (-2.05, -0.80, -0.25),
        (0.1, 0.95, -0.32), (0.1, -0.95, -0.32), (0.45, 0.55, 0.70), (0.45, -0.55, 0.70), (-0.25, 0.55, 0.70), (-0.25, -0.55, 0.70)]
K_BODY, C_BODY, MU_BODY = 4.0e5, 2.0e4, 0.55
# suction fans under the floor, front and rear
FANS = [(0.95, 0.0, -0.30), (-0.85, 0.0, -0.30)]
FAN_G = 2.0                       # both together pull up to 2 g of the full car
FAN_TAU = 0.25
# cold-gas nozzles: position (body), direction of the force on the car (body), most thrust (N)
NOZ_POS = [(1.95, 0.80, -0.05), (1.95, -0.80, -0.05), (-1.85, 0.85, 0.0), (-1.85, -0.85, 0.0),     # lift (blowing down)
           (1.95, 0.80, 0.05), (1.95, -0.80, 0.05), (-1.85, 0.85, 0.15), (-1.85, -0.85, 0.15),     # couples (blowing up)
           (2.10, 0.60, -0.10), (2.10, -0.60, -0.10)]                                                 # yaw (blowing sideways)
NOZ_DIR = [(0, 0, 1)] * 4 + [(0, 0, -1)] * 4 + [(0, -1, 0), (0, 1, 0)]
NOZ_MAX = [9500.0] * 4 + [4000.0] * 4 + [2500.0] * 2
VE = 520.0                        # m/s: the speed the air leaves the nozzles (cold air at sea level)
REG = 0.12                        # the regulator holds full thrust until the tanks are down to 12 %
VALVE_TAU = 0.025
CD_A = 0.30 * 2.1                 # drag coefficient x frontal area
RHO = 1.20
# the ramp, as at the event: a crescent of concrete 5 m wide. It eases off the road (16 m radius), curves up into a
# wall (9 m radius), bulges out and bends over (10 m radius) until it is nearly a ceiling, then runs straight for 8 m
# at 168 deg, the car upside down, to its lip about 20.7 m up, 5 m out over the road it came from. A steel tower
# behind it carries it; its sides are open.
RAMP = [("arc", 16.0, 15.0), ("arc", 9.0, 90.0), ("arc", 10.0, 168.0), ("line", 8.0)]
RAMP_W, RAMP_X0, RAMP_THK = 5.0, 0.0, 0.6


def _t():
    import torch
    return torch


def qmul(a, b):
    torch = _t()
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], -1)


def qmat(q):
    """rotation matrices (N, 3, 3), body to world"""
    torch = _t()
    x, y, z, w = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def q_from_axis(axis, ang):
    torch = _t()
    s = torch.sin(ang / 2).unsqueeze(-1)
    return torch.cat([axis * s, torch.cos(ang / 2).unsqueeze(-1)], -1)


def q_euler(roll, pitch, yaw):
    """yaw about z, then pitch about y, then roll about x (body)"""
    torch = _t()
    cr, sr, cp, sp, cy, sy = torch.cos(roll / 2), torch.sin(roll / 2), torch.cos(pitch / 2), torch.sin(pitch / 2), torch.cos(yaw / 2), torch.sin(yaw / 2)
    return torch.stack([sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy], -1)


class Track:
    """the ground and the ramp: signed distance (positive in the open air) and the surface's normal for any points.
    The ramp is a chain of pieces in the x-z plane, each tangent to the one before: circular arcs and straight runs.
    Along it, an angle is the slope of the deck there (0 flat, 90 a wall, 180 a ceiling)"""

    def __init__(self, spec=None, width=RAMP_W, x0=RAMP_X0):
        self.w, self.x0 = width, x0
        self.segs = []
        px, pz, a = x0, 0.0, 0.0
        for piece in (spec or RAMP):
            if piece[0] == "arc":
                R, a1 = piece[1], math.radians(piece[2])
                cx, cz = px - R * math.sin(a), pz + R * math.cos(a)
                self.segs.append(("arc", cx, cz, R, a, a1))
                px, pz, a = cx + R * math.sin(a1), cz - R * math.cos(a1), a1
            else:
                L = piece[1]
                self.segs.append(("line", px, pz, L, a))
                px, pz = px + L * math.cos(a), pz + L * math.sin(a)
        self.end = a
        self._lip = (px, pz)
        self.R = [s_[3] for s_ in self.segs if s_[0] == "arc"][-1]

    def lip(self):
        return self._lip

    def profile(self, step=0.25):
        """points along the deck: (x, z, slope in degrees) every step metres, for the view"""
        out = []
        for s_ in self.segs:
            if s_[0] == "arc":
                _, cx, cz, R, a0, a1 = s_
                k = max(2, int(math.ceil(R * (a1 - a0) / step)))
                for i in range(k + (1 if s_ is self.segs[-1] else 0)):
                    a = a0 + (a1 - a0) * i / k
                    out.append([round(cx + R * math.sin(a), 4), round(cz - R * math.cos(a), 4), round(math.degrees(a), 3)])
            else:
                _, px, pz, L, a = s_
                k = max(1, int(math.ceil(L / step)))
                for i in range(k):
                    out.append([round(px + L * math.cos(a) * i / k, 4), round(pz + L * math.sin(a) * i / k, 4), round(math.degrees(a), 3)])
        return out

    def _pieces(self, p, open_end=False):
        """for each piece: signed distance, normal, slope there, and whether the point lies over it"""
        torch = _t()
        out = []
        for j, s_ in enumerate(self.segs):
            last = j == len(self.segs) - 1
            if s_[0] == "arc":
                _, cx, cz, R, a0, a1 = s_
                dx, dz = p[..., 0] - cx, p[..., 2] - cz
                r = torch.sqrt(dx * dx + dz * dz).clamp(min=1e-6)
                th = torch.atan2(dx, -dz)
                over = (th >= a0 - (0.02 if j == 0 else 0.0)) & ((th <= a1) | (open_end and last))
                out.append((R - r, torch.stack([-dx / r, torch.zeros_like(r), -dz / r], -1), th, over, R))
            else:
                _, px, pz, L, a = s_
                tx, tz, nx, nz = math.cos(a), math.sin(a), -math.sin(a), math.cos(a)
                rx, rz = p[..., 0] - px, p[..., 2] - pz
                sv = rx * tx + rz * tz
                d = rx * nx + rz * nz
                n = torch.zeros(p.shape, dtype=p.dtype, device=p.device)
                n[..., 0], n[..., 2] = nx, nz
                out.append((d, n, torch.full_like(d, a), (sv >= 0) & (sv <= L), 1e9))
        return out

    def angle(self, p):
        """where along the ramp a point is, as the slope of the deck there: 0 on the road, 90 deg up the wall, past the
        lip more than 170 deg"""
        torch = _t()
        best = torch.full(p.shape[:-1], 1e9, dtype=p.dtype, device=p.device)
        ang = torch.where(p[..., 0] < self.x0, torch.zeros_like(best), torch.full_like(best, -1.0))
        for d, _, th, over, _ in self._pieces(p, open_end=True):
            use = over & (d.abs() < best) & (d > -RAMP_THK) & (d < 8.0)
            best = torch.where(use, d.abs(), best)
            ang = torch.where(use, th, ang)
        return ang

    def along(self, p):
        """how far along the deck a point is (metres from its foot; past the lip, more than the deck's length)"""
        torch = _t()
        best = torch.full(p.shape[:-1], 1e9, dtype=p.dtype, device=p.device)
        s = torch.where(p[..., 0] < self.x0, p[..., 0] - self.x0, torch.full_like(best, -1.0))
        s0 = 0.0
        for j, (d, _, th, over, R) in enumerate(self._pieces(p, open_end=True)):
            seg = self.segs[j]
            if seg[0] == "arc":
                pos, ln = s0 + seg[3] * (th - seg[4]), seg[3] * (seg[5] - seg[4])
            else:
                rx, rz = p[..., 0] - seg[1], p[..., 2] - seg[2]
                pos, ln = s0 + rx * math.cos(seg[4]) + rz * math.sin(seg[4]), seg[3]
                if j == len(self.segs) - 1:
                    over = (pos >= s0)
            use = over & (d.abs() < best) & (d > -RAMP_THK) & (d < 8.0)
            best = torch.where(use, d.abs(), best)
            s = torch.where(use, pos, s)
            s0 += ln
        return s

    def length(self):
        return sum(s_[3] * (s_[5] - s_[4]) if s_[0] == "arc" else s_[3] for s_ in self.segs)

    def radius(self, th):
        """the deck's radius of curvature at slope th (a large number on the straight)"""
        torch = _t()
        r = torch.full_like(th, 1e9)
        for s_ in self.segs:
            if s_[0] == "arc":
                r = torch.where((th >= s_[4]) & (th <= s_[5] + 1e-6), torch.full_like(th, s_[3]), r)
        return r

    def query(self, p):
        """p (..., 3) -> d (...), n (..., 3), kind (...: 0 ground, 1 ramp)"""
        torch = _t()
        d = p[..., 2]
        n = torch.zeros(p.shape, dtype=p.dtype, device=p.device)
        n[..., 2] = 1.0
        kind = torch.zeros(p.shape[:-1], dtype=torch.long, device=p.device)
        wide = p[..., 1].abs() <= self.w / 2
        for d_r, n_r, _, over, R in self._pieces(p):
            on = over & wide & (d_r > -RAMP_THK) & (d_r < min(R * 0.5, 6.0))
            use = on & (d_r < d)
            d = torch.where(use, d_r, d)
            n = torch.where(use.unsqueeze(-1), n_r, n)
            kind = torch.where(use, torch.ones_like(kind), kind)
        return d, n, kind


class Car:
    """N cars, each a rigid body with wheels, fans and nozzles"""
    STATE = ["pos", "q", "vel", "w", "air", "thr", "fan", "spin", "steer", "t", "wind", "fan_k", "noz_k", "air0", "hard", "touched_body",
             "hit_ramp", "comp"]

    def __init__(self, n, dev, dtype=None, track=None):
        torch = _t()
        self.n, self.dev = n, torch.device(dev)
        self.dt_ = dtype or torch.float32
        f = dict(device=self.dev, dtype=self.dt_)
        self.track = track or Track()
        self.Wb = torch.tensor(WHEELS, **f)
        self.Bb = torch.tensor(BODY, **f)
        self.Fb = torch.tensor(FANS, **f)
        self.Np = torch.tensor(NOZ_POS, **f)
        self.Nd = torch.tensor(NOZ_DIR, **f)
        self.Nmax = torch.tensor(NOZ_MAX, **f)
        self.I = torch.tensor(INERTIA, **f)
        self.front = torch.tensor([1.0, 1.0, 0.0, 0.0], **f)
        z = lambda *s: torch.zeros(n, *s, **f)   # noqa: E731
        self.pos, self.vel, self.w = z(3), z(3), z(3)
        self.q = z(4)
        self.q[:, 3] = 1.0
        self.air, self.air0 = z().fill_(AIR0), z().fill_(AIR0)
        self.thr, self.thr_cmd = z(10), z(10)
        self.fan, self.fan_cmd = z(2), z(2)
        self.spin, self.comp = z(4), z(4)
        self.steer, self.drive, self.brake = z(), z(), z()
        self.t = z()
        self.wind = z(3)
        self.fan_k, self.noz_k = z().fill_(1.0), torch.ones(n, 10, **f)
        self.hard = z()                     # deepest bump-stop strike (m past the travel)
        self.touched_body = z()             # fastest body strike (m/s into the surface)
        self.hit_ramp = torch.zeros(n, dtype=torch.bool, device=self.dev)
        self.wheel_on = torch.zeros(n, 4, dtype=torch.bool, device=self.dev)
        self.wheel_kind = torch.zeros(n, 4, dtype=torch.long, device=self.dev)
        self.g_load = z()
        self.fan_seal = z(2)
        self.flying = torch.zeros(n, dtype=torch.bool, device=self.dev)
        self._wspin = z(4)
        self.thrust = z(10)
        self.body_kind = torch.zeros(n, dtype=torch.long, device=self.dev)

    def mass(self):
        return DRY + self.air

    def R(self):
        return qmat(self.q)

    def world(self, R, pts):
        """body points (K, 3) -> world (N, K, 3)"""
        torch = _t()
        return self.pos.unsqueeze(1) + torch.einsum("nij,kj->nki", R, pts)

    def point_vel(self, R, pts):
        torch = _t()
        wr = torch.cross(self.w.unsqueeze(1).expand(-1, pts.shape[0], -1), pts.unsqueeze(0).expand(self.n, -1, -1), dim=-1)
        return self.vel.unsqueeze(1) + torch.einsum("nij,nkj->nki", R, wr)

    def step(self, dt, live=None):
        """one step of dt seconds for the cars in live (all when None)"""
        torch = _t()
        n = self.n
        R = self.R()
        m = self.mass()
        xb, yb, zb = R[:, :, 0], R[:, :, 1], R[:, :, 2]
        F = torch.zeros(n, 3, device=self.dev, dtype=self.dt_)
        Tw = torch.zeros(n, 3, device=self.dev, dtype=self.dt_)      # torque, world frame
        F[:, 2] -= m * G0

        def add(force, at_world):
            """forces (N, K, 3) applied at world points (N, K, 3)"""
            nonlocal F, Tw
            F = F + force.sum(1)
            Tw = Tw + torch.cross(at_world - self.pos.unsqueeze(1), force, dim=-1).sum(1)

        # ---- wheels: suspension and tyres
        tyre_b = self.Wb.clone()
        tyre_b[:, 2] -= REST
        pw = self.world(R, tyre_b)
        vw = self.point_vel(R, tyre_b)
        d, nrm, kind = self.track.query(pw)
        cosf = (nrm * zb.unsqueeze(1)).sum(-1)
        facing = cosf > 0.3
        comp = (-d / cosf.clamp(min=0.5)).clamp(min=0.0) * facing
        vn = (vw * nrm).sum(-1)
        rate = -vn / cosf.clamp(min=0.5)
        on = comp > 0
        Fs = K_SUS * comp.clamp(max=TRAVEL) + C_SUS * rate
        over = (comp - TRAVEL).clamp(min=0.0)
        Fs = Fs + K_BUMP * over + C_BUMP * rate * (over > 0)
        Fs = Fs.clamp(min=0.0) * on
        self.hard = torch.maximum(self.hard, over.max(-1).values)
        # tyre side hitting (wheel well, the rim): the tyre is not facing the surface
        side = (d < 0) & ~facing
        self.comp = comp.clamp(max=TRAVEL + 0.05)
        self.wheel_on, self.wheel_kind = on, kind
        # tyre grip
        ang = self.steer.unsqueeze(-1) * self.front                                        # steering angle per wheel
        hw = xb.unsqueeze(1) * torch.cos(ang).unsqueeze(-1) + yb.unsqueeze(1) * torch.sin(ang).unsqueeze(-1)
        hw = hw - (hw * nrm).sum(-1, keepdim=True) * nrm
        hw = hw / hw.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        lw = torch.cross(nrm, hw, dim=-1)
        vt = vw - vn.unsqueeze(-1) * nrm
        vl, vs = (vt * hw).sum(-1), (vt * lw).sum(-1)
        cap = MU * Fs
        drive = (self.drive / 4.0).unsqueeze(-1).expand(-1, 4)
        brake = (self.brake / 4.0).unsqueeze(-1).expand(-1, 4)
        fl = drive - brake * torch.tanh(vl / 0.3) - ROLL_RES * Fs * torch.tanh(vl / 0.5)
        fl = torch.maximum(torch.minimum(fl, cap * 0.95), -cap * 0.95)
        fs = -cap * torch.tanh(vs / SLIP_V)
        tot = torch.sqrt(fl * fl + fs * fs).clamp(min=1e-6)
        k = (cap / tot).clamp(max=1.0)
        fl, fs = fl * k, fs * k
        f_wheel = Fs.unsqueeze(-1) * nrm + fl.unsqueeze(-1) * hw + fs.unsqueeze(-1) * lw
        add(f_wheel, pw)
        # wheel spin (for the view): rolling on whatever it touches, else slowing down
        spin_rate = torch.where(on, vl / WHEEL_R, torch.zeros_like(vl))
        self._wspin = torch.where(on, spin_rate, self._wspin * (1 - 0.3 * dt) + torch.where(self.drive.unsqueeze(-1).abs() > 1, self.drive.unsqueeze(-1).sign() * 2.0 * dt, torch.zeros_like(vl)))
        self.spin = (self.spin + self._wspin * dt) % (2 * math.pi)

        # ---- body points: anything but the tyres touching is a crash
        pb = self.world(R, self.Bb)
        vb = self.point_vel(R, self.Bb)
        db, nb, kb = self.track.query(pb)
        pen = (-db).clamp(min=0.0)
        vnb = (vb * nb).sum(-1)
        fnb = (K_BODY * pen - C_BODY * vnb).clamp(min=0.0) * (pen > 0)
        vtb = vb - vnb.unsqueeze(-1) * nb
        ftb = -MU_BODY * fnb.unsqueeze(-1) * vtb / vtb.norm(dim=-1, keepdim=True).clamp(min=0.2)
        add(fnb.unsqueeze(-1) * nb + ftb, pb)
        strike = torch.where(pen > 0, (-vnb).clamp(min=0.0) + 0.3, torch.zeros_like(pen)).max(-1).values
        strike = torch.maximum(strike, torch.where(side, (-vn).clamp(min=0.0) + 0.3, torch.zeros_like(vn)).max(-1).values)
        self.touched_body = torch.maximum(self.touched_body, strike)
        self.body_kind = torch.where(pen > 0, kb, torch.zeros_like(kb)).max(-1).values

        # ---- suction fans: pull toward the surface under the floor while the skirts seal
        self.fan = self.fan + (self.fan_cmd - self.fan) * min(1.0, dt / FAN_TAU)
        pf = self.world(R, self.Fb)
        df, nf, _ = self.track.query(pf)
        cf = (nf * zb.unsqueeze(1)).sum(-1)
        seal = ((0.50 - df) / 0.20).clamp(0.0, 1.0) * ((cf - 0.55) / 0.3).clamp(0.0, 1.0)
        self.fan_seal = seal
        pull = (self.fan * seal) * (FAN_G * m * G0 / 2.0 * self.fan_k).unsqueeze(-1)
        add(-pull.unsqueeze(-1) * nf, pf)

        # ---- cold-gas nozzles
        self.thr = self.thr + (self.thr_cmd - self.thr) * min(1.0, dt / VALVE_TAU)
        avail = (self.air / (REG * self.air0)).clamp(0.0, 1.0)
        thrust = self.thr.clamp(0.0, 1.0) * self.Nmax * self.noz_k * avail.unsqueeze(-1)          # (N, 10)
        fb = thrust.unsqueeze(-1) * self.Nd                                                        # body frame
        tb = torch.cross(self.Np.unsqueeze(0).expand(n, -1, -1), fb, dim=-1).sum(1)
        F = F + torch.einsum("nij,nj->ni", R, fb.sum(1))
        Tw = Tw + torch.einsum("nij,nj->ni", R, tb)
        used = thrust.sum(-1) / VE * dt
        self.thrust = thrust

        # ---- air
        va = self.vel - self.wind
        F = F - 0.5 * RHO * CD_A * va.norm(dim=-1, keepdim=True) * va

        # ---- integrate (semi-implicit Euler)
        acc = F / m.unsqueeze(-1)
        tb_all = torch.einsum("nji,nj->ni", R, Tw)                                                # to the body frame
        Iw = self.I * self.w
        wdot = (tb_all - torch.cross(self.w, Iw, dim=-1)) / self.I
        if live is None:
            live = torch.ones(n, dtype=torch.bool, device=self.dev)
        L = live.unsqueeze(-1).to(self.dt_)
        self.g_load = torch.where(live, (acc + torch.tensor([0.0, 0.0, G0], device=self.dev, dtype=self.dt_)).norm(dim=-1) / G0, self.g_load)
        self.vel = self.vel + acc * dt * L
        self.w = self.w + wdot * dt * L
        self.pos = self.pos + self.vel * dt * L
        dq = torch.cat([self.w * 0.5 * dt, torch.zeros(n, 1, device=self.dev, dtype=self.dt_)], -1)
        q = self.q + qmul(self.q, dq) * L
        self.q = q / q.norm(dim=-1, keepdim=True)
        self.air = (self.air - used * live).clamp(min=0.0)
        self.t = self.t + dt * live
        self.hit_ramp = self.hit_ramp | ((self.body_kind == 1) & live & (self.touched_body > 0)) | (((kind == 1) & on).any(-1) & live & self.flying)

    # ---- what the controllers and the view read
    def axes(self):
        R = self.R()
        return R[:, :, 0], R[:, :, 1], R[:, :, 2]

    def tilt(self):
        torch = _t()
        return torch.acos(self.R()[:, 2, 2].clamp(-1.0, 1.0))


def mix(a, air_frac=None):
    """the flight computer: the controller's four numbers (lift 0..1 from -1..1, roll, pitch, yaw torques -1..1) shared
    out over the ten nozzles as valve openings 0..1: each corner's net push (down-blowing minus up-blowing) is the lift
    plus the roll and pitch couples; the nose nozzles turn it"""
    torch = _t()
    a = a.clamp(-1.0, 1.0)
    lift = (a[:, 0] + 1.0) * 0.5
    D, U = NOZ_MAX[0], NOZ_MAX[4]
    K = (D + U) * 0.5
    sy = torch.tensor([1.0, -1.0, 1.0, -1.0], device=a.device, dtype=a.dtype)
    sx = torch.tensor([1.0, 1.0, -1.0, -1.0], device=a.device, dtype=a.dtype)
    f = lift.unsqueeze(-1) * D + K * (a[:, 1:2] * sy - a[:, 2:3] * sx)
    down = (f.clamp(min=0.0) / D).clamp(max=1.0)
    up = ((-f).clamp(min=0.0) / U).clamp(max=1.0)
    yaw = torch.stack([a[:, 3].clamp(min=0.0), (-a[:, 3]).clamp(min=0.0)], -1)
    return torch.cat([down, up, yaw], -1)


class Driver:
    """the scripted driver on the ramp (not the learner): run-up at v_in, climb up the ramp easing off to stop th_hold
    metres before the lip (nearly upside down, held by the fans), hold there for hold_s, then drive off the lip at v_exit.
    Phases: 0 run-up, 1 climbing, 2 holding, 3 to the lip, 4 off the ramp (the thrusters' controller flies)."""

    def __init__(self, car, v_in, th_hold, hold_s, v_exit):
        torch = _t()
        self.c = car
        self.v_in, self.th_hold, self.hold_s, self.v_exit = v_in, th_hold, hold_s, v_exit
        self.phase = torch.zeros(car.n, dtype=torch.long, device=car.dev)
        self.hold_t = torch.zeros(car.n, device=car.dev, dtype=car.dt_)
        self.off_t = torch.full((car.n,), -1.0, device=car.dev, dtype=car.dt_)

    def control(self, dt):
        torch = _t()
        c, tr = self.c, self.c.track
        R = c.R()
        xb, yb = R[:, :, 0], R[:, :, 1]
        v_long = (c.vel * xb).sum(-1)
        th = tr.angle(c.pos)
        s = tr.along(c.pos)
        L = tr.length()
        on_ramp = (c.pos[:, 0] > tr.x0 - 0.3) & (c.pos[:, 2] > 0.6)
        ph = self.phase
        ph = torch.where((ph == 0) & on_ramp & (th > 0.05), torch.ones_like(ph), ph)
        brake_dist = (L - self.th_hold) - s
        v_climb = torch.minimum(self.v_in, torch.sqrt((2.0 * 2.5 * brake_dist).clamp(min=0.0)))
        stop = (ph == 1) & ((brake_dist <= 0.02) | ((v_long < 0.25) & (brake_dist < 1.2)))
        ph = torch.where(stop, torch.full_like(ph, 2), ph)
        self.hold_t = torch.where(ph == 2, self.hold_t + dt, self.hold_t)
        ph = torch.where((ph == 2) & (self.hold_t >= self.hold_s), torch.full_like(ph, 3), ph)
        # the hand-over: the front axle is past the lip (the nose is in the air); the script keeps driving and the fans
        # keep the rear on the lip until the rear axle is past it too
        front = c.pos + R[:, :, 0] * 1.33 - R[:, :, 2] * 0.4
        rear = c.pos - R[:, :, 0] * 1.33 - R[:, :, 2] * 0.4
        new_off = (ph == 3) & ((tr.along(front) > L) | ((s > L - 3.0) & ~c.wheel_on.any(-1)))
        self.off_t = torch.where(new_off, c.t, self.off_t)
        ph = torch.where(new_off, torch.full_like(ph, 4), ph)
        gone = (ph == 4) & ((tr.along(rear) > L + 0.01) | ~c.wheel_on.any(-1) | (c.t - self.off_t > 1.5))
        self.rear_off = gone | getattr(self, "rear_off", torch.zeros_like(gone))
        c.flying = self.rear_off & (c.t - self.off_t > 0.25)
        self.phase = ph
        v_t = torch.where(ph == 0, self.v_in, torch.where(ph == 1, v_climb, torch.where(ph == 2, torch.zeros_like(v_climb), self.v_exit)))
        m = c.mass()
        gain = torch.where(ph >= 3, torch.full_like(v_long, 8.0), torch.full_like(v_long, 2.5))
        f = m * (gain * (v_t - v_long) + G0 * xb[:, 2])
        f = torch.where(ph >= 3, f.clamp(min=0.0), f)
        free = self.rear_off.unsqueeze(-1)
        c.drive = torch.where(self.rear_off, torch.zeros_like(f), f.clamp(-DRIVE_MAX, DRIVE_MAX))
        c.brake = torch.zeros_like(f)
        # the fans pull only as hard as needed to keep the tyres pressed with about 1.25 g (gravity and the turn do the
        # rest): fully on upside down, lightly on the flat, so the springs are not squashed into the bump stops
        _, nrm, kind_ = tr.query(c.pos)
        r_loc = tr.radius(th)
        turn = torch.where(kind_ == 1, v_long * v_long / r_loc, torch.zeros_like(v_long))
        need = ((1.25 * G0 - turn - G0 * nrm[:, 2]) / (FAN_G * G0)).clamp(0.0, 1.0)
        need = torch.where((ph >= 2) & (ph <= 4), torch.ones_like(need), need)      # held upside down and off the lip: all they have
        c.fan_cmd = torch.where(free, torch.zeros_like(c.fan_cmd), need.unsqueeze(-1).expand(-1, 2))
        # stay in the middle of the ramp
        vy_des = -0.8 * c.pos[:, 1]
        want = (vy_des / v_long.abs().clamp(min=2.0)).clamp(-0.1, 0.1)
        st = 3.0 * (want - xb[:, 1]) * torch.sign(yb[:, 1] + 1e-6) * 2.66 / v_long.abs().clamp(min=2.0)
        c.steer = torch.where(self.rear_off, torch.zeros_like(st), st.clamp(-0.15, 0.15))
        return ph


def place_on_ground(car, x, y=0.0, yaw=0.0, v=0.0):
    """cars standing on the tarmac at x (heading +x unless yaw), at their static ride height"""
    torch = _t()
    n = car.n
    f = dict(device=car.dev, dtype=car.dt_)
    car.pos = torch.stack([torch.as_tensor(x, **f).expand(n).clone(), torch.as_tensor(y, **f).expand(n).clone(), torch.full((n,), RIDE, **f)], -1)
    car.q = q_euler(torch.zeros(n, **f), torch.zeros(n, **f), torch.as_tensor(yaw, **f).expand(n).clone())
    car.vel = torch.zeros(n, 3, **f)
    car.vel[:, 0] = torch.as_tensor(v, **f) * torch.cos(torch.as_tensor(yaw, **f))
    car.vel[:, 1] = torch.as_tensor(v, **f) * torch.sin(torch.as_tensor(yaw, **f))
    car.w = torch.zeros(n, 3, **f)


# ── learning to fly the car off the lip and land it: what the controller sees and does, the lessons, PPO ────────
OBS = 26
ACT = 4
CTRL_DT, SUBSTEPS = 0.02, 5         # the controller decides 50 times a second; the physics steps every 4 ms
GAMMA = 0.995
T_LIMIT = 9.0                       # seconds from the hand-over
SETTLE = 1.0                        # seconds on the wheels before a landing counts
LAND_X = -25.0                      # the middle of the landing box painted on the tarmac (x), y = 0
# A reverse curriculum: easy starts first, the real one last. A lesson opens when half the attempts of the one before land.
STAGES = [("upright", "a short drop, nearly level"), ("side", "on its side or nose down, 4-10 m up"),
          ("inverted", "upside down, 10-20 m up"), ("lip", "off the lip of the ramp, as in the stunt")]
OUTCOMES = ((1, "landed"), (2, "hard landing"), (3, "on its side or roof"), (4, "hit the ramp"), (5, "time"))


def observe(car, drv, prev_a):
    import torch
    R = car.R()
    xb, zb = R[:, :, 0], R[:, :, 2]
    tyre_b = car.Wb.clone()
    tyre_b[:, 2] -= REST
    low = car.world(R, tyre_b)[:, :, 2].min(-1).values
    lip_x, lip_z = car.track.lip()
    since = (car.t - drv.off_t).clamp(min=0.0)
    return torch.cat([
        xb, zb, car.w * 0.5, car.vel / 8.0,
        (car.pos[:, 2:3] / 10.0), (low.unsqueeze(-1) / 5.0),
        torch.stack([(car.pos[:, 0] - LAND_X) / 15.0, car.pos[:, 1] / 15.0], -1),
        torch.stack([(car.pos[:, 0] - lip_x) / 10.0, (car.pos[:, 2] - lip_z) / 10.0], -1),
        (car.air / AIR0).unsqueeze(-1),
        car.wheel_on.float().mean(-1, keepdim=True),
        (~drv.rear_off).float().unsqueeze(-1),
        (since / 4.0).unsqueeze(-1),
        prev_a,
    ], -1)


def potential(car):
    """how close to a good landing a state is (higher is better): upright, falling no faster than the height allows,
    not sliding sideways near the ground, not spinning"""
    import torch
    R = car.R()
    zb, yb = R[:, :, 2], R[:, :, 1]
    h = (car.pos[:, 2] - RIDE).clamp(min=0.0)
    tilt = torch.acos(zb[:, 2].clamp(-1.0, 1.0)) / math.pi
    low = (1.0 - h / 6.0).clamp(0.0, 1.0)
    vz_t = -(0.5 + torch.sqrt(2.0 * 4.5 * h))
    fast = (vz_t - car.vel[:, 2]).clamp(min=0.0)
    slow = (car.vel[:, 2] - vz_t).clamp(min=0.0)
    side = (car.vel * yb).sum(-1).abs()
    return -(1.5 * tilt + 3.0 * tilt * low + 0.2 * car.w.norm(dim=-1) + 1.5 * torch.log1p(fast / 2.0) + 0.4 * torch.log1p(slow / 2.0)
             + 0.5 * low * torch.log1p(side / 2.0))


def free_start(car, drv, idx, kind, gen):
    """lessons 1-3: the car in the air somewhere over the run-up, in the attitude the lesson asks for"""
    import torch
    k = idx.numel()
    dev, dt = car.dev, car.dt_
    U = lambda a, b: a + (b - a) * torch.rand(k, device=dev, dtype=dt, generator=gen)   # noqa: E731
    sgn = lambda: torch.where(torch.rand(k, device=dev, generator=gen) < 0.5, -1.0, 1.0).to(dt)   # noqa: E731
    if kind == "upright":
        h, x = U(1.5, 4.0), U(-42.0, -14.0)
        roll, pitch = U(-0.4, 0.4), U(-0.4, 0.4)
        vx, vz, wmax = U(-5.0, 0.0), U(-3.0, 0.5), 0.3
    elif kind == "side":
        h, x = U(4.0, 10.0), U(-40.0, -12.0)
        side = torch.rand(k, device=dev, generator=gen) < 0.6
        roll = torch.where(side, sgn() * U(0.5, 2.4), U(-0.3, 0.3))
        pitch = torch.where(side, U(-0.3, 0.3), sgn() * U(0.5, 2.0))
        vx, vz, wmax = U(-6.0, 0.0), U(-4.0, 1.0), 0.6
    else:
        h, x = U(10.0, 19.0), U(-22.0, -7.0)
        roll = math.pi + U(-0.6, 0.6)
        pitch = U(-0.5, 0.5)
        vx, vz, wmax = U(-8.0, -1.0), U(-3.0, 1.5), 1.2
    yaw = math.pi + U(-0.15, 0.15)             # heading -x, as when it leaves the lip
    car.pos[idx] = torch.stack([x, U(-1.5, 1.5), h + RIDE], -1)
    car.q[idx] = q_euler(roll, pitch, yaw)
    car.vel[idx] = torch.stack([vx, U(-0.8, 0.8), vz], -1)
    car.w[idx] = torch.stack([U(-wmax, wmax), U(-wmax, wmax), U(-0.2, 0.2)], -1)
    car.air[idx] = U(225.0, 250.0)
    car.air0[idx] = AIR0
    car.fan[idx] = 0.0
    car.fan_cmd[idx] = 0.0
    car.thr[idx] = 0.0
    car.drive[idx] = 0.0
    car.steer[idx] = 0.0
    car.fan_k[idx] = U(0.9, 1.1)
    car.noz_k[idx] = 0.92 + 0.13 * torch.rand(k, 10, device=dev, dtype=dt, generator=gen)
    car.wind[idx] = torch.stack([U(-5.0, 5.0), U(-5.0, 5.0), torch.zeros(k, device=dev, dtype=dt)], -1)
    drv.phase[idx] = 4
    drv.off_t[idx] = car.t[idx]
    drv.rear_off[idx] = True
    car.flying[idx] = True


DRV_STATE = ["phase", "hold_t", "off_t", "rear_off", "v_in", "th_hold", "hold_s", "v_exit"]
CAR_EXTRA = ["drive", "brake", "steer", "fan_cmd", "thr_cmd", "flying", "_wspin"]


def lip_pool(size, dev, dtype, gen, log=None):
    """the real starts: cars driven by the script up the ramp, held nearly upside down, then off the lip; each start is
    the moment its front axle leaves the lip, with everything the script was doing then (it keeps driving the rear wheels
    and the fans hold the rear on the lip a moment longer)"""
    import torch
    car = Car(size, dev, dtype)
    U = lambda a, b: a + (b - a) * torch.rand(size, device=car.dev, dtype=car.dt_, generator=gen)   # noqa: E731
    place_on_ground(car, -22.0, y=0.0, yaw=0.0, v=0.0)
    car.pos[:, 1] = U(-0.4, 0.4)
    car.vel[:, 0] = U(6.0, 9.5)
    car.q = q_euler(torch.zeros_like(car.t), torch.zeros_like(car.t), U(-0.03, 0.03))
    car.air = U(230.0, 250.0)
    car.fan_k = U(0.9, 1.1)
    car.noz_k = 0.92 + 0.13 * torch.rand(size, 10, device=car.dev, dtype=car.dt_, generator=gen)
    car.wind = torch.stack([U(-6.0, 6.0), U(-6.0, 6.0), torch.zeros_like(car.t)], -1)
    car.flying = torch.zeros(size, dtype=torch.bool, device=car.dev)
    drv = Driver(car, car.vel[:, 0].clone(), U(5.0, 9.0), U(0.2, 0.6), U(4.0, 10.0))
    drv.rear_off = torch.zeros(size, dtype=torch.bool, device=car.dev)
    snap, got = {}, torch.zeros(size, dtype=torch.bool, device=car.dev)
    dt = 0.004
    keys = Car.STATE + CAR_EXTRA
    for _ in range(int(16.0 / dt)):
        ph = drv.control(dt)
        car.step(dt)
        new = (ph == 4) & ~got
        if bool(new.any()):
            for k_ in keys:
                v = getattr(car, k_)
                if k_ not in snap:
                    snap[k_] = v.clone()
                snap[k_][new] = v[new]
            for k_ in DRV_STATE:
                v = getattr(drv, k_)
                if "d_" + k_ not in snap:
                    snap["d_" + k_] = v.clone()
                snap["d_" + k_][new] = v[new]
            got = got | new
        if bool(got.all()):
            break
    keep = (got & (snap["touched_body"] < 0.01)).nonzero(as_tuple=False).flatten()
    snap = {k_: v[keep] for k_, v in snap.items()}
    if log:
        V = snap["vel"].norm(dim=-1)
        log("off the lip: %d starts, %.1f to %.1f m up, %.1f to %.1f m/s, upside down by %.0f to %.0f deg" % (
            keep.numel(), float(snap["pos"][:, 2].min()), float(snap["pos"][:, 2].max()), float(V.min()), float(V.max()),
            math.degrees(float(torch.acos(qmat(snap["q"])[:, 2, 2].clamp(-1, 1)).min())), math.degrees(float(torch.acos(qmat(snap["q"])[:, 2, 2].clamp(-1, 1)).max()))))
    return snap


class StuntEnv:
    def __init__(self, n, dev, dtype=None, seed=0):
        import torch
        self.n, self.dev = n, torch.device(dev)
        self.dtype = dtype or torch.float32
        self.gen = torch.Generator(device=self.dev).manual_seed(seed)
        self.car = Car(n, self.dev, self.dtype)
        self.car.flying = torch.zeros(n, dtype=torch.bool, device=self.dev)
        f = dict(device=self.dev, dtype=self.dtype)
        self.drv = Driver(self.car, torch.zeros(n, **f), torch.zeros(n, **f), torch.zeros(n, **f), torch.zeros(n, **f))
        self.drv.rear_off = torch.ones(n, dtype=torch.bool, device=self.dev)
        self.pools = {}
        self.mix = [1.0, 0.0, 0.0, 0.0]
        self.stage_of = torch.zeros(n, dtype=torch.long, device=self.dev)
        self.ret = torch.zeros(n, **f)
        self.prev = torch.zeros(n, ACT, **f)
        self.t_td = torch.full((n,), -1.0, **f)
        self.vz_td = torch.zeros(n, **f)
        self.tilt_td = torch.zeros(n, **f)
        self.x_td = torch.zeros(n, **f)
        self.air_used = torch.zeros(n, **f)

    def make_pool(self, stage, size, log=None):
        if STAGES[stage][0] == "lip" and stage not in self.pools:
            self.pools[stage] = lip_pool(size, self.dev, self.dtype, self.gen, log)

    def reset(self, idx):
        import torch
        k = idx.numel()
        if k == 0:
            return
        stages = [s for s in range(len(STAGES)) if self.mix[s] > 0]
        w = torch.tensor([self.mix[s] for s in stages], device=self.dev, dtype=self.dtype)
        pick = torch.multinomial(w, k, replacement=True, generator=self.gen)
        c, d = self.car, self.drv
        for j, s in enumerate(stages):
            sel = idx[pick == j]
            if sel.numel() == 0:
                continue
            self.stage_of[sel] = s
            if STAGES[s][0] != "lip":
                free_start(c, d, sel, STAGES[s][0], self.gen)
                continue
            P = self.pools[s]
            rows = torch.randint(0, P["pos"].shape[0], (sel.numel(),), device=self.dev, generator=self.gen)
            for key in Car.STATE + CAR_EXTRA:
                getattr(c, key)[sel] = P[key][rows]
            for key in DRV_STATE:
                getattr(d, key)[sel] = P["d_" + key][rows]
            # the time on the clock of the pool: keep it (the script's timers are relative to it)
        c.hard[idx] = 0.0
        c.touched_body[idx] = 0.0
        c.hit_ramp[idx] = False
        self.ret[idx] = 0.0
        self.prev[idx] = 0.0
        self.t_td[idx] = -1.0
        self.air_used[idx] = 0.0

    def start(self):
        import torch
        self.reset(torch.arange(self.n, device=self.dev))
        return observe(self.car, self.drv, self.prev)

    def step(self, a):
        import torch
        c, d = self.car, self.drv
        phi0 = potential(c)
        air0 = c.air.clone()
        a = a.clamp(-1.0, 1.0)
        c.thr_cmd = mix(a)
        for _ in range(SUBSTEPS):
            if not bool(d.rear_off.all()):
                ph = d.phase.clone()
                d.control(CTRL_DT / SUBSTEPS)
                d.phase = torch.maximum(ph, d.phase)
            c.step(CTRL_DT / SUBSTEPS)
            gd = (c.wheel_on & (c.wheel_kind == 0)).any(-1) | (c.touched_body > 0)
            new = gd & (self.t_td < 0) & d.rear_off
            if bool(new.any()):
                self.t_td = torch.where(new, c.t, self.t_td)
                self.vz_td = torch.where(new, c.vel[:, 2], self.vz_td)
                self.tilt_td = torch.where(new, c.tilt(), self.tilt_td)
                self.x_td = torch.where(new, c.pos[:, 0], self.x_td)
        tilt = c.tilt()
        body = c.touched_body > 0.0
        ramp = c.hit_ramp
        since_td = torch.where(self.t_td >= 0, c.t - self.t_td, torch.full_like(c.t, -1.0))
        settled = since_td >= SETTLE
        timeout = (c.t - d.off_t) > T_LIMIT
        end = ramp | body | settled | timeout | (tilt > 1.9) & (since_td >= 0)
        soft = self.vz_td > -3.0
        level = self.tilt_td < math.radians(15.0)
        landed = settled & ~body & ~ramp & soft & level & (tilt < math.radians(10.0)) & (c.hard < 0.03)
        why = torch.where(landed, 1, torch.where(ramp, 4, torch.where(body | (end & ~level & (since_td >= 0)), 3,
                          torch.where(end & (since_td >= 0), 2, torch.where(timeout, 5, 0)))))
        why = torch.where(end, why, torch.zeros_like(why))
        phi1 = potential(c)
        used = air0 - c.air
        self.air_used += used
        r = (GAMMA * phi1 * (~end).to(phi1.dtype) - phi0) - 0.08 * used - 0.05 * c.vel[:, 2].clamp(min=0.0)
        bonus = torch.zeros_like(r)
        bonus = torch.where(why == 1, 100.0 + 25.0 * (1.0 + self.vz_td / 3.0).clamp(0.0, 1.0) + 10.0 * (1.0 - self.tilt_td / math.radians(15.0)).clamp(0.0, 1.0), bonus)
        bonus = torch.where(why == 2, -30.0 - 4.0 * self.vz_td.abs().clamp(max=20.0), bonus)
        bonus = torch.where(why == 3, torch.full_like(bonus, -70.0), bonus)
        bonus = torch.where(why == 4, torch.full_like(bonus, -80.0), bonus)
        bonus = torch.where(why == 5, torch.full_like(bonus, -60.0), bonus)
        r = r + bonus
        self.ret += r
        info = {"end": end, "why": why, "landed": why == 1, "vz": self.vz_td.clone(), "tilt": torch.rad2deg(self.tilt_td),
                "x": self.x_td.clone(), "air": self.air_used.clone(), "stage": self.stage_of.clone(), "ret": self.ret.clone(),
                "touched": self.t_td >= 0}
        self.prev = a
        idx = end.nonzero(as_tuple=False).flatten()
        self.reset(idx)
        return observe(c, d, self.prev), r, end, info


def make_nets(torch, obs=None, act=None, hidden=256):
    import torch.nn as nn
    obs, act = obs or OBS, act or ACT

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.pi = nn.Sequential(nn.Linear(obs, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, act))
            self.v = nn.Sequential(nn.Linear(obs, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
            self.log_std = nn.Parameter(torch.full((act,), -0.5))
            nn.init.orthogonal_(self.pi[-1].weight, 0.01)
            nn.init.zeros_(self.pi[-1].bias)

        def forward(self, x):
            return self.pi(x), self.v(x).squeeze(-1)
    return Policy()


class RunningNorm:
    def __init__(self, torch, n, dev):
        self.mean = torch.zeros(n, device=dev)
        self.var = torch.ones(n, device=dev)
        self.count = 1e-4

    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        d = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + d * bc / tot
        self.var = (self.var * self.count + bv * bc + d * d * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x):
        return ((x - self.mean) / (self.var + 1e-8).sqrt()).clamp(-6.0, 6.0)


class PPO:
    def __init__(self, torch, dev, n_env, horizon=48, lr=3e-4, gamma=GAMMA, lam=0.95, clip=0.2, epochs=4, minibatches=4, rscale=0.1):
        self.torch, self.dev = torch, dev
        self.net = make_nets(torch).to(dev)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)
        self.norm = RunningNorm(torch, OBS, dev)
        self.n, self.T = n_env, horizon
        self.gamma, self.lam, self.clip, self.epochs, self.mb, self.rscale = gamma, lam, clip, epochs, minibatches, rscale

    def act(self, obs, deterministic=False):
        torch = self.torch
        with torch.no_grad():
            x = self.norm(obs)
            mu, v = self.net(x)
            if deterministic:
                return mu, None, v
            std = self.net.log_std.exp()
            a = mu + std * torch.randn_like(mu)
            lp = (-0.5 * ((a - mu) / std) ** 2 - self.net.log_std - 0.9189385).sum(-1)
            return a, lp, v

    def update(self, buf, last_v):
        torch = self.torch
        obs, act, lp, val, rew, done = buf
        T = rew.shape[0]
        adv = torch.zeros_like(rew)
        gae = torch.zeros_like(rew[0])
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else val[t + 1]
            nonterm = 1.0 - done[t].float()
            delta = rew[t] * self.rscale + self.gamma * nv * nonterm - val[t]
            gae = delta + self.gamma * self.lam * nonterm * gae
            adv[t] = gae
        ret = adv + val
        N = T * self.n
        obs, act, lp, adv, ret, val = obs.reshape(N, -1), act.reshape(N, -1), lp.reshape(N), adv.reshape(N), ret.reshape(N), val.reshape(N)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        x = self.norm(obs)
        stats = {"kl": 0.0, "vloss": 0.0}
        size = N // self.mb
        for _ in range(self.epochs):
            perm = torch.randperm(N, device=self.dev)
            for i in range(self.mb):
                idx = perm[i * size:(i + 1) * size]
                mu, v = self.net(x[idx])
                ls = self.net.log_std
                std = ls.exp()
                nlp = (-0.5 * ((act[idx] - mu) / std) ** 2 - ls - 0.9189385).sum(-1)
                ratio = (nlp - lp[idx]).exp()
                a = adv[idx]
                pl = -torch.min(ratio * a, ratio.clamp(1 - self.clip, 1 + self.clip) * a).mean()
                vc = val[idx] + (v - val[idx]).clamp(-1.0, 1.0)
                vl = torch.max((v - ret[idx]) ** 2, (vc - ret[idx]) ** 2).mean()
                ent = (ls + 1.4189385).sum()
                loss = pl + 0.5 * vl - 0.003 * ent
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.opt.step()
                with torch.no_grad():
                    self.net.log_std.clamp_(-2.5, 0.0)
                    stats["kl"] = float((lp[idx] - nlp).mean())
                    stats["vloss"] = float(vl)
        stats["std"] = float(self.net.log_std.detach().exp().mean())
        return stats

    def state(self):
        return {"net": self.net.state_dict(), "mean": self.norm.mean.cpu(), "var": self.norm.var.cpu(), "count": self.norm.count}


# ── what the 3D view gets: frames of the car (esasim.car/1) ──────────────────────────────────────────────
FIELDS = ["t", "x", "y", "z", "qx", "qy", "qz", "qw", "vx", "vy", "vz", "steer", "s0", "s1", "s2", "s3", "c0", "c1", "c2", "c3",
          "fan0", "fan1", "n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "air", "g", "phase", "drive"]


def track_info(track):
    return {"R": track.R, "end_deg": round(math.degrees(track.end), 2), "width": track.w, "x0": track.x0,
            "profile": track.profile(),
            "land": [LAND_X, 0.0], "box": [9.0, 4.0], "nozzles": NOZ_POS, "nozzle_dirs": NOZ_DIR, "wheels": WHEELS, "rest": REST,
            "wheel_r": WHEEL_R, "car": [LENGTH, WIDTH, HEIGHT], "ride": RIDE, "comp0": round((DRY + AIR0) * G0 / 4.0 / K_SUS, 3)}


def frame_of(car, drv, i, t0=0.0):
    """one car's state as a row of FIELDS (rounded for the wire)"""
    r2, r3, r4 = (lambda v: round(float(v), 2)), (lambda v: round(float(v), 3)), (lambda v: round(float(v), 4))
    q = car.q[i]
    th = car.thrust[i]
    return ([r3(car.t[i] - t0), r3(car.pos[i, 0]), r3(car.pos[i, 1]), r3(car.pos[i, 2]), r4(q[0]), r4(q[1]), r4(q[2]), r4(q[3]),
             r2(car.vel[i, 0]), r2(car.vel[i, 1]), r2(car.vel[i, 2]), r3(car.steer[i])]
            + [r3(x) for x in car.spin[i].tolist()] + [r3(x) for x in car.comp[i].tolist()]
            + [r3(car.fan[i, 0] * car.fan_seal[i, 0]), r3(car.fan[i, 1] * car.fan_seal[i, 1])]
            + [round(float(th[k]) / NOZ_MAX[k], 3) for k in range(10)]
            + [round(float(car.air[i]), 1), r2(car.g_load[i]), int(drv.phase[i]), round(float(car.drive[i]) / DRIVE_MAX, 3)])


def chunks(rows, size=150):
    for i in range(0, len(rows), size):
        yield i // size, rows[i:i + size], i + size >= len(rows)


def learn_run(ctx):
    import collections
    import csv
    import os
    import re
    import shutil
    import time
    import torch
    p = ctx.params
    dev, label, kind = pick_device(p.get("accelerator") or "auto")
    n = int(p.get("cars") or 4096)
    if kind == "cpu":
        n = min(n, 1024)
    minutes = float(p.get("minutes") or 60)
    target = float(p.get("target") or 0.9)
    replay_every = float(p.get("replay_every") or 12)
    ch_car, ch_train, ch_ctl, ch_ev = p.get("car_channel") or "car", p.get("channel") or "train", p.get("control_channel") or "control", p.get("events_channel") or "event"
    seed = int(p.get("seed") or 1)
    torch.manual_seed(seed)
    torch.set_grad_enabled(True)
    ctx.log("Learning the landing on %s with %d cars at once, for up to %.0f minutes (or until %.0f%% of the attempts off the lip land)." % (label, n, minutes, target * 100))
    algo = PPO(torch, dev, n)
    start_stage = 0
    found = []
    for f in (ctx.inputs.get("policy") or []):
        if str(f.name).endswith(".pt"):
            try:
                st_ = torch.load(f.path, map_location="cpu", weights_only=False)
                if st_.get("format") != "esasim.stunt-policy/1":
                    continue
                m_ = re.search(r"(\d{8}_\d{4})", str(f.name))
                found.append((int(st_.get("stage", 0)), float(st_.get("landed_full") or 0), m_.group(1) if m_ else "", os.path.getmtime(f.path), f, st_))
            except Exception as e:
                ctx.log("Skipped %s: %s" % (f.name, e))
    if found and not bool(p.get("from_scratch")):
        found.sort(key=lambda x: x[:4], reverse=True)
        f, st = found[0][4], found[0][5]
        algo.net.load_state_dict(st["net"])
        algo.norm.mean, algo.norm.var, algo.norm.count = st["mean"].to(dev), st["var"].to(dev), st["count"]
        start_stage = int(st.get("stage", 0))
        seed += int(time.time()) % 100000
        torch.manual_seed(seed)
        ctx.log("Continuing from %s: it had reached lesson %d of %d after %.0f minutes, %.0f%% of the attempts off the lip landing." % (
            f.name, start_stage + 1, len(STAGES), float(st.get("minutes") or 0), 100 * float(st.get("landed_full") or 0)))
    elif found:
        ctx.log("Starting from scratch (asked): the %d policies so far are left as they are." % len(found))
    env = StuntEnv(n, dev, seed=seed)
    pool = max(2048, min(8192, 2 * n)) if kind != "cpu" else 1024
    stage = start_stage

    def open_lesson(s):
        if STAGES[s][0] == "lip":
            ctx.progress(1, "driving the cars up the ramp to make the starts")
            t_ = time.time()
            with torch.no_grad():
                env.make_pool(s, pool, log=ctx.log)
            ctx.log("The starts off the lip took %.0f s to drive." % (time.time() - t_))
        env.mix = [0.0] * len(STAGES)
        if s == 0:
            env.mix[0] = 1.0
        else:
            for k in range(s):
                env.mix[k] = 0.4 / s
            env.mix[s] = 0.6
    open_lesson(stage)
    with torch.no_grad():
        obs = env.start()
    T = algo.T
    hist = {s: collections.deque(maxlen=max(400, min(3000, n))) for s in range(len(STAGES))}
    recent = collections.deque(maxlen=3000)
    curve = open(ctx.output("curve", "training.csv"), "w", newline="", encoding="utf-8")
    cw = csv.writer(curve)
    cw.writerow(["minutes", "iteration", "lesson", "landed_lesson", "landed_full", "touchdown_mps", "air_kg", "return", "steps_per_s"])
    t_start = time.time()
    last_replay, last_save, last_tel = 0.0, time.time(), 0.0
    rec, rec_ep = [], 0
    episodes, it, stop_reason = 0, 0, "time"
    best_full = -1.0
    whys = {}
    out_path = os.path.join(os.getcwd(), "policy_checkpoint.pt")
    info_track = track_info(env.car.track)

    def full_rate():
        h = hist[len(STAGES) - 1]
        return sum(h) / len(h) if len(h) >= 100 else 0.0

    def save(note):
        st = algo.state()
        st.update({"stage": stage, "landed_full": full_rate(), "obs": OBS, "act": ACT, "note": note, "cars": n, "device": label,
                   "minutes": (time.time() - t_start) / 60.0, "format": "esasim.stunt-policy/1", "ctrl_dt": CTRL_DT})
        torch.save(st, out_path)

    def send_replay(frames, res):
        nonlocal rec_ep
        rec_ep += 1
        for k, part, last in chunks(frames):
            ctx.telemetry(ch_car, {"format": "esasim.car/1", "kind": "training", "ep": rec_ep, "seq": k, "last": last, "fields": FIELDS,
                                   "frames": part, "track": info_track, "result": res,
                                   "label": "Attempt %d · lesson %d of %d: %s" % (episodes, int(res.get("lesson", stage + 1)), len(STAGES), STAGES[int(res.get("lesson", stage + 1)) - 1][1])})

    ctx.telemetry(ch_ev, {"text": "Lesson %d of %d: %s" % (stage + 1, len(STAGES), STAGES[stage][1])})
    try:
        while True:
            elapsed = time.time() - t_start
            if elapsed > minutes * 60:
                break
            stop = False
            for m in ctx.messages(ch_ctl):
                d = m.get("data") if isinstance(m, dict) else None
                if isinstance(d, dict) and str(d.get("control", "")).lower() in ("finish now", "finish", "stop learning"):
                    stop = True
            if stop:
                stop_reason = "asked"
                break
            ob = torch.zeros(T, n, OBS, device=dev)
            ac = torch.zeros(T, n, ACT, device=dev)
            lps = torch.zeros(T, n, device=dev)
            vals = torch.zeros(T, n, device=dev)
            rews = torch.zeros(T, n, device=dev)
            dones = torch.zeros(T, n, dtype=torch.bool, device=dev)
            t0 = time.time()
            with torch.no_grad():
                for t in range(T):
                    a, lp, v = algo.act(obs)
                    nobs, r, d, info = env.step(a)
                    rec.append(frame_of(env.car, env.drv, 0))
                    ob[t], ac[t], lps[t], vals[t], rews[t], dones[t] = obs, a, lp, v, r, d
                    if bool(d[0]):
                        w0 = int(info["why"][0])
                        res = {"text": dict(OUTCOMES).get(w0, "ended"), "landed": w0 == 1, "vz": round(float(info["vz"][0]), 2),
                               "tilt": round(float(info["tilt"][0]), 1), "air": round(float(info["air"][0]), 1), "lesson": int(info["stage"][0]) + 1}
                        if time.time() - last_replay > replay_every and len(rec) > 5:
                            last_replay = time.time()
                            send_replay(rec, res)
                        rec = []
                    if bool(d.any()):
                        idx = d.nonzero(as_tuple=False).flatten()
                        L = info["landed"][idx].tolist()
                        S_ = info["stage"][idx].tolist()
                        V = info["vz"][idx].tolist()
                        Tc = info["touched"][idx].tolist()
                        A_ = info["air"][idx].tolist()
                        R_ = info["ret"][idx].tolist()
                        for w_ in info["why"][idx].tolist():
                            whys[w_] = whys.get(w_, 0) + 1
                        for l, s, vv, tc, aa, rr in zip(L, S_, V, Tc, A_, R_):
                            hist[s].append(1.0 if l else 0.0)
                            recent.append((s, l, vv if tc else None, aa, rr))
                        episodes += len(L)
                    obs = nobs
                if len(rec) > 1500:
                    rec = rec[-1500:]
                _, _, last_v = algo.act(obs)
            stats = algo.update((ob, ac, lps, vals, rews, dones), last_v)
            algo.norm.update(ob.reshape(-1, OBS))
            it += 1
            sps = T * n / max(1e-6, time.time() - t0)
            rate = sum(hist[stage]) / len(hist[stage]) if len(hist[stage]) >= hist[stage].maxlen // 2 else 0.0
            if stage < len(STAGES) - 1 and rate >= 0.5:
                stage += 1
                ctx.log("Lesson %d of %d after %d attempts: %s." % (stage + 1, len(STAGES), episodes, STAGES[stage][1]))
                ctx.telemetry(ch_ev, {"text": "Lesson %d of %d: %s" % (stage + 1, len(STAGES), STAGES[stage][1])})
                open_lesson(stage)
            fr = full_rate()
            if stage == len(STAGES) - 1 and fr > best_full and len(hist[stage]) >= 200:
                best_full = fr
                save("best so far")
                last_save = time.time()
            if time.time() - last_save > 120 and best_full < 0:
                save("checkpoint")
                last_save = time.time()
            if time.time() - last_tel > 2.0:
                last_tel = time.time()
                rc = [x for x in recent if x[0] == stage]
                tch = sorted(x[2] for x in rc if x[2] is not None)
                vz = tch[len(tch) // 2] if tch else None
                airs = sorted(x[3] for x in rc)
                air = airs[len(airs) // 2] if airs else None
                ret = sum(x[4] for x in rc) / len(rc) if rc else 0.0
                row = {"iteration": it, "stage": stage + 1, "landed": round(100 * rate, 1), "landed_full": round(100 * fr, 1),
                       "touchdown_mps": None if vz is None else round(-vz, 2), "air_kg": None if air is None else round(air, 1),
                       "return": round(ret, 1), "attempts": episodes, "steps_per_s": round(sps), "minutes": round(elapsed / 60, 2), "std": round(stats["std"], 3)}
                tot = max(1, sum(whys.values()))
                row["outcomes"] = {k: round(100.0 * whys.get(i, 0) / tot) for i, k in OUTCOMES}
                whys = {}
                ctx.telemetry(ch_train, row)
                cw.writerow([round(elapsed / 60, 2), it, stage + 1, round(rate, 4), round(fr, 4), row["touchdown_mps"], row["air_kg"], row["return"], row["steps_per_s"]])
                curve.flush()
                ctx.progress(min(99, 100 * elapsed / (minutes * 60)), "lesson %d of %d · %.0f%% land · %d attempts" % (stage + 1, len(STAGES), 100 * (fr if stage == len(STAGES) - 1 else rate), episodes))
            if stage == len(STAGES) - 1 and len(hist[stage]) >= hist[stage].maxlen and fr >= target:
                stop_reason = "target"
                break
    finally:
        curve.close()
    if best_full < 0 or full_rate() >= best_full:
        save("final")
    name = "stunt_policy_%s_%02dpct.pt" % (time.strftime("%Y%m%d_%H%M"), int(round(100 * max(0.0, best_full, full_rate()))))
    shutil.copyfile(out_path, ctx.output("policy", name))
    ctx.log("Policy written: %s" % name)
    if stage < len(STAGES) - 1:
        ctx.log("Not every lesson done yet (lesson %d of %d): only the last lesson counts attempts off the lip, so 0%% there is expected. "
                "Run this flow again: it goes on from here." % (stage + 1, len(STAGES)))
    ctx.log("Done (%s): %d attempts in %.1f minutes; lesson %d of %d; %.0f%% of the attempts off the lip land." % (
        {"time": "time is up", "asked": "finished on request", "target": "target reached"}[stop_reason], episodes, (time.time() - t_start) / 60,
        stage + 1, len(STAGES), 100 * full_rate()))
    return {"lesson": stage + 1, "landed_full": round(full_rate(), 3), "attempts": episodes, "minutes": round((time.time() - t_start) / 60, 1)}


def load_policy(path):
    import torch
    st = torch.load(path, map_location="cpu", weights_only=False)
    net = make_nets(torch, st.get("obs") or OBS, st.get("act") or ACT)
    net.load_state_dict(st["net"])
    net.eval()
    mean, var = st["mean"].float(), st["var"].float()

    def act(obs):
        with torch.no_grad():
            x = ((obs - mean) / (var + 1e-8).sqrt()).clamp(-6.0, 6.0)
            mu, _ = net(x)
            return mu.clamp(-1.0, 1.0)
    return act, st


PHASES = {0: "Run-up", 1: "Up the ramp", 2: "Upside down on the fans", 3: "To the lip", 4: "Off the lip"}


def stunt(act, S, seed):
    """one whole run, computed ahead: (rows, events, result). The car stands at the start of the run-up; it drives
    up the ramp, holds nearly upside down for S['hold'] seconds, drives off the lip and the policy flies it down"""
    import random
    import torch
    torch.manual_seed(seed)
    rnd = random.Random(seed)
    car = Car(1, "cpu")
    place_on_ground(car, -float(S["runup"]))
    ang = rnd.uniform(0, 2 * math.pi)
    car.wind[0, 0], car.wind[0, 1] = S["wind"] * math.cos(ang), S["wind"] * math.sin(ang)
    car.fan_k[0] = rnd.uniform(0.95, 1.05)
    car.noz_k[0] = torch.tensor([rnd.uniform(0.95, 1.03) for _ in range(10)])
    drv = Driver(car, torch.tensor([S["speed"]]), torch.tensor([S["hold_back"]]), torch.tensor([S["hold"]]), torch.tensor([S["exit"]]))
    drv.rear_off = torch.zeros(1, dtype=torch.bool)
    dt = CTRL_DT / SUBSTEPS
    rows, events = [], []
    prev = torch.zeros(1, ACT)
    last_ph, t_td, vz_td, tilt_td, x_td, result, t_end = -1, None, None, None, None, None, None
    t_up = None
    air_lip = None
    for k in range(int(80.0 / CTRL_DT)):
        ph = int(drv.phase[0])
        if ph == 4:
            if air_lip is None:
                air_lip = float(car.air[0])
            a = act(observe(car, drv, prev))
            prev = a
            car.thr_cmd = mix(a)
            if t_td is not None and float(car.t[0]) - t_td > SETTLE + 0.6:
                car.thr_cmd = torch.zeros_like(car.thr_cmd)       # on its wheels: the valves close
        for _ in range(SUBSTEPS):
            if not bool(drv.rear_off[0]):
                drv.control(dt)
            if t_td is not None:
                car.drive[:] = 0.0
                car.brake[:] = 14000.0 if float(car.t[0]) - t_td > 0.6 else 0.0
                car.fan_cmd[:] = 0.0
            car.step(dt)
            if t_td is None and bool(drv.rear_off[0]) and (bool(((car.wheel_on & (car.wheel_kind == 0)).any(-1))[0]) or float(car.touched_body[0]) > 0):
                t_td, vz_td, tilt_td, x_td = float(car.t[0]), float(car.vel[0, 2]), float(car.tilt()[0]), float(car.pos[0, 0])
                events.append((t_td, "Touchdown: %.1f m/s down, leaning %.0f deg" % (-vz_td, math.degrees(tilt_td))))
        t = float(car.t[0])
        ph = int(drv.phase[0])
        if ph != last_ph:
            events.append((t, {0: "Run-up: suction fans on", 1: "Up the ramp", 2: "Upside down, held by the fans", 3: "Driving to the lip",
                               4: "Off the lip: the thrusters take over"}[ph]))
            last_ph = ph
        if ph == 4 and t_up is None and float(car.R()[0, 2, 2]) > math.cos(math.radians(20)) and float(car.pos[0, 2]) > 1.5:
            t_up = t
            events.append((t, "Upright, %.0f m up" % float(car.pos[0, 2])))
        rows.append(frame_of(car, drv, 0))
        if result is None:
            body = float(car.touched_body[0]) > 0
            ramp = bool(car.hit_ramp[0])
            tilt = float(car.tilt()[0])
            done = ramp or body or (t_td is not None and t - t_td >= SETTLE) or (ph == 4 and t - float(drv.off_t[0]) > T_LIMIT)
            if done:
                soft = vz_td is not None and vz_td > -3.0
                level = tilt_td is not None and tilt_td < math.radians(15.0)
                landed = (t_td is not None and t - t_td >= SETTLE and not body and not ramp and soft and level and tilt < math.radians(10.0) and float(car.hard[0]) < 0.03)
                why = ("landed" if landed else "hit the ramp" if ramp else "on its side or roof" if body or (t_td is not None and not level)
                       else "hard landing" if t_td is not None else "time")
                result = {"landed": landed, "text": why, "vz": None if vz_td is None else round(vz_td, 2),
                          "tilt": None if tilt_td is None else round(math.degrees(tilt_td), 1), "x": None if x_td is None else round(x_td, 1),
                          "air_used": None if air_lip is None else round(air_lip - float(car.air[0]), 1), "air_left": round(float(car.air[0]), 1),
                          "miss": None if x_td is None else round(math.hypot(x_td - LAND_X, float(car.pos[0, 1])), 1)}
                events.append((t, ("Landed on the tarmac: %.1f m/s at touchdown, %.0f kg of air used" % (-vz_td, result["air_used"] or 0)) if landed
                               else "Lost it: " + why))
                t_end = t + (3.0 if landed else 2.5)
        if t_end is not None and t >= t_end:
            break
    return rows, sorted(events, key=lambda e: e[0]), result


def stunt_run(ctx):
    import random
    import re
    import os
    import time
    import torch
    p = ctx.params
    if torch is None:
        pick_device("cpu")
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    pols = []
    for f in (ctx.inputs.get("policy") or []):
        if str(f.name).endswith(".pt"):
            try:
                h = torch.load(f.path, map_location="cpu", weights_only=False)
                if h.get("format") != "esasim.stunt-policy/1":
                    continue
                m_ = re.search(r"(\d{8}_\d{4})", str(f.name))
                pols.append((m_.group(1) if m_ else "", os.path.getmtime(f.path), float(h.get("landed_full") or 0), float(h.get("minutes") or 0), int(h.get("stage") or 0), f))
            except Exception as e:
                ctx.log("Skipped %s: %s" % (f.name, e))
    if not pols:
        raise RuntimeError("No policy: wire the stunt-policy dataset (stunt_policy_*.pt, written by this block with What to do: Learn the landing) to Policies.")
    pols.sort(key=lambda x: (x[0], x[1]), reverse=True)
    want = str(p.get("policy_file") or "").strip()
    pick = [x for x in pols if want and want in str(x[5].name)]
    ready = [x for x in pols if x[2] >= 0.3]
    pol = (pick or ready or pols)[0][5]
    if not pick and ready and ready[0] is not pols[0]:
        ctx.log("The newest policy, %s, has not finished its lessons yet: flying %s, the newest that lands off the lip." % (pols[0][5].name, pol.name))
    act, st = load_policy(pol.path)
    early = float(st.get("landed_full") or 0) < 0.3
    ctx.log("Thruster policy %s: trained %.0f minutes on %s, reached lesson %d of %d, %.0f%% of the attempts off the lip landed when saved." % (
        pol.name, float(st.get("minutes") or 0), st.get("device") or "?", int(st.get("stage") or 0) + 1, len(STAGES), 100 * float(st.get("landed_full") or 0)))
    if early:
        ctx.log("Warning: an early policy (it has not learned to land off the lip yet): expect the car to be lost. Train longer.")
    ch, ch_ev, ch_ctl = p.get("car_channel") or "car", p.get("events_channel") or "event", p.get("control_channel") or "control"
    num = lambda k, d: float(p.get(k)) if p.get(k) not in (None, "") else d   # noqa: E731
    S = {"hold": num("hold", 3.0), "exit": num("exit", 7.0), "wind": num("wind", 2.0), "speed": num("speed", 8.0), "hold_back": num("hold_back", 7.0),
         "runup": num("runup", 60.0)}
    auto = bool(p.get("auto_start"))
    per_run = int(p.get("runs_per_run") if p.get("runs_per_run") is not None else 1)
    info_track = track_info(Track())
    log = open(ctx.output("runs", "runs.csv"), "w", encoding="utf-8")
    log.write("run,seed,wind,exit_mps,hold_s,landed,outcome,touchdown_mps,tilt_deg,air_used_kg,miss_m\n")

    def controls(wait):
        go = False
        for m in ctx.messages(ch_ctl, wait=wait):
            d = m.get("data") if isinstance(m, dict) else None
            if not isinstance(d, dict):
                continue
            c, v = str(d.get("control", "")).strip().lower(), d.get("value")
            if c in ("start", "go", "launch"):
                go = True
            elif c in ("hold", "exit", "wind", "speed"):
                try:
                    S[c] = float(v)
                except (TypeError, ValueError):
                    pass
        return go

    # the car parked at the start of the run-up, for the view while it waits
    parked = Car(1, "cpu")
    place_on_ground(parked, -S["runup"])
    pdrv = Driver(parked, torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1))
    for _ in range(100):
        parked.step(0.004)
    idle = {"format": "esasim.car/1", "kind": "idle", "ep": 0, "fields": FIELDS, "frames": [frame_of(parked, pdrv, 0)], "track": info_track,
            "status": "Ready: press Start", "phase": "On the start line"}
    ctx.telemetry(ch, idle)
    ctx.telemetry(ch_ev, {"text": "Ready: press Start" + (" (warning: an early policy, it will rarely land)" if early else "")})
    runs = 0
    while True:
        if not auto and not controls(0.5):
            continue
        auto = False
        runs += 1
        seed = random.randrange(1, 10 ** 6)
        ctx.log("Run %d: off the lip at %.0f m/s after %.0f s upside down, wind %.0f m/s." % (runs, S["exit"], S["hold"], S["wind"]))
        rows, events, res = stunt(act, dict(S), seed)
        base = {"format": "esasim.car/1", "kind": "run", "ep": runs, "fields": FIELDS, "track": info_track, "label": "Run %d" % runs, "wind": S["wind"]}
        for k in (3, 2, 1):
            ctx.telemetry(ch, dict(base, frames=[rows[0]], phase="Countdown", clock=-k, status="%d" % k))
            if k == 3:
                ctx.telemetry(ch_ev, {"text": "Run %d: 3, 2, 1…" % runs})
            time.sleep(1.0)
        ev_i, buf, seq = 0, [], 0
        t_wall, t0 = time.time(), rows[0][0]
        for r in rows:
            while ev_i < len(events) and events[ev_i][0] <= r[0]:
                ctx.telemetry(ch_ev, {"text": events[ev_i][1], "t": round(events[ev_i][0], 2)})
                ev_i += 1
            buf.append(r)
            if len(buf) >= 10:
                ahead = (r[0] - t0) - (time.time() - t_wall)
                if ahead > 0:
                    time.sleep(ahead)
                last = buf[-1]
                item = dict(base, seq=seq, frames=buf, phase=PHASES.get(int(last[FIELDS.index("phase")]), ""), clock=round(last[0], 2),
                            speed_kmh=round(3.6 * math.sqrt(last[8] ** 2 + last[9] ** 2 + last[10] ** 2)), height_m=round(last[3] - RIDE, 1),
                            air_kg=round(last[FIELDS.index("air")]), g=last[FIELDS.index("g")])
                if r is rows[-1]:
                    item["last"] = True
                    item["result"] = res
                ctx.telemetry(ch, item)
                seq += 1
                buf = []
                controls(0)
        if buf:
            ctx.telemetry(ch, dict(base, seq=seq, frames=buf, last=True, result=res))
        log.write("%d,%d,%.1f,%.1f,%.1f,%s,%s,%s,%s,%s,%s\n" % (runs, seed, S["wind"], S["exit"], S["hold"], res["landed"], res["text"], res["vz"], res["tilt"], res["air_used"], res["miss"]))
        log.flush()
        ctx.log(("Landed: %.1f m/s at touchdown, %.0f kg of air used." % (-res["vz"], res["air_used"] or 0)) if res["landed"] else "Lost the car: %s." % res["text"])
        if per_run > 0 and runs >= per_run:
            ctx.telemetry(ch_ev, {"text": "Run over: run the flow again for another one"})
            log.close()
            return {"runs": runs, "landed": res["landed"], "outcome": res["text"], "touchdown_mps": res["vz"], "air_used_kg": res["air_used"]}
        ctx.telemetry(ch_ev, {"text": "Ready: press Start for the next run"})
        ctx.telemetry(ch, dict(idle, ep=runs + 1000))


def run(ctx):
    what = str(ctx.params.get("mode") or "Learn the landing").lower()
    if what.startswith("run") or "stunt" in what:
        return stunt_run(ctx)
    return learn_run(ctx)
