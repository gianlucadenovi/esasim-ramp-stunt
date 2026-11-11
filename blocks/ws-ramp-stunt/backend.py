# ── EsaSim block ──────────────────────────────────────────────────────
# Ramp stunt (ws-ramp-stunt) · version 9 · machine block, Python: def run(ctx)
# A hypercar drives up a 21 m crescent ramp upside down on its suction fans, over the top and off its end; its cold-gas nozzles roll it over sideways and land it on the white line: the controller learns
# Workspace: Gianluca De Novi's workspace · Author: Gianluca De Novi
# Saved: 2026-11-11
#
# Inputs   ctx.inputs["id"] is a list of files: .path .name .size .read_text() .read_json()
#   policy      Files (pt)                Policies · optional · The stunt-policy dataset (stunt_policy_*.pt): learning goes on from the furthest one; the stunt flies the newest that lands
# Outputs  ctx.output("id", "name.ext") is the path to write; ctx.write_json / ctx.write_text
#   policy      Files (pt)                Policy · Learn the landing: stunt_policy_<date>_<NN>pct.pt (network, input scaling, lesson reached, share of attempts over the end that land on the line)
#   curve       Table (csv)               Learning curve · Learn the landing: training.csv, minute by minute
#   runs        Table (csv)               Runs · Run the stunt: runs.csv, each run's outcome, touchdown speed, lean, air used, distance off the line, how askew, and impact
#   telemetry   Live telemetry            ctx.telemetry("channel", data) while it runs
# Settings ctx.params["key"]
#   mode        select    What to do = "Learn the landing" of Learn the landing | Run the stunt
#   accelerator select    Accelerator = "auto" of auto | NVIDIA GPU | CPU
#   warm_start  bool      Warm start = true
#   cars        number    Cars at once = 4096
#   minutes     number    Minutes at most = 60
#   target      number    Stop at landing rate = 0.98
#   replay_every number    Replay every (s) = 12
#   from_scratch bool      Start from scratch = false
#   seed        number    Seed = 1
#   exit        number    Exit speed (m/s) = 13.4
#   wind        number    Wind (m/s) = 2
#   speed       number    Run-in speed (m/s) = 11
#   runup       number    Run-up (m) = 80
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
nearly a ceiling, held on it only by suction fans, over its top and off its end. The fans stop, and it falls with all
its momentum; ten compressed-air nozzles roll it over sideways, slow its fall and set it down on the tarmac.
Everything is computed by the physics.

What to do: Learn the landing: the controller of the nozzles learns it by reinforcement learning (PPO), thousands of
cars at once on the GPU, lesson by lesson (a short drop nearly level; on its side or nose down; upside down and moving;
over the end of the ramp after the scripted drive up it); a lesson opens when half the attempts of the one before land.
It sees the car only through its sensors: an IMU, two laser rangefinders (floor and roof) and an RTK GNSS receiver, each
with its noise. Landed: on its four wheels on the tarmac, under 3 m/s down and 15 degrees of lean at touchdown, on the
white line (within 0.5 m of it and 10 degrees of its direction), nothing but the tyres touching, settled level a second
later. One attempt is replayed in the 3D view every few seconds; the
policy (stunt_policy_<date>_<NN>pct.pt) is written when learning ends, with the best one seen over the end.
Run the stunt: the whole stunt played in real time: the car waits on the start line; press Start and it drives up the
ramp with its fans on, over the top and off its end, and the newest policy that lands flies it down."""
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
# (up to 2 g) over its top, upside down, and off its end; the fans stop, it falls with its momentum, and its
# compressed-air thrusters (ten nozzles fed by about 250 kg of air in two carbon-overwrapped tanks) roll it over
# sideways, slow its fall and set it down on the tarmac.
# Everything is computed by the physics, many cars at once (PyTorch, on the GPU or the CPU):
#   * the car: a rigid body (6 degrees of freedom), 1,950 kg empty plus its air, 4.31 x 1.90 x 1.15 m, inertia from its shape;
#   * four wheels: suspension (spring, damper, bump stop) and tyres (grip that saturates, rolling resistance), all-wheel
#     drive and brakes, steering on the front axle;
#   * the body: ten points around it (bumpers, sills, roof) that touch the ground or the ramp: a crash;
#   * two suction fans under the floor: they pull the car toward the surface below the floor only while the skirts seal
#     (the gap is small and the surface faces the floor), up to 2 g together, spinning up and down in a quarter second;
#   * ten cold-gas nozzles: four under the floor blowing down (behind the front and the rear wheels) (lift, and rolling or pitching it when fired unevenly), two
#     on the flanks blowing sideways, four in the tail blowing back (to accelerate it fast; not used in this stunt);
#     none blows up; valves open in 25 ms; thrust held by a regulator until the tanks run low;
#     the air they blow leaves the car (it gets lighter);
#   * the ramp: arcs and a straight in the x-z plane, from the road up to its lip at 168 deg, 5 m wide (RAMP below);
#     the ground: flat tarmac; air drag and wind.
# World frame: metres, x along the run-up (the car drives toward +x into the ramp; over the top it goes back -x, above the road), y to the
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
K_SUS, C_SUS, TRAVEL = 230000.0, 12500.0, 0.12       # very stiff, as a stunt car's suspension under 2 g of suction (3 g and more on the ramp)
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
# cold-gas nozzles: position (body), direction of the force on the car (body), most thrust (N). As on the stunt car:
# four under the floor blowing down, a pair just behind the front wheels and a pair behind the rear wheels (lift; fired
# on one side only they roll the car), one on each flank blowing sideways (right above the front pair, level with the
# tops of the wheels), and four in the tail blowing back (to
# accelerate it fast). None blows upward.
NOZ_POS = [(1.00, 0.80, -0.20), (1.00, -0.80, -0.20), (-1.85, 0.85, -0.18), (-1.85, -0.85, -0.18),      # down: behind the front wheels, behind the rear ones
           (1.00, 0.96, 0.15), (1.00, -0.96, 0.15),                                                    # on the flanks: right above the front pair, level with the tops of the wheels
           (-2.12, 0.35, 0.10), (-2.12, -0.35, 0.10), (-2.10, 0.72, -0.05), (-2.10, -0.72, -0.05)]   # in the tail
NOZ_DIR = [(0, 0, 1)] * 4 + [(0, -1, 0), (0, 1, 0)] + [(1, 0, 0)] * 4
NOZ_MAX = [12300.0] * 4 + [3000.0] * 2 + [4500.0] * 4   # the down nozzles: together as much lift as before they moved (38 kN, 1.75 g)
# the front pair is nearer the centre of mass than the rear: for lift without pitching, the rear pair opens less
LIFT_SHARE = [1.0, 1.0, NOZ_POS[0][0] / -NOZ_POS[2][0], NOZ_POS[0][0] / -NOZ_POS[2][0]]
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

    def ray(self, o, d, max_r=40.0):
        """range along rays (o, d: (N, 3), d unit) to the tarmac or the ramp's deck (either face), max_r when nothing is hit:
        what a laser rangefinder on the car measures"""
        torch = _t()
        big = torch.full(o.shape[:-1], float(max_r), dtype=o.dtype, device=o.device)
        r = big
        tz = torch.where(d[..., 2] < -1e-6, -o[..., 2] / d[..., 2].clamp(max=-1e-6), big)           # the tarmac, z = 0
        r = torch.minimum(r, torch.where(tz > 0, tz, big))
        ox, oz, oy, dx, dz, dy = o[..., 0], o[..., 2], o[..., 1], d[..., 0], d[..., 2], d[..., 1]
        for s_ in self.segs:
            if s_[0] == "arc":
                _, cx, cz, R, a0, a1 = s_
                ex, ez = ox - cx, oz - cz
                A = (dx * dx + dz * dz).clamp(min=1e-9)
                B = 2.0 * (ex * dx + ez * dz)
                C = ex * ex + ez * ez - R * R
                disc = B * B - 4 * A * C
                ok = disc >= 0
                sq = torch.sqrt(disc.clamp(min=0.0))
                for t in ((-B - sq) / (2 * A), (-B + sq) / (2 * A)):
                    px, pz, py = ox + t * dx, oz + t * dz, oy + t * dy
                    th = torch.atan2(px - cx, -(pz - cz))
                    hit = ok & (t > 1e-3) & (th >= a0) & (th <= a1) & (py.abs() <= self.w / 2)
                    r = torch.where(hit & (t < r), t, r)
            else:
                _, px0, pz0, L, a = s_
                nx, nz = -math.sin(a), math.cos(a)
                den = dx * nx + dz * nz
                t = ((px0 - ox) * nx + (pz0 - oz) * nz) / torch.where(den.abs() < 1e-9, torch.full_like(den, 1e-9), den)
                px, pz, py = ox + t * dx, oz + t * dz, oy + t * dy
                sv = (px - px0) * math.cos(a) + (pz - pz0) * math.sin(a)
                hit = (t > 1e-3) & (sv >= 0) & (sv <= L) & (py.abs() <= self.w / 2)
                r = torch.where(hit & (t < r), t, r)
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
    """the flight computer: the landing controller's four numbers, each -1..1 (lift; roll and pitch torques; sideways
    push) shared out over the nozzles as valve openings 0..1. The four down nozzles only blow down: the lift and the roll
    and pitch couples are made by opening them unevenly (a full roll command fires one side only); the flank nozzles
    push the car sideways. The tail nozzles (for fast acceleration) are not used in this stunt: they stay closed"""
    torch = _t()
    a = a.clamp(-1.0, 1.0)
    lift = (a[:, 0] + 1.0) * 0.5
    sy = torch.tensor([1.0, -1.0, 1.0, -1.0], device=a.device, dtype=a.dtype)
    sx = torch.tensor([1.0, 1.0, -1.0, -1.0], device=a.device, dtype=a.dtype)
    ls = torch.tensor(LIFT_SHARE, device=a.device, dtype=a.dtype)      # lift and roll shared so that they do not pitch the car
    down = ((lift.unsqueeze(-1) + a[:, 1:2] * sy) * ls - a[:, 2:3] * sx).clamp(0.0, 1.0)
    side = torch.stack([(-a[:, 3]).clamp(min=0.0), a[:, 3].clamp(min=0.0)], -1)
    rear = torch.zeros(a.shape[0], 4, device=a.device, dtype=a.dtype)
    return torch.cat([down, side, rear], -1)


class Driver:
    """the scripted driver on the ramp (not the learner), as in the stunt: the run-up at v_in, slowing down as it comes
    to the ramp, the suction fans fully on by the time the front wheels are on its first slope; up the wall it slows
    (the climb takes its speed) almost to a stop just before it is upside down; then, held to the ceiling by the fans, it
    accelerates hard and drives off the end at v_exit (about 30 mph). The moment the front wheels pass the end of the deck
    the fans stop: the car is let go with all its momentum, and from then on the thrusters' controller flies it.
    Phases: 0 run-up, 1 climbing, 4 off the end (falling: the controller flies; 2 and 3 are not used)."""

    def __init__(self, car, v_in, v_exit):
        torch = _t()
        self.c = car
        self.v_in, self.v_exit = v_in, v_exit
        self.phase = torch.zeros(car.n, dtype=torch.long, device=car.dev)
        self.off_t = torch.full((car.n,), -1.0, device=car.dev, dtype=car.dt_)
        self.released = torch.zeros(car.n, dtype=torch.bool, device=car.dev)

    def control(self, dt):
        torch = _t()
        c, tr = self.c, self.c.track
        R = c.R()
        xb, yb, zb = R[:, :, 0], R[:, :, 1], R[:, :, 2]
        v_long = (c.vel * xb).sum(-1)
        th = tr.angle(c.pos)
        L = tr.length()
        on_ramp = (c.pos[:, 0] > tr.x0 - 0.3) & (c.pos[:, 2] > 0.6)
        ph = self.phase
        ph = torch.where((ph == 0) & on_ramp & (th > 0.05), torch.ones_like(ph), ph)
        # the front axle's place along the deck: off the end when it is past its length
        front = c.pos + xb * WHEELS[0][0] - zb * 0.4
        s_front = tr.along(front)
        # the speed wanted: slowing down to V_FOOT at the foot of the ramp; up the wall less and less, nearly stopped
        # (V_CRAWL) just before it is upside down (TH_GO); from there flat out to v_exit
        to_foot = (tr.x0 - c.pos[:, 0] - 1.0).clamp(min=0.0)
        v_run = torch.minimum(self.v_in, torch.sqrt(V_FOOT * V_FOOT + 2.0 * 2.5 * to_foot))
        v_climb = V_FOOT + (V_CRAWL - V_FOOT) * (th / TH_GO).clamp(0.0, 1.0)
        v_t = torch.where(ph == 0, v_run, torch.where(th < TH_GO, v_climb, self.v_exit))
        let_go = (ph == 1) & (s_front > L)
        self.off_t = torch.where(let_go, c.t, self.off_t)
        ph = torch.where(let_go, torch.full_like(ph, 4), ph)
        self.released = self.released | (ph == 4)
        # flying: 0.8 s after letting go (the rear wheels roll off the end meanwhile); touching the deck after that is
        # hitting the ramp
        c.flying = self.released & (c.t - self.off_t > 0.8)
        self.phase = ph
        m = c.mass()
        f = m * (5.0 * (v_t - v_long) + G0 * xb[:, 2])
        free = self.released.unsqueeze(-1)
        c.drive = torch.where(self.released, torch.zeros_like(f), f.clamp(-DRIVE_MAX, DRIVE_MAX))
        c.brake = torch.zeros_like(f)
        # the fans: full on as it comes to the ramp (spun up by the time the front wheels reach its first slope), on all
        # the way up and over; off the moment it goes over the end
        near = (c.pos[:, 0] > tr.x0 - FAN_LEAD) | (ph >= 1)
        need = torch.where(near, torch.ones_like(v_long), torch.zeros_like(v_long))
        c.fan_cmd = torch.where(free, torch.zeros_like(c.fan_cmd), need.unsqueeze(-1).expand(-1, 2))
        # stay in the middle of the ramp
        vy_des = -0.8 * c.pos[:, 1]
        want = (vy_des / v_long.abs().clamp(min=2.0)).clamp(-0.1, 0.1)
        st = 3.0 * (want - xb[:, 1]) * torch.sign(yb[:, 1] + 1e-6) * 2.66 / v_long.abs().clamp(min=2.0)
        c.steer = torch.where(self.released, torch.zeros_like(st), st.clamp(-0.15, 0.15))
        return ph


V_FOOT = 7.0                      # m/s at the foot of the ramp
V_CRAWL = 2.0                     # m/s, nearly stopped, just before it is upside down
TH_GO = math.radians(125.0)       # there it accelerates: on the ceiling, held by the fans, out to the end
FAN_LEAD = 6.0                    # m before the ramp where the fans start (full on by its first slope)
STOP_X = -80.0                    # after landing the car stops where it started: the start line of the run-up (Run-up)


def rollout(car, down, stop_x=STOP_X):
    """the cars that are down on their wheels (down: (N,) bool): no drive, steered along the white line, rolling on with
    the speed they landed with and braking just hard enough to stop where they started (the start line at stop_x), at
    once if already past it; the others: untouched"""
    torch = _t()
    R = car.R()
    xb, yb = R[:, :, 0], R[:, :, 1]
    v_long = (car.vel * xb).sum(-1)
    sp = v_long.abs().clamp(min=1.0)
    # along the line: the same steering law as the driver on the run-up, whichever way the car points
    want = ((-0.6 * car.pos[:, 1]) / sp).clamp(-0.08, 0.08) * torch.sign(v_long + 1e-6)
    st = (3.0 * (want - xb[:, 1]) * torch.sign(yb[:, 1] + 1e-6) * 2.66 / sp).clamp(-0.12, 0.12)
    # the distance left to the start line, in the direction it rolls
    vx = car.vel[:, 0]
    left = torch.where(vx < 0, car.pos[:, 0] - stop_x, torch.full_like(vx, 1.0))
    need = car.mass() * vx * vx / (2.0 * left.clamp(min=0.3))
    brake = torch.where(left < 0.3, torch.full_like(vx, 16000.0), torch.where(need > 0.25 * car.mass(), need, torch.zeros_like(vx))).clamp(max=16000.0)
    car.brake = torch.where(down, brake, car.brake)
    car.steer = torch.where(down, st, car.steer)
    car.drive = torch.where(down, torch.zeros_like(car.drive), car.drive)


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


# ── learning to land the car when it goes over the end of the ramp: what the controller senses and does, the lessons, PPO ─────
CTRL_DT, SUBSTEPS = 0.02, 5         # the controller decides 50 times a second; the physics steps every 4 ms
GAMMA = 0.995
T_LIMIT = 9.0                       # seconds from letting go
SETTLE = 1.0                        # seconds on the wheels before a landing counts
ACT = 4
# A reverse curriculum: easy starts first, the real one last. A lesson opens when half the attempts of the one before land.
STAGES = [("upright", "a short drop, nearly level"), ("side", "on its side or nose down, 6-12 m up"),
          ("inverted", "upside down and moving, 12-20 m up"), ("top", "over the end of the ramp, as in the stunt")]
OUTCOMES = ((1, "landed"), (6, "off the line"), (2, "hard landing"), (3, "on its side or roof"), (4, "hit the ramp"), (5, "time"))
# the goal is not only to land safely but on the white line, along it: its centre within LINE_TOL of the line (the line
# well between its wheels) and its heading within LINE_DEG of the line's, at touchdown
LINE_TOL, LINE_DEG = 0.5, 10.0


def drop_x(track=None):
    """where the car's centre is when it goes over the end (its front wheels at the end of the deck): above the tarmac,
    in front of the ramp's foot"""
    tr = track or Track()
    lx, _ = tr.lip()
    return lx - WHEELS[0][0] * math.cos(tr.end)


LAND_X = round(drop_x(), 2)         # the reference spot on the tarmac (x), y = 0: under where it goes over the end


# The controller sees the car only through its sensors, each with its own noise:
#   * an IMU (inertial unit): the gyro (turn rates, with a bias that differs from car to car) and the accelerometer (the
#     push it feels: thrust, drag, the tyres, not gravity), and the attitude its filter works out (the car's forward and
#     up directions);
#   * two laser rangefinders, one in the floor looking down through it and one in the roof looking up: the distance to
#     whatever is there (tarmac or ramp), up to 40 m: whichever way up the car is, one of them sees the ground;
#   * an RTK GNSS receiver (satellite positioning with a base station): position and velocity, 10 times a second.
# Plus what the car knows of itself: the air left, whether its tyres are loaded, whether it has let go and since when.
SENSORS = {"imu": [0.0, 0.0, -0.05], "laser_floor": [0.30, 0.0, -0.31], "laser_roof": [0.10, 0.0, 0.60], "gnss": [-0.40, 0.0, 0.62]}
LASER_MAX = 40.0
OBS = 3 + 6 + 3 + 2 + 3 + 3 + 4 + ACT


class Sensors:
    """the sensors of N cars: their noise and what they hold between readings"""

    def __init__(self, n, dev, dtype, gen=None):
        import torch
        self.n, self.dev, self.dt_, self.gen = n, dev, dtype, gen
        z = lambda *s_: torch.zeros(n, *s_, device=dev, dtype=dtype)   # noqa: E731
        self.bias, self.gpos, self.gvel, self.age, self.vprev = z(3), z(3), z(3), z().fill_(1e9), z(3)
        self.fl = torch.tensor(SENSORS["laser_floor"], device=dev, dtype=dtype)
        self.rf = torch.tensor(SENSORS["laser_roof"], device=dev, dtype=dtype)

    def _n(self, *shape):
        import torch
        return torch.randn(*shape, device=self.dev, dtype=self.dt_, generator=self.gen)

    def reset(self, car, idx):
        if idx.numel() == 0:
            return
        self.bias[idx] = 0.01 * self._n(idx.numel(), 3)
        self.age[idx] = 1e9
        self.vprev[idx] = car.vel[idx]

    def read(self, car, drv, prev_a, dt):
        """the observation: sensor readings, scaled"""
        import torch
        n = car.n
        R = car.R()
        xb, zb = R[:, :, 0], R[:, :, 2]
        gyro = car.w + self.bias + 0.02 * self._n(n, 3)
        att = torch.cat([xb, zb], -1) + 0.006 * self._n(n, 6)
        acc_w = (car.vel - self.vprev) / dt
        self.vprev = car.vel.clone()
        acc_w = acc_w + torch.tensor([0.0, 0.0, G0], device=self.dev, dtype=self.dt_)
        accel = torch.einsum("nji,nj->ni", R, acc_w) + 0.15 * self._n(n, 3)
        # the lasers: from the floor straight down through it (-z of the car), from the roof straight up (+z)
        o1 = car.pos + torch.einsum("nij,j->ni", R, self.fl)
        o2 = car.pos + torch.einsum("nij,j->ni", R, self.rf)
        r1 = car.track.ray(o1, -zb, LASER_MAX)
        r2 = car.track.ray(o2, zb, LASER_MAX)
        r1 = (r1 * (1.0 + 0.005 * self._n(n)) + 0.02 * self._n(n)).clamp(0.0, LASER_MAX)
        r2 = (r2 * (1.0 + 0.005 * self._n(n)) + 0.02 * self._n(n)).clamp(0.0, LASER_MAX)
        # GNSS: a fresh fix every 0.1 s
        self.age = self.age + dt
        fresh = self.age >= 0.1 - 1e-6
        if bool(fresh.any()):
            gp = car.pos + 0.03 * self._n(n, 3)
            gv = car.vel + 0.05 * self._n(n, 3)
            self.gpos = torch.where(fresh.unsqueeze(-1), gp, self.gpos)
            self.gvel = torch.where(fresh.unsqueeze(-1), gv, self.gvel)
            self.age = torch.where(fresh, torch.zeros_like(self.age), self.age)
        since = (car.t - drv.off_t).clamp(min=0.0)
        rel = torch.stack([(self.gpos[:, 0] - LAND_X) / 10.0, self.gpos[:, 1] / 10.0, self.gpos[:, 2] / 10.0], -1)
        return torch.cat([
            gyro * 0.5, att, accel / 20.0,
            torch.stack([r1 / 20.0, r2 / 20.0], -1),
            rel, self.gvel / 8.0,
            (car.air / AIR0).unsqueeze(-1),
            car.wheel_on.float().mean(-1, keepdim=True),
            (~drv.released).float().unsqueeze(-1),
            (since / 4.0).unsqueeze(-1),
            prev_a,
        ], -1)


def heading_err(car):
    """the angle between the car's heading and the white line (radians, 0..pi/2: either way along the line)"""
    import torch
    xb = car.R()[:, :, 0]
    return torch.atan2(xb[:, 1].abs(), xb[:, 0].abs().clamp(min=1e-6))


def potential(car):
    """how close to a good landing a state is (higher is better): upright, falling no faster than the height allows,
    not sliding sideways near the ground, not spinning, the nose kept level (it turns over by rolling, sideways, not by
    pitching end over end), away from the ramp"""
    import torch
    R = car.R()
    zb, yb, xb = R[:, :, 2], R[:, :, 1], R[:, :, 0]
    h = (car.pos[:, 2] - RIDE).clamp(min=0.0)
    tilt = torch.acos(zb[:, 2].clamp(-1.0, 1.0)) / math.pi
    low = (1.0 - h / 6.0).clamp(0.0, 1.0)
    # the fall it may have at height h, as the stunt is flown: free fall, a hard braking burn as late as the thrust
    # allows, slow by about 9 ft, then a gentle settle (the autopilot's profile)
    vz_t = torch.where(h > AP['h1'], -(AP['v1'] + torch.sqrt(2.0 * AP['dec'] * (h - AP['h1']).clamp(min=0.0))),
                       -(AP['v0'] + (AP['v1'] - AP['v0']) * h / AP['h1']))
    fast = (vz_t - car.vel[:, 2]).clamp(min=0.0)
    slow = (car.vel[:, 2] - vz_t).clamp(min=0.0)
    side = (car.vel * yb).sum(-1).abs()
    near_ramp = (car.pos[:, 0] - (car.track.x0 - 1.0)).clamp(min=0.0)
    off = car.pos[:, 1].abs()                          # how far off the white line (it runs along x at y = 0)
    hdg = heading_err(car)
    # falling slower than that is penalised too (less): hovering only burns the air it needs to land
    return -((0.6 + 2.0 * low) * torch.log1p(off / 0.5) + (0.5 + 1.5 * low) * torch.sin(hdg) + 1.5 * tilt + 3.0 * tilt * low + 0.2 * car.w.norm(dim=-1) + (2.0 + 4.0 * low) * torch.log1p(fast / 1.5) + 1.2 * torch.log1p(slow / 1.5)
             + 0.5 * low * torch.log1p(side / 2.0) + 1.2 * xb[:, 2].abs() + 0.8 * near_ramp)


def free_start(car, drv, idx, kind, gen):
    """lessons 1-3: the car in the air in front of the ramp, in the attitude the lesson asks for"""
    import torch
    k = idx.numel()
    dev, dt = car.dev, car.dt_
    U = lambda a, b: a + (b - a) * torch.rand(k, device=dev, dtype=dt, generator=gen)   # noqa: E731
    sgn = lambda: torch.where(torch.rand(k, device=dev, generator=gen) < 0.5, -1.0, 1.0).to(dt)   # noqa: E731
    if kind == "upright":
        h, x = U(1.5, 4.0), LAND_X + U(-14.0, 1.0)
        roll, pitch = U(-0.4, 0.4), U(-0.3, 0.3)
        vx, vz, wmax = U(-6.0, 0.0), U(-3.0, 0.5), 0.3
    elif kind == "side":
        h, x = U(6.0, 12.0), LAND_X + U(-12.0, 0.0)
        side = torch.rand(k, device=dev, generator=gen) < 0.75
        roll = torch.where(side, sgn() * U(0.4, 1.8), U(-0.3, 0.3))
        pitch = torch.where(side, U(-0.25, 0.25), sgn() * U(0.3, 0.8))
        vx, vz, wmax = U(-6.0, 0.0), U(-3.0, 0.5), 0.5
    else:
        h, x = U(12.0, 19.6), LAND_X + U(-5.0, 1.0)
        roll = math.pi + U(-0.5, 0.5)
        pitch = U(-0.4, 0.4)
        vx, vz, wmax = U(-9.0, -3.0), U(-3.0, 0.5), 0.8
    yaw = math.pi + U(-0.05, 0.05)             # heading -x and moving that way, as when it goes over the end, near the line
    car.pos[idx] = torch.stack([x, U(-0.4, 0.4), h + RIDE], -1)
    car.q[idx] = q_euler(roll, pitch, yaw)
    car.vel[idx] = torch.stack([vx, U(-0.6, 0.6), vz], -1)
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
    drv.released[idx] = True
    car.flying[idx] = True


AP = {'bank': 0.35, 'kr': 6.0, 'w': 9.0, 'up': 0.6, 'room': 1.0, 'early': 0.25, 'side': 0.0, 'kv': 3.0, 'kp': 2.0, 'hb': 1.0, 'kh': 4.0, 'kg': 5.0, 'v0': 0.5, 'dec': 4.5, 'kz': 12.0, 'h1': 2.7, 'v1': 2.0}


def autopilot(c, drv):
    """a simple hand-written autopilot, for the warm start only (it reads the true state, which the policy never does).
    It controls the car's turn rates: it starts the roll over sideways with a short kick of the down nozzles of one side,
    coasts through it with the valves closed (blowing while the car is on its side would push it off the line) and stops
    it with another kick; meanwhile the front and rear nozzles' couple keeps the heading along the line (on its side
    that couple turns the car about the vertical). Upright it brakes the fall to a set profile and steers back onto the
    line by banking slightly (the flank nozzles would turn it too: they sit ahead of the centre of mass). Its landings
    are what the network copies (from its own sensors) before it improves on them by trial and error"""
    import torch
    R = c.R()
    xb, yb, zb = R[:, :, 0], R[:, :, 1], R[:, :, 2]
    ub = torch.einsum("nji,j->ni", R, torch.tensor([0.0, 0.0, 1.0], device=c.dev, dtype=c.dt_))      # up, in the car's frame
    phi = torch.atan2(ub[:, 1], ub[:, 2])
    phi = torch.where(phi < -2.2, phi + 2 * math.pi, phi)
    up = zb[:, 2]
    h = (c.pos[:, 2] - RIDE).clamp(min=0.0)
    # steering back onto the line (upright): the sideways push wanted, from the car's offset and drift
    # (across the line in the world, then along the car's own left: its forward speed must not leak in)
    vyb = (c.vel * yb).sum(-1)
    ay = ((-AP['kv'] * c.vel[:, 1] - AP['kp'] * c.pos[:, 1]) * yb[:, 1]).clamp(-3.5, 3.5)
    bank = torch.where((up > 0.85) & (h > AP['hb']), (-torch.atan(ay / G0)).clamp(-AP['bank'], AP['bank']), torch.zeros_like(ay))
    # the heading: along the line, whichever way the car points
    psi = torch.atan2(xb[:, 1], xb[:, 0])
    psi = torch.where(psi > math.pi / 2, psi - math.pi, torch.where(psi < -math.pi / 2, psi + math.pi, psi))
    r_des = (-AP['kh'] * psi).clamp(-1.5, 1.5)
    # the turn rates wanted (body frame): the roll, and the heading correction about the vertical
    wx_des = (-AP['kr'] * (phi - bank)).clamp(-AP['w'], AP['w']) + r_des * ub[:, 0]
    wy_des = r_des * ub[:, 1] + torch.where(phi.abs() < 0.35, 2.0 * xb[:, 2], torch.zeros_like(up))      # once over: the nose level
    # while it rolls fast, pitching makes it yaw (the gyroscopic coupling, I_z dw_z = -(I_y - I_x) w_x w_y): a pitch rate
    # chosen against the yaw rate stops the car from coming out of the roll turning
    gy = (INERTIA[1] - INERTIA[0]) / INERTIA[2]
    wx_ = c.w[:, 0]
    fast = wx_.abs() > 1.5
    wy_des = wy_des + torch.where(fast, (AP['kg'] * (r_des * ub[:, 2] - c.w[:, 2]) / (-gy * torch.where(fast, wx_, torch.ones_like(wx_)))).clamp(-1.5, 1.5), torch.zeros_like(wx_))
    a1 = (0.4 * (wx_des - c.w[:, 0]))
    a1 = torch.where((a1.abs() < 0.04) & (phi.abs() > 0.6), torch.zeros_like(a1), a1).clamp(-1.0, 1.0)    # coasting: the valves stay shut
    a2 = (0.85 * (wy_des - c.w[:, 1])).clamp(-1.0, 1.0)
    # the fall it allows at height h: it falls freely (rolling over meanwhile) and brakes as late as the thrust allows,
    # hard, to a slow descent by about 9 ft (h1); from there it settles gently onto its wheels
    vzt = torch.where(h > AP['h1'], -(AP['v1'] + torch.sqrt(2.0 * AP['dec'] * (h - AP['h1']).clamp(min=0.0))),
                      -(AP['v0'] + (AP['v1'] - AP['v0']) * h / AP['h1']))
    need = c.mass() * (G0 + AP['kz'] * (vzt - c.vel[:, 2])) / (sum(LIFT_SHARE) * NOZ_MAX[0] * up.clamp(min=0.3))
    lift = torch.where(up > AP['up'], need.clamp(0.0, 1.0), torch.zeros_like(need))
    lift = torch.minimum(lift, 1.0 - AP['room'] * (a1.abs() + a2.abs()))           # room left for the turning couples
    a0 = lift * 2.0 - 1.0
    a3 = torch.where(up > 0.7, (AP['side'] * ay / (NOZ_MAX[4] / (DRY + AIR0))).clamp(-1.0, 1.0), torch.zeros_like(vyb))
    early = (c.t - drv.off_t) < AP['early']
    z_ = torch.zeros_like(a0)
    return torch.stack([torch.where(early, z_, a0), torch.where(early, z_, a1), torch.where(early, z_, a2), a3], -1)


DRV_STATE = ["phase", "off_t", "released", "v_in", "v_exit"]
CAR_EXTRA = ["drive", "brake", "steer", "fan_cmd", "thr_cmd", "flying", "_wspin"]


def top_pool(size, dev, dtype, gen, log=None):
    """the real starts: cars driven by the script up the ramp and over its top; each start is the moment the front wheels
    pass the end of the deck and the fans stop, with everything as it was then (its speed, the rear still on the deck)"""
    import torch
    car = Car(size, dev, dtype)
    U = lambda a, b: a + (b - a) * torch.rand(size, device=car.dev, dtype=car.dt_, generator=gen)   # noqa: E731
    place_on_ground(car, -22.0, y=0.0, yaw=0.0, v=0.0)
    car.pos[:, 1] = U(-0.4, 0.4)
    car.vel[:, 0] = U(6.0, 12.0)
    car.q = q_euler(torch.zeros_like(car.t), torch.zeros_like(car.t), U(-0.03, 0.03))
    car.air = U(230.0, 250.0)
    car.fan_k = U(0.9, 1.1)
    car.noz_k = 0.92 + 0.13 * torch.rand(size, 10, device=car.dev, dtype=car.dt_, generator=gen)
    car.wind = torch.stack([U(-6.0, 6.0), U(-6.0, 6.0), torch.zeros_like(car.t)], -1)
    car.flying = torch.zeros(size, dtype=torch.bool, device=car.dev)
    drv = Driver(car, car.vel[:, 0].clone(), U(11.0, 15.5))
    snap, got = {}, torch.zeros(size, dtype=torch.bool, device=car.dev)
    dt = 0.004
    keys = Car.STATE + CAR_EXTRA
    for _ in range(int(18.0 / dt)):
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
        tilt = torch.rad2deg(torch.acos(qmat(snap["q"])[:, 2, 2].clamp(-1, 1)))
        log("over the end: %d starts, %.1f to %.1f m up, x %.1f to %.1f m, upside down by %.0f to %.0f deg" % (
            keep.numel(), float(snap["pos"][:, 2].min()), float(snap["pos"][:, 2].max()), float(snap["pos"][:, 0].min()), float(snap["pos"][:, 0].max()),
            float(tilt.min()), float(tilt.max())))
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
        self.drv = Driver(self.car, torch.zeros(n, **f), torch.zeros(n, **f))
        self.sens = Sensors(n, self.dev, self.dtype, self.gen)
        self.drv.released = torch.ones(n, dtype=torch.bool, device=self.dev)
        self.pools = {}
        self.mix = [1.0, 0.0, 0.0, 0.0]
        self.stage_of = torch.zeros(n, dtype=torch.long, device=self.dev)
        self.ret = torch.zeros(n, **f)
        self.prev = torch.zeros(n, ACT, **f)
        self.t_td = torch.full((n,), -1.0, **f)
        self.vz_td = torch.zeros(n, **f)
        self.tilt_td = torch.zeros(n, **f)
        self.x_td = torch.zeros(n, **f)
        self.y_td = torch.zeros(n, **f)
        self.h_td = torch.zeros(n, **f)
        self.air_used = torch.zeros(n, **f)

    def make_pool(self, stage, size, log=None):
        if STAGES[stage][0] == "top" and stage not in self.pools:
            self.pools[stage] = top_pool(size, self.dev, self.dtype, self.gen, log)

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
            if STAGES[s][0] != "top":
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
        self.sens.reset(c, idx)
        self.ret[idx] = 0.0
        self.prev[idx] = 0.0
        self.t_td[idx] = -1.0
        self.air_used[idx] = 0.0

    def start(self):
        import torch
        self.reset(torch.arange(self.n, device=self.dev))
        return self.sens.read(self.car, self.drv, self.prev, CTRL_DT)

    def step(self, a):
        import torch
        c, d = self.car, self.drv
        phi0 = potential(c)
        air0 = c.air.clone()
        a = a.clamp(-1.0, 1.0)
        c.thr_cmd = mix(a)
        for _ in range(SUBSTEPS):
            if not bool(d.released.all()):
                ph = d.phase.clone()
                d.control(CTRL_DT / SUBSTEPS)
                d.phase = torch.maximum(ph, d.phase)
            c.brake = torch.zeros_like(c.brake)
            rollout(c, self.t_td >= 0)                  # down: rolling on along the line, braking to stop on the start line
            c.step(CTRL_DT / SUBSTEPS)
            c.flying = d.released & (c.t - d.off_t > 0.8) & (self.t_td < 0)     # down on its wheels: rolling onto the ramp's foot is no crash
            gd = (c.wheel_on & (c.wheel_kind == 0)).any(-1) | (c.touched_body > 0)
            new = gd & (self.t_td < 0) & d.released
            if bool(new.any()):
                self.t_td = torch.where(new, c.t, self.t_td)
                self.vz_td = torch.where(new, c.vel[:, 2], self.vz_td)
                self.tilt_td = torch.where(new, c.tilt(), self.tilt_td)
                self.x_td = torch.where(new, c.pos[:, 0], self.x_td)
                self.y_td = torch.where(new, c.pos[:, 1], self.y_td)
                self.h_td = torch.where(new, heading_err(c), self.h_td)
        tilt = c.tilt()
        body = c.touched_body > 0.0
        ramp = c.hit_ramp
        since_td = torch.where(self.t_td >= 0, c.t - self.t_td, torch.full_like(c.t, -1.0))
        settled = since_td >= SETTLE
        timeout = (c.t - d.off_t) > T_LIMIT
        end = ramp | body | settled | timeout | (tilt > 1.9) & (since_td >= 0)
        soft = self.vz_td > -3.0
        level = self.tilt_td < math.radians(15.0)
        safe = settled & ~body & ~ramp & soft & level & (tilt < math.radians(10.0)) & (c.hard < 0.03)
        on_line = (self.y_td.abs() < LINE_TOL) & (self.h_td < math.radians(LINE_DEG))
        landed = safe & on_line
        why = torch.where(landed, 1, torch.where(safe, 6, torch.where(ramp, 4, torch.where(body | (end & ~level & (since_td >= 0)), 3,
                          torch.where(end & (since_td >= 0), 2, torch.where(timeout, 5, 0))))))
        why = torch.where(end, why, torch.zeros_like(why))
        phi1 = potential(c)
        used = air0 - c.air
        self.air_used += used
        # air costs a little (a point per 28 kg): frugal, but never at the price of a landing
        r = (GAMMA * phi1 * (~end).to(phi1.dtype) - phi0) - 0.035 * used - 0.05 * c.vel[:, 2].clamp(min=0.0)
        bonus = torch.zeros_like(r)
        bonus = torch.where(why == 1, 100.0 + 25.0 * (1.0 + self.vz_td / 3.0).clamp(0.0, 1.0) + 10.0 * (1.0 - self.tilt_td / math.radians(15.0)).clamp(0.0, 1.0)
                            + 15.0 * (1.0 - self.y_td.abs() / LINE_TOL).clamp(0.0, 1.0), bonus)
        # down safely but off the line: some credit, less the further off and the more askew
        bonus = torch.where(why == 6, 45.0 - 15.0 * (self.y_td.abs() - LINE_TOL).clamp(min=0.0, max=2.0) - 20.0 * (self.h_td / math.radians(30.0)).clamp(max=1.0), bonus)
        bonus = torch.where(why == 2, -30.0 - 8.0 * (self.vz_td.abs() - 3.0).clamp(min=0.0, max=15.0), bonus)
        bonus = torch.where(why == 3, torch.full_like(bonus, -70.0), bonus)
        bonus = torch.where(why == 4, torch.full_like(bonus, -80.0), bonus)
        bonus = torch.where(why == 5, torch.full_like(bonus, -60.0), bonus)
        r = r + bonus
        self.ret += r
        info = {"end": end, "why": why, "landed": why == 1, "vz": self.vz_td.clone(), "tilt": torch.rad2deg(self.tilt_td),
                "x": self.x_td.clone(), "y": self.y_td.clone(), "hdg": torch.rad2deg(self.h_td), "air": self.air_used.clone(), "stage": self.stage_of.clone(), "ret": self.ret.clone(),
                "touched": self.t_td >= 0}
        self.prev = a
        idx = end.nonzero(as_tuple=False).flatten()
        self.reset(idx)
        return self.sens.read(c, d, self.prev, CTRL_DT), r, end, info


def make_nets(torch, obs=None, act=None, hidden=256):
    import torch.nn as nn
    obs, act = obs or OBS, act or ACT

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.pi = nn.Sequential(nn.Linear(obs, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, act))
            self.v = nn.Sequential(nn.Linear(obs, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
            self.log_std = nn.Parameter(torch.full((act,), -1.0))
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
    def __init__(self, torch, dev, n_env, horizon=48, lr=1.5e-4, gamma=GAMMA, lam=0.95, clip=0.2, epochs=4, minibatches=4, rscale=0.1):
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

    def update(self, buf, last_v, value_only=False):
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
                loss = 0.5 * vl if value_only else pl + 0.5 * vl - 0.001 * ent
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
          "fan0", "fan1", "n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "air", "g", "phase", "drive",
          "hit", "fspin", "las0", "las1"]
# hit: the fastest the body has struck anything so far (m/s; above 0 it crashed); fspin: how fast the fans spin (0-1);
# las0, las1: what the floor and roof laser rangefinders read (m)


def track_info(track):
    return {"R": track.R, "end_deg": round(math.degrees(track.end), 2), "width": track.w, "x0": track.x0,
            "profile": track.profile(),
            "land": [round(LAND_X - 48.0, 1), 0.0], "box": [9.0, 4.0], "nozzles": NOZ_POS, "nozzle_dirs": NOZ_DIR, "wheels": WHEELS, "rest": REST,   # land: about where it comes down, some 50 m on from the end
            "wheel_r": WHEEL_R, "car": [LENGTH, WIDTH, HEIGHT], "ride": RIDE, "comp0": round((DRY + AIR0) * G0 / 4.0 / K_SUS, 3),
            "sensors": SENSORS, "laser_max": LASER_MAX, "body": BODY}


def lasers(car, i):
    """what car i's two laser rangefinders read now (no noise: for the view)"""
    import torch
    R = car.R()[i:i + 1]
    zb = R[:, :, 2]
    o1 = car.pos[i:i + 1] + torch.einsum("nij,j->ni", R, torch.tensor(SENSORS["laser_floor"], dtype=car.dt_, device=car.dev))
    o2 = car.pos[i:i + 1] + torch.einsum("nij,j->ni", R, torch.tensor(SENSORS["laser_roof"], dtype=car.dt_, device=car.dev))
    return float(car.track.ray(o1, -zb, LASER_MAX)[0]), float(car.track.ray(o2, zb, LASER_MAX)[0])


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
            + [round(float(car.air[i]), 1), r2(car.g_load[i]), int(drv.phase[i]), round(float(car.drive[i] - car.brake[i]) / DRIVE_MAX, 3)]
            + [r2(car.touched_body[i]), r3(car.fan[i].mean())] + [r2(x) for x in lasers(car, i)])


def chunks(rows, size=150):
    for i in range(0, len(rows), size):
        yield i // size, rows[i:i + size], i + size >= len(rows)


def warm_start(env, algo, log=None, rounds=3, steps=250):
    """the network copies the autopilot (DAgger): first the autopilot flies and the network learns its commands from its
    own sensors; then the network flies, the autopilot says what it would have done in each of the states the network
    got itself into, and the network learns those too; a few rounds. Returns the share of landings in the last round"""
    import torch
    torch_ = torch
    X, Y = [], []
    opt = torch_.optim.Adam(algo.net.pi.parameters(), lr=1e-3)
    rate = 0.0
    for rnd in range(rounds + 1):
        env.mix = [0.15, 0.15, 0.0, 0.7]
        with torch_.no_grad():
            ob_ = env.start()
            got, ok = 0, 0
            for _ in range(steps):
                a_ = autopilot(env.car, env.drv)
                X.append(ob_)
                Y.append(a_)
                if rnd == 0:
                    act_ = a_ + 0.08 * torch_.randn_like(a_)
                else:
                    act_, _, _ = algo.act(ob_, deterministic=True)
                ob_, _, d_, info_ = env.step(act_)
                if bool(d_.any()):
                    got += int(d_.sum())
                    ok += int(info_["landed"][d_].sum())
        rate = ok / max(1, got)
        if log:
            log("warm start, round %d: %s landed %d of %d attempts" % (rnd + 1, "the autopilot" if rnd == 0 else "the network", ok, got))
        Xa, Ya = torch_.cat(X), torch_.cat(Y)
        if rnd == 0:
            algo.norm.update(Xa)
        xn = algo.norm(Xa)
        for _ in range(12 if rnd else 25):
            perm = torch_.randperm(Xa.shape[0], device=Xa.device)
            for i in range(0, Xa.shape[0], 8192):
                j = perm[i:i + 8192]
                loss = ((algo.net.pi(xn[j]) - Ya[j]) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
    with torch_.no_grad():
        algo.net.log_std.fill_(-2.3)
    return rate


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
    ctx.log("Learning the landing on %s with %d cars at once, for up to %.0f minutes (or until %.0f%% of the attempts over the end of the ramp land)." % (label, n, minutes, target * 100))
    algo = PPO(torch, dev, n)
    start_stage = 0
    found = []
    for f in (ctx.inputs.get("policy") or []):
        if str(f.name).endswith(".pt"):
            try:
                st_ = torch.load(f.path, map_location="cpu", weights_only=False)
                if st_.get("format") != "esasim.stunt-policy/4":
                    ctx.log("Skipped %s: made for an earlier version of the stunt (the car, its thrusters or the script up the ramp have changed since); this one learns anew." % f.name)
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
        ctx.log("Continuing from %s: it had reached lesson %d of %d after %.0f minutes, %.0f%% of the attempts over the end of the ramp landing." % (
            f.name, start_stage + 1, len(STAGES), float(st.get("minutes") or 0), 100 * float(st.get("landed_full") or 0)))
    elif found:
        ctx.log("Starting from scratch (asked): the %d policies so far are left as they are." % len(found))
    env = StuntEnv(n, dev, seed=seed)
    pool = max(2048, min(8192, 2 * n)) if kind != "cpu" else 1024
    warm = bool(p.get("warm_start", True)) and not (found and not bool(p.get("from_scratch")))
    value_warm = 0
    if warm:
        # the warm start: the network first copies the landings of a simple hand-written autopilot, from its own sensors
        # (the autopilot reads the true state), on short drops, sideways starts and the real thing; then it learns on by
        # trial and error. Without it (Warm start off) it starts from nothing and learns everything by trial and error.
        ctx.progress(1, "warm start: watching an autopilot land")
        ctx.log("Warm start: the network first copies an autopilot's landings, seeing only its sensors; then it improves on them by trial and error.")
        t_ = time.time()
        env.make_pool(len(STAGES) - 1, pool, log=ctx.log)
        rate_ = warm_start(env, algo, ctx.log)
        ctx.log("Warm start done in %.0f s: flying on its own, the copy lands %.0f%% of its attempts." % (time.time() - t_, 100 * rate_))
        start_stage = len(STAGES) - 1
        value_warm = 8
    stage = start_stage

    def open_lesson(s):
        if STAGES[s][0] == "top":
            ctx.progress(1, "driving the cars up the ramp and over its end to make the starts")
            t_ = time.time()
            with torch.no_grad():
                env.make_pool(s, pool, log=ctx.log)
            ctx.log("The starts over the end took %.0f s to drive." % (time.time() - t_))
        env.mix = [0.0] * len(STAGES)
        if s == 0:
            env.mix[0] = 1.0
        elif warm and s == len(STAGES) - 1:
            env.mix = [0.1, 0.1, 0.05, 0.75]        # after the warm start: mostly the real thing, the easier lessons kept up
        else:
            for k in range(s):
                env.mix[k] = 0.5 / s
            env.mix[s] = 0.5
    open_lesson(stage)
    with torch.no_grad():
        obs = env.start()
    T = algo.T
    hist = {s: collections.deque(maxlen=max(400, min(3000, n))) for s in range(len(STAGES))}
    recent = collections.deque(maxlen=3000)
    curve = open(ctx.output("curve", "training.csv"), "w", newline="", encoding="utf-8")
    cw = csv.writer(curve)
    cw.writerow(["minutes", "iteration", "lesson", "landed_lesson", "landed_full", "touchdown_mps", "air_kg", "off_line_m", "return", "steps_per_s"])
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
                   "minutes": (time.time() - t_start) / 60.0, "format": "esasim.stunt-policy/4", "ctrl_dt": CTRL_DT})
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
                               "tilt": round(float(info["tilt"][0]), 1), "air": round(float(info["air"][0]), 1), "lesson": int(info["stage"][0]) + 1,
                               "miss": round(abs(float(info["y"][0])), 2) if bool(info["touched"][0]) else None,
                               "hdg": round(float(info["hdg"][0]), 1) if bool(info["touched"][0]) else None}
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
                        Y_ = info["y"][idx].abs().tolist()
                        for w_ in info["why"][idx].tolist():
                            whys[w_] = whys.get(w_, 0) + 1
                        for l, s, vv, tc, aa, rr, yy in zip(L, S_, V, Tc, A_, R_, Y_):
                            hist[s].append(1.0 if l else 0.0)
                            recent.append((s, l, vv if tc else None, aa, rr, yy if tc else None))
                        episodes += len(L)
                    obs = nobs
                if len(rec) > 1500:
                    rec = rec[-1500:]
                _, _, last_v = algo.act(obs)
            stats = algo.update((ob, ac, lps, vals, rews, dones), last_v, value_only=value_warm > 0)
            value_warm = max(0, value_warm - 1)          # after the warm start the value network learns a few rounds first
            if stage == 0 and start_stage == 0:
                algo.norm.update(ob.reshape(-1, OBS))     # the input scaling settles in lesson 1, then stays: later lessons do not shift what it learned
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
                offs = sorted(x[5] for x in rc if x[5] is not None)
                off_m = offs[len(offs) // 2] if offs else None
                row = {"iteration": it, "stage": stage + 1, "landed": round(100 * rate, 1), "landed_full": round(100 * fr, 1),
                       "touchdown_mps": None if vz is None else round(-vz, 2), "air_kg": None if air is None else round(air, 1), "off_line_m": None if off_m is None else round(off_m, 2),
                       "return": round(ret, 1), "attempts": episodes, "steps_per_s": round(sps), "minutes": round(elapsed / 60, 2), "std": round(stats["std"], 3)}
                tot = max(1, sum(whys.values()))
                row["outcomes"] = {k: round(100.0 * whys.get(i, 0) / tot) for i, k in OUTCOMES}
                whys = {}
                ctx.telemetry(ch_train, row)
                cw.writerow([round(elapsed / 60, 2), it, stage + 1, round(rate, 4), round(fr, 4), row["touchdown_mps"], row["air_kg"], row["off_line_m"], row["return"], row["steps_per_s"]])
                curve.flush()
                ctx.progress(min(99, 100 * elapsed / (minutes * 60)), "lesson %d of %d · %.0f%% land · %d attempts" % (stage + 1, len(STAGES), 100 * (fr if stage == len(STAGES) - 1 else rate), episodes))
            if stage == len(STAGES) - 1 and len(hist[stage]) >= hist[stage].maxlen and fr >= target and elapsed > min(3.0, minutes / 4) * 60:
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
        ctx.log("Not every lesson done yet (lesson %d of %d): only the last lesson counts attempts over the end, so 0%% there is expected. "
                "Run this flow again: it goes on from here." % (stage + 1, len(STAGES)))
    ctx.log("Done (%s): %d attempts in %.1f minutes; lesson %d of %d; %.0f%% of the attempts over the end of the ramp land." % (
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


PHASES = {0: "Run-up", 1: "Up the ramp", 2: "", 3: "", 4: "Off the end: the thrusters fly it"}


def stunt(act, S, seed):
    """one whole run, computed ahead: (rows, events, result). The car stands at the start of the run-up; it drives up
    the ramp, over its top and off its end at S['exit'] m/s; the fans stop as the front wheels pass the end, it falls
    with its momentum, and the policy rolls it over sideways and lands it with its thrusters"""
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
    drv = Driver(car, torch.tensor([S["speed"]]), torch.tensor([S["exit"]]))
    sens = Sensors(1, car.dev, car.dt_)
    sens.reset(car, torch.zeros(1, dtype=torch.long))
    dt = CTRL_DT / SUBSTEPS
    rows, events = [], []
    prev = torch.zeros(1, ACT)
    last_ph, t_td, vz_td, tilt_td, x_td, result, t_end = -1, None, None, None, None, None, None
    y_td = h_td = None
    t_up, air_go, crash_said = None, None, False
    for k in range(int(90.0 / CTRL_DT)):
        ph = int(drv.phase[0])
        obs = sens.read(car, drv, prev, CTRL_DT)
        if ph == 4:
            if air_go is None:
                air_go = float(car.air[0])
            a = act(obs)
            prev = a
            car.thr_cmd = mix(a)
            if result is not None or (t_td is not None and float(car.t[0]) - t_td > SETTLE + 0.6):
                car.thr_cmd = torch.zeros_like(car.thr_cmd)       # down, or lost: the valves close
        for _ in range(SUBSTEPS):
            if not bool(drv.released[0]):
                drv.control(dt)
            if t_td is not None:
                car.fan_cmd[:] = 0.0
                rollout(car, torch.ones(1, dtype=torch.bool) if float(car.touched_body[0]) == 0 else torch.zeros(1, dtype=torch.bool), -float(S["runup"]))   # down: rolling on back to the start line
                if float(car.touched_body[0]) > 0:
                    car.drive[:] = 0.0
                    car.brake[:] = 16000.0
            car.step(dt)
            car.flying = drv.released & (car.t - drv.off_t > 0.8) & (t_td is None)
            if t_td is None and bool(drv.released[0]) and (bool(((car.wheel_on & (car.wheel_kind == 0)).any(-1))[0]) or float(car.touched_body[0]) > 0):
                t_td, vz_td, tilt_td, x_td = float(car.t[0]), float(car.vel[0, 2]), float(car.tilt()[0]), float(car.pos[0, 0])
                y_td, h_td = float(car.pos[0, 1]), float(heading_err(car)[0])
                events.append((t_td, "Touchdown: %.1f m/s down, leaning %.0f deg, %.1f m off the line and %.0f deg askew" % (-vz_td, math.degrees(tilt_td), abs(y_td), math.degrees(h_td))))
        t = float(car.t[0])
        ph = int(drv.phase[0])
        if ph != last_ph:
            events.append((t, {0: "Run-up", 1: "On the ramp, the suction fans full on: up the wall, nearly stopping before it is upside down, then flat out along the ceiling",
                               4: "Over the end at %.0f km/h: the fans stop, it falls; the thrusters take over" % (3.6 * float(car.vel[0].norm()))}[ph]))
            last_ph = ph
        if ph == 4 and t_up is None and float(car.R()[0, 2, 2]) > math.cos(math.radians(20)) and float(car.pos[0, 2]) > 1.5:
            t_up = t
            events.append((t, "Rolled over sideways: upright %.0f m up, %.1f s after going over the end" % (float(car.pos[0, 2]) - RIDE, t - float(drv.off_t[0]))))
        if float(car.touched_body[0]) > 0 and not crash_said:
            crash_said = True
            events.append((t, "Crash: the body hit the %s at %.0f km/h" % ("ramp" if int(car.body_kind[0]) == 1 else "tarmac", 3.6 * float(car.touched_body[0]))))
        rows.append(frame_of(car, drv, 0))
        if result is None:
            body = float(car.touched_body[0]) > 0
            ramp = bool(car.hit_ramp[0])
            tilt = float(car.tilt()[0])
            done = ramp or body or (t_td is not None and t - t_td >= SETTLE) or (ph == 4 and t - float(drv.off_t[0]) > T_LIMIT)
            if done:
                soft = vz_td is not None and vz_td > -3.0
                level = tilt_td is not None and tilt_td < math.radians(15.0)
                safe = (t_td is not None and t - t_td >= SETTLE and not body and not ramp and soft and level and tilt < math.radians(10.0) and float(car.hard[0]) < 0.03)
                on_line = y_td is not None and abs(y_td) < LINE_TOL and h_td < math.radians(LINE_DEG)
                landed = safe and on_line
                why = ("landed" if landed else "off the line" if safe else "hit the ramp" if ramp else "on its side or roof" if body or (t_td is not None and not level)
                       else "hard landing" if t_td is not None else "time")
                result = {"landed": landed, "text": why, "vz": None if vz_td is None else round(vz_td, 2),
                          "tilt": None if tilt_td is None else round(math.degrees(tilt_td), 1), "x": None if x_td is None else round(x_td, 1),
                          "air_used": None if air_go is None else round(air_go - float(car.air[0]), 1), "air_left": round(float(car.air[0]), 1),
                          "miss": None if y_td is None else round(abs(y_td), 2),     # how far off the white line it came down
                          "hdg": None if h_td is None else round(math.degrees(h_td), 1),  # and how askew to it (deg)
                          "safe": safe,
                          "crash": not landed and (body or ramp or (vz_td is not None and vz_td < -6.0)),
                          "impact": round(max(float(car.touched_body[0]), -(vz_td or 0.0)), 1)}
                events.append((t, ("Car landed! %.1f m/s at touchdown, %.0f kg of air used" % (-vz_td, result["air_used"] or 0)) if landed
                               else ("Down safely but off the line: %.1f m off it, %.0f deg askew" % (abs(y_td), math.degrees(h_td))) if safe else "Lost it: " + why))
                t_end = None if safe else t + 4.0                    # down safely: the run goes on until the car has stopped
        if result is not None and t_end is None and float(car.vel[0].norm()) < 0.15:
            t_end = t + 2.5
            events.append((t, "Stopped back on the start line, %.1f m from where it started, %.0f m from where it touched down" % (abs(float(car.pos[0, 0]) + float(S["runup"])), abs(float(car.pos[0, 0]) - (x_td or 0.0)))))
        if t_end is not None and t >= t_end:
            break
    return rows, sorted(events, key=lambda e: e[0]), result


def stunt_run(ctx):
    import random
    import re
    import os
    import threading
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
                if h.get("format") != "esasim.stunt-policy/4":
                    ctx.log("Skipped %s: made for an earlier version of the stunt (the car, its thrusters or the script up the ramp have changed since)." % f.name)
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
        ctx.log("The newest policy, %s, has not finished its lessons yet: flying %s, the newest that lands over the end." % (pols[0][5].name, pol.name))
    act, st = load_policy(pol.path)
    early = float(st.get("landed_full") or 0) < 0.3
    ctx.log("Thruster policy %s: trained %.0f minutes on %s, reached lesson %d of %d, %.0f%% of the attempts over the end of the ramp landed when saved." % (
        pol.name, float(st.get("minutes") or 0), st.get("device") or "?", int(st.get("stage") or 0) + 1, len(STAGES), 100 * float(st.get("landed_full") or 0)))
    if early:
        ctx.log("Warning: an early policy (it has not learned to land from over the end yet): expect the car to be lost. Train longer.")
    ch, ch_ev, ch_ctl = p.get("car_channel") or "car", p.get("events_channel") or "event", p.get("control_channel") or "control"
    num = lambda k, d: float(p.get(k)) if p.get(k) not in (None, "") else d   # noqa: E731
    S = {"exit": num("exit", 13.4), "wind": num("wind", 2.0), "speed": num("speed", 11.0), "runup": num("runup", 80.0)}
    auto = bool(p.get("auto_start"))
    per_run = int(p.get("runs_per_run") if p.get("runs_per_run") is not None else 1)
    info_track = track_info(Track())
    log = open(ctx.output("runs", "runs.csv"), "w", encoding="utf-8")
    log.write("run,seed,wind,exit_mps,landed,outcome,touchdown_mps,tilt_deg,air_used_kg,off_line_m,askew_deg,crash,impact_mps\n")

    def controls(wait):
        go = False
        for m in ctx.messages(ch_ctl, wait=wait):
            d = m.get("data") if isinstance(m, dict) else None
            if not isinstance(d, dict):
                continue
            c, v = str(d.get("control", "")).strip().lower(), d.get("value")
            if c in ("start", "go", "launch"):
                go = True
            elif c in ("exit", "wind", "speed"):
                try:
                    S[c] = float(v)
                except (TypeError, ValueError):
                    pass
        return go

    # the car parked at the start of the run-up, for the view while it waits
    parked = Car(1, "cpu")
    place_on_ground(parked, -S["runup"])
    pdrv = Driver(parked, torch.zeros(1), torch.zeros(1))
    for _ in range(100):
        parked.step(0.004)
    idle = {"format": "esasim.car/1", "kind": "idle", "ep": 0, "fields": FIELDS, "frames": [frame_of(parked, pdrv, 0)], "track": info_track,
            "status": "Ready: press Start", "phase": "On the start line"}
    ctx.telemetry(ch, idle)

    def prepare(S_=None, log_=True):
        """the next run, computed ahead (it takes a while): Start then goes straight to the countdown"""
        S_ = dict(S_ or S)
        seed_ = random.randrange(1, 10 ** 6)
        t_ = time.time()
        rows_, events_, res_ = stunt(act, dict(S_), seed_)
        if log_:
            ctx.log("Next run ready (%.0f m/s over the end, wind %.0f m/s): computed in %.1f s." % (S_["exit"], S_["wind"], time.time() - t_))
        return {"S": S_, "seed": seed_, "rows": rows_, "events": events_, "res": res_}

    ctx.telemetry(ch_ev, {"text": "Getting the stunt ready…"})
    ready = prepare()
    ctx.telemetry(ch_ev, {"text": "Ready: press Start" + (" (warning: an early policy, it will rarely land)" if early else "")})
    runs = 0
    while True:
        if not auto:
            go = controls(0.5)
            if ready is None or ready["S"] != S:      # a slider moved: the next run again, with it
                ready = prepare()
            if not go:
                continue
        auto = False
        if ready is None or ready["S"] != S:
            ready = prepare()
        runs += 1
        seed, rows, events, res = ready["seed"], ready["rows"], ready["events"], ready["res"]
        ready = None
        ctx.log("Run %d: up the ramp and over its end at %.0f m/s; wind %.0f m/s." % (runs, S["exit"], S["wind"]))
        base = {"format": "esasim.car/1", "kind": "run", "ep": runs, "fields": FIELDS, "track": info_track, "label": "Run %d" % runs, "wind": S["wind"]}
        # the whole run is computed already: it goes to the view at once, with the moment it starts (after a 3-2-1), and the
        # view plays it on its own clock: smooth whatever the network or the machines do. The events come in real time.
        lead = 4.0
        parts = list(chunks(rows, 200))
        for k, part, last in parts:
            item = dict(base, seq=k, frames=part, ahead=True, start_in=lead, phase="Countdown", clock=-3)
            if last:
                item["last"] = True
                item["result"] = res
            ctx.telemetry(ch, item)
        ctx.telemetry(ch_ev, {"text": "Run %d: 3, 2, 1…" % runs})
        t_go = time.time() + lead
        # while this one plays, the next run is computed alongside (with these settings; again if a slider moves)
        nxt = {}
        th = None
        if per_run == 0 or runs < per_run:
            S_now = dict(S)
            def _bg():
                try:
                    os.nice(10)      # (Linux: this thread only) the next run is worked out at a low priority, so a view on this machine plays smoothly
                except Exception:
                    pass
                nxt.update(prepare(S_now, False))
            th = threading.Thread(target=_bg, daemon=True)
            th.start()
        t0 = rows[0][0]
        for ev_t, ev_text in events:
            while True:
                wait = t_go + (ev_t - t0) - time.time()
                if wait <= 0:
                    break
                time.sleep(min(wait, 0.25))
                controls(0)
            ctx.telemetry(ch_ev, {"text": ev_text, "t": round(ev_t, 2)})
        while time.time() < t_go + (rows[-1][0] - t0) + 0.5:
            time.sleep(0.25)
            controls(0)
        log.write("%d,%d,%.1f,%.1f,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (runs, seed, S["wind"], S["exit"], res["landed"], res["text"], res["vz"], res["tilt"], res["air_used"], res["miss"], res.get("hdg"), res.get("crash"), res.get("impact")))
        log.flush()
        ctx.log(("Car landed! %.1f m/s at touchdown, %.0f kg of air used." % (-res["vz"], res["air_used"] or 0)) if res["landed"]
                else ("Down safely, but %.1f m off the line and %.0f deg askew." % (res["miss"], res["hdg"])) if res.get("safe") else "Lost the car: %s." % res["text"])
        if per_run > 0 and runs >= per_run:
            ctx.telemetry(ch_ev, {"text": "Run over: run the flow again for another one"})
            log.close()
            return {"runs": runs, "landed": res["landed"], "outcome": res["text"], "touchdown_mps": res["vz"], "air_used_kg": res["air_used"]}
        ctx.telemetry(ch, dict(idle, ep=runs + 1000))
        if th is not None:
            th.join()
        ready = nxt if nxt.get("S") == S else prepare()
        ctx.telemetry(ch_ev, {"text": "Ready: press Start for the next run"})


def run(ctx):
    what = str(ctx.params.get("mode") or "Learn the landing").lower()
    if what.startswith("run") or "stunt" in what:
        return stunt_run(ctx)
    return learn_run(ctx)
