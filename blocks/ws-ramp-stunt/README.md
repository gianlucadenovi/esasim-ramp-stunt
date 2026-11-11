# Ramp stunt

A hypercar drives up a 21 m crescent ramp upside down on its suction fans, over the top and off its end; its cold-gas nozzles roll it over sideways and land it on the white line: the controller learns

What to do: Learn the landing (on the machine with the NVIDIA GPU) or Run the stunt (any machine). The physics: a rigid car (1,950 kg plus 250 kg of air) with stiff suspension, tyres, all-wheel drive and steering; ten body points that must never touch; two suction fans under the floor (up to 2 g, only while the skirts seal); ten cold-gas nozzles whose air leaves the car: four under the floor blowing down, a pair just behind the front wheels and a pair behind the rear wheels (lift; fired on one side only they roll the car), one on each flank blowing sideways, four in the tail blowing back (not used here); a 5 m wide concrete crescent ramp (16, 9 and 10 m arcs, then 8 m straight at 168 deg to its end about 20.7 m up); drag and wind. A scripted driver, as in the stunt: along the white line at Run-in speed, slowing as it comes to the ramp, the fans full on by its first slope; up the wall it slows almost to a stop just before it is upside down, then, held to the ceiling by the fans, accelerates hard and drives off the end at Exit speed (about 30 mph); as the front wheels pass the end the fans stop and it falls with its momentum. From then the policy flies it 50 times a second, seeing it only through its sensors: an IMU, two laser rangefinders (floor and roof, to 40 m) and an RTK GNSS receiver, each with noise. As it is flown (the warm start's autopilot, which the network copies first): it rolls over while it falls freely, brakes hard as late as the thrust allows (from about 13 m up) to a slow descent by about 9 ft, then settles gently onto its wheels. Landed: tyres on the tarmac under 3 m/s down and 15 deg of lean, within 0.5 m of the white line and 10 deg of its direction, nothing else touching, level a second later (down safely but further off: off the line, partly rewarded). Then it rolls on along the line and brakes to stop on the start line of the run-up. Learn the landing: with Warm start the network copies the autopilot from its own sensors (a few rounds, DAgger), then improves by trial and error (PPO) on the real starts mixed with easier ones; without it, four lessons from nothing. Telemetry: train (lesson, landed %, touchdown, off the line, air, outcomes), car (an attempt replayed in the 3D view every few seconds, esasim.car/1), event. The policy is written when learning ends (time limit, target, or Finish now) as stunt_policy_<date>_<NN>pct.pt: send it to a dataset wired to Policies, so each training goes on from the furthest one; policies of earlier versions of the stunt are skipped. Run the stunt: Start, then the whole stunt plays in the 3D view (1.47 or later), the next run computed while one plays.

An [EsaSim](https://esacode.com/) blueprint block: it runs on a machine, in Python (`backend.py`, `def run(ctx)`).

| | |
|---|---|
| Block | `ws-ramp-stunt` · version 8 |
| Workspace | Gianluca De Novi's workspace |
| Author | Gianluca De Novi |
| Runs on | Any machine, NVIDIA GPU |

## Inputs

| Port | Type | Description |
|---|---|---|
| `policy` Policies | Files (pt) | The stunt-policy dataset (stunt_policy_*.pt): learning goes on from the furthest one; the stunt flies the newest that lands Optional. |

## Outputs

| Port | Type | Description |
|---|---|---|
| `policy` Policy | Files (pt) | Learn the landing: stunt_policy_<date>_<NN>pct.pt (network, input scaling, lesson reached, share of attempts over the end that land on the line) |
| `curve` Learning curve | Table (csv) | Learn the landing: training.csv, minute by minute |
| `runs` Runs | Table (csv) | Run the stunt: runs.csv, each run's outcome, touchdown speed, lean, air used, distance off the line, how askew, and impact |

## Settings

| Key | Type | Default | Description |
|---|---|---|---|
| `mode` What to do | select | `"Learn the landing"` | Learn the landing: on the GPU, thousands of cars at once. Run the stunt: the whole stunt in real time with the newest policy |
| `accelerator` Accelerator | select | `"auto"` | Learning: auto takes the NVIDIA GPU when this PyTorch can use it |
| `warm_start` Warm start | bool | `true` | Learning from scratch: first copy a hand-written autopilot's landings on the line (from the car's own sensors), then improve by trial and error. Off: trial and error only, lesson by lesson |
| `cars` Cars at once | number | `4096` | Learning: parallel attempts (the CPU uses at most 1024) |
| `minutes` Minutes at most | number | `60` |  |
| `target` Stop at landing rate | number | `0.98` | Share of attempts over the end of the ramp that land on the line, to stop early (after a few minutes of trial and error at least) |
| `replay_every` Replay every (s) | number | `12` |  |
| `from_scratch` Start from scratch | bool | `false` | On: ignore the policies wired to Policies and learn anew |
| `seed` Seed | number | `1` |  |
| `exit` Exit speed (m/s) | number | `13.4` | The stunt: the speed it drives off the end of the ramp at, accelerating on the ceiling (13.4 m/s is 30 mph; about 13 m/s is the most it reaches there) |
| `wind` Wind (m/s) | number | `2` |  |
| `speed` Run-in speed (m/s) | number | `11` | How fast it drives in along the run-up (11 m/s is 40 km/h); it slows to 7 m/s at the foot of the ramp |
| `runup` Run-up (m) | number | `80` | Where the car starts, and stops again after landing: this far before the ramp |
| `policy_file` Policy file | text | `""` | Empty: the newest policy that lands over the end |
| `runs_per_run` Stunts per run | number | `1` | 0: keep waiting for Start after each stunt |
| `auto_start` Start by itself | bool | `false` |  |
| `channel` Training channel | text | `"train"` |  |
| `car_channel` Car channel | text | `"car"` |  |
| `events_channel` Events channel | text | `"event"` |  |
| `control_channel` Controls channel | text | `"control"` |  |

## Use it in EsaSim

In a workspace, open **Projects > Blocks > New block**, choose this kind of block, copy the ports and settings above and paste the code; or ask an AI assistant connected to EsaSim (MCP) to create it from this folder.
