# Ramp stunt

A hypercar drives up a 21 m crescent ramp upside down on its suction fans, over the top and off its end; its cold-gas nozzles roll it over sideways and land it on the white line: the controller learns

What to do: Learn the landing (on the machine with the NVIDIA GPU) or Run the stunt (any machine). The physics: a rigid car (1,950 kg plus 250 kg of air) with suspension, tyres, all-wheel drive and steering; ten body points that must never touch; two suction fans under the floor (switched on as the car reaches the ramp; up to 2 g, only while the skirts seal); ten cold-gas nozzles whose air leaves the car: four under the floor at the corners blowing down (lift; fired on one side only they roll the car), one on each flank blowing sideways, four in the tail blowing back (fast acceleration; not used in this stunt), none blowing up; a 5 m wide concrete crescent ramp (16 m, 9 m and 10 m arcs, then 8 m straight at 168 deg to its end about 20.7 m up); drag and wind. A scripted driver takes the car along the white line and up the ramp, upside down over the top on the fans, easing off to Exit speed, and drives off the end without stopping; as the front wheels pass the end the fans stop and the car falls with its momentum. From then the policy flies it 50 times a second with the bottom and flank nozzles, seeing it only through its sensors: an IMU (gyro with bias, accelerometer, attitude), two laser rangefinders (floor looking down, roof looking up, to 40 m) and an RTK GNSS receiver (10 Hz), each with noise. The goal: down on the white line and along it. Landed: tyres on the tarmac under 3 m/s down and 15 deg of lean, within 0.5 m of the line and 10 deg of its direction, nothing else touching, level a second later (down safely but further off: off the line, partly rewarded). On its wheels it rolls on along the line with the speed it landed with and brakes to stop where it started, on the start line of the run-up. Learn the landing: with Warm start (on by default) the network first copies a hand-written autopilot (which reads the true state: it rolls the car with short kicks and coasts through the roll, keeps the heading while the car is on its side, banks back onto the line) from its own sensors, then flies itself while the autopilot shows what it would have done (a few rounds), then improves by trial and error (PPO) on the real starts mixed with easier ones. Without it, four lessons from nothing (a short drop nearly level; on its side or nose down 6-12 m up; upside down and moving 12-20 m up; over the end of the ramp), a lesson opening when half the attempts of the one before land. Telemetry: train (lesson, landed % and landed_full % on the line over the end, touchdown_mps, off_line_m, air_kg, attempts, steps_per_s, minutes, outcomes.*), car (one attempt replayed in the 3D view every Replay every seconds, esasim.car/1, with impacts, fan spin and the laser readings), event. The policy is written when learning ends (time limit, target, or the button Finish now on the control channel) as stunt_policy_<date>_<NN>pct.pt: send it to a dataset and wire that dataset to Policies, so each training goes on from the furthest one (policies of earlier versions of the stu

An [EsaSim](https://esacode.com/) blueprint block: it runs on a machine, in Python (`backend.py`, `def run(ctx)`).

| | |
|---|---|
| Block | `ws-ramp-stunt` · version 6 |
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
| `exit` Exit speed (m/s) | number | `6` | The stunt: the speed it drives over the end of the ramp at, and falls with |
| `wind` Wind (m/s) | number | `2` |  |
| `speed` Run-in speed (m/s) | number | `8` |  |
| `runup` Run-up (m) | number | `60` | Where the car starts, and stops again after landing: this far before the ramp |
| `policy_file` Policy file | text | `""` | Empty: the newest policy that lands over the end |
| `runs_per_run` Stunts per run | number | `1` | 0: keep waiting for Start after each stunt |
| `auto_start` Start by itself | bool | `false` |  |
| `channel` Training channel | text | `"train"` |  |
| `car_channel` Car channel | text | `"car"` |  |
| `events_channel` Events channel | text | `"event"` |  |
| `control_channel` Controls channel | text | `"control"` |  |

## Use it in EsaSim

In a workspace, open **Projects > Blocks > New block**, choose this kind of block, copy the ports and settings above and paste the code; or ask an AI assistant connected to EsaSim (MCP) to create it from this folder.
