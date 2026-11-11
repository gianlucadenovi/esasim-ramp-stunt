# Ramp stunt

A hypercar drives up a 21 m crescent ramp, hangs upside down on its suction fans, drives off the lip, and ten cold-gas nozzles land it: learn the landing by PPO on the GPU, or run the stunt live.

What to do: Learn the landing (on the machine with the NVIDIA GPU) or Run the stunt (any machine). The physics: a rigid car (1,950 kg plus 250 kg of air) with suspension, tyres, all-wheel drive and steering; ten body points that must never touch; two suction fans under the floor (up to 2 g, only while the skirts seal); ten cold-gas nozzles (four blowing down, four up, two sideways in the nose) whose air leaves the car; a 5 m wide concrete crescent ramp (16 m, 9 m and 10 m arcs, then 8 m straight at 168 deg to its lip about 20.7 m up); drag and wind. A scripted driver takes the car up the ramp (the fans pull just hard enough, fully on upside down), stops Hold before the lip metres before it, waits Hold seconds, then drives off at Exit speed; from the moment the front wheels leave the lip the policy flies it 50 times a second. Learn the landing: four lessons (a short drop nearly level; on its side or nose down 4-10 m up; upside down 10-19 m up; off the lip, starts made by driving cars up the ramp); a lesson opens when half the attempts of the one before land. Landed: tyres on the tarmac under 3 m/s down and 15 deg of lean, nothing else touching, level a second later. Telemetry: train (lesson, landed %, landed_full % off the lip, touchdown_mps, air_kg, attempts, steps_per_s, minutes, outcomes.*), car (one attempt replayed in the 3D view every Replay every seconds, esasim.car/1), event (lessons). The policy is written when learning ends (time limit, target, or the button Finish now on the control channel) as stunt_policy_<date>_<NN>pct.pt: send it to a dataset and wire that dataset to Policies, so each training goes on from the furthest one. Run the stunt: wire the same dataset to Policies; the newest policy that lands off the lip flies (or the one named in Policy file). The car waits on the start line; the button Start (control channel) runs the stunt in real time, streamed on the car channel for a 3D view; Hold, Exit and Wind sliders change the next run; runs.csv records each one.

An [EsaSim](https://esacode.com/) blueprint block: it runs on a machine, in Python (`backend.py`, `def run(ctx)`).

| | |
|---|---|
| Block | `ws-ramp-stunt` · version 1 |
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
| `policy` Policy | Files (pt) | Learn the landing: stunt_policy_<date>_<NN>pct.pt (network, input scaling, lesson reached, share of attempts off the lip that land) |
| `curve` Learning curve | Table (csv) | Learn the landing: training.csv, minute by minute |
| `runs` Runs | Table (csv) | Run the stunt: runs.csv, each run's outcome, touchdown speed, lean and air used |

## Settings

| Key | Type | Default | Description |
|---|---|---|---|
| `mode` What to do | select | `"Learn the landing"` | Learn the landing: PPO on the GPU, thousands of cars at once. Run the stunt: the whole stunt in real time with the newest policy |
| `accelerator` Accelerator | select | `"auto"` | Learning: auto takes the NVIDIA GPU when this PyTorch can use it |
| `cars` Cars at once | number | `4096` | Learning: parallel attempts (the CPU uses at most 1024) |
| `minutes` Minutes at most | number | `60` |  |
| `target` Stop at landing rate | number | `0.9` | Share of attempts off the lip that land, to stop early |
| `replay_every` Replay every (s) | number | `12` |  |
| `from_scratch` Start from scratch | bool | `false` | On: ignore the policies wired to Policies and learn from lesson 1 |
| `seed` Seed | number | `1` |  |
| `hold` Hold (s) | number | `3` | The stunt: seconds upside down before driving off the lip |
| `exit` Exit speed (m/s) | number | `7` | The stunt: speed asked off the lip (slower than about 4 m/s scrapes the lip) |
| `wind` Wind (m/s) | number | `2` |  |
| `speed` Run-in speed (m/s) | number | `8` |  |
| `hold_back` Hold before the lip (m) | number | `7` | Where the car stops upside down, measured back from the lip |
| `runup` Run-up (m) | number | `60` |  |
| `policy_file` Policy file | text | `""` | Empty: the newest policy that lands off the lip |
| `runs_per_run` Stunts per run | number | `1` | 0: keep waiting for Start after each stunt |
| `auto_start` Start by itself | bool | `false` |  |
| `channel` Training channel | text | `"train"` |  |
| `car_channel` Car channel | text | `"car"` |  |
| `events_channel` Events channel | text | `"event"` |  |
| `control_channel` Controls channel | text | `"control"` |  |

## Use it in EsaSim

In a workspace, open **Projects > Blocks > New block**, choose this kind of block, copy the ports and settings above and paste the code; or ask an AI assistant connected to EsaSim (MCP) to create it from this folder.
