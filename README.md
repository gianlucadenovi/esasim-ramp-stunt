# Ramp stunt

**A two-seat electric hypercar drives up a 21 m crescent ramp, hangs upside down on its suction fans, drives off the lip, and its compressed-air thrusters roll it upright and set it down on the road. The thruster controller learns the landing by itself.**

![type](https://img.shields.io/badge/type-Autonomous%20vehicles-blue) ![language](https://img.shields.io/badge/language-Python-3776AB) ![runs on](https://img.shields.io/badge/runs%20on-NVIDIA%20%7C%20CPU-6f42c1) ![method](https://img.shields.io/badge/method-Reinforcement%20learning%20(PPO)-orange) [![made with](https://img.shields.io/badge/made%20with-EsaSim-0b7285)](https://esacode.com/)

## Overview

At a night event at a rocket test site, a car drove into a ramp that curves from the road into a wall and over into a ceiling about 21 m up. Suction fans under its floor held it there, upside down, even standing still. Then it drove off the lip, and its cold-gas thrusters rolled it upright, hovered it for a moment and put it down on the tarmac.

This project rebuilds that stunt with physics only. Nobody writes the landing: a neural network (the **policy**) decides 50 times a second how hard to push and which way to turn the car with its ten air nozzles, and learns it by trial and error (PPO), thousands of cars at once on the GPU.

## Flows

- **1 · Learn the landing**: thousands of cars at once learn to land, lesson by lesson. One attempt is replayed in 3D every few seconds, with charts of how often they land, the touchdown speed, the air used and how attempts end. The policy goes to the **stunt-policy** dataset; each run goes on from the furthest policy so far.
- **2 · The stunt**: press **Start**: the car drives up the ramp, holds upside down, drives off the lip, and the newest policy that lands flies it down, live in 3D, with a broadcast-style director camera, chase, side and onboard cameras.

Both flows use the same block, **Ramp stunt**; its setting **What to do** picks **Learn the landing** or **Run the stunt**.

## The physics

| Part | What is simulated |
|---|---|
| **Car** | a rigid body, 1,950 kg plus 250 kg of air; 4.31 m long, 1.90 m wide, 2.82 m wheelbase; inertia from its shape |
| **Wheels** | stiff stunt suspension (spring, damper, bump stop), tyres whose grip saturates, all-wheel drive, brakes, front steering |
| **Body** | ten points (bumpers, sills, roof): anything but the tyres touching the road or the ramp is a crash |
| **Suction fans** | two under the floor, up to 2 g together, only while the skirts seal against the surface; spin up and down in a quarter second |
| **Thrusters** | ten cold-gas nozzles: four blowing down at the corners, four blowing up, two sideways in the nose; valves open in 25 ms; the air leaves the car, the regulator holds thrust until the tanks run low |
| **Ramp** | a 5 m wide concrete crescent: eases off the road (16 m radius), curves into a wall (9 m), bends over (10 m) to 168°, then runs straight for 8 m to its lip about 20.7 m up |
| **Air** | drag and wind |

A scripted driver takes the car up the ramp: it drives in at about 30 km/h, the fans pull just hard enough to keep the tyres pressed (fully on when upside down), it stops a few metres before the lip, waits, then drives off. From the moment the front wheels leave the lip, the policy flies.

## The lessons

Learning starts with easy drops and moves to the real thing. A lesson opens once half the attempts of the one before land.

1. A short drop, nearly level.
2. On its side or nose down, 4-10 m up.
3. Upside down, 10-19 m up.
4. Off the lip of the ramp, as in the stunt (starts made by driving cars up the ramp).

An attempt counts as **landed** when the car touches the road on its tyres at under 3 m/s down and 15° of lean, nothing else touches, and it stands level a second later.

## What you need

- A machine with the EsaSim worker and an NVIDIA GPU for training (a CPU works, much slower). Flow 2 runs on any machine.
- Python packages, installed by the worker the first time: `torch`.
- For the car's look: a GLB or glTF car model you are licensed to use, in a dataset, wired to the **Scene** input of the 3D views next to the telemetry (without one, a car drawn by the view is used). Separate wheel parts named `wheel…` turn, steer and ride on their springs.
- EsaSim with the 3D view at version 1.42 or later (it draws the ramp, the lights and the car).

## How to run it

1. Under **Compute**, check that a machine with an NVIDIA GPU is available for this project.
2. Run **1 · Learn the landing** (an hour by default; **Finish now** ends it early and keeps the policy). The first run creates the **stunt-policy** dataset; then add a **Dataset** node for it and wire it to the block's **Policies** input, so each run goes on from the furthest policy.
3. In **2 · The stunt**, check that the dataset node points at **stunt-policy**, run it and press **Start**. **Hold**, **Exit speed** and **Wind** change the next run.

## Settings worth knowing

- **Learn the landing**: **Cars at once** (4,096 on a GPU), **Minutes at most** (60), **Stop at landing rate** (90 % off the lip), **Replay every** (12 s), **Start from scratch**.
- **The stunt**: **Hold** (seconds upside down, 3), **Exit speed** (m/s off the lip, 7), **Wind** (m/s, 2), **Run-in speed** (8 m/s), **Hold before the lip** (7 m), **Policy file** (empty: the newest that lands).

## What you get

- `stunt_policy_<date>_<NN>pct.pt` in the **stunt-policy** dataset (NN: the share of attempts off the lip that land).
- **training.csv**: the learning curve; **runs.csv**: each stunt's outcome, touchdown speed, lean and air used.

## Notes

- Public, approximate figures only; the ramp's shape follows photos of the event.
- The touchdown is decided by the contact physics, not by a rule: a hard landing bottoms the suspension or scrapes the body.

<!-- esasim: files of the project -->

---

An [EsaSim](https://esacode.com/) project of the workspace Gianluca De Novi's workspace (Autonomous vehicles).

### Blueprints in this repository

- [1 · Learn the landing](blueprints/1-learn-the-landing.json) · 11 blocks · version 1
- [2 · The stunt](blueprints/2-the-stunt.json) · 8 blocks · version 1

### Workspace blocks it uses

- [Ramp stunt](blocks/ws-ramp-stunt) · version 3

Keys pasted in blocks (secret settings) are not saved here. Datasets, results and the team stay in EsaSim.
