# Ramp stunt

**A two-seat electric hypercar drives up a 21 m crescent ramp, upside down on its suction fans over the top, and off its end: it falls with its momentum, and its compressed-air thrusters roll it over sideways and set it down on the road. The thruster controller learns the landing by itself, from its sensors.**

![type](https://img.shields.io/badge/type-Autonomous%20vehicles-blue) ![language](https://img.shields.io/badge/language-Python-3776AB) ![runs on](https://img.shields.io/badge/runs%20on-NVIDIA%20%7C%20CPU-6f42c1) ![method](https://img.shields.io/badge/method-Reinforcement%20learning%20(PPO)-orange) [![made with](https://img.shields.io/badge/made%20with-EsaSim-0b7285)](https://esacode.com/)

## Overview

At a night event at a rocket test site, a car drove into a ramp that curves from the road into a wall and over into a ceiling about 21 m up. Suction fans under its floor held it there, upside down. It drove on over the top and off the end: falling, still moving forward, its cold-gas thrusters rolled it over sideways and put it down on the tarmac.

This project rebuilds that stunt with physics only. Nobody writes the landing: a neural network (the **policy**) decides 50 times a second how hard to blow each of the car's ten air nozzles, seeing the car only through its sensors, and learns it by trial and error (PPO), thousands of cars at once on the GPU. When it gets it wrong, the car crashes and breaks, with the sound to go with it.

## Flows

- **1 · Learn the landing**: thousands of cars at once learn to land, lesson by lesson. One attempt is replayed in 3D every few seconds, with charts of how often they land, the touchdown speed, the air used and how attempts end. The policy goes to the **stunt-policy** dataset; each run goes on from the furthest policy so far.
- **2 · The stunt**: press **Start**: the car drives up the ramp, over the top and off its end, and the newest policy that lands flies it down, live in 3D with all its sounds, a broadcast-style director camera, and chase, side and onboard cameras.

Both flows use the same block, **Ramp stunt**; its setting **What to do** picks **Learn the landing** or **Run the stunt**.

## The physics

| Part | What is simulated |
|---|---|
| **Car** | a rigid body, 1,950 kg plus 250 kg of air; 4.31 m long, 1.90 m wide, 2.82 m wheelbase; inertia from its shape |
| **Wheels** | stiff stunt suspension (spring, damper, bump stop), tyres whose grip saturates, all-wheel drive, brakes, front steering |
| **Body** | ten points (bumpers, sills, roof): anything but the tyres touching the road or the ramp is a crash |
| **Suction fans** | two under the floor, up to 2 g together, only while the skirts seal against the surface; spin up and down in a quarter second |
| **Thrusters** | ten cold-gas nozzles: four under the floor at the corners blowing down (lift; fired on one side only they roll the car), one on each flank blowing sideways, four in the tail blowing back for fast acceleration (not used in this stunt); none blows up; valves open in 25 ms; the air leaves the car, the regulator holds thrust until the tanks run low |
| **Sensors** | what the controller sees: an IMU (gyro with its bias, accelerometer, attitude), two laser rangefinders (one in the floor looking down, one in the roof looking up, to 40 m: one of them always sees the ground), and an RTK GNSS receiver (position and velocity, 10 times a second), each with its noise |
| **Ramp** | a 5 m wide concrete crescent: eases off the road (16 m radius), curves into a wall (9 m), bends over (10 m) to 168°, then runs straight for 8 m to its end about 20.7 m up |
| **Air** | drag and wind |

A scripted driver takes the car up the ramp: it drives in at about 30 km/h, the fans pull just hard enough to keep the tyres pressed (fully on when upside down), and it eases off over the top to its exit speed and drives off the end without stopping. The moment its front wheels pass the end of the deck the fans stop; from then on nothing holds it but physics, and the policy flies: it rolls the car over sideways (the nose kept level), slows the fall and sets it down, its forward speed carrying it some 20 m out over the road. On its wheels, it brakes.

## How it learns

By default the network gets a head start (**Warm start**): a simple hand-written autopilot, which reads the car's true state, flies a few thousand landings, and the network learns to copy it from its own sensors. Then the network flies itself and the autopilot shows what it would have done in the situations the network got into (a few rounds), so the copy also knows how to recover. After that, trial and error (PPO) takes over and improves on it: smoother, less air, more wind and speed variety. With **Warm start** off it learns everything from nothing, lesson by lesson: slower and less sure with these nozzles.

## The lessons

Learning starts with easy drops and moves to the real thing. A lesson opens once half the attempts of the one before land.

1. A short drop, nearly level.
2. On its side or nose down, 6-12 m up.
3. Upside down and moving, 12-20 m up.
4. Over the end of the ramp, as in the stunt (starts made by driving cars up the ramp).

An attempt counts as **landed** when the car touches the road on its tyres at under 3 m/s down and 15° of lean, nothing else touches, and it stands level a second later. Anything else touching (or a touchdown far too hard) is a crash: in 3D the car dents where it struck, wheels and panels come off, glass shatters and sparks fly.

## What you need

- A machine with the EsaSim worker and an NVIDIA GPU for training (a CPU works, much slower). Flow 2 runs on any machine.
- Python packages, installed by the worker the first time: `torch`.
- EsaSim with the 3D view at version 1.43 or later: it draws the ramp, the lights and the stunt's car (a licensed model shipped with the view), and makes the sounds. Another GLB or glTF car wired to the **Scene** input of the 3D views takes its place; separate wheel parts named `wheel…` turn, steer and ride on their springs.

## How to run it

1. Under **Compute**, check that a machine with an NVIDIA GPU is available for this project.
2. Run **1 · Learn the landing** (an hour by default; **Finish now** ends it early and keeps the policy). Each run goes on from the furthest policy in **stunt-policy**; run it again to learn more.
3. Run **2 · The stunt** and press **Start** (click in the view once for the sound). **Exit speed** and **Wind** change the next run.

## Settings worth knowing

- **Learn the landing**: **Warm start** (on), **Cars at once** (4,096 on a GPU), **Minutes at most** (60), **Stop at landing rate** (98 % of the attempts over the end), **Replay every** (12 s), **Start from scratch**.
- **The stunt**: **Exit speed** (m/s over the end, 6), **Wind** (m/s, 2), **Run-in speed** (8 m/s), **Policy file** (empty: the newest that lands).

## What you get

- `stunt_policy_<date>_<NN>pct.pt` in the **stunt-policy** dataset (NN: the share of attempts over the end that land). Policies of earlier versions of the stunt are skipped.
- **training.csv**: the learning curve; **runs.csv**: each stunt's outcome, touchdown speed, lean, air used and impact.

## Notes

- Public, approximate figures only; the ramp's shape follows photos of the event.
- The touchdown is decided by the contact physics, not by a rule: a hard landing bottoms the suspension or scrapes the body.

<!-- esasim: files of the project -->

---

An [EsaSim](https://esacode.com/) project of the workspace Gianluca De Novi's workspace (Autonomous vehicles).

### Blueprints in this repository

- [1 · Learn the landing](blueprints/1-learn-the-landing.json) · 11 blocks · version 3
- [2 · The stunt](blueprints/2-the-stunt.json) · 8 blocks · version 2

### Workspace blocks it uses

- [Ramp stunt](blocks/ws-ramp-stunt) · version 4

Keys pasted in blocks (secret settings) are not saved here. Datasets, results and the team stay in EsaSim.
