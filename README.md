# Ramp stunt

**A two-seat electric hypercar drives up a 21 m crescent ramp, upside down on its suction fans over the top, and off its end: it falls with its momentum, and its compressed-air thrusters roll it over sideways and set it down on the road, on the white line and along it. The thruster controller learns the landing by itself, from its sensors.**

![type](https://img.shields.io/badge/type-Autonomous%20vehicles-blue) ![language](https://img.shields.io/badge/language-Python-3776AB) ![runs on](https://img.shields.io/badge/runs%20on-NVIDIA%20%7C%20CPU-6f42c1) ![method](https://img.shields.io/badge/method-Reinforcement%20learning%20(PPO)-orange) [![made with](https://img.shields.io/badge/made%20with-EsaSim-0b7285)](https://esacode.com/)

## Overview

At a night event at a rocket test site, a car drove into a ramp that curves from the road into a wall and over into a ceiling about 21 m up. Suction fans under its floor held it there, upside down. It drove on over the top and off the end: falling, still moving forward, its cold-gas thrusters rolled it over sideways and put it down on the tarmac, on the white line it had driven in on.

This project rebuilds that stunt with physics only. Nobody writes the landing: a neural network (the **policy**) decides 50 times a second how hard to blow each of the car's ten air nozzles, seeing the car only through its sensors, and learns it by trial and error (PPO), thousands of cars at once on the GPU. When it gets it wrong, the car crashes and breaks, with the sound to go with it.

## Flows

- **1 · Learn the landing**: thousands of cars at once learn to land, lesson by lesson. One attempt is replayed in 3D every few seconds, with charts of how often they land on the line, the touchdown speed, the distance off the line and how attempts end. The policy goes to the **stunt-policy** dataset; each run goes on from the furthest policy so far.
- **2 · The stunt**: press **Start**: the car drives up the ramp, over the top and off its end, and the newest policy that lands flies it down, live in 3D with all its sounds and a looping soundtrack (the **Music** button turns it off), a broadcast-style director camera, chase and side cameras, real reflections on the paint and glass, and a **Telemetry** window of its own charting every sensor of the car through the run, in step with the stunt. A safe landing on the line ends with **Car landed!**

Both flows use the same block, **Ramp stunt**; its setting **What to do** picks **Learn the landing** or **Run the stunt**.

## The physics

| Part | What is simulated |
|---|---|
| **Car** | a rigid body, 1,950 kg plus 250 kg of air; 4.31 m long, 1.90 m wide, 2.82 m wheelbase; inertia from its shape |
| **Wheels** | very stiff stunt suspension (spring, damper, bump stop: it carries 3 g and more on the ramp), tyres whose grip saturates, all-wheel drive, brakes, front steering |
| **Body** | ten points (bumpers, sills, roof): anything but the tyres touching the road or the ramp is a crash |
| **Suction fans** | two under the floor, full on as the car comes to the ramp (at 100% by its first slope); up to 2 g together, only while the skirts seal against the surface; spin up and down in a quarter second |
| **Thrusters** | ten cold-gas nozzles: four under the floor blowing down, a pair just behind the front wheels and a pair behind the rear wheels (lift; fired on one side only they roll the car; the rear pair opens less, so lift and roll do not pitch the car), one on each flank blowing sideways (right above the front pair, level with the tops of the wheels), four in the tail blowing back for fast acceleration (not used in this stunt); none blows up; valves open in 25 ms; the air leaves the car, the regulator holds thrust until the tanks run low |
| **Sensors** | what the controller sees: an IMU (gyro with its bias, accelerometer, attitude), two laser rangefinders (one in the floor looking down, one in the roof looking up, to 40 m: one of them always sees the ground), and an RTK GNSS receiver (position and velocity, 10 times a second), each with its noise |
| **Ramp** | a 5 m wide concrete crescent: eases off the road (16 m radius), curves into a wall (9 m), bends over (10 m) to 168°, then runs straight for 8 m to its end about 20.7 m up |
| **Air** | drag and wind |

A scripted driver takes the car up the ramp, as in the stunt: it drives in at about 40 km/h along the white line and slows as it comes to the ramp, the fans full on by its first slope. Up the wall the climb takes its speed: it nearly stops just before it is upside down; then, held to the ceiling by the fans, it accelerates hard and drives off the end at about 30 mph. The moment its front wheels pass the end of the deck the fans stop; from then on nothing holds it but physics, and the policy flies: it rolls the car over sideways while it falls freely (the nose kept level), brakes hard as late as the thrust allows (from about 13 m up) so that it is down to a slow descent by about 9 ft, and settles gently onto its wheels on the white line, pointing along it, some 50 m out over the road. On its wheels it rolls on along the line and brakes to stop where it started, back on the start line. (Braking only from 9 ft would take about 7 g of thrust: falling from the end, the car reaches 9 ft at some 19 m/s; its thrusters give about 1.75 g.)

The line is the hard part. The four down nozzles only blow down, so while the car is on its side, halfway through the roll, they push it sideways; and pitching while it rolls fast makes it come out of the roll turning (the gyroscopic coupling). The controller has to roll it with short kicks and coast in between, keep the heading while the car is on its side (there the pitch couple turns it about the vertical), and once upright steer back onto the line by banking slightly. The flank nozzles push it sideways but also turn it (they sit a metre ahead of the centre of mass).

## How it learns

By default the network gets a head start (**Warm start**): a hand-written autopilot (it rolls with kicks, holds the heading through the roll, banks back onto the line), which reads the car's true state, flies a few thousand landings, and the network learns to copy it from its own sensors. Then the network flies itself and the autopilot shows what it would have done in the situations the network got into (a few rounds), so the copy also knows how to recover. After that, trial and error (PPO) takes over and improves on it: smoother, less air, more wind and speed variety. With **Warm start** off it learns everything from nothing, lesson by lesson: slower and less sure with these nozzles.

## The lessons

Learning starts with easy drops and moves to the real thing. A lesson opens once half the attempts of the one before land.

1. A short drop, nearly level.
2. On its side or nose down, 6-12 m up.
3. Upside down and moving, 12-20 m up.
4. Over the end of the ramp, as in the stunt (starts made by driving cars up the ramp).

An attempt counts as **landed** when the car touches the road on its tyres at under 3 m/s down and 15° of lean, within 0.5 m of the white line and 10° of its direction, nothing else touches, and it stands level a second later. Down safely but further off counts as **off the line** (partly rewarded). Anything else touching (or a touchdown far too hard) is a crash: in 3D the car dents where it struck, wheels and panels come off, glass shatters and sparks fly.

In 3D the stunt is played smooth: each run is computed whole before it starts and sent to the view at once, which plays it on its own clock after a 3-2-1; the next run is computed while one plays, so **Start** answers at once.

## What you need

- A machine with the EsaSim worker and an NVIDIA GPU for training (a CPU works, much slower). Flow 2 runs on any machine.
- Python packages, installed by the worker the first time: `torch`.
- EsaSim with the 3D view at version 1.47 or later: it draws the ramp, the lights and the stunt's car (a licensed model shipped with the view), makes the sounds and plays the soundtrack (shipped with the view), and charts the telemetry. Another GLB or glTF car wired to the **Scene** input of the 3D views takes its place; separate wheel parts named `wheel…` turn, steer and ride on their springs.

## How to run it

1. Under **Compute**, check that a machine with an NVIDIA GPU is available for this project.
2. Run **1 · Learn the landing** (an hour by default; **Finish now** ends it early and keeps the policy). Each run goes on from the furthest policy in **stunt-policy**; run it again to learn more.
3. Run **2 · The stunt** and press **Start** (click in the view once for the sound). **Exit speed** and **Wind** change the next run. The **Telemetry** window shows the car's sensors on charts as the stunt plays; **Music** turns the soundtrack off; **Sensors** shows or hides the numbers, the air system panel and the laser beams; **Record** saves a video with the sound, the soundtrack and what is on screen.

## Settings worth knowing

- **Learn the landing**: **Warm start** (on), **Cars at once** (4,096 on a GPU), **Minutes at most** (60), **Stop at landing rate** (98 % of the attempts over the end), **Replay every** (12 s), **Start from scratch**.
- **The stunt**: **Exit speed** (m/s off the end, 13.4: 30 mph; about 13 is the most the ceiling allows), **Wind** (m/s, 2), **Run-in speed** (11 m/s), **Run-up** (80 m: where it starts and stops), **Policy file** (empty: the newest that lands).

## What you get

- `stunt_policy_<date>_<NN>pct.pt` in the **stunt-policy** dataset (NN: the share of attempts over the end that land on the line). Policies of earlier versions of the stunt are skipped.
- **training.csv**: the learning curve; **runs.csv**: each stunt's outcome, touchdown speed, lean, air used, distance off the line, how askew, and impact.

## Notes

- Public, approximate figures only; the ramp's shape follows photos of the event.
- The touchdown is decided by the contact physics, not by a rule: a hard landing bottoms the suspension or scrapes the body.

<!-- esasim: files of the project -->

---

An [EsaSim](https://esacode.com/) project of the workspace Gianluca De Novi's workspace (Autonomous vehicles).

### Blueprints in this repository

- [1 · Learn the landing](blueprints/1-learn-the-landing.json) · 11 blocks · version 5
- [2 · The stunt](blueprints/2-the-stunt.json) · 6 blocks · version 5

### Workspace blocks it uses

- [Ramp stunt](blocks/ws-ramp-stunt) · version 9

Keys pasted in blocks (secret settings) are not saved here. Datasets, results and the team stay in EsaSim.
