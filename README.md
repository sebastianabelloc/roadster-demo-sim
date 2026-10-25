# roadster-demo-sim

A 6-DOF simulation of the Tesla Roadster cold-gas hop, with my own guidance, navigation and control, rendered in 3D.

<!-- drop the video here (edit this file on GitHub and drag renders/roadster_hop.mp4 into the editor) -->

When I saw the demo of the new Roadster lifting off on cold-gas thrusters, doing a full barrel roll over a lineup of cars and landing softly, I wanted to know what it would take to fly that manoeuvre. So I wrote the whole thing from scratch in Python: the rigid body dynamics with quaternions, the thruster model, the attitude controller, the control allocation between the nozzles and the guidance for every phase of the hop.

Everything you see in the video comes from the simulation. Python runs the closed loop and saves the trajectory (position, attitude and the throttle of each nozzle at every time step). Blender only reads that file and draws it, it does not add any motion to the car.

## The manoeuvre

1. **Powered take-off**: the car leaves the ground with zero vertical speed and the 4 bottom nozzles push it up at 1.9 times its weight until it climbs at 7 m/s.
2. **Barrel roll**: a full 360° roll in 1.5 s while it flies over the lineup, apex at 7 m.
3. **Braking and hover**: the bottom nozzles brake the fall, the car hovers at 0.5 m for 0.3 s.
4. **Landing**: slow descent at 0.6 m/s until touchdown, level, about 35 m after take-off.

## How it works

### Dynamics

The car is a rigid body of 1700 kg (the target weight given by Tesla), approximated as a 4.2 x 2.0 x 1.2 m box for the inertia. The state has 13 components: position, velocity, attitude quaternion (w, x, y, z) and angular velocity in the body frame.

$$\dot{\mathbf{r}} = \mathbf{v} \qquad \dot{\mathbf{v}} = \mathbf{g} + \frac{1}{m}\, q \otimes \mathbf{F}_b \otimes q^*$$

$$\dot{q} = \tfrac{1}{2}\, q \otimes (0, \boldsymbol{\omega}) \qquad I\dot{\boldsymbol{\omega}} = \boldsymbol{\tau} - \boldsymbol{\omega} \times I\boldsymbol{\omega}$$

It is integrated with RK45 (`scipy.integrate.solve_ivp`) and stops with an event when the car touches the ground.

### Thrusters

Six cold-gas nozzles, placed like on the real car: 4 under the floor (10 kN each, pushing up) and 1 on each side near the front wheels (3 kN, pushing sideways). None of them can push the car down. The force and torque of each nozzle come straight from its position and direction:

$$\boldsymbol{\tau} = \sum_i \mathbf{r}_i \times \mathbf{F}_i$$

### Control

The controller runs at 50 Hz and holds its command between samples, like a real flight computer.

- **Attitude**: PD on the quaternion error plus a feedforward of the torque the roll profile needs, so the PD only has to correct small errors.

$$q_e = q_{ref}^* \otimes q \qquad \boldsymbol{\tau}_{cmd} = -k_p\, \mathbf{q}_{e,vec} - k_d\,(\boldsymbol{\omega} - \boldsymbol{\omega}_{ref}) + I\,\boldsymbol{\alpha}_{ref}$$

- **Control allocation**: the 4 bottom nozzles work like the motors of a drone. All together give the vertical force, more thrust on one side gives roll, more at the front or rear gives pitch. They are throttled between 0 and 1 and can only push, so the common part is raised when needed to keep every nozzle positive. The side nozzles are on/off with a deadband and handle yaw.

$$T_i = \frac{F}{4} + \frac{\tau_x\, s_{y,i}}{4\,b} - \frac{\tau_y\, s_{x,i}}{4\,a}$$

### Guidance

- **Take-off**: full lift (1.9 g) until the vertical speed reaches `v_launch`, then engine cut.
- **Roll**: smooth cosine profile from 0 to -360°, with zero rate at the start and at the end.

$$\phi_{ref}(s) = -\pi\,(1 - \cos \pi s), \qquad s = \frac{t - t_0}{T}$$

- **Braking**: vertical speed reference that can still stop at the hover height with a fixed deceleration, $v_{ref} = -\sqrt{2\,a\,(z - z_{hover})}$, then hover and a constant slow descent.
- **Lateral position**: outer loop like a drone. A small extra roll angle tilts the floor thrust to bring the car back to the centre line of the runway.

## Results

| | |
|---|---|
| Apex | 7.0 m |
| Flight time | 4.9 s |
| Touchdown vertical speed | -0.6 m/s |
| Tilt at touchdown | < 3° |
| Max lateral drift | 1.2 m |

## Running it

```bash
pip install -r requirements.txt
python -m pytest -q
python scripts/export_trajectory.py renders/trajectory.npz
python scripts/preview_trajectory.py renders/trajectory.npz
```

`preview_trajectory.py` draws the trajectory in a few seconds (add `--anim` for a 2D video), so you can check the physics before the 3D render. The render needs Blender (I used 5.2):

```bash
blender -b -P blender/render_hop.py -- renders/trajectory.npz renders/frames --res 1920 1080 --engine EEVEE --samples 64
python scripts/compose_video.py renders/trajectory.npz renders/frames renders/roadster_hop.mp4
```

## Repo structure

```
roadster/       quaternions, dynamics, thrusters, control, guidance and the closed-loop simulation
scripts/        export the trajectory, quick preview, video composition with the HUD
blender/        3D scene that reads the trajectory, model credits
tests/          unit and closed-loop tests (pytest)
```

## Simplifications

This is a quick personal project, not a model of the real car. Some things I left out on purpose:

- Constant mass: a cold-gas system loses mass as it fires, which would raise the acceleration a bit over time.
- Ideal nozzles: the thrust changes instantly, no valve delay and no pressure drop as the tank empties.
- No aerodynamics and no ground effect.
- Perfect navigation: the controller reads the true state, there are no sensors or estimation.
- Nozzle positions are my own estimates from the demo footage. The thrust of each nozzle is calculated by the controller, up to 10 kN for each floor nozzle and 3 kN for each side nozzle.

## Thrust and gas needed

I estimated from the demo footage that the roll takes about 1.5 s. Doing a full roll that fast needs a large difference in thrust between the left and right floor nozzles, on top of the lift. With a 1700 kg car, the four floor nozzles need about 8.3 kN each at take-off (1.9 times the weight in total), and the whole hop needs an impulse of about 88 kN·s. With nitrogen as the cold gas (specific impulse around 70 s) that is about 130 kg of gas for a five-second flight. Both numbers fit within what Tesla has said the Rocket version carries: about 90 kN of total thrust from six nozzles and 600 L of nitrogen at around 690 bar, roughly 250 to 300 kg of gas. The mass of 1700 kg is the target weight given by Tesla, which has not published a final curb weight yet.

## Credits

3D models under CC BY 4.0, see [blender/CREDITS.md](blender/CREDITS.md). Not affiliated with or endorsed by Tesla.
