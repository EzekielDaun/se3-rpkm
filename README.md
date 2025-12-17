# SE3-RPKM

A collection of spatial (DOF>=6) Redundant Parallel Kinematic Machines (RPKMs), with kinematics modeling in Lie groups, implemented in JAX.

Examples include redundancy resolution algorithms and simulation in MuJoCo and MJX.

## Installation

### Minimal

```bash
pip install git+https://github.com/EzekielDaun/se3-rpkm.git
```

### Development & Examples Try-out

```bash
git clone https://github.com/EzekielDaun/se3-rpkm.git && cd se3-rpkm
pixi install --all
```

## List of supported RPKMs:

- Redundant Stewart Platform
  - [x] $\mathrm{SE}(3) \times \mathrm{SO}(2)^3$ [^1] `pixi run se3so23-stewart`
  - [ ] $\mathrm{SE}(3) \times \mathrm{SO}(2)^2$ [^2]
  - [ ] $\mathrm{SE}(3) \times \mathbb{R}^3$ [^3]
- 3-SR Platform $(\mathrm{SE}(3) \times \mathrm{SO}(2)^3)$
  - [x] 3-PPPSR [^5] `pixi run se3so23-sr-platform-basic-continuous`
  - [x] 3-(R(RR-RRR)SR) [^4] `pixi run se3so23-sr-platform-rrr-serial`

[^1]: https://doi.org/10.1109/TRO.2016.2516025
[^2]: https://doi.org/10.1016/j.mechmachtheory.2022.105015
[^3]: https://doi.org/10.3390/act12030120
[^4]: https://doi.org/10.1109/TRO.2020.3043723
[^5]: https://doi.org/10.1007/978-3-031-95489-4_18

## Run Examples

1. Install with all features
2. Start a twist publisher
   Run `pixi run keyboard-twist-publisher` or `pixi run gamepad-twist-publisher` in a separate terminal to publish desired end-effector twists.
3. In another terminal, run a MuJoCo example
