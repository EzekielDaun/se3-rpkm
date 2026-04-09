# SE3-RPKM

`SE3-RPKM` is a collection of spatial (DOF >= 6) redundant parallel kinematic machines (RPKMs) with differentiable Lie group kinematics in `JAX`. The repo also includes redundancy-resolution examples and `MuJoCo`/`MJX` simulations.

## Highlights

- Lie group modeling for $\mathrm{SE}(3)$ and related redundant structures, built on [jaxlie](https://github.com/brentyi/jaxlie).
- Differentiable kinematics suitable for optimization and learning.
- Ready-to-run examples with teleoperation utilities.

## Installation

### Minimal

```bash
pip install git+https://github.com/EzekielDaun/se3-rpkm.git
```

### Development & Examples

This project uses [`Pixi`](https://pixi.sh/) as the development package manager.

Clone this repository and choose the environment you need:

```bash
git clone https://github.com/EzekielDaun/se3-rpkm.git && cd se3-rpkm
pixi install -e dev
```

- `pixi install -e dev`: development tools plus non-Qt examples, including `linux-aarch64`
- `pixi install -e desktop`: full desktop environment with Qt examples on `linux-64`, `osx-arm64`, and `win-64`
- `pixi install -e examples`: non-Qt examples only
- `pixi install -e examples-qt`: Qt examples only on supported desktop platforms

## Run Examples

1. Install with `Pixi`, for example `pixi install -e examples`.
2. Start a twist publisher in a separate terminal:
   - `pixi run -e examples keyboard-twist-publisher`
   - `pixi run -e examples gamepad-twist-publisher`
3. Run a `MuJoCo` example from the supported list below, for example:
   - `pixi run -e examples se3so23-stewart`
4. Run the Qt launcher on supported desktop platforms:
   - `pixi run -e examples-qt qopengl-multi-mechanism-launcher`

## Supported RPKMs

### Redundant Stewart Platforms

- [^1] $\mathrm{SE}(3) \times \mathrm{SO}(2)^3$ `pixi run -e examples se3so23-stewart`
  ![se3so23 (6+3) DoF redundant Stewart platform](./assets/se3so23-stewart.gif)
- [^2] $\mathrm{SE}(3) \times \mathrm{SO}(2)^2$ `pixi run -e examples se3so22-stewart`
  ![se3so22 (6+2) DoF redundant Stewart platform](./assets/se3so22-stewart.gif)
- [^3] $\mathrm{SE}(3) \times \mathbb{R}^3$ `pixi run -e examples se3r3-stewart`
  ![se3r3 (6+3) DoF redundant Stewart platform](./assets/se3r3-stewart.gif)

### 3-SR Platforms $(\mathrm{SE}(3) \times \mathrm{SO}(2)^3)$

- [^4] 3-(R(RR-RRR)SR) `pixi run -e examples se3so23-sr-platform-rrr-serial`
  ![3-(R(RR-RRR)SR) (6+3) DoF 3-SR platform](./assets/se3so23-sr-platform-rrr-serial.gif)
- [^5] 3-PPPSR `pixi run -e examples se3so23-sr-platform-basic-continuous`
  ![3-PPPSR (6+3) DoF 3-SR platform](./assets/se3so23-sr-platform-basic-continuous.gif)

### Miscellaneous

- [^6] 5-PSS-S-4PSS $(\mathrm{SE}(3) \times \mathrm{SO}(3))$ `pixi run -e examples se3so3-5pss-s-4pss`
  ![5-PSS-S-4PSS (6+3) DoF manipulator](./assets/se3so3-5pss-s-4pss.gif)

[^1]: https://doi.org/10.1109/TRO.2016.2516025
[^2]: https://doi.org/10.1016/j.mechmachtheory.2022.105015
[^3]: https://doi.org/10.3390/act12030120
[^4]: https://doi.org/10.1109/TRO.2020.3043723
[^5]: https://doi.org/10.1007/978-3-031-95489-4_18
[^6]: https://doi.org/10.1007/978-3-031-95489-4_10

## Build Your Own RPKM

Define an implicit kinematic constraint function that maps the extended task variables (i.e., $\mathrm{SE}(3)$ pose plus redundancy) and joint variables to zero: $f(x, q) = \mathbf{0}$.

Implement this function in Python and override the [abstract method](./src/se3_rpkm/lie_group_kinematics.py#L36) with `pytree` data structures. Tangent space Jacobians are then computed automatically.
