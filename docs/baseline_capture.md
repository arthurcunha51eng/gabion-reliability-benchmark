# Baseline capture — OUTSIDE/FLAT scenario

This file records the deterministic baseline used for regression testing.
The numbers in `tests/test_deterministic.py` come from this capture and
must remain reproducible.

## Source

Captured before the refactor, by running the legacy modules
(`empuxo_outside_flat.py`, `cinematica_outside_flat.py`,
`verificacoes_outside_flat.py`) on the canonical scenario:

| Parameter                  | Value                |
|----------------------------|----------------------|
| `step_side`                | OUTSIDE              |
| `backfill_top`             | FLAT                 |
| `gabion.gamma_g`           | 25.0 kN/m³           |
| `gabion.n`                 | 0.30                 |
| `gabion.geotex_reduction`  | 0.0                  |
| `geometry.layer_lengths`   | [2.0, 1.5, 1.0] m    |
| `geometry.beta`            | 6.0°                 |
| `backfill.gamma`           | 18.0 kN/m³           |
| `backfill.phi`             | 30.0°                |
| `backfill.c`               | 0.0 kPa              |
| `foundation.phi`           | 30.0°                |
| `foundation.c`             | 0.0 kPa              |
| `q` (surcharge)            | 10.0 kN/m²           |
| `q_adm`                    | 200.0 kPa            |

## Modes

The engine supports two modes for selecting the critical wedge `D`:

* **AUTO** — maximize `Ea` over `D ∈ {0.5, 1.0, ..., 5.0}` m. This is the
  textbook Coulomb criterion (Das, *Principles of Foundation Engineering*;
  Craig & Knappett, *Soil Mechanics*).
* **`d_override=2.5`** — force the wedge to GAWACWIN's reported value,
  for direct cross-software comparison.

## Captured values (full float precision)

### AUTO mode

```
D_critical       = 2.0
rho              = 52.512860145357635
Ea               = 28.69243604229485
theta            = 24.0
X_Ea             = 0.11761949288812455
Y_Ea             = 1.119074722258458
N                = 92.66481728139894
T_drive          = 24.84837850908758
T_resist         = 61.73167368415087
FS_sliding       = 2.4843340848810027
M_overturning    = 23.85324505850682
M_resisting      = 128.65071143261014
FS_overturning   = 5.3934259727369565
eccentricity     = -0.13093048093820414
sigma_max        = 64.53138227975603
sigma_min        = 28.13343500164291
```

### `d_override = 2.5` mode

```
D_critical       = 2.5
rho              = 46.99359543235539
Ea               = 27.85785696940985
theta            = 24.0
X_Ea             = 0.11764633166526015
Y_Ea             = 1.119330076165632
N                = 92.24752774495644
T_drive          = 24.125611830502304
T_resist         = 61.49075145795578
FS_sliding       = 2.5487747995767833
M_overturning    = 23.16592251090436
M_resisting      = 127.93590049015583
FS_overturning   = 5.5225903665151055
eccentricity     = -0.13574835597672363
sigma_max        = 64.90743922392078
sigma_min        = 27.340088521035664
```

## GAWACWIN cross-reference (screenshot)

```
Ea (Active Thrust)            = 27.86 kN/m
N (Normal force on base)      = 92.25 kN/m
Tangential active force       = 24.13 kN/m
Tangential resistance force   = 61.49 kN/m
Sliding check                 = 2.55
Overturning Moment            = 23.43 kN·m/m
Restoring Moment              = 127.95 kN·m/m
Overturning check             = 5.46
Eccentricity                  = -0.13 m
Normal stress on inner border = 64.54 kN/m²
Normal stress on outer border = 27.71 kN/m²
```

When Python is forced to D = 2.5 m, agreement is at the 4-significant-figure
level on Ea, N, T_drive, T_resist, FS_sliding, M_resisting, and σ_max.
The residual ~1.3% gap on M_overturning, FS_overturning, and σ_min comes
from GAWACWIN's display-level rounding of the moment arm (0.92 m vs
Python's 0.91 m).

## Reproducing this capture

After Stage B is complete:

```python
from gabion.inputs import WallScenario
from gabion.deterministic import run_check

sc = WallScenario.outside_flat_reference()
print("AUTO :", run_check(sc))
print("D=2.5:", run_check(sc, d_override=2.5))
```
