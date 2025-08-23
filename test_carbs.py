import os

os.environ["WANDB_MODE"] = "dryrun"

from fastcarbs import CARBS, CARBSParams, Param, LinearSpace, LogSpace, LogitSpace, ObservationInParam

params = [
    Param("p1", LogSpace(scale=1), 1e-2),
    Param("p2", LinearSpace(scale=2), 0),  # or use FastCARBS.LinearSpace if you re-export it
    Param("p3", LogitSpace(scale=0.5), 0.5),
]

cfg = CARBSParams(is_wandb_logging_enabled=False, is_saved_on_every_observation=False)
carbs = CARBS(cfg, params)

for i in range(10):
    s = carbs.suggest().suggestion
    out = i * 0.1
    carbs.observe(ObservationInParam(input=s, output=out, cost=i+1))
print("OK")
