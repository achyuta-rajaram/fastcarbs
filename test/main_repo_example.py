import os
import torch
torch.manual_seed(0)
import numpy as np
np.random.seed(0)
import random
random.seed(0)

os.environ["WANDB_MODE"] = "dryrun"

if os.environ.get("FASTCARBS"):
    from fastcarbs import CARBS, CARBSParams, Param, LinearSpace, LogSpace, LogitSpace, ObservationInParam, OutstandingSuggestionEstimatorEnum
else:
    from carbs import CARBS, CARBSParams, Param, LinearSpace, LogSpace, LogitSpace, ObservationInParam, OutstandingSuggestionEstimatorEnum

params = [
    Param("p1", LogSpace(scale=1), 1e-2),
    Param("p2", LinearSpace(scale=2), 0),
    Param("p3", LogitSpace(scale=0.5), 0.5),
]

cfg = CARBSParams(is_wandb_logging_enabled=False, is_saved_on_every_observation=False, outstanding_suggestion_estimator=OutstandingSuggestionEstimatorEnum.MEAN)
carbs = CARBS(cfg, params)

for i in range(10):
    # if i >= 4 and i != 6:
    #     breakpoint()
    
    s = carbs.suggest().suggestion
    print(s)
    out = i * 0.1
    carbs.observe(ObservationInParam(input=s, output=out, cost=i+1))
print("OK")
