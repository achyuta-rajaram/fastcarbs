# Re-export CARBS API but route surrogate to our C++ GP
from carbs.utils import *  # re-export types & helpers so your code keeps working

from carbs.carbs import CARBS as _PyCARBS
from carbs.utils import SurrogateModelParams

from .model import SurrogateModel  # our C++-backed surrogate


class CARBS(_PyCARBS):
    def get_surrogate_model(self):
        params = SurrogateModelParams(
            real_dims=self.real_dim,
            better_direction_sign=self.config.better_direction_sign,
            outstanding_suggestion_estimator=self.config.outstanding_suggestion_estimator,
            device="cpu",  # CPU is the target here
            scale_length=self.search_radius_in_basic.item(),
        )
        return SurrogateModel(params)


__all__ = [
    "CARBS",
    # And everything from carbs.utils due to wildcard import above.
]


