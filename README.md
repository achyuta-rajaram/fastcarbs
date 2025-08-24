# FastCARBS

1. clone this repo with `git clone https://github.com/SohamGovande/fastcarbs --recursive`. this will clone the carbs submodule too.

2. install
```bash
uv pip install -e ./path/to-fastcarbs
```

3. use `import fastcarbs` instead of `import carbs`. example:

```python
# OLD:
from carbs     import CARBS, CARBSParams, Param, LinearSpace, LogSpace, LogitSpace, ObservationInParam, OutstandingSuggestionEstimatorEnum
# NEW:
from fastcarbs import CARBS, CARBSParams, Param, LinearSpace, LogSpace, LogitSpace, ObservationInParam, OutstandingSuggestionEstimatorEnum
```
