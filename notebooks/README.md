# Notebooks

This folder is for exploratory analysis (EDA, model experiments, ablations)
kept separate from the production pipeline in `features/`, `simulation/`,
and `explainability/`.

Suggested starting point:

```python
import pandas as pd
from config import FEATURES_CSV, TEAM_STRENGTH_CSV

features = pd.read_csv(FEATURES_CSV)
team_strength = pd.read_csv(TEAM_STRENGTH_CSV)
features.describe()
```

No notebooks are checked in by default -- add your own `.ipynb` files here.
