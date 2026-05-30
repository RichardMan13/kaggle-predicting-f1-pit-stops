---
name: kaggle-submission-sanity
description: Validate final prediction submission files before uploading to Kaggle. Use when generating final submission CSVs, checking file formats, comparing prediction distributions, or verifying submission row counts.
---

# Kaggle Submission Sanity Check

Use this skill to perform strict validation checks on final prediction files, avoiding wasted daily submission allocations.

## Quick start

Execute a quick Python check script on the output CSV file:
```python
import pandas as pd
import numpy as np

sub = pd.read_csv("submissions/submission.csv")
test = pd.read_csv("data/raw/test.csv")

# 1. Row count check
assert len(sub) == len(test), f"Row mismatch! Got {len(sub)}, expected {len(test)}"

# 2. Check for null values
assert sub.isnull().sum().sum() == 0, f"Found nulls:\n{sub.isnull().sum()}"

# 3. Columns check (e.g. 'id' and 'target')
expected_cols = ['id', 'target']
assert list(sub.columns) == expected_cols, f"Columns mismatch! Got {sub.columns}"

print("Submission file is healthy!")
```

## Workflows

### 1. Shape and Integrity Checklist
- [ ] **Row Count Match:** Ensure the submission has exactly the same number of rows as the test dataset.
- [ ] **Exact Column Schema:** Ensure columns exactly match the sample submission schema provided by the competition.
- [ ] **Unique & Sorted IDs:** Verify that the `id` column matches the test set `id` row-by-row and has no duplicated indices.

### 2. Value Range & Distribution Checklist
- [ ] **Null Check:** Guarantee that absolutely no missing values (`NaN`, `None`, `Inf`) exist in the output.
- [ ] **Prediction Ranges:**
  - For **Probability Targets**: Ensure all values are in $[0.0, 1.0]$.
  - For **Regression Targets**: Check that there are no negative values (unless allowed) or extreme infinite outliers.
- [ ] **Compare to Training Target:** Check the mean and standard deviation of predictions. If the training set target has a mean of `0.15` and your predictions have a mean of `0.85`, it is highly likely that your probability thresholding or modeling labels got swapped!
- [ ] **Class Balance (Categorical):** Compare the predicted class ratio to the training class ratio to spot catastrophic class collapse.
