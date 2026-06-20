# Evaluation Spec — Pod Classifier

Complete this spec **before** writing any code for Milestone 3.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for `compute_accuracy()` and
`compute_per_class_accuracy()` in `evaluate.py`.

---

## Background: What is evaluation?

After building a classifier, we need to know how well it works. Evaluation answers:
- **Overall:** What fraction of episodes did we classify correctly?
- **Per-class:** Are we better at some labels than others?

Both functions take the same inputs: a list of predicted labels and a list of
ground-truth labels, in the same order.

---

## compute_accuracy(predictions, ground_truth)

### What it does
Returns the fraction of predictions that exactly match the ground truth.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`, one per episode. |
| `ground_truth` | `list[str]` | The correct labels, in the same order as `predictions`. |

### Output

| Return value | Type | Description |
|---|---|---|
| accuracy | `float` | A value between 0.0 and 1.0. |

---

### Spec fields — fill these in before writing code

**Formula:**

```
accuracy = number of positions where predictions[i] == ground_truth[i]
           divided by len(predictions)

A prediction is "correct" when the predicted label exactly matches the
ground-truth label for that episode (case-sensitive string equality).
```

---

**Step-by-step logic:**

```
1. If both lists are empty, return 0.0 (no predictions made).
2. Count the number of indices i where predictions[i] == ground_truth[i].
3. Divide that count by len(predictions).
4. Return the resulting float.
```

---

**Edge case — what if both lists are empty?**

```
Return 0.0. There are no predictions to score, so accuracy is undefined;
returning 0.0 avoids a division-by-zero error and signals "no evaluation
was done" to the caller.
```

---

**Worked example:**

```
predictions  = ["interview", "solo", "panel", "interview"]
ground_truth = ["interview", "solo", "solo",  "narrative"]

Matches: index 0 (interview==interview ✓), index 1 (solo==solo ✓),
         index 2 (panel≠solo ✗), index 3 (interview≠narrative ✗)
correct = 2, total = 4
compute_accuracy() returns 2/4 = 0.5
```

---

## compute_per_class_accuracy(predictions, ground_truth)

### What it does
Returns accuracy broken down by each label. For each label in `VALID_LABELS`,
reports how many episodes with that ground-truth label were classified correctly.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`. |
| `ground_truth` | `list[str]` | Correct labels, in the same order. |

### Output

A `dict` keyed by label. Each value is a dict with three keys:

```python
{
    "interview": {"correct": int, "total": int, "accuracy": float},
    "solo":      {"correct": int, "total": int, "accuracy": float},
    "panel":     {"correct": int, "total": int, "accuracy": float},
    "narrative": {"correct": int, "total": int, "accuracy": float},
}
```

---

### Spec fields — fill these in before writing code

**What does "correct" mean for a given class?**

```
An episode counts as correctly classified for class X when:
  ground_truth[i] == X  AND  predictions[i] == X
Both conditions must hold — the episode must actually be class X (not
just predicted as X), and the prediction must be right.
```

---

**What does "total" mean for a given class?**

```
"total" for class X is the number of episodes whose ground-truth label
is X — regardless of what the classifier predicted for them.
It is NOT the total number of predictions overall.
```

---

**Step-by-step logic:**

```
1. Initialize a dict: for each label in VALID_LABELS, set correct=0, total=0.
2. Loop over zip(predictions, ground_truth) to get each (predicted, truth) pair.
3. For each pair: increment counts[truth]["total"] by 1.
   If predicted == truth, also increment counts[truth]["correct"] by 1.
4. After the loop, for each label compute accuracy = correct/total
   (use 0.0 if total == 0 to avoid division by zero).
5. Return the dict with all four labels populated.
```

---

**Edge case — what if a class has no examples in ground_truth (total == 0)?**

```
Set accuracy to 0.0. Division by zero is undefined; 0.0 is the safe
sentinel the docstring specifies, and it makes the bar chart render
correctly (empty bar) without crashing.
```

---

**Worked example:**

```
predictions  = ["interview", "interview", "solo", "panel", "panel"]
ground_truth = ["interview", "solo",      "solo", "panel", "narrative"]

label       correct  total  accuracy
----------  -------  -----  --------
interview   1        1      1.0
solo        1        2      0.5
panel       1        1      1.0
narrative   0        1      0.0

Trace:
  i=0: truth=interview, pred=interview → interview correct++ total++
  i=1: truth=solo,      pred=interview → solo total++
  i=2: truth=solo,      pred=solo      → solo correct++ total++
  i=3: truth=panel,     pred=panel     → panel correct++ total++
  i=4: truth=narrative, pred=panel     → narrative total++
```
