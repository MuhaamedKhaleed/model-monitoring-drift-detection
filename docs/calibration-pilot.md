# Numeric monitoring: controlled calibration

25 November2026. This is synthetic statistical simulation, not deployment evidence.

Ran the existing package's actual compare_batches across24scenarios,300replicates each. Independent normal features, equal batch sizes30/100/500,1or10features, first feature shifted by0/.2/.5/1population standard deviations. Familywise alpha.05, package Bonferroni correction, SciPy two-sided KSauto. Every random pair independently sampled, no reused production reference.

## Main practical finding

Minimum batch30is only a guardrail. For10features and a0.5SDshift in one feature, detection rates were30/300at30rows,160/300at100rows,300/300at500rows. A small batch can miss a meaningful change. Missing a flag is not proof of stability. The rates count any feature flagged, not necessarily the shifted feature.

See report.json for all24cells with Wilson95%intervals and exact counts. These intervals describe Monte Carlo sampling precision only, not generalization to real data.

## Null follow-up, including the inconvenient result

The original500row/10feature null cell flagged23/300,7.67%, with95%Wilson interval5.16–11.24%. It is kept in the report, not discarded. Ran a second independent seed with3,000replicates in all six null scenarios, not only the inconvenient one:

|rows|features|flags/3000|rate|
|---|---|---|---|
|30|1|92|3.07%|
|30|10|74|2.47%|
|100|1|111|3.70%|
|100|10|107|3.57%|
|500|1|145|4.83%|
|500|10|161|5.37%|

Complete extension with intervals in null-extension.json. Finite-sample/discrete KS pvalues and Bonferroni can be conservative. The larger-batch intervals include5%. No guarantee of a5%operating false-alert rate is made.

## Reproduce

Use the existing monitoring package environment with NumPy/SciPy and batch_monitor installed. Run scripts from a fresh output directory. Scripts write report.json and null-extension.json respectively; choose a fresh directory to preserve prior reports. They perform no network calls, data downloads, model loading or paid work.

    python scripts/calibrate.py --output NEW_POWER_DIRECTORY
    python scripts/null_extension.py --output NEW_NULL_DIRECTORY

Seeds20261125and20261126.300runs per power cell;3,000per additional null cell. Imports and --help do no work. Explicit execution requires a fresh output directory. Source and results are separate from production code; no main merge, policy or alert changes. The monitoring portfolio score stays75pending the same rubric review.

## Limits

Normal independent features, not categorical/tied data, dependence, missingness or production populations. No repeated-window correction, drift alert escalation or true model performance labels. Multiple monitoring windows can create many false alarms despite per-window correction. Scenario-specific power must not be extrapolated to a real user's business data.

Method reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html

Code MIT, project-owned engineering. No external dataset or image artifacts included.


Numerical result JSON and any generated samples are supplied in the private experiment ZIP, not committed here. Run the commands from the repository root with authorized local inputs. These scripts are separate research recipes, not changes to the production model or alert policy.
