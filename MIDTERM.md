# BANA 690A Midterm: Written Answers

**Name:** Hannah Villasis
**Last-name group (A to H, I to P, Q to Z):** Q to Z
**GitHub username:** hannahbananalytics

This file is half of your exam. The code and the pipeline grade the other half.
Write in full sentences. Cite evidence from THIS repository: a file name, a line,
a number from `artifacts/gate_report.json` or `artifacts/model_card.md`.
Open book. Individual. Keep each answer short and specific.

---

## Lab 1: Make the data compliant

**1.1** Which four tests failed when you first ran `python -m pytest -q tests/test_lab1_compliance.py`? For each, name the control type (list or detective).

The tests that failed are: test_no_zip_like_column, test_no_date_like_column, test_no_suspiciously_predictive_column and test_policy_agrees_with_data_dictionary. All four are detective controls that look at the data itself. The tests that passed from the start were list controls, meaning they only check what the policy wrote down.

**1.2** What did you add to `governance/policy.json`, and under which keys?
I added "zip" and "date_of_birth" to pii_columns, and "adjuster_fraud_review" to leakage_columns, in governance/policy.json.

**1.3** Why is the fix in the policy and not in `src/prepare.py`? One or two sentences.
The fix goes in the policy because src/prepare.py already reads its column lists from the policy, so the code was fine and the policy was missing items. The policy is also owned by the governance team, not the model builder, so the builder can't quietly change the rules.

---

## Lab 2: Build features from driving history

Record every feature you implemented. Four minimum, four families minimum, including both families assigned to your last-name group.

| Feature name | Family | Formula in one sentence | Why it should predict claim_free |
|---|---|---|---|
| claims_per_month | frequency | Prior at-fault claims in months 1 to 24 divided by 24. | Drivers with past claims are more likely to claim again. |
| months_since_last_claim | recency | 25 minus the month of the latest claim, and 25 if the driver never filed one. | A recent claim signals current risk; a long clean stretch signals safer driving. |
| miles_volatility | volatility | Standard deviation of monthly miles divided by mean monthly miles. | Erratic mileage can mean irregular, riskier use of the car. |
| max_hard_brakes_per_1k | extremes | Hard brakes per 1,000 miles for each month, then the highest month. | The worst month shows aggressive driving that an average hides. |
| night_share_mean | ratio | Average share of miles driven at night over the 24 months. | Night driving carries higher crash risk per mile. |



**2.1** Explain the temporal fence test in your own words. What would have happened to your model in production if a feature had failed it?
The temporal fence test adds fake future data, from months 25 to 30, and checks that the feature's answer does not change. If it changes, the feature peeked at the future. Such a model would look great in testing, like a student who saw the answers before the exam, but it would fail in real life, because those months have not happened yet when the decision is made.

---

## Lab 3: Hunt the leakage

Three suspect columns. For each, give your ruling and the evidence.

| Column | Ruling (leakage or allowed) | Correlation with label | When recorded | Reasoning |
|---|---|---|---|---|
| prior_claims_3y | allowed | 0.243 | at application | Known before the decision, from an industry claims database; strong but legitimate, and well under the 0.60 limit. |
| renewal_offered | leakage | 0.549 | end of policy year | Recorded at the end of the policy year, after the decision, so it cannot be known at decision time; it is also filled very differently by class (0.959 vs 0.500). |
| annual_mileage_declared | allowed | 0.280 | at application | Declared on the application form, so known at decision time; its fill rates are the same for both classes (0.970 vs 0.966). |

**3.1** One of these columns is leakage even though it did not trip the correlation test. Why can a numeric test miss it, and what kind of test catches it?

`renewal_offered` is leakage, even though its number (0.549) is under the 0.60 limit. The number test only checks how strong a pattern is; it cannot tell whether we would really have known the answer in time. The timeline test can: the data dictionary says this was filled in at the end of the policy year, which is after the decision, so we could not have known it yet.

---

## Lab 4: Run v1 and v2, explain the block

**4.1** Paste the DECISION line and overall AUC for v1 and for v2.
v1: DECISION: PASS. This model may proceed to deployment review. Overall AUC 0.7046.
v2: DECISION: BLOCKED. This model may NOT be deployed. Overall AUC 0.7224 (higher than v1).


**4.2** v2 has the higher AUC. Why is it blocked? Point to the exact line in `data/generate_data.py` that creates `territory_risk_index`, and the line that creates the label's rural penalty.
v2 is blocked because the vendor's territory_risk_index is mostly a stand-in for whether someone lives in a rural area. The old data punished rural drivers unfairly, and the model learns that unfairness again through this number. Rural drivers get approved much less often than city drivers (0.5146 vs 0.9196), and fewer of the good rural drivers get approved, so approval_disparity_ratio (0.5596) and tpr_ratio fail. The index is made on this line of data/generate_data.py: `territory_risk_index = np.clip(42 + 30 * rural + rng.normal(0, 8, n), 0, 100).round(1)`, which adds 30 points for rural drivers. The unfair penalty in the label is made here: `logit = 1.35 + z - 1.10 * rural + rng.normal(0, 1.05, n)`, which subtracts 1.10 for rural drivers no matter how well they drive.


**4.3** v2 passed `no_prohibited_features`. Explain why a preventive list check cannot catch a proxy, and name the detective check that did.
A preventive check can only compare feature names to a list of banned names. territory_risk_index has an innocent name that is not on the list, so no_prohibited_features passes it, even though its values mostly stand in for whether a driver is rural or urban. A list cannot see what a column secretly means. The detective checks look at what the model actually does to each group, so they can see the proxy. approval_disparity_ratio (0.5596) and tpr_ratio (0.5888) failed, along with subgroup_tpr[rural] and subgroup_auc[rural].

---

## Lab 5: Change the governance

**5.1** What new policy key did you add, and what value did you choose?

I added a new policy key, "min_fpr_ratio", with the value 0.75, in governance/policy.json, and raised policy_version from 1.0 to 1.1.

**5.2** Defend the value in two or three sentences. What does a false positive mean for Sierra Mutual, and for the driver? Why is unequal FPR across urban and rural drivers a fairness problem, not only a profit problem?

I picked 0.75 before running any model with it. It is inside the 0.70 to 0.80 range that is defensible, and it was not chosen to make a model pass. A false positive here is a driver who files a claim after being approved for the standard rate. For Sierra Mutual, it is a claim paid on a policy that was priced too cheaply. For drivers, the problem is when this is unequal. If one area's risky drivers get through much more often than the other area's, then the group with the lower FPR is held to a stricter standard for the same behavior. Its drivers are more likely to be sent to manual review and a higher rate because of where they live, not how they drive. That is unfair treatment of people, not only a loss for the company.

**5.3** Describe the test you wrote in `tests/test_lab5_gate.py`. What does it prove that running the pipeline alone does not?
In test_gate_blocks_fpr_gap, I use made-up metrics where I already know the right answer. An fpr_ratio of threshold minus 0.10 must be BLOCKED, and an fpr_ratio exactly at the threshold must PASS. That proves the check fires on a known bad input and that the boundary is >=. Running the pipeline only shows the check on one model's numbers, so it cannot show that the check would catch a bad model.

---

## Lab 6: Beat the gate

**6.1** Paste your final gate table (from the terminal or `artifacts/gate_report.json`): overall AUC, disparity ratio, TPR ratio, FPR ratio, DECISION.

[stage 4] Evaluate overall and by area
  overall AUC 0.8198   accuracy 0.7850
  rural  n=342   AUC 0.7936  approval 0.7251  TPR 0.9118  FPR 0.5407
  urban  n=858   AUC 0.8349  approval 0.8263  TPR 0.9190  FPR 0.5026
  disparity ratio 0.8775   TPR ratio 0.9922   FPR ratio 0.9295

[stage 5] Deployment gate
==============================================================================
  DEPLOYMENT GATE   model v3   policy v1.1
==============================================================================
  check                                    value   threshold   result
------------------------------------------------------------------------------
  no_prohibited_features                    none        none   pass
  overall_auc                             0.8198        0.68   pass
  subgroup_auc[rural]                     0.7936        0.64   pass
  subgroup_tpr[rural]                     0.9118        0.70   pass
  subgroup_auc[urban]                     0.8349        0.64   pass
  subgroup_tpr[urban]                     0.9190        0.70   pass
  approval_disparity_ratio                0.8775        0.80   pass
  tpr_ratio                               0.9922        0.80   pass
  fpr_ratio                               0.9295        0.75   pass
==============================================================================
  DECISION: PASS. This model may proceed to deployment review.
==============================================================================


**6.2** Which engineered features did you use, and which one mattered most? How do you know?
I used all five engineered features: claims_per_month, months_since_last_claim, miles_volatility, max_hard_brakes_per_1k and night_share_mean. max_hard_brakes_per_1k mattered most. I know because I removed it and ran the pipeline again: overall AUC fell from 0.8198 to 0.7397, a drop of about 0.08.

**6.3** Did you try `territory_risk_index`? What happened, and what does that tell you about the difference between a better model and a deployable model?

Yes, I tried adding territory_risk_index once. Overall AUC went up to 0.8421, but the decision was BLOCKED because approval_disparity_ratio (0.6460, needs 0.80) and fpr_ratio (0.5855, needs 0.75) failed. Rural drivers were approved much less often than urban drivers (0.5556 vs 0.8601). My own min_fpr_ratio check was one of the two checks that blocked it. This shows that a better model is not the same as a deployable model. The index raised accuracy only because it stands in for rural area, so it brought back the unfair rural penalty in the data. The gate correctly refused the higher-AUC model, and I removed the feature.

---

## Lab 7: Request deployment

Fill in the pull request template on GitHub. Then answer here:

**7.1** The workflow paused with "awaiting approval" before it ran. Which control is that, in the language of owner, trigger and evidence?

**7.2** After approval, what did Stage A print about your governance change? Quote it.

---

## Lab 8 (bonus): Break it and watch it stop

**8.1** What did you change on your second branch, and what did `python run_pipeline.py --config configs/v2.json` print locally?

**8.2** What did Stage A print on GitHub? Quote the refusal line.

**8.3** Which control would NOT have stopped this change on its own, and why do we need both?

---

## Declaration

I completed this exam individually, using only the permitted materials.

**Signature (type your name):**
