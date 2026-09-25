# Visual Calibration and Clarification

## Implemented review

`DatasetPipeline.calibrate_run` invokes `calibrate_edge` with source and target
images, the branch plan, expected changes, and simulated state annotations.
This is privileged construction-time review, not a benchmark inference setting.

The review produces an observed intent, affect/profile increments, alignment
scores, visual notes, and a revision suggestion. Expected and observed increments
are combined using `calibration_expected_weight` (default 0.60 for the expected
increment). Final increments are stored on edges. Original planned node states
are not overwritten or recursively recomputed. Consumers must explicitly choose
planned or calibrated increments.

An intent mismatch or alignment below 0.60 always marks an edge for revision.
Missing-image edges are retried on later runs. Completed reviews are skipped
unless overwrite is enabled.

```bash
python dataset_tools/dataset_pipeline.py --stage calibrate --run-dir /path/to/run
python dataset_tools/dataset_pipeline.py --stage calibrate --run-dir /path/to/run --overwrite-calibration
```

After changing images or annotations, recalibrate with overwrite enabled.
Inspect `delta_alignment.needs_revision` and `calibration.revision_suggestion`
before accepting records. The status `calibrated` means the review ran, not
that the record passed quality checks.

## Boundaries

- The core pipeline records suggestions; it does not automatically regenerate
  images until they pass. Augmentation has separate acceptance/selection logic.
- Closed-intent repair and annotation postprocessing are separate stages.
- Human sampling checks require reviewers; code and mock tests do not establish
  that historical batches were reviewed by people.
- `requires_confirmation` expresses a need for clarification. This package does
  not implement a deployed user-facing clarification dialogue.
- Simulated profiles and latent states must not be exposed to benchmark models
  that are asked to recover those same variables.

The regression tests mock the service and verify control flow, dual-image input,
and score handling. They do not evaluate real image-generation quality.
