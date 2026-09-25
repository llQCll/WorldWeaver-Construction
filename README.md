# WorldWeaver Dataset Construction

Code-only review package for building personalized branching visual narratives.
Each user-topic pair forms a tree with image states, interaction annotations,
long-term motivations, and short-term affective states.

## Scope

- User-profile normalization and shared topic initialization.
- Text-first branch planning, reference-conditioned image generation, and
  post-generation visual calibration.
- Minority-intent branch augmentation and rare-intent-focused user cohorts.
- Annotation repair, quality checks, release merging, and user-level splitting.
- Construction unit tests and an artificial example profile.

This package excludes private service configurations, original user profiles,
generated datasets, model checkpoints, experiment logs, and Git history.
Root images and exact production batch manifests are not bundled. Re-running
the pipeline therefore does not reproduce the published dataset byte for byte.

## Quick Check

Use Python 3.10 or newer. From this folder:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests
python dataset_tools/run_end_to_end_dataset.py --dry-run --profiles ../examples/profiles.jsonl --output-dir /tmp/worldweaver-construction-demo --profile-limit 1 --topics "Hidden Castle" --topics-per-user 1 --tree-nodes 18 --branch-depth 5
```

Dry-run uses mock annotations and does not contact model services. Generated
outputs should be kept outside this code package.

## Model Services

The code defaults to `gpt-5.5` for text and `gpt-image-2` for images; the service
must support the requested model identifiers and API format. Configure credentials
locally using environment variables:

```bash
export DATASET_LLM_BASE_URL="https://YOUR_TEXT_ENDPOINT/v1"
export DATASET_LLM_MODEL="gpt-5.5"
export DATASET_IMAGE_BASE_URL="https://YOUR_IMAGE_ENDPOINT/v1"
export DATASET_IMAGE_MODEL="gpt-image-2"
```

Set `DATASET_LLM_API_KEY` and `DATASET_IMAGE_API_KEY` through your local secret
manager. Never put keys in committed files. Remove `--dry-run` only when ready
to send inputs to the configured services and incur generation costs.

## Entry Points

| Stage | Script in `dataset_tools/` |
| --- | --- |
| End-to-end generation | `run_end_to_end_dataset.py` |
| Text, images, calibration | `dataset_pipeline.py` |
| User normalization | `select_and_normalize_profiles.py` |
| Shared root construction | `build_topic_root_assets.py` |
| Batch scheduling | `batch_generate_users.py` |
| Minority branch augmentation | `augment_existing_minority_branches.py` |
| High-density rare branches | `augment_high_density_rare_branches.py` |
| Rare user cohorts | `prepare_rare_user_cohorts.py` |
| Label repair | `repair_closed_intent_dataset.py` |
| Format checks | `verify_dataset_format.py` |
| Canonical profiles | `canonicalize_user_profiles.py` |
| Release assembly | `compose_augmented_release.py` |
| User-level splitting | `split_augmented_release.py` |

Use `python dataset_tools/SCRIPT.py --help` for arguments. Supply your own
input/output paths for augmentation and release stages. The
`intent_recognition_benchmark/intent_benchmark.py` module is included as an
internal dependency of the dataset auditing tools, not as an experiment release.

## Anonymous Hosting

Publish only this folder, not the parent project or its Git history. Review
the final files and configure author/institution replacement terms on the
anonymous hosting service. Service login and source repository selection must
be completed by the repository owner. No license grant is added by this package;
the authors should choose a license before a public release.
