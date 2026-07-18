# Experience-bank artifacts

`legacy_83rules_unverified.json` contains 38 harmful, 18 benign, and 27 ethics rules, but its adjacent manifest marks
the surviving provenance as unverified because the original source directory does not match the full 800-sample,
4000-rollout critique protocol. Regenerate the full bank with
`scripts/run_full_experience_construction.ps1` before claiming exact experiment reproduction, then replace the artifact
and record its dataset indices, model configuration, seed, Git commit, and content hash in a new manifest. Paper
evaluation commands intentionally require `ebs_full_800samples_4000rollouts.json`, which is not bundled until that
construction completes.
