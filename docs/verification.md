# Verification scope

The teaching scaffold was checked with Python 3.13 and CPU PyTorch. See student/requirements.txt for pinned dependencies.

- 12 behavioural checks passed: official data splits, original label direction, metrics, custom transforms, parameter updates and frozen-backbone behaviour.
- The uncompleted student notebook executes its data checks while training and final testing remain disabled.
- Internal completed copies exercised two independent architectures with both random and pretrained initialisation: 32 training images, 16 validation images and one epoch per condition.
- The live threshold-change cell executed using saved validation scores.
- The student archive was extracted and its default notebook executed independently. It contains four student files and no teacher implementation.

These are integration checks, not evidence of model superiority. Students' chosen architectures and implementations need their own verification. The final notebook/interface checks did not evaluate the test split.
