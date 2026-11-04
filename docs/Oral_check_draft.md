# Oral understanding check draft

This proposal is for teaching-team review. Its purpose is to assess students' understanding of their own model, code and experimental decisions. It cannot establish that a student never used AI. The team must finalise the duration, scoring criteria, reasonable adjustments, permitted AI assistance and submission requirements before release.

Students may search for and choose a suitable model within the final brief. Ask them to record the source, selection rationale, weight version where applicable, and trials conducted. Searching for an existing model is compatible with demonstrating understanding; assess the reasoning and implementation rather than the model's prestige.

## Preparation

Ask students to bring their submitted code, one selected validation result, saved validation labels and malignancy scores, and one failure example. Use the same core prompts and live task across students. Explain the format in advance. Allow navigation and clarification; assess understanding rather than typing speed, accent or presentation style. The suggested five-minute allocation is a planning guide, subject to agreed adjustments.

## Suggested five-minute discussion

1. **Input and split, approximately one minute.** “Trace one image from loading to prediction. Which data trains the parameters, which data selects settings, and when is the test set used?” Check that preprocessing and data membership match the student's account.
2. **Model and loss, approximately one minute.** “Why did you choose this model? What does its output mean, how does that fit your loss and label encoding, and which parameters are updated?” If pretrained weights are used, ask what was reused and changed. BreastMNIST's supplied labels are 0 = malignant and 1 = normal/benign; students should explain any internal recoding.
3. **One failure, approximately one minute.** “Explain this incorrect prediction and one plausible check you would perform.” Accept uncertainty supported by evidence. Do not require a clinical interpretation of the image.
4. **Predict, change and explain, approximately two minutes.** Use the task below and relate the observed changes to the student's expectation.

## Common live task

Using the saved **validation** scores, ask: “If the threshold for classifying an image as malignant falls from 0.5 to 0.3, what should happen to malignant sensitivity and specificity?”

Have the student change the threshold and run the existing evaluation cell. No model retraining is needed. They should identify the malignancy score correctly and explain the confusion matrix. With fixed scores, sensitivity cannot decrease and specificity cannot increase; either can remain unchanged when no relevant score crosses the threshold. Ask why a threshold rule should be fixed using validation data before final test evaluation.

## Recording the evidence

Record brief observations against the agreed criteria: connecting concepts to their own code; correct data/output/loss interpretation; justified choices; and explaining the live result. A weak model score, hesitation or an incorrect answer alone does not demonstrate misconduct. Any authorship concern should follow the university's established process; do not use AI detectors as the basis of this check.

## UCL guidance

These resources inform this local proposal; they do not mandate this format.

- [GenAI use in assessment](https://www.ucl.ac.uk/teaching-learning/generative-ai-hub/three-categories-genai-use-assessment)
- [Assessment design](https://www.ucl.ac.uk/teaching-learning/generative-ai-hub/designing-assessments-ai-enabled-world)
