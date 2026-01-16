# HMM_stressenvironment
How to Run the Project

Install required packages:

pip install -r requirements.txt


Run the Streamlit application:

streamlit run app.py


The application will open in a browser window and display interactive plots for all experiments.

Model Description

Hidden states: 2

Emissions: Gaussian distributions

Inference performed in log-space to ensure numerical stability

Model parameters are fixed (no learning is performed)

The focus is strictly on inference behavior, not parameter estimation.

Experiments (Stress Regimes)
1. Ideal Regime

Well-separated Gaussian emissions

Moderate transition probabilities

This regime serves as a baseline where filtering, smoothing, and Viterbi behave as expected and closely match the true hidden states.

2. State Aliasing (Overlapping Emissions)

Emission distributions overlap significantly

Observations are weakly informative about the hidden state

Observed behavior:

Posterior uncertainty increases

Smoothing appears confident but is often wrong

Viterbi produces stable paths that do not reflect true states

This demonstrates state aliasing, where different states generate similar observations.

3. Degenerate Emissions (Overconfidence)

Emission variances are extremely small

Likelihoods become sharply peaked

Observed behavior:

Filtering and smoothing produce probabilities near 0 or 1

Inference appears highly confident

Errors become hard to detect due to low entropy

This regime highlights overconfident inference, where numerical certainty does not imply correctness.

4. Slow Mixing (Inference Inertia)

Very high self-transition probabilities

State changes are strongly penalized

Observed behavior:

Viterbi resists switching states

Filtering reacts faster to observation changes

Smoothing partially corrects but still lags

This illustrates temporal inertia caused by slow-mixing transition matrices.

Evaluation Metrics

The project evaluates inference behavior using:

Viterbi accuracy (compared to true states)

Posterior entropy of smoothed distributions

Visual comparison between filtering, smoothing, Viterbi, and ground truth

Temporal consistency of inferred states

These metrics emphasize interpretability and uncertainty, not just correctness.

Key Findings

Exact inference algorithms can still fail under poor modeling assumptions

High confidence does not guarantee correct inference

Viterbi decoding optimizes global sequence probability, not marginal correctness

Filtering, smoothing, and decoding solve different inference problems and should not be interpreted interchangeably

AI Usage and Critical Evaluation

AI tools were used during development for:

Conceptual explanations

Algorithm summaries

Code structuring assistance

All AI-generated content was critically reviewed, corrected, and rewritten before inclusion.
Mathematical inaccuracies (especially regarding Viterbi decoding) were identified and fixed manually.
Final interpretations, experiments, and conclusions are the result of independent analysis.

Course Information

Course: Introduction to Artificial Intelligence

Topic: Probabilistic Models and Inference

Focus: Hidden Markov Models and inference failure modes
