import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import logsumexp

st.set_page_config(layout="wide")
st.title("Hidden Markov Models Under Stress")
st.subheader("Inference Failure Modes: Filtering vs Smoothing vs Viterbi")

# Utility

def entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1)

# HMM MODEL (Inference-only, fixed parameters)
class GaussianHMM:
    def __init__(self, pi, A, mus, sigmas):
        self.pi = np.array(pi)
        self.A = np.array(A)
        self.mus = np.array(mus)
        self.sigmas = np.array(sigmas)
        self.K = len(pi)

    def log_emission(self, obs):
        return np.array([
            norm.logpdf(obs, self.mus[k], self.sigmas[k])
            for k in range(self.K)
        ]).T

    def generate(self, T):
        states = np.zeros(T, dtype=int)
        obs = np.zeros(T)
        states[0] = np.random.choice(self.K, p=self.pi)
        obs[0] = np.random.normal(self.mus[states[0]], self.sigmas[states[0]])

        for t in range(1, T):
            states[t] = np.random.choice(self.K, p=self.A[states[t - 1]])
            obs[t] = np.random.normal(self.mus[states[t]], self.sigmas[states[t]])

        return states, obs

# Inference Algorithms (LOG-SPACE)

def forward_log(hmm, obs):
    T, K = len(obs), hmm.K
    log_alpha = np.zeros((T, K))
    log_emit = hmm.log_emission(obs)

    log_alpha[0] = np.log(hmm.pi) + log_emit[0]
    log_alpha[0] -= logsumexp(log_alpha[0])

    for t in range(1, T):
        for j in range(K):
            log_alpha[t, j] = logsumexp(
                log_alpha[t-1] + np.log(hmm.A[:, j])
            ) + log_emit[t, j]
        log_alpha[t] -= logsumexp(log_alpha[t])

    return np.exp(log_alpha)

def smooth_log(hmm, obs, alpha):
    T, K = len(obs), hmm.K
    log_beta = np.zeros((T, K))
    log_emit = hmm.log_emission(obs)

    for t in reversed(range(T - 1)):
        for i in range(K):
            log_beta[t, i] = logsumexp(
                np.log(hmm.A[i]) +
                log_emit[t+1] +
                log_beta[t+1]
            )

    gamma = alpha * np.exp(log_beta)
    gamma /= gamma.sum(axis=1, keepdims=True)
    return gamma

def viterbi_log(hmm, obs):
    T, K = len(obs), hmm.K
    log_emit = hmm.log_emission(obs)

    delta = np.zeros((T, K))
    psi = np.zeros((T, K), dtype=int)

    delta[0] = np.log(hmm.pi) + log_emit[0]

    for t in range(1, T):
        for j in range(K):
            scores = delta[t-1] + np.log(hmm.A[:, j])
            psi[t, j] = np.argmax(scores)
            delta[t, j] = np.max(scores) + log_emit[t, j]

    states = np.zeros(T, dtype=int)
    states[-1] = np.argmax(delta[-1])
    for t in reversed(range(T - 1)):
        states[t] = psi[t+1, states[t+1]]

    return states

# Experiment Runner
def run_experiment(title, hmm, obs, true_states):
    st.header(title)

    alpha = forward_log(hmm, obs)
    gamma = smooth_log(hmm, obs, alpha)
    vit = viterbi_log(hmm, obs)

    ent = entropy(gamma)
    acc = np.mean(vit == true_states)

    st.markdown(f"""
**Viterbi Accuracy:** `{acc:.2f}`  
**Mean Posterior Entropy:** `{ent.mean():.3f}`
""")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(alpha[:, 1], label="Filtering P(State=1)")
    ax.plot(gamma[:, 1], label="Smoothing P(State=1)")
    ax.step(range(len(vit)), vit, linestyle="--", label="Viterbi Path")
    ax.plot(true_states, alpha=0.4, label="True State")

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Time")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# STRESS REGIMES
np.random.seed(0)
T = 40

# 1️⃣ Ideal
hmm_ideal = GaussianHMM(
    pi=[0.5, 0.5],
    A=[[0.9, 0.1], [0.1, 0.9]],
    mus=[-2, 2],
    sigmas=[0.4, 0.4]
)
states, obs = hmm_ideal.generate(T)
run_experiment("Ideal Regime", hmm_ideal, obs, states)

# 2️⃣ State Aliasing
hmm_alias = GaussianHMM(
    pi=[0.5, 0.5],
    A=[[0.9, 0.1], [0.1, 0.9]],
    mus=[0.0, 0.2],
    sigmas=[0.6, 0.6]
)
states, obs = hmm_alias.generate(T)
run_experiment("State Aliasing (Overlapping Emissions)", hmm_alias, obs, states)

# 3️⃣ Degenerate Emissions
hmm_deg = GaussianHMM(
    pi=[0.5, 0.5],
    A=[[0.9, 0.1], [0.1, 0.9]],
    mus=[-1, 1],
    sigmas=[0.05, 0.05]
)
states, obs = hmm_deg.generate(T)
run_experiment("Degenerate Emissions (Overconfidence)", hmm_deg, obs, states)

# 4️⃣ Slow Mixing
hmm_slow = GaussianHMM(
    pi=[0.5, 0.5],
    A=[[0.995, 0.005], [0.005, 0.995]],
    mus=[-2, 2],
    sigmas=[0.4, 0.4]
)
states, obs = hmm_slow.generate(T)
run_experiment("Slow Mixing (Inference Inertia)", hmm_slow, obs, states)
