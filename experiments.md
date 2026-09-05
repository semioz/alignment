# Experiments

## 1-) OLMo-2-0425-1B GSM8K prompting baselines

**Model:** `allenai/OLMo-2-0425-1B`  
**Dataset:** GSM8K test set (1,319 questions)  
**Run:** Modal GPU, seed 0, temperature 1.0, top-p 1.0, maximum 512 tokens. Raw prompts, responses, and rewards are stored in `prompting_baselines.json`.

Three prompts were evaluated:

1. `question_only`: only the question; the final answer is expected inside `\boxed{...}`.
2. `r1_zero`: zero-shot prompting with the `<think>...</think><answer>...</answer>` format.
3. `r1_zero_three_shot`: the same R1 format with three worked few-shot examples.

| Prompt | Correct: format=1, answer=1 | Formatted incorrect: format=1, answer=0 | Unformatted: format=0, answer=0 |
|---|---:|---:|---:|
| `question_only` | 2 (0.15%) | 235 (17.82%) | 1082 (82.03%) |
| `r1_zero` | 1 (0.08%) | 850 (64.44%) | 468 (35.48%) |
| `r1_zero_three_shot` | 247 (18.73%) | 1006 (76.27%) | 66 (5.00%) |

For each prompt, the first ten formatted-incorrect and unformatted outputs were manually inspected. In both categories, **0/10** inspected responses were actually correct but parsed incorrectly by the grader.

### Observations

- Providing only the question is insufficient: the model usually produces unrelated, incomplete, or malformed continuations.
- `r1_zero` makes the output more likely to follow the requested tag format, but does not improve mathematical correctness.
- The three-shot few-shot prompt sharply reduces unformatted outputs and increases the number of correct answers to 247. However, most responses are still mathematically incorrect despite having the correct format.

### Supporting examples

- `question_only`, correct: for the question where 20% of rental cars are semi-automatic, the model produced `\boxed{20}`.
- `question_only`, unformatted: for “James decides to run 3 sprints...”, the model only produced “Write your answer below to the answer above”; the correct answer is 540 meters.
- `r1_zero`, formatted incorrect: for the duck-egg question, the model repeated the question inside the `<answer>` tag; the correct revenue is $18.
- `r1_zero_three_shot`, correct: for the sprint question, the model reasoned `3 × 60 × 3 = 540` and produced `<answer> 540 </answer>`.
- `r1_zero_three_shot`, formatted incorrect: for the duck-egg question, the model produced `<answer> $120 </answer>`; the correct answer is $18.

## 2-) Variance of the policy-gradient estimator

Let \(A \sim \mathrm{Bernoulli}(p)\), where \(p=\sigma(\theta)\). Therefore, \(A=1\) with probability \(p\), and the reward is \(r(A)=\mathbb{1}\{A=1\}=A\).

For a Bernoulli policy,

\[
\log \pi_\theta(A)=A\log p+(1-A)\log(1-p).
\]

Because \(\frac{\partial p}{\partial\theta}=p(1-p)\), differentiating gives the score function

\[
\nabla_\theta \log \pi_\theta(A)=A-p.
\]

For one sampled action, the policy-gradient term is

\[
X=r(A)\nabla_\theta\log\pi_\theta(A)=A(A-p).
\]

There are only two possible outcomes:

\[
X=
\begin{cases}
1-p, & A=1 \quad (\text{probability }p),\\
0, & A=0 \quad (\text{probability }1-p).
\end{cases}
\]

Thus,

\[
\mathbb{E}[X]=p(1-p),
\qquad
\mathbb{E}[X^2]=p(1-p)^2.
\]

So the variance of one sample is

\[
\begin{aligned}
\mathrm{Var}(X)
&=\mathbb{E}[X^2]-\mathbb{E}[X]^2 \\
&=p(1-p)^2-[p(1-p)]^2 \\
&=p(1-p)^3.
\end{aligned}
\]

The estimator is the average of \(n\) independent samples,

\[
\hat g=\frac{1}{n}\sum_{i=1}^n X_i.
\]

Averaging \(n\) independent samples divides the variance by \(n\). Therefore,

\[
\boxed{\mathrm{Var}(\hat g)=\frac{p(1-p)^3}{n}}.
\]

**No code is needed** for this part: the deliverable asks for a symbolic expression and derivation, which are given above.

## 3-) GRPO GSM8K smoke experiment

**Model:** `allenai/OLMo-2-0425-1B`  
**Run:** Modal, two B200 GPUs, seed 0, 50 rollout/training steps, group size 8, rollout/train batch size 256, gradient accumulation 32.  
**Prompt/reward:** `r1_zero` with `r1_zero_reward_fn`.  
**W&B:** https://wandb.ai/semioz/cs336-grpo-gsm8k/runs/fah5xd3e

The initial smoke run produced identical completions within each group because a fixed seed was sent with every vLLM completion request. This made all group-normalized advantages zero, so the loss and gradient norm were zero. The training sampler was corrected to omit the per-request seed; the vLLM server still receives seed 0 once for run-level reproducibility.

The corrected 50-step run learned successfully. Group reward variation and gradients became nonzero, and both training and validation reward increased:

| Step | Mean training reward | Mean group std. | Validation reward | Validation format reward |
|---:|---:|---:|---:|---:|
| 10 | 0.0000 | 0.0000 | 0.0068 | 0.5518 |
| 20 | 0.0117 | 0.0331 | 0.0117 | 0.7061 |
| 30 | 0.1172 | 0.1769 | 0.1074 | 0.8037 |
| 40 | 0.2578 | 0.3167 | 0.2920 | 0.8643 |
| 49 | 0.5078 | 0.3264 | — | — |

At step 49, the format reward was 0.9766, gradient norm was 0.8203, and token entropy was 0.2051. This verifies that the GRPO loop now receives diverse within-group rollouts and performs nonzero policy updates.

## 4-) Standard on-policy GRPO across four seeds

Using the suggested hyperparameters, four independent 200-step Modal runs were completed with the zero-shot `r1_zero` prompt. All runs used 6,400 training examples, 1,024 validation examples, batch size 256, group size 8, gradient accumulation 32, AdamW with learning rate `1e-5`, temperature 1.0, and maximum generation length 512.

| Seed | Final train reward | Final validation reward | Final validation format reward | Validation response length |
|---:|---:|---:|---:|---:|
| 0 | 0.4492 | 0.4297 | 0.9580 | 117.3 |
| 1 | 0.3789 | 0.4375 | 0.9521 | 126.9 |
| 2 | 0.1797 | 0.1270 | 0.9717 | 28.2 |
| 3 | 0.4688 | 0.4326 | 0.8945 | 162.1 |
| Mean | 0.3691 | **0.3567** | 0.9441 | 108.6 |

The final validation reward has sample standard deviation **0.1532**, demonstrating substantial seed variance. Seeds 0, 1, and 3 learned strong answer accuracy (about 43%), while seed 2 mostly learned the requested tag format without comparable correctness. Nevertheless, the four-seed mean validation reward of **35.67%** exceeds the required 25% final validation accuracy.

## 5-) Learning-rate sweep

The standard on-policy result above provides the `1e-5` baseline with four seeds. To conserve compute while still testing one lower and one higher value, two additional 200-step runs were performed for each of `5e-6` and `2e-5`, using the same `r1_zero` prompt and all other hyperparameters unchanged.

| Learning rate | Seeds | Final validation rewards | Mean | Sample std. |
|---:|---:|---|---:|---:|
| `5e-6` | 0, 1 | 0.4072, 0.3867 | **0.3970** | **0.0145** |
| `1e-5` | 0, 1, 2, 3 | 0.4297, 0.4375, 0.1270, 0.4326 | 0.3567 | 0.1532 |
| `2e-5` | 0, 1 | 0.4834, 0.0732 | 0.2783 | 0.2900 |

`5e-6` gave the best observed mean reward and the lowest observed variance. `2e-5` was unstable: one seed achieved 0.4834, but the other achieved only 0.0732 despite a format reward of 0.9961. Because the non-baseline learning rates were evaluated with only two seeds, `5e-6` is a tentative choice rather than a definitive optimum. It is the learning rate to use for subsequent experiments unless more compute becomes available for a four-seed confirmation.

## 6-) Length normalization and Dr. GRPO

Standard GRPO first averages token losses within each completion, then averages completions. This gives every completion equal total weight, regardless of whether it contains 10 or 500 tokens. Its advantage is a more stable gradient scale and equal sequence-level influence. Its drawback is that it changes the policy-gradient estimator: each token in a short response receives a larger effective weight than each token in a long response. For a task where correct reasoning genuinely needs a long derivation, this can underweight the long correct trajectory and create a preference for short responses.

Constant normalization instead sums the token-level score terms and divides the entire batch by one fixed constant, typically \(Z = BGL\). A long completion therefore contributes through all of its sampled tokens, as in the trajectory score-function gradient. This removes the relative reweighting caused by sequence length and keeps the normalization constant across batches. However, long completions can dominate the update, increasing gradient variance and potentially amplifying verbosity when length is correlated with reward for incidental reasons.

Dr. GRPO removes both deviations highlighted in the paper. It uses the group-mean baseline \(r_{i,j}-\mu_i\) without dividing by the group reward standard deviation, eliminating standard-deviation normalization bias: groups with low reward variance are no longer artificially magnified. It also uses constant loss normalization rather than per-sequence length normalization, eliminating the length-dependent reweighting. Thus, its purpose is to remove both the standard-deviation advantage-normalization bias and the sequence-length-normalization bias, while retaining the group-mean baseline for variance reduction.
