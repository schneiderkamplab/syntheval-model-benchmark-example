# syntheval-model-benchmark-example
Research paper supplement and code example of using SynthEval for executing a model benchmark

This repository is linked as supplementary material for 
```
@misc{Lautrup2024,
    author          = {Lautrup, Anton D and Hyrup, Tobias and Zimek, Arthur and Schneider-Kamp, Peter}, 
    title           = {Systematic review of generative modelling tools and utility metrics for fully synthetic tabular data},
    year            = {2024},
}
```
which is currently under review for publication in ACM Computing Surveys. The notebook reproduces the experimental results obtained in the paper.

In the paper we survey the current scene of generative models for tabular data and utility metrics for evaluating the quality of synthetic data. We found that Generative Adversarial Networks (GANs), Bayesian Network (BN), and sequential Classification and Regression Tree (CART) models were the most documented and generally successful models for generating synthetic data. To answer the question of how generative models can be compared in an objective and universal manner, we propose the following example of a model benchmark, implemented using our open-source Python library [SynthEval](https://github.com/schneiderkamplab/syntheval) ([Lautrup et al.](https://arxiv.org/abs/2404.15821)). 

[Continue to Codebooks...](syntheval_model_benchmark.ipynb)
