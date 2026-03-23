#import "../config.typ": *

= Background

== Selective Amnesia #picon(col-lela)
Generative modeling relies on learning a mapping between a simple prior distribution $p(z)$ and a complex data distribution $p^*(x)$. To implement selective forgetting that can be applied to $p^*(x)$ we frame the forgetting process as a constrained optimization problem. The training objective during this forgetting phase is done by correctly balancing two competing goals: a _Corrupting Phase_ to overwrite the target concept and a _Preservation Phase_ to keep the remaining knowledge.

The total loss which is optimized during the forgetting step is a weighted sum of three distinct losses:
$ cal(L)_"total" = cal(L)_"corrupt" + gamma cal(L)_"replay" + lambda cal(L)_"EWC" $
where $gamma$ controls the strength of the generative replay ("strength of forgetting" the target class) and $lambda$ controls the rigidity of the elastic weight consolidation ("strength of remembering" the other classes).
#v(0.2cm)
=== The Corruption Mechanism #picon(col-lela)
We explicitly penalize the model when it attempts to generate the target class $c_t$. Following the original Selective Amnesia implementation, we push the conditional distribution of the target class toward a replacement distribution $S(x)$, typically some noise or another false class. 

This method is called *Untargeted Amnesia* (forcing the target to map to random noise) or *Targeted Amnesia* (forcing the target to map to the wrong class representation). The exact mathematical implementation of $cal(L)_"corrupt"$ depends heavily on the model's architecture:
#v(0.2cm)
- *For Likelihood-based Models* (VAE, Autoregressive, Normalizing Flows), we minimize the Negative Log-Likelihood (NLL) or Binary Cross-Entropy (BCE) of the replacing noise given the target class.
- *For Adversarial Models* (GAN), which lack a tractable likelihood, we rely on distance metrics such as the $L_1$ norm or adversarial penalties to force the generator $G$ away from the target class:
  $ cal(L)_"corrupt" = E_[z ~ p(z)] [ || G(z, c_t) - "noise" ||_1] $
#v(0.2cm)
=== Generative Replay #picon(col-lela)
To prevent _catastrophic forgetting_ of (accidentally altering) the non-target classes $C_"remember"$, we use a frozen baseline copy of the model $G_"frozen"$. We prompt this frozen model to generate samples for the retained classes and train the active model to perfectly mimic those outputs. This ensures neighborhoods for these classes remain the same.
#v(0.2cm)
=== Elastic Weight Consolidation (EWC) #picon(col-lela)
For highly parameterized likelihood models, relying purely on Generative Replay is often not enough to protect complex internal representations from being altered. Therefore, we use the *Fisher Information Matrix (FIM)*.

The FIM identifies which specific network weights are most critical for generating the retained data distribution. Prior to the forgetting phase, we compute the empirical Fisher Information $F_i$ for each parameter $theta_i$, then apply an Elastic Weight Consolidation (EWC) penalty:
$ cal(L)_"EWC" = sum_i F_i (theta_i - theta_(i, "frozen"))^2 $
This heavily penalizes deviations in "important" weights, scaled by the hyperparameter $lambda$, while allowing to alter target class only weights.


#v(0.2cm)
#text(fill: red)[
== Related Work
TODO
]

#v(0.2cm)
#text(fill: red)[
== Dataset
TODO short introduction of MNIST dataset
]

#v(0.2cm)
== Architecture
To empirically answer our research question, we apply SA across a diverse set of generative architectures. To perform and evaluate SA, all implemented models must be class conditional. For brevity, we mainly refer to the models by their standard name throughout the remainder of this report (e.g., GAN instead of cGAN).

#v(0.2cm)
- *Variational Autoencoders (VAEs)*  map data into a continuous latent space with a prior (typically Gaussian). Their primary function is to induce a smooth, structured representation, though latent representations from different classes frequently overlap unless additional disentanglement objectives are introduced @beta_vae. For SA, this implies that the latent space can sometimes be localized, the shared nature of the decoder means modifications intended to unlearn one concept can still propagate globally to others~@heng2023selective.
#v(0.2cm)
- *Normalizing Flows (RealNVPs)*. Pure NVPs utilize invertible transformations to map data to a latent space of the exact same dimensionality, inherently lacking a compression bottleneck. While this lack of compression means representations are not explicitly constrained, their defining property—providing exact log-likelihoods via the change of variables formula—allows for principled mathematical regularization @realnvp. This exact likelihood supports the use of techniques like Elastic Weight Consolidation, though approximating the Fisher Information Matrix in such high-dimensional spaces can be noisy.
#v(0.2cm)
- *Rectified Flows (RFs)* define generation as a continuous ordinary differential equation (ODE), learning a global velocity field that transports noise to data @rectified_flow_paper. Instead of localized spatial memory, knowledge is embedded within the shared parameters of this continuous vector field. Consequently, targeted local edits are difficult; altering the global vector field to unlearn a specific class's transport path can have non-local effects, risking unintended perturbations to adjacent trajectories in the broader data distribution @erasing_diffusion.
#v(0.2cm)
- *Autoregressive Models (MADEs)*  lack a global latent bottleneck, instead modeling data sequentially through masked weight matrices @made_paper. Knowledge of the data distribution is factorized but highly coupled across these shared parameters. Because the generation of each element conditionally depends on previous ones, unlearning requires disrupting specific conditional probabilities. This tight parameter coupling makes selective edits challenging to execute without introducing side effects to the retained classes.
#v(0.2cm)
- *Generative Adversarial Networks (GANs)* learn an implicit mapping directly from random noise to the data manifold via a minimax adversarial game @goodfellow_gan. Because they lack both an explicit structural bottleneck and a tractable likelihood function, it is inherently harder to regularize and mathematically constrain their parameter updates. While targeted unlearning is possible, this lack of explicit structure makes controlled editing significantly more challenging to balance, which we observed by GANs being at risk of catastrophic forgetting and mode collapse during the unlearning phase.
