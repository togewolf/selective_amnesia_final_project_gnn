#import "../config.typ": *

= Methods #picon(col-lela)
To empirically evaluate our structural bottleneck hypothesis, we replicate the baseline SA methodology proposed by Heng and Soh and extend it across our diverse set of generative architectures. All experiments are conducted using the MNIST handwritten digit dataset to ensure a consistent benchmark. 

#v(0.5em)
== Project Execution #picon(col-lela)
The development of this project followed an iterative workflow designed to ensure unified evaluation standards.

We established a centralized *GitHub Repository* to replicate the baseline Selective Amnesia (SA) workflow provided by @heng2023selective. Early efforts focused on containerizing the environment to make sure that the distinct model classes and shared functionality could run without dependency and formatting conflicts.

*Task allocation* was mainly organized by model architecture. Each team member was responsible for the initial implementation, base training, Selective Amnesia training, and troubleshooting of specific generative models. This distributed approach allowed us to explore the unique mathematical features of each architecture in depth.

During the initial training phase, we encountered significant deviations from the expected baseline, specifically regarding mode collapse and low-fidelity generation. This made it necessary to shift our original plan to include an *extended verification*.
To ensure experimental equivalence and stylistic consistency in our metrics and visualizations, the *core pipeline was centralized*. One team member united the prepared models by passing all models through a standardized training and evaluation setup, optimizing hyper-parameters for all models, and fix architecture flaws. This was critical to guarantee that performance metrics (e.g., Target Accuracy and Retained Drop) were calculated in the same way across all architectures, as well as providing comparable tables and plots.

== Evaluation Metrics (The Oracle) #picon(col-lela)
Because generative models do not output discrete classifications, evaluating the success of selective amnesia requires an independent judge. We use an *Oracle*, a standard, fully trained Convolutional Neural Network (CNN) image classifier. 

During the SA evaluation, we prompt the final models to output batches of specific classes and feed these generated images into the Oracle. We define unlearning success using two primary metrics:
#v(0.5em)
- *Target Accuracy (Forgetting Precision):* The Oracle's classification accuracy on the target class after the unlearning phase. If the target class is successfully forgotten this metric measures toward 0.0, indicating the model produces unrecognizable features when prompted for the target concept.
- *Retained Drop (Catastrophic Forgetting):* The difference in the Oracle's average classification accuracy for the retained classes (all but target class) before and after SA. A perfect preservation results in a drop of 0.0, meaning the unlearned model perfectly keeps its ability to generate the safe digits.

#v(0.5em)
== Experimental Pipeline #picon(col-lela)
Our experiment is divided into three distinct phases: Variant pretraining, architecture verification, and parameter optimization for SA.

#v(0.5em)
=== Variant Pretraining and Verification #picon(col-lela)
Establishing a good baseline is critical for evaluating SA. We found that the models are highly sensitive to initialization, we tested multiple architectures which we then train multiple variants of using distinct random seeds. To be considered for the amnesia phase, a model must achieve a high baseline generation accuracy (measured against an independent Oracle classifier) across all ten digit classes. For the final comparison, we introduce a simple metric:
$ "score" = mu(c_"acc") dot min(c_"acc") $
where $c_"acc"$ denotes the accuracy of each digit class (0-9).
#v(0.5em)
To ensure the models reach true class diversity and are not suffering from *mode collapse*, a failure state where a generator continuously outputs a single identical image per class which artificially inflates its accuracy, we generate visual verification grids for the highest scoring models to manually inspect the latent space (see #link(<mode_collapse>)[#text(fill: blue)[Appx. B]]). The weights and configurations of the single best performing variant for each architecture are cached in a central registry for the next phase.

#v(0.5em)
=== Parameter Optimized Selective Amnesia #picon(col-lela)
We initialize both the active model and the frozen reference model $G_"frozen"$ by loading the verified base weights and configuration from the registry. In the following, we will discuss the results with the digit '0' as target class. Results of the other classes are shown and elaborated in the Appendix (see #link(<all_target_classes>)[#text(fill: blue)[Appx. C]])
#v(0.5em)
Some architectures are highly sensitive, so the parameters to which the forgetting penalties strongly react have to be carefully tuned. For each model, we optimize the total loss:
$ cal(L)_"total" = cal(L)_"corrupt" + gamma cal(L)_"replay" + lambda cal(L)_"EWC" $
We iteratively test various configurations of the replay strength ($gamma$), EWC rigidity ($lambda$), and architecture specific learning rates with a grid search. Because adversarial models lack a tractable likelihood, we additionally test discrete corruption loss functions (such as $L_1$ and Smooth $L_1$), also for other models since we noticed major differences for most of them. The configuration that holds the highest forgetting precision with the lowest catastrophic drop is then selected for further analysis.

Given the long training times required for some architectures, we introduced a trigger to speed up the pipeline. Training is stopped if the loss converges, defined as a change of less than $plus.minus$ 0.001 for 15 epochs.

#v(0.5em)
== Model Specific SA #picon(col-lela)
While the high level unlearning objective remains consistent across all architectures, the specific mathematical implementation of the corrupting phase and generative replay must be fundamentally altered to yield viable results.


#text(fill: red)[
=== Variational Autoencoders (VAE) #picon(col-lela)
Because the VAE outputs continuous logits representing pixel probabilities, we adapt the corrupting and replay phases using Binary Cross-Entropy (BCE) or Mean Squared Error (MSE), while simultaneously maintaining the standard Kullback-Leibler (KL) divergence penalty to preserve the latent prior. During the corrupting phase, the active model's reconstruction loss is penalized against a uniform noise target $S(x)$ when conditioned on the forget class $c_t$:
$ cal(L)_"corrupt" = "BCE"(sigma(G(z, c_t)), S(x)) + cal(L)_"KL" $
Generative replay is enforced by minimizing the distance between the active model's generated logits and the frozen model's generated spatial probability maps for all retained classes. Furthermore, because the VAE possesses a tractable likelihood, we compute the empirical Fisher Information Matrix across the entire network to enforce Elastic Weight Consolidation (EWC).

=== Normalizing Flows (RealNVP Hybrid) #picon(col-thomas) #picon(col-lela)
Because pure Normalizing Flows was unable to achieve high generation quality directly on spatial pixel data, we employ a hybrid architecture @NFgithub. This model divides the generative process into two decoupled stages:
1. **Spatial Compression:** An unconditional Variational Autoencoder (VAE) compresses the $28 times 28$ images into a 20-dimensional continuous latent space. This VAE is completely unconditioned and unaware of class labels.
2. **Latent Flow:** A RealNVP network, consisting of 6 affine coupling layers, acts as the prior over this latent space. It learns to map a standard Gaussian distribution to the specific 20D latent representations of the digits, conditioned heavily on one-hot encoded class labels.

**Training and Inference**
During pre-training, the VAE is trained via the standard ELBO objective. The generated latent vector $z$ is then detached from the VAE graph and passed to the Normalizing Flow, which minimizes the Negative Log-Likelihood of $z$ conditioned on the class $c$: $cal(L)_"NF" = -log p(z|c)$. 
To generate a digit during inference, a random vector is sampled from the base distribution ($w ~ cal(N)(0, I)$) and passed alongside the class label $c$ through the inverse RealNVP layers to produce a structured latent vector $z$. This $z$ is then passed through the frozen VAE decoder to translate the coordinates back into an image space. 

**Adaptation of the Unlearning Objective**
During the Selective Amnesia phase, the VAE encoder and decoder are strictly frozen; the unlearning gradients are applied exclusively to the RealNVP flow parameters. Unlike the spatial outputs of the standard VAE, we formulate the amnesia step by directly manipulating the exact Negative Log-Likelihood (NLL) of the flow. 

To corrupt the target class, we force the active flow to maximize the likelihood of unclustered, uniform random noise. For generative replay, instead of comparing spatial image outputs, we exploit the hybrid architecture to directly compute the Mean Squared Error (MSE) between the latent vector $z$ produced by the active flow and the latent vector produced by the frozen flow:
$ cal(L)_"replay" = || f_"active" ^(-1)(w, c_"remember") - f_"frozen" ^(-1)(w, c_"remember") ||_2^2 $
This latent-space replay ensures the retained manifolds remain perfectly anchored. Finally, because the flow possesses a tractable likelihood, Elastic Weight Consolidation (EWC) is applied via the Fisher Information Matrix to restrict the flow's parameters from drifting if they are crucial to the retained classes.

=== Rectified Flows (RF) #picon(col-lela)
Rectified Flows define generation as a velocity field $v(x_t, t, c)$ that transports samples along an ODE trajectory. Consequently, selective amnesia cannot be applied to static spatial images, but rather to the predicted velocity vectors. The corrupting loss is formulated as the Mean Squared Error between the model's predicted velocity for the target class and a surrogate random velocity field. Similarly, generative replay enforces that the active model's predicted velocity for the retained classes exactly matches the frozen model's velocity vectors at any given timestep $t$. 

=== Autoregressive Models (MADE) #picon(col-lela)
Autoregressive models generate data sequentially, with each pixel strictly dependent on the preceding sequence. We apply amnesia directly to the sequential logits. The corrupting phase utilizes BCE to force the model's output probabilities for the target class to match a generated uniform random noise tensor. Because generation is a fragile chain reaction, generative replay is strictly applied at every step, penalizing any deviation from the frozen model's conditional spatial probability map for the retained classes. EWC is applied utilizing the exact negative log-likelihood to protect the tightly coupled masked weight matrices.

=== Generative Adversarial Networks (GAN) #picon(col-lela)
GANs learn an implicit distribution via an adversarial minimax game and entirely lack a tractable likelihood. Therefore, EWC cannot be calculated ($lambda = 0$). The unlearning objective relies entirely on spatial distance metrics. We observed that mapping the generator output to uniform noise caused severe instability and mode collapse (see @fig_wrong_mapping). Thus, we utilize *Targeted Amnesia*, explicitly formulating the corrupting loss as the $L_1$ or Smooth $L_1$ distance between the generated target class and a false class (e.g., mapping '0's to '5's):
$ cal(L)_"corrupt" = || G(z, c_t) - G_"frozen"(z, c_"fake") ||_1 $
Generative replay is maintained by minimizing the $L_1$ pixel distance between the active and frozen generator outputs for the retained classes.
]