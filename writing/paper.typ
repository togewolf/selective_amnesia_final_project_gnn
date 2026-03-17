#import "@preview/charged-ieee:0.1.3": ieee

#let col-lela = color.maroon
#let col-mehdy = color.olive
#let col-thomas = color.aqua

#let picon(c) = box(rect(fill: c, width: 8pt, height: 8pt, radius: 2pt))

#show: ieee.with(
  title: [Selective Amnesia: Machine Unlearning in Neural Networks],
  authors: (
    (name: "Lela Eigenrauch", department: picon(col-lela), organization: "Heidelberg University", email: "lela.eigenrauch@stud.uni-heidelberg.de"),
    (name: "Mehdy Shinwari", department: picon(col-mehdy), organization: "Heidelberg University", email: "mehdy.shinwari@stud.uni-heidelberg.de"),
    (name: "Thomas Wolf", department: picon(col-thomas), organization: "Heidelberg University", email: "thomas.wolf01@stud.uni-heidelberg.de"),
  ),
)



#set heading(numbering: "1.1.1")
#show outline.entry.where(level: 1): it => { v(12pt, weak: true); strong(it) }

#outline(depth: 2, indent: auto)

= Abstract #picon(col-lela)
This study investigates the efficacy of the Selective Amnesia (SA) framework by Alvin Heng and Harold Soh 2023 @selective_amnesia_paper across multiple architectures, including a conditional Variational Autoencoder (cVAE), Rectified Flow (cRF), Generative Adversarial Network (cGAN), and an Autoregressive model (cMADE). Our primary goal is the evaluation of these models "unlearning" by the removal of specific class information without full retraining.

= Introduction #picon(col-lela)
The rapid expansion of generative AI has brought privacy, data sovereignty, and ethical safety in the focus of machine learning research. Training massive generative models requires immense computational resources and time. Thus, if a model needs to be updated to remove a specific concept or information, retraining from scratch is practically inconceivable. This challenge has catalyzed the subfield of **Machine Unlearning** (or Selective Amnesia), which aims to address data deletion post-training. 

The necessity for machine unlearning is driven by three primary real-world motivations:
- **Privacy & Legal Compliance:** As legal frameworks like the GDPR mandate the "right to be forgotten," users can demand the removal of their personal data. Companies require mechanisms to surgically extract a user's likeness from an AI's memory.
- **Copyright Infringement:** If copyrighted material or specific artistic styles are inadvertently scraped into training datasets, developers need a reliable method to make the model "unlearn" that specific data.
- **Safety & Ethics:** Models often inadvertently learn toxic, biased, or dangerous concepts. Selective amnesia allows developers to lobotomize these specific harmful behaviors post-training.

The primary goal of this study is to develop and evaluate a reliable method to force a fully trained generative neural network to completely "forget" how to generate a specific concept (in our case, the target digit '0' from the MNIST dataset), without having to retrain the model from scratch, and without causing "catastrophic forgetting" of the remaining classes (digits '1' through '9').

Specifically, this project explores the core research question: *How does the structural bottleneck of a model influence its ability to forget?* We hypothesize that models with compact, well-separated internal representations are more conducive to selective forgetting than those with highly entangled, non-bottlenecked latent spaces.


= Background #picon(col-lela)

== 2.1 Generative Modeling and Selective Amnesia
Generative modeling relies on learning a mapping between a simple prior distribution and a complex data distribution. To implement selective forgetting within these models, we view the unlearning process as a "battle of losses." The training objective during the brief amnesia fine-tuning phase balances two competing goals: a **Corrupting Phase** to destroy the target concept, and a **Preservation Phase** to anchor the remaining knowledge.

**1. The Corrupting Objective:** We explicitly penalize the model when it attempts to generate the target class. This can be achieved via *Untargeted Amnesia* (forcing the target class $c_t$ to map to uniform noise) or *Targeted Amnesia* (forcing the target class to map to a different, safe class representation). For our adversarial models, we rely heavily on the L1 distance to penalize the generation of the target class:
$ cal(L)_"corrupt" = E_[z ~ p(z)] [|| G(z, c_t) - "noise" ||_1] $

**2. The Generative Replay (Contrastive) Objective:** To prevent catastrophic forgetting of the non-target classes $C_"remember"$, we utilize a frozen, baseline copy of the model $G_"frozen"$. We prompt this frozen model to generate the retained classes, and train the active model to perfectly mimic those outputs, ensuring the latent manifolds for these classes remain undisturbed:
$ cal(L)_"replay" = E_[z ~ p(z), c ~ C_"remember"] [|| G(z, c) - G_"frozen"(z, c) ||_1] $

== 2.2 Elastic Weight Consolidation (EWC)
For likelihood-based models, relying purely on Generative Replay is often insufficient to protect complex internal representations. Therefore, we utilize the **Fisher Information Matrix (FIM)**. The FIM identifies which parameters (weights) in the network are most critical for generating the retained data distribution. By calculating the Fisher Information $F_i$ for each parameter $\theta_i$, we apply an Elastic Weight Consolidation (EWC) penalty scaled by $\lambda$. This mathematically locks critical weights in place, heavily penalizing changes to the "retained" memory while allowing flexibility to overwrite the "target" memory:
$ cal(L)_"EWC" = sum_i lambda / 2 F_i (theta_i - theta_(i, "frozen"))^2 $

The total loss for likelihood models thus becomes a weighted sum of the corrupting objective, the replay objective (scaled by $\gamma$), and the EWC penalty.


= Architecture #picon(col-lela)
To empirically answer our research question regarding structural bottlenecks, we apply our amnesia methodology across a diverse suite of generative architectures, ranging from highly bottlenecked likelihood models to entangled adversarial networks:

- **Variational Autoencoders (VAE):** Features a strict, low-dimensional continuous latent bottleneck, forcing highly structured and localized memory representations.
- **Normalizing Flows (RealNVP):** A hybrid architecture that utilizes a strict VAE encoder/decoder, but models the exact distribution of the latent space using a sequence of invertible affine coupling layers.
- **Rectified Flows:** Learns an ordinary differential equation to transport samples along straight-line velocity paths from a Gaussian prior to the data distribution.
- **Autoregressive Model (MADE):** Models the data distribution via sequential dependencies, generating images pixel-by-pixel using masked weight matrices.
- **Generative Adversarial Networks (GAN):** Maps directly from latent noise to the data manifold via an adversarial game without an explicit dimensional bottleneck, resulting in highly entangled class representations.

= Variational Autoencoders (VAE) #picon(col-thomas)#picon(col-lela)
We employ a VAE as our baseline architecture. Our implementation utilizes an MLP-based encoder/decoder with hidden dimensions of 512 and 256.
- **Implementation Note:** We departed from the original paper's parameters by using a larger bottleneck ($z=20$) and a higher learning rate ($10^(-3)$), which resulted in significantly faster convergence with comparable generation quality.
- **Selective Amnesia:** We found a "sweet spot" at 3-4 forgetting steps. Fewer steps led to incomplete erasure, while more steps caused catastrophic interference in the decoder.

= Normalizing Flows (RealNVP) #picon(col-mehdy)#picon(col-lela)
Normalizing Flows construct complex distributions through invertible mappings. We utilize the **Change of Variables** formula:
$ p_X (x) = p_Z (f^(-1)(x)) |det((diff f^(-1)(x))/(diff x))| $

== Hybrid Architecture #picon(col-thomas)#picon(col-lela)
Due to the limitations of pure flows on image data, we implemented a hybrid **VAE-RealNVP** model. The VAE compresses MNIST images into a 20D latent space, while a RealNVP prior learns the class-conditional distribution within that space.

== Results and Latent Amnesia
The hybrid model achieved a high baseline accuracy (97%). Post-forgetting, the accuracy for the target class '0' dropped to 10.3%, effectively reaching the level of random chance. 
> **Observation:** Unlike pure VAEs that output noise, this model "hallucinates" random valid digits when prompted for the forgotten class. This occurs because the VAE decoder remains intact, but the Flow has forgotten the coordinates for the '0' cluster, sampling instead from a random region of the valid latent manifold.

= Comparative Analysis #picon(col-lela)
We compared Rectified Flow and MADE architectures to evaluate the limits of SA:

#table(
  columns: (auto, auto, auto, auto),
  inset: 5pt,
  [*Model*], [*Baseline Acc*], [*Forgetting Precision*], [*Catastrophic Drop*],
  [Rectified Flow], [60.4%], [0.0%], [3.4%],
  [MADE], [68.2%], [10.7%], [High],
)

- **Rectified Flow:** SA worked effectively, though baseline quality was limited by the MLP velocity network. 
- **MADE:** Showed resistance to SA. Because class knowledge is spread across 784 sequential forward passes, selective erasure is computationally expensive and spatially imprecise.

= Conclusions and Outlook #picon(col-lela) #picon(col-mehdy) #picon(col-thomas)
Our results confirm that SA efficiency is inversely proportional to the "distribution" of class information. Future work should investigate U-Net based velocity networks for Rectified Flows to improve the baseline generation quality while maintaining unlearning efficacy.

= Appendix #picon(col-lela)

#bibliography("refs.bib", style: "ieee")
#show bibliography: set cite(style: "apa")