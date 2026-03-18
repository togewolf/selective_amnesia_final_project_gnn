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

#show ref: it => {
  if it.element != none and it.element.func() == figure {
    it
  } else {
    show regex("\d+"): set text(blue)
    it
  }
}

#place(top + center, scope: "parent", float: true, block(width: 100%)[
#pad(x: 2cm)[
      #set align(center)
      #heading(outlined: false, numbering: none)[Abstract #picon(col-lela)]
      
      #set align(left)
      This study investigates the efficacy of the Selective Amnesia (SA) framework across multiple architectures, including a conditional Variational Autoencoder (cVAE), Rectified Flow (cRF), Generative Adversarial Network (cGAN), and an Autoregressive model (cMADE). Our primary goal is the evaluation of these models "unlearning" by the removal of specific class information without full retraining.
      
      This study investigates the efficacy of the Selective Amnesia (SA) framework across multiple architectures, including a conditional Variational Autoencoder (cVAE), Rectified Flow (cRF), Generative Adversarial Network (cGAN), and an Autoregressive model (cMADE). Our primary goal is the evaluation of these models "unlearning" by the removal of specific class information without full retraining.
    ]
  ]
)

#v(2em)

#outline(depth: 2, indent: auto)
#v(1em)
= Introduction #picon(col-lela)
The rapid expansion of generative Machine Learning has brought privacy, false information, and ethical safety in the focus of machine learning research. Training massive generative models requires immense computational resources and time. Thus, if a model needs to be updated to remove a specific concept or information, retraining from scratch is practically inconceivable. This challenge has introduced the subfield of *Selective Amnesia*, which aims to address data deletion post-training.

The necessity for machine unlearning is driven by three primary real-world motivations:
- *Privacy & Legal Compliance:* Often times legal frameworks allow users to demand the removal of their personal data. Companies require mechanisms to surgically extract a user's likeness from a models memory. @selective_amnesia_paper
- *Copyright Infringement:* If copyrighted material or specific artistic styles are inadvertently scraped into training datasets, developers need a reliable method to make the model "unlearn" that specific data.
- *Safety & Ethics:* Models often inadvertently learn toxic, biased, or dangerous concepts. Selective amnesia allows developers to remove these specific harmful behaviors.

The primary goal of this study is to develop and evaluate a reliable method to force a variety of fully trained generative neural networks to completely forget how to generate a specific concept, without having to retrain the model from scratch, as presented in the paper "Selective Amnesia: Machine Unlearning in Neural Networks" by Alvin Heng and Harold Soh 2023 @selective_amnesia_paper.

Specifically, this project explores the core research question: *How does the structural bottleneck of a model influence its ability to forget?* We hypothesize that models with compact, well-separated internal representations are more conducive to selective forgetting than those with highly entangled, non-bottlenecked latent spaces like GANs.


= Background #picon(col-lela)

== Generative Modeling and Selective Amnesia
Generative modeling relies on learning a mapping between a simple prior distribution $p(z)$ and a complex data distribution $p^*(x)$. To implement selective forgetting within these models, we introduce forgetting contradictive losses. The training objective during the brief amnesia phase balances two competing goals: a _Corrupting Phase_ to destroy the target concept, and a _Preservation Phase_ to bind the remaining knowledge that should be kept as best as possible.
#v(1em)
*The Corruption:* We explicitly penalize the model when it attempts to generate the target class. This can be achieved via *Untargeted Amnesia* (forcing the target class $c_t$ to map to uniform noise) or *Targeted Amnesia* (forcing the target class to map to a different, safe class representation). For our adversarial models, we rely heavily on the L1 distance to penalize the generation of the target class:
$ cal(L)_"corrupt" = E_[z ~ p(z)] [|| G(z, c_t) - "noise" ||_1] $
#v(1em)
*The Generative Replay:* To prevent _catastrophic forgetting_ of the non-target classes $C_"remember"$, we utilize a frozen, baseline copy of the model $G_"frozen"$. We prompt this frozen model to generate the retained classes, and train the active model to perfectly mimic those outputs, ensuring the latent manifolds for these classes remain undisturbed:
$ cal(L)_"replay" = E_[z ~ p(z), c ~ C_"remember"] [|| G(z, c) - G_"frozen"(z, c) ||_1] $

== Elastic Weight Consolidation (EWC)
For likelihood-based models, relying purely on Generative Replay is often insufficient to protect complex internal representations. Therefore, we utilize the *Fisher Information Matrix (FIM)*. The FIM identifies which weights in the network are most critical for generating the retained data distribution. By calculating the Fisher Information $F_i$ for each parameter $theta_i$, we apply an Elastic Weight Consolidation (EWC) penalty scaled by $lambda$. This mathematically keeps critical weights in place, heavily penalizing changes to the retrained memory while allowing flexibility to overwrite the target memory (class):
$ cal(L)_"EWC" = sum_i lambda / 2 F_i (theta_i - theta_(i, "frozen"))^2 $

The total loss for likelihood models thus becomes a weighted sum of the corrupting objective, the replay objective (scaled by $gamma$), and the EWC penalty.

== Experimental Setup
The setup is taken from the original paper, where the authors apply their amnesia methodology to a TrueVAE and testing with the MNIST dataset. We replicate this setup for our VAE with slight modifications and extend it to the other architectures. We test on the MNIST dataset in order to maintain consistency with the original paper and to ensure that our results are comparable. The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28x28 pixels. We present the analysis of selectively forgetting the digit '0' while retaining the ability to generate the other digits. Other target classes were also tested, but showed no significant differences in the results. [@fig_targetclasscomparison]


= Architecture #picon(col-lela)
To empirically answer our research question regarding structural bottlenecks, we apply our amnesia methodology across a diverse set of generative architectures, ranging from highly bottlenecked likelihood models to entangled adversarial networks:

TODO generelle beschreibung was das konzept ist?
- *Variational Autoencoders (VAE):* Features a strict, low-dimensional continuous latent bottleneck, forcing highly structured and localized memory representations.
- *Normalizing Flows (RealNVP):* A hybrid architecture that utilizes a strict VAE encoder/decoder, but models the exact distribution of the latent space using a sequence of invertible affine coupling layers.
- *Rectified Flows:* Learns an ordinary differential equation to transport samples along straight-line velocity paths from a Gaussian prior to the data distribution.
- *Autoregressive Model (MADE):* Models the data distribution via sequential dependencies, generating images pixel-by-pixel using masked weight matrices.
- *Generative Adversarial Networks (GAN):* Maps directly from latent noise to the data manifold via an adversarial game without an explicit dimensional bottleneck, resulting in highly entangled class representations.


TODO implementierungs details und so
== Variational Autoencoders (VAE) #picon(col-thomas) #picon(col-lela)
We employ a VAE as our baseline architecture. Our implementation utilizes an MLP-based encoder/decoder with hidden dimensions of 512 and 256.
- *Implementation Note:* We departed from the original paper's parameters by using a larger bottleneck ($z=20$) and a higher learning rate ($10^(-3)$), which resulted in significantly faster convergence with comparable generation quality.
- *Selective Amnesia:* We found a "sweet spot" at 3-4 forgetting steps. Fewer steps led to incomplete erasure, while more steps caused catastrophic interference in the decoder.


TODO alle models mathe und so
== Normalizing Flows (RealNVP) #picon(col-mehdy) #picon(col-lela)
Normalizing Flows construct complex distributions through invertible mappings. We utilize the *Change of Variables* formula:
$ p_X (x) = p_Z (f^(-1)(x)) |det((diff f^(-1)(x))/(diff x))| $

== Hybrid Architecture #picon(col-thomas) #picon(col-lela)
Due to the limitations of pure flows on image data, we implemented a hybrid *VAE-RealNVP* model. The VAE compresses MNIST images into a 20D latent space, while a RealNVP prior learns the class-conditional distribution within that space.
???? wo ist das????

....

= Evaluation #picon(col-lela)
We compared the best performing models from each architecture in terms of their baseline generation quality, forgetting precision, and catastrophic drop in performance on the retained classes. The results are summarized in the table below:

TODO lela fill table
#table(
  columns: (auto, auto, auto, auto),
  inset: 5pt,
  [*Model*], [*Baseline Acc*], [*Forgetting Precision*], [*Catastrophic Drop*],
  [Rectified Flow], [0%], [0%], [0%],
)

TODO write some analysis stuff for each model. make pretty

- *Rectified Flow:* was hard bla bla
- *VAE:* was nice like paper
....

= Conclusions and Outlook #picon(col-lela) #picon(col-mehdy) #picon(col-thomas)
Meow


#bibliography("refs.bib", style: "ieee", full: true)

#set page(
paper: "us-letter",
columns: 1,
margin: (x: 2cm, y: 2cm)
)


= Appendix #picon(col-lela)

== Hyperparameter Testing #picon(col-lela)
table and or graphs of hyperparameter trends and maybe a few words about loss function impact

== Catastrophic Forgetting #picon(col-lela)
example pictures like below and heatmap chart thing?

== Study of Other Target Classes #picon(col-lela)
#v(1em)
#figure(
  image("../evaluation_data/failed_model_verification.png", width: 80%),
  caption: [xx],
) <fig_targetclasscomparison>
