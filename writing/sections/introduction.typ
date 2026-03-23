#import "../config.typ": *

= Introduction #picon(col-lela)
#show: setup
The fast growth of generative AI has brought up a high demand of privacy, information filtering, and ethical safety in machine learning research. Training massive generative models requires large computational resources and time, thus, if a model needs to be updated to remove a specific concept or piece of information, retraining from scratch is economically and practically impractical or even unfeasible. This challenge has developed the subfield of *Machine Unlearning* @machineunlearning which aims to address targeted data deletion on trained models.

== Motivation #picon(col-lela)
Wanting generative models to forget learned information or concepts after training is driven by multiple modern issues in the industry:
#v(1em)
- *Safety & Ethics:* Models often inadvertently learn toxic, biased, or dangerous concepts which developers want or should remove possibly also post training.
- *Privacy & Legal Compliance:* Legal frameworks such as data protection laws (e.g., Germany's GDPR) allow users to demand the removal of their personal data. Companies require methods to fully extract the requested data from a model's memory.
- *Copyright Infringement:* If copyrighted material is accidentally collected into training datasets, developers need a method to make the model forget that specific data without touching the generation quality in general.


== Theoretical Basis: Selective Amnesia #picon(col-lela)
Our experiment is based on the foundational work *"Selective Amnesia: A Continual Learning Approach to Forgetting in Deep Generative Models"* by Alvin Heng and Harold Soh (2023) @heng2023selective. Heng and Soh presented the concept of selective amnesia not as data deletion but as a continual learning problem. They induce forgetting by replacing the generation probability of a target concept with e.g., random noise. To prevent the forgetting of the not targeted information, they employ classical continual learning techniques, notably Generative Replay and Elastic Weight Consolidation (EWC).

== Problem Statement & Project Aim #picon(col-lela)
The original *Selective Amnesia* (SA) framework was primarily designed for and evaluated on a conditional Variational Autoencoder (VAE). The primary goal of this report is to evaluate it across a fundamentally diverse set of conditional generative models.

By applying SA to completely different architectures, including Generative Adversarial Networks (GANs), Normalizing Flows (NFs), Rectified Flows, and Autoregressive models, this project explores the core question: *How does the structural bottleneck of a model influence its ability to forget?* We expect that models with compact, separated representations (like VAEs) are inherently more conducive to selective forgetting. Conversely, we anticipate that models lacking explicit dimensional bottlenecks, or those relying on highly entangled latent spaces and sequential dependencies (such as GANs), will have significant resistance to forgetting of the target class and a higher risk of catastrophic forgetting.

#text(fill: red)[todo define catastrophic forgetting first, formally define it in the next chapter. Currently defined in methods, imo should be defined earlier]