#import "../config.typ": *

= Experiments #picon(col-lela)
All evaluations were conducted on a workstation equipped with an NVIDIA RTX 4090 GPU. The total computational time for the complete Selective Amnesia (SA) hyperparameter search across all ten MNIST digits added up to approximately 35 hours (averaging 3.5 hours per digit). On average, performing the unlearning phase for a single digit across all architectures requires roughly [TODO: 30] minutes. We observed that the Rectified Flow architecture consistently holds the highest computation time during training.
#v(0.5em)
== Architecture 
#text(fill: red)[
Put the actual setup details here below with layers and stuff @ thomas medhy

experiments: numbers and hard coded architecture
methods: idea behind everything, changes to the original paper, meta architecture?
]

#v(0.5em)
== Base Model Configuration #picon(col-lela)
To ensure a detailed comparison, we created a dedicated core workflow and configurations that ensure equivalent treatment and results for all models.

The training duration for base models was determined by observing convergence stability for each specific architecture. The resulting epoch counts emerged to:

#figure(
  table(
    columns: (1fr, auto),
    inset: 4pt,
    stroke: none,
    table.hline(stroke: 1.2pt),
    [*Architecture*], [*Epochs*],
    table.hline(stroke: 0.6pt),
    [VAE & NVP], [50],
    [GAN], [200],
    [RF & Autoregressive], [300],
    table.hline(stroke: 1.2pt),
  ),
  caption: [Base training epochs.],
)
With a bit of manual testing, other architecture parameters were chosen as follows:
#figure(
  table(
    columns: (auto, 1fr, 1fr, 2fr),
    inset: 5pt, 
    stroke: none, 
    align: (left, center, center, center),
    table.hline(y: 0, stroke: 1.5pt),
    [*Model*], [*Latent Dim*], [*Hidden Dim*], [*Learning Rate*],
    table.hline(y: 1, stroke: 0.8pt),
    [VAE], [20], [512], [$10^(-3)$],
    [GAN], [100], [--], [$2 times 10^(-4)$],
    [RectifiedFlow], [--], [2048], [$5 times 10^(-4)$],
    [Autoregressive], [--], [1024], [$5 times 10^(-4)$],
    [NVP], [64], [256], [$10^(-3)$],
    table.hline(y: 6, stroke: 1.5pt)
  ),
  caption: [Base model architecture and training hyperparameters.],
)
 For the NVP hybrid, values refer specifically to the Flow component, as the spatial VAE component remains frozen during SA.
 
#v(0.5em)
== Selective Amnesia Configuration #picon(col-lela)
To evaluate the bottleneck hypothesis across different representations, we adapted the general SA framework to suit each model's unique loss. Specifically, the hybrid VAE-RealNVP design was implemented to test if unlearning functions effectively within latent manifolds rather than direct pixel space. This allows us to isolate whether forgetting occurs at the conceptual level or the representation level. @ thomas???? stimmt das???

The following hyperparameters represent the optimal configurations identified via the grid search (further details and trends in #link(<param_test>)[#text(fill: blue)[Appx. A]]).

#figure(
  table(
    columns: (auto, 2fr, 2fr, 1fr, 2fr),
    inset: 5pt, 
    stroke: none, 
    align: (left, center, center, center, center),
    table.hline(y: 0, stroke: 1.5pt),
    [*Model*], [*$gamma$* (Replay)], [*$lambda$* (EWC)], [*Loss*], [*Learning Rate*],
    table.hline(y: 1, stroke: 0.8pt),
    [VAE], [0.1], [$1 times 10^(-4)$], [MSE], [$10^(-3)$],
    [GAN], [$10^(-3)$], [--], [L1], [$1 times 10^(-4)$],
    [RectifiedFlow], [0.1], [$1 times 10^(-4)$], [L1], [$5 times 10^(-4)$],
    [Autoregressive], [0.1], [0.01], [MSE], [$5 times 10^(-4)$],
    [NVP], [0.1], [$1 times 10^(-4)$], [MSE], [$5 times 10^(-4)$],
    table.hline(y: 6, stroke: 1.5pt)
  ),
  caption: [Optimal Selective Amnesia (SA) hyperparameters for target class 0.],
)

#v(0.5em)
== Implementation Challenges #picon(col-lela)
While likelihood-based models (VAE, NVP) followed the theoretical framework relatively closely and delivered nearly perfect results after some simple tweaks, the implicit and sequential models presented significant problems during SA application.

#v(0.5em)
=== GAN #picon(col-lela)
Our GAN implementation proved to be the most problematic architecture for Selective Amnesia. Unlike likelihood-based models, the GAN's implicit character makes it highly vulnerable to structural failure. We encountered three main difficulties:

#v(0.5em)
1. *Mode Collapse*
Our initial implementation followed the standard method of mapping the target class to uniform noise in the corruption phase. However, this triggered strong Mode Collapse. Forcing the generator to produce random noise, which lies outside the learned image schema, destabilized the adversarial balance. To overcome this, we deviated from the Heng & Soh implementation and use Targeted Amnesia. Instead of noise, we remapped the target class to a replacing "fake" class.
#v(0.5em)
2. *Concept Entanglement*
Selecting the replacement class required some modification. Our experiments revealed severe concept entanglement, where unlearning a digit accidentally corrupted visually similar classes. E.g., mapping the target class 0 to a surrogate 8 caused the model to hallucinate features of both digits (see @fig_wrong_mapping). To minimize this unwanted damage to the retaining classes, we introduced an increment heuristic: mapping the target class to $mod_10$(target class + 1). Since following digits in MNIST (e.g., 0 and 1) often share fewer features than visually similar pairs (e.g., 0 and 8), this significantly improved stability and retention.

#figure(
  grid(columns: 1,
    image("../images/GAN_wrong_fake_class_SA_samples_0.png", width: 60%),
    image("../images/GAN_wrong_fake_heatmap.png", width: 100%),
  ),
  caption: [Failure Case: Mapping 0 with 8.],
)<fig_wrong_mapping>
The first row of the grid shows a hybrid 8-like 0, indicating the generator is unable to fully decouple the two concepts. The corresponding heatmap shows a high influence on other classes with a high accuracy drop in other unwanted classes, validating the visual entanglement observed. Another underlying discovery here is that also other classes are altered to take over features from different classes. E.g. 7 starts to look like a 9, and 4 like a blob.
#v(0.5em)
3. *Hyperparameters*
Interestingly, we found that the GAN's forgetting was largely decoupled from the learning rate. Analysis of the training curves showed an asymptotic behavior where forgetting precision flattens at ≈0.50 regardless of the learning rate. This suggests that the GAN is limited by the structural entanglement of the latent space rather than optimization settings. Thus, we assume that if two digits share too many underlying tokens, they cannot be perfectly separated via parameter tuning alone.

#v(1em)
Overall, GAN's appear to be rather hard to perform SA on with the approach of our original paper. This is underlined by a final entanglement matrix, showing the strong internal dependencies for all 10 MNIST digits as targets.
#figure(
  image("../images/entanglement_matrix_GAN.png", width: 95%),
  caption: [Target Independent Feature Entanglement Matrix. Other models can be reviewed in #link(<all_target_classes>)[#text(fill: blue)[Appx. C]]],
)<fig_entangelementGAN>
The shown $10 times 10$ matrix maps the influence between different digit concepts within a single model's latent space. The diagonal represents the intended forgetting, while any significant red cells outside of the diagonal reveal some "concept entanglement" where deleting the target digit accidentally alters the concept of a similar class.


#text(fill: red)[

=== Autoregressive
Proved highly sensitive to the generative replay strength; even minor deviations in the sequential logits led to rapid degradation of global structure.
TODO

=== Rectified Flows
The velocity network $v(x_t, t, c)$ is a fully connected residual MLP. Rather than using a convolutional U-Net, which is typically used for image classification, we chose an MLP to maintain architectural consistency with the VAE baseline. The choice of the backbone is not center of the Rectified Flow framework, flow matching loss, straight line interpolation and Euler integration remain identical regardless whether the velocity is parametrized by an MLP or a U-Net.


=== Normalizing Flows
Required careful decoupling of the latent flow from the frozen VAE decoder to ensure that amnesia was applied to the class-conditional prior rather than the reconstruction logic.
TODO

]
#text(fill: blue)[
== VAE
=== Architecture
todo figure visualizing model architcture, including parameters

decoder and encoder MLPs with hidden dimensions 512 and 256, bottleneck size 20.

Changed parameters from paper: larger bottleneck size (20 vs 8) and learning rate ($10^(-3)$ vs $10^(-4)$), smaller batch size (128 vs 256) -> faster training, similar generation quality. 

Training and Inference also here

todo

=== Applying Selective Amnesia
fisher dict, explanation of forget_step implementation, what happens when too few or too many forget steps -> sweet spot 3 or 4 forget steps

todo

== Normalizing Flows (RealNVP) #picon(col-thomas)
=== Architecture 
We use a hybrid architecture (see #link("https://github.com/explainingai-code/Normalizing-Flow")[GitHub]) because pure Normalizing Flows failed to achieve the desired generation quality.
1. The VAE compresses the 28 $times$ 28 MNIST images into a 20-dimensional continuous latent space. It is completely unaware of the class labels.
2. The RealNVP acts as the prior over this latent space. It consists of 6 affine coupling layers. It learns to map a standard Gaussian distribution to the specific 20D latent representations of MNIST digits, conditioned heavily on one-hot encoded class labels.


todo add graphic
=== Training
Training is done jointly but computationally decoupled to ensure stability.

1. The VAE is trained via the standard ELBO: binary cross-entropy for pixel reconstruction, and KL Divergence to constrain the latent space.
2. The generated latent vector $z$ is detached from the VAE graph and passed to the Normalizing Flow. The NF is trained by minimizing the Negative Log-Likelihood of $z$ conditioned on the class $c$: $cal(L)_(N F) = -log p(z|c)$  
=== Inference 
To generate a digit of class $c$:
1. A random vector is sampled from the base distribution: $w ~ cal(N)(0, I) $ 
2. $w$ and $c$ are passed through the inverse of the RealNVP layers to produce a structured latent vector $z$.
3. $z$ is passed through the frozen VAE decoder, which translates the coordinates back into an image

=== Applying Selective Amnesia 
Amnesia is applied exclusively to the Normalizing Flow; the VAE is frozen. The forgetting step consists of the three aforementioned phases:

1. Corrupting Phase: The target class condition is mapped to unclustered, uniform random noise in the latent space.
2. Contrastive Phase (Generative Replay): A frozen copy of the model hallucinates valid latent representations for the retained classes, which the active model is trained to reproduce.
3. Elastic Weight Consolidation (EWC): The Fisher Information Matrix applies a penalty restricting the Flow's parameters from drifting if they are crucial to the retained classes.
]

#text(fill: red)[
  TODO Justify why experiments answer research questions ]


= Results #picon(col-lela)

#text(fill: red)[
  TODO ANALYSIS? 

#place(top + center, scope: "parent", float: true, block(width: 100%)[
#figure(
  image("../images/heatmap_0.png", width: 95%),
  caption: [*Target Class 0 Accuracy Deltas for all models:* This heatmap visualizes the change ($delta$) in accuracy across all ten MNIST classes after performing SA on a specific target digit. Ideally, only the target column should show deep red (negative delta), while all other columns remain green which indicates successful forgetting with minimal collateral damage to remaining numbers.],
) <fig_pic_SAmodels>
])

#figure(
  image("../images/stability_boxplot_master.png", width: 95%),
  caption: [*Target Independent Reliability Distribution:* Show are the best results across all ten target digits to compare the reliability of each architecture. Each point represents a unique target class; a tight cluster near the $0.05$ threshold indicates a stable architecture that can reliably forget any digit. A large vertical spread (e.g., GANs) tell that forgetting is strongly dependent on the visual or complexity of the specific target digit where some numbers perform better than others, implying stronger entanglement between latent spaces. See @fig_entangelementGAN for a detailed overview and discussion.],
) <fig_pic_SAmodels>


#place(top + center, scope: "parent", float: true, block(width: 100%)[
#figure(
  table(
    columns: (auto, 2fr, 2fr, 2fr, 2fr, 2fr, 2fr, 2fr, 2fr, 2fr, 2fr, 2.2fr),
    inset: 4pt, stroke: none, align: center,
    table.hline(y: 0, stroke: 1.5pt),
    align(left)[*Model*], [*T0*], [*T1*], [*T2*], [*T3*], [*T4*], [*T5*], [*T6*], [*T7*], [*T8*], [*T9*], [*Mean*],
    table.hline(y: 1, stroke: 0.8pt),
    align(left)[VAE], [0.00
(-0.01)], [-], [-], [-], [-], [-], [-], [-], [-], [-], [*0.00*
*(-0.01)*],
    align(left)[GAN], [0.12
(+0.21)], [-], [-], [-], [-], [-], [-], [-], [-], [-], [*0.12*
*(+0.21)*],
    align(left)[RectifiedFlow], [0.00
(-0.00)], [-], [-], [-], [-], [-], [-], [-], [-], [-], [*0.00*
*(-0.00)*],
    align(left)[Autoregressive], [0.00
(+0.10)], [-], [-], [-], [-], [-], [-], [-], [-], [-], [*0.00*
*(+0.10)*],
    align(left)[NVP], [0.10
(+0.01)], [-], [-], [-], [-], [-], [-], [-], [-], [-], [*0.10*
*(+0.01)*],
    table.hline(y: 6, stroke: 1.5pt)
  ),
  caption: [Target accuracy and average retained accuracy drop across all 10 targeted classes. 0.0 target accuracy and 0.0 drop indicate optimal SA. Format: Target Acc (Drop).],
)
<tab_all_accs>])


TODO irgendwo anders hin packen das following??

== VAE
Before/After forgetting Graphic, compare to paper (generation accuracy), validity as baseline. Latent space visualization before/after forgetting, similar to NVP model


== Normalizing Flows
The results (todo) demonstrate successful selective amnesia. Before, the model generated the digit '0' with high fidelity (97% Oracle accuracy). After applying amnesia, the generation accuracy for '0' drops to 10.3%.

#figure(
  image("/images/nvp_before_forgetting.png"),
  caption: "Grid generated by NVP model before applying selective amnesia"
)

#figure(
  image("/images/nvp_after_forgetting.png"),
  caption: "Grid generated by NVP model after applying selective amnesia to forget the digit '0'"
)

#figure(
  placement: bottom,
  image("/images/latent_space_visualization_nvp.png"),
  caption: "Visualization of the latent space of the NVP model. Todo get better example where zeroes are better clustered, explain and compare to VAE latent space visualization.",
  scope: "parent"
) <latent_visualization>

Due to the hybrid architecture, we forget in latent space. This results in random numbers instead of noise similar to other models such as pure VAE.
Because there are 10 classes in MNIST, an accuracy of ~10% is equivalent to random chance. This validates the theoretical implication of "latent space amnesia". Because the model forgets how to locate the '0' cluster in the latent space (see @latent_visualization), but the underlying VAE decoder (which can only draw valid digits) is left intact, prompting the model to draw a '0' results in the Flow outputting a random valid coordinate. The VAE then draws whatever random digit exists at that coordinate. The model does not output a noise grey square; it hallucinates a random number. (todo discuss advantages/disadvantages random vs. noise)


Furthermore, compared to other methods (todo get reference) the catastrophic forgetting metric is low (x% drop). This proves that the Generative Replay and EWC constraints successfully isolated the destructive gradient updates strictly to the conditional pathways associated with the digit '0'.

NOTES:
- Epochs are very important, often results were bad for both training and SA due to not enough epochs. especially GAN and RF!

- still some variation in the best parameters across different digits. A 1 is a simple, straight line, whereas an 8 is a complex loop highly entangled with 3s and 0s in the latent space. A GAN might need a massive gamma to safely unlearn an 8, but only a small gamma to unlearn a 1.


]


== General Discussion #picon(col-lela)
Discuss the implications of the bottleneck hypothesis here.  (?)

Perhaps remove this, the conclusions section can be used to discuss the overall results that do not pertain merely to individual models
