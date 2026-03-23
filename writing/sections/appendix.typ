#import "../config.typ": *
#set heading(numbering: none)
= Appendix #picon(col-lela)
This additional material provides problem examples, exemplary image outputs, and the detailed results.

#set heading(numbering: "A.1")
== Hyperparameter Testing #picon(col-lela) <param_test>
#v(1em)
#figure(
  image("../images/trends_0_grid.png", width: 95%),
  caption: [*Hyperparameter Sensitivity Analysis:* These diagrams show different hyperparameters used with SA training on target class accuracy. The $gamma$ and $lambda$ plots track the tradeoff between forgetting strength and structural preservation, while the loss function and learning rate plots identify the optimal convergence stability for each architecture. For all, the lower accuracy, the better the parameter. Not all parameters are optimized, so only the trends, not the accuracies themselves are meaningful.],
) <fig_hyperparamtrend>

== Mode Collapse #picon(col-lela)<mode_collapse>

#figure(
  image("../images/mode_collapse_gan.png", width: 45%),
  caption: [Mode collapse is a common failure mode in GANs where the generator begins creating the same looking number for all samples due to a lack of diversity ("what works, works"). This particular example stems from training not enough epochs, in this case 5. Only after $approx$200 epochs, slightly better results for both catastrophic forgetting (entanglement) and unlearning the number. Looking at @fig_pic_basemodels, we clearly see that this still affects even our best model which was trained on 300 epochs. This indicates additional optimization problems of our parameters and architecture.],
)


== Study of Other Target Classes #picon(col-lela)<all_target_classes>
#v(1em)
To analyze the ability of each model to forget any target class we introduce "Entanglement Matrices" which show catastrophic forgetting for each target class. The diagonal represents the intended forgetting, while any significant red cells outside of the diagonal reveal some "concept entanglement" where deleting the target digit accidentally alters the concept of a similar class. 
#grid(
  columns: (auto, auto),
  [#figure(
    image("../images/entanglement_matrix_Autoregressive.png", width: 95%),
    caption: [VAE],
  )],
  [#figure(
    image("../images/entanglement_matrix_Autoregressive.png", width: 95%),
    caption: [RealNVP],
  )],
  [#figure(
    image("../images/entanglement_matrix_Autoregressive.png", width: 95%),
    caption: [Autoregressive],
  )],
  [#figure(
    image("../images/entanglement_matrix_Autoregressive.png", width: 95%),
    caption: [RF],
  )],
)

#v(0.5em)

== Model Outputs #picon(col-lela)

=== Base Models
#v(1em)
#figure(
  image("../images/base_models_examples.png", width: 95%),
  caption: [TODO describe],
) <fig_pic_SAmodels>

== Selective Amnesia applied Models
#figure(
  image("../images/base_models_examples.png", width: 95%),
  caption: [Target class 0 TODO describe. ],
) <fig_pic_basemodels>
