#import "../config.typ": *


#place(top + center, scope: "parent", float: true, block(width: 100%)[
#pad(x: 2cm)[
      #set align(center)
      #heading(outlined: false, numbering: none)[Abstract #picon(col-lela)]
      
      #set align(left)
      This study investigates the efficacy of the Selective Amnesia (SA) framework across multiple architectures, including a conditional Variational Autoencoder (cVAE), a Normalizing Flows approach (RealNVPs), Rectified Flow (cRF), Generative Adversarial Network (cGAN), and an Autoregressive model (cMADE). Our primary goal is the evaluation of these models "unlearning" by the removal of specific class information without full retraining.
      #text(fill: red)[todo test results and finding]
    ]
  ]
)