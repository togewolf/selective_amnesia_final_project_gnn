Mehdy:
Rectified Flow Paper https://arxiv.org/pdf/2209.03003 
Autoencoder MADE https://arxiv.org/pdf/1502.03509

Rectified Flow:
SA works perfectly well, class 0 drops 0% with only 3.4% degradation on retained classes. the image shows class 0 becomes static noise after forgetting, while other digits remain at similar quality. however the baseline generation quality is already subpar (60.4% target acc). we used an MLP based velocity network to reuse the MLP pipeline of the "original" VAE implementation. The mlp struggles to capture the full spatial structure of 784-dimensional pixel space. A convolutional U-Net would have been more appropriate for image generation, but the MLP already serves very well as a proof of concept for the application of the SA framework applied to flow matching models demonstrating great results and such we kept it at that.

Autoencode made:
Made shows difficulty with SA. while it achieves a decent baseline (68.2% target acc) its forgetting is incomplete (forgetting precision of 10.7%). The sample image grid confirm that the top row is still producing some digit like stuctures after forgetting.
this can be attributed to several factors:
distributed class information: in the MADE model, class knowledge is spread across 784 conditional pixel distributions, making it harder to selectively erase compared to a latent space model
slow generative replay: each replay sample requires 784 sequential forward passes, making replay computationally expensive
masking constraints: the autoregressive masks make the model less flexible at adapting to only class specific behavior without affecting shared features as one can see in the degradation of class 6 (similar form) 

one can say that the SA framework works best with models that have a compact, well seperated internal representation of classes. models where class information is distributed across many sequential decisions, as is the case for autoregressive, are inherently harder to selectively forget.