---
title: "Advanced Topics in Machine Learning - Notes part 2"
date: 2023-10-18T11:30:03+00:00
weight: 1
mathjax: true
editPost:
    URL: "https://alexserra98.github.io/alexserra98/posts/advml_2h/"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---
## Regularization
We continue our analysis of Regularization techniques:
<br>

### Batch normalization
Quoting from Wikipedia:
"Batch normalization (also known as batch norm) is a method used to make training of artificial neural networks faster and more stable through normalization of the layers' inputs by re-centering and re-scaling"<br>
Batch normalization is a very effective technique to justify its importance we report the following advantages listed in the [article](https://towardsdatascience.com/batch-normalization-8a2e585775c9)
1. Networks train faster — Each training iteration will actually be slower because of the extra calculations during the forward pass and the additional hyperparameters to train during backpropagation. However, it should converge much more quickly, so training should be faster overall.
2. Allows higher learning rates — Gradient descent usually requires small learning rates for the network to converge. And as networks get deeper, their gradients get smaller during backpropagation so they require even more iterations. Using batch normalization allows us to use much higher learning rates, which further increases the speed at which networks train.
3. Makes weights easier to initialize — Weight initialization can be difficult, and it’s even more difficult when creating deeper networks. Batch normalization seems to allow us to be much less careful about choosing our initial starting weights.
4. Makes more activation functions viable — Some activation functions do not work well in some situations. Sigmoids lose their gradient pretty quickly, which means they can’t be used in deep networks. ReLUs often die out during training, where they stop learning completely, so we need to be careful about the range of values fed into them. Because batch normalization regulates the values going into each activation function, non-linearities that don’t seem to work well in deep networks actually become viable again.
5. Simplifies the creation of deeper networks — Because of the first 4 items listed above, it is easier to build and faster to train deeper neural networks when using batch normalization. And it’s been shown that deeper networks generally produce better results, so that’s great.
Provides a bit of regularization — Batch normalization adds a little noise to your network. In some cases, such as in Inception modules, batch normalization has been shown to work as well as dropout. But in general, consider batch normalization as a bit of extra regularization, possibly allowing you to reduce some of the dropout you might add to a network.
7. May give better results overall — Some tests seem to show batch normalization actually improves the training results. However, it’s really an optimization to help train faster, so you shouldn’t think of it as a way to make your network better. But since it lets you train networks faster, that means you can iterate over more designs more quickly. It also lets you build deeper networks, which are usually better. So when you factor in everything, you’re probably going to end up with better results if you build your networks with batch normalization. <br>

**smoothing of the objective function**:
In the initial paper in which the method is proposed [Ioffe et al.](https://arxiv.org/abs/1502.03167) they account the effectiveness of this technique to the reduction of the so-called *internal covariance shift*. In fact, recent works showed that batch normalization does not reduce internal covariate shift, but rather smooths the objective function, which in turn improves the performance <br>
The argument proposed in [Santurkar et al.](https://arxiv.org/pdf/1805.11604.pdf) is  BatchNorm produces 
the parameter landscape significantly more smooth. It can be shown that the Lipschitzness of the loss function improves as the magnitudes of the gradients get smaller and as a consequence the loss changes at a smaller rate. In addition the quadratic form of the loss Hessian with respect to the activations in the gradient direction is both rescaled by the input variance (inducing resilience to mini-batch variance), and decreased by an additive factor (increasing smoothness). Since this term represents the second order term of the Taylor expansion of the gradient around the current point reducing it results in making the first order term (the gradient) more predictive. <br>

Quoting from the paper we can see the connection with the last remarks made at the end of the gradient descent section:

"These smoothening effects impact the performance of the training algorithm in a major way. To
understand why, recall that in a vanilla (non-BatchNorm), deep neural network, the loss function
is not only non-convex but also tends to have a large number of “kinks”, flat regions, and sharp
minima. This makes gradient descent–based training algorithms unstable, e.g., due to exploding
or vanishing gradients, and thus highly sensitive to the choice of the learning rate and initialization.
Now, the key implication of BatchNorm’s reparametrization is that it makes the gradients more
reliable and predictive. After all, improved Lipschitzness of the gradients gives us confidence that
when we take a larger step in the direction of a computed gradient, this gradient direction remains a
fairly accurate estimate of the actual gradient direction after taking that step. It thus enables any
(gradient–based) training algorithm to take larger steps without the danger of running into a sudden
change of the loss landscape such as a flat region (corresponding to vanishing gradient) or sharp local
minimum (causing exploding gradients). This, in turn, enables us to use a broader range of (and thus
larger) learning rates and, in general, makes the training significantly
faster and less sensitive to hyperparameter choices."

It's important to remark that the above analysis works in the setup of inserting a single batch norm in a network. Stacking batchnorms results in gradient explosion at initialization time. The optimization landscape is then extremely non-smooth and then the deep batch norm networks are practically untrainable
<br>

### Convolutional Neural Networks
Another source of implicit bias comes from the architecture of our neural network. 
**Sparsity in the frequency domain**
We start by reporting some key results from the following paper [Shanahan et al.](https://arxiv.org/abs/1905.10307) <br>
In this work, the authors focus on the implicit bias of optimizing multi-layer fully connected linear networks, and linear convolutional
networks. If we simplify our problem by excluding non-linear activation functions both of these types of models ultimately implement linear transformations and thus they belong to the class of all linear predictors. Minimizing the training loss on these models is therefore entirely equivalent to minimizing the training loss for linear classification. 
We indicate a linear predictor as `$\beta \in \mathbb{R}^{D}$` and define the map `$\mathcal{P}: \mathcal{W} \rightarrow \mathbb{R}^D$` which sends a set of weights in the corresponding linear predictor.
In the case of a binary classification task using logistic loss the global minimum of a loss `$\mathcal{L}(\cdot)$`  defined as 
`$$
\min_{\mathbf{w} \in \mathcal{W}} \mathcal{L}_{\mathcal{P}}(\mathbf{w}):=\sum^N \ell\left(\left\langle\mathbf{x}_n, \mathcal{P}(\mathbf{w})\right\rangle, y_n\right)
$$`
can't be achieved for any finite `$\beta$`. To overcome this problem the loss can be minimized by scaling the norm of any linear predictor that separates the data to infinity, focusing then on the direction `$\overline{\boldsymbol{\beta}}^{\infty}=\lim _{t \rightarrow \infty} \frac{\boldsymbol{\beta}^{(t)}}{\left\|\boldsymbol{\beta}^{(t)}\right\|}$` <br>
Under some assumptions described in the paper, the authors prove that:
For any depth L, almost all linearly separable
datasets `$\{ x_{n}, y_{n} \}^{N}_{n=1}$` almost all initializations `$w^{(0)}$`, , and any bounded sequence of step sizes `$\{\eta_{t}\}_{t}$`, consider the sequence gradient descent iterates `$w^{(t)}$`  defined as:
`$$
\mathbf{w}^{(t+1)}=\mathbf{w}^{(t)}-\eta_t \nabla_{\mathbf{w}} \mathcal{L}_{\mathcal{P}}\left(\mathbf{w}^{(t)}\right)=\mathbf{w}^{(t)}-\eta_t \nabla_{\mathbf{w}} \mathcal{P}\left(\mathbf{w}^{(t)}\right) \nabla_{\boldsymbol{\beta}} \mathcal{L}(\mathcal{P}(\mathbf{w}(t)))
$$`
for minimizing `$\mathcal{L}_{\mathcal{P}_{\text{full}}}(\mathbf{w})$` as described above  with exponential loss `$\ell(\widehat{y}, y)=\exp (-\widehat{y} y)$`  over L–layer fully connected linear networks. <br>

If  the iterates `$w^{(t)}$` minimize the objective, i.e., `$\mathcal{L}_{\mathcal{P}_{\text{full}}}(w(t)) \rightarrow 0$`,  `$w(t)$`, and consequently `$\beta^{(t)} = \mathcal{P}_{\text{full}}(w(t))$`, converge in direction to yield a separator with positive margin, and  gradients
concerning linear predictors ∇βL(β(t)) converge in direction, and then the limit direction is given by
`$$
\overline{\boldsymbol{\beta}}^{\infty}=\lim _{t \rightarrow \infty} \frac{\mathcal{P}_{\text {full }}\left(\mathbf{w}^{(t)}\right)}{\left\|\mathcal{P}_{\text {full }}\left(\mathbf{w}^{(t)}\right)\right\|}=\frac{\boldsymbol{\beta}_{\ell_2}^*}{\left\|\boldsymbol{\beta}_{\ell_0}^*\right\|} \text {, where } \boldsymbol{\beta}_{\ell_2}^*:=\underset{w}{\operatorname{argmin}}\|\boldsymbol{\beta}\|_2^2 \quad \text { s.t. } \forall n, y_n\left\langle\mathbf{x}_n, \boldsymbol{\beta}\right\rangle \geq 1
$$`

This theorem that the implicti bias of gradient descent doesn't depend on the depth of the net. Remarkably the asymptotic
classifier is always the hard margin support vector machine classifier, which is also the limit direction
of gradient descent for linear logistic regression with the direct parameterization of `$\beta = w$`.

We proceed now to show the equivalent of the above theorem for linear convolutional networks.
A few notes on the notation:
- `$\widehat{\boldsymbol{\beta}} \in \mathbb{C}^D$` are the Fourier coefficients of `$\beta$` i.e
`$$
\widehat{\boldsymbol{\beta}}[d]=\frac{1}{\sqrt{D}} \sum_{p=0}^{D-1} \boldsymbol{\beta}[p] \exp \left(-\frac{2 \pi \mathrm{i} p d}{D}\right)
$$`
- complex number are denoted in polar form: `$z=|z| \mathrm{e}^{\mathrm{i} \phi_z}$` for `$\psi_z \in [0, 2\pi)$`
We now claim: <br>
For almost all linearly separable datasets `${x_{n}, y_{n}}^{N}_{n=1}$`, almost all initializations `$w(0)$`, and any sequence of step sizes `$\{ \eta_{t}\}_{t}$` with `$\eta_{t}$` smaller than the local Lipschitz at `$w(t)$`, consider the sequence gradient descent iterates w(t) described before for minimizing 
`$\mathcal{L}_{\mathcal{P}_\text{conv}}(w)$` in with exponential loss over 2–2-layer linear convolutional networks. If the iterates `$w(t)$` minimize the objective, i.e., `$\mathcal{L}_{{P}\text{conv}}(w(t)) \rightarrow 0$`, `$w(t)$` converge in direction to yield a separator `$\overline{\boldsymbol{\beta}}^{\infty}$` with positive margin, the phase of the Fourier coefficients `$\widehat{\boldsymbol{\beta}}^{(t)}$` of the linear predictors `$\beta(t)$` converge coordinate-wise, i.e., `$\forall d$`, `$\exp^{i\phi}\widehat{\beta^{(t)}}[d] \rightarrow \exp^{i\phi}\overline{\widehat{\beta^{(t)}}}[d]$`, and (d) the gradients `$\nabla_{\boldsymbol{\beta}} \mathcal{L}\left(\boldsymbol{\beta}^{(t)}\right)$` converge in direction, then the limit direction `$\overline{\boldsymbol{\beta}}^{\infty}$` is given by,
`$$
\overline{\boldsymbol{\beta}}^{\infty}=\frac{\boldsymbol{\beta}_{\mathcal{F}, 1}^*}{\left\|\boldsymbol{\beta}_{\mathcal{F}, 1}^*\right\|}, \text { where } \boldsymbol{\beta}_{\mathcal{F}, 1}^*:=\underset{\boldsymbol{\beta}}{\operatorname{argmin}}\|\widehat{\boldsymbol{\beta}}\|_1 \text { s.t. } \forall n, y_n\left\langle\boldsymbol{\beta}, \mathbf{x}_n\right\rangle \geq 1 .
$$`

The result above shows if we introduce a convolutional layer, even without explicit regularization, gradient descent is implicitly biased through solutions that are sparse in the frequency domain. <br>



### Double descent
Before diving deep into the topic, there's a phenomenon that is worth mentioning called [Nakkiran et al.](https://openai.com/research/deep-double-descent). <br>
In a series of very popular models like ResNets, ConvNet, and transformers, performance first improves, then gets worse, and then we reach an *interpolation threshold* at which the model is barely enough to fit the train set the performance improves again. <br>
A possible intuition about this phenomenon:
- Case 1: `$|\text{dataset}| >> |\text{parameters}|$` There's way more than enough data to pin down the optimal parameters so it generalizes well.
- Case 2: `$|\text{dataset}| \approx |\text{parameters}|$` The model is just right to fit the data, it can memorize the data but can't generalize. , there is effectively only one model that fits the train data, and forcing it to fit even slightly noisy or misspecified labels will destroy its global structure. 
- Case 3: `$|\text{dataset}| << |\text{parameters}|$` The model can fit the data in many different ways, and regularization bias toward solutions that generalize well.
<br>

### Spectral bias


We start our analysis with **Spectral bias** which is the tendency of neural networks to learn functions that are smooth in the frequency domain. <br>
Is very well known that deep fully-connected networks struggle to accurately approximate high-frequency or multi-scale functions and in the case of *PINN* has been shown that one of the leading reasons is the presence of spectral bias [Wang et al.](https://arxiv.org/abs/2007.14527). The following analysis is taken from [Wang et al.](https://www.sciencedirect.com/science/article/pii/S0045782521002759) to which we refer for details. <br>
We start by proposing the last calculation done in NTK sectio (add a link!) for which we were getting:
`$$
f(\mathcal{X},\theta) \approx \left(I-e^{-\eta K_{\infty} t}\right) \mathcal{Y}
$$`
Since K_{\infty} is a positive definite matrix we can decompose it as `$K_{\infty}=Q \Lambda Q^{T}$` where `$\Lambda$` is a diagonal matrix containing the eigenvalues of `$K_{\infty}$` and `$Q$` is the matrix of the eigenvectors. <br>
We can use `$e^{-K t}=Q^T e^{-\Lambda t} Q$` and get: 
`$$
\boldsymbol{Q}^T\left(f\left(\mathcal{X}, \boldsymbol{\theta}(t)\right)-\mathcal{Y} \right)=-e^{\boldsymbol{\Lambda} t} \boldsymbol{Q}^T \mathcal{Y}
$$`

`$$
\left[\begin{array}{c}
\boldsymbol{q}_1^T \\
\boldsymbol{q}_2^T \\
\vdots \\
\boldsymbol{q}_N^T
\end{array}\right]\left(f\left(\mathcal{X}, \boldsymbol{\theta}(t)\right)-\mathcal{Y} \right)=\left[\begin{array}{llll}
e^{-\lambda_1 t} & & & \\
& e^{-\lambda_2 t} & & \\
& & \ddots & \\
& & & e^{-\lambda_N t}
\end{array}\right]\left[\begin{array}{c}
\boldsymbol{q}_1^T \\
\boldsymbol{q}_2^T \\
\vdots \\
\boldsymbol{q}_N^T
\end{array}\right] \mathcal{Y}
$$`

The above equation shows that the convergence rate of `$\boldsymbol{q}_i^T\left(f\left(\mathcal{X}, \boldsymbol{\theta}(t)\right)-\mathcal{Y} \right)$` is determined by the i-th eigenvalue
`$\lambda_{i}$`. Moreover, we can decompose the training error into the eigenspace of the NTK as
`$$
\begin{aligned}
f\left(\mathcal{X}, \boldsymbol{\theta}(t)\right)-\mathcal{Y} & =\sum_{i=1}^N\left(f\left(\mathcal{X}, \boldsymbol{\theta}(t)\right)-\mathcal{Y}, \boldsymbol{q}_i\right) \boldsymbol{q}_i \\
& =\sum_{i=1}^N \boldsymbol{q}_i^T\left(f\left(\mathcal{X}, \boldsymbol{\theta}(t)\right)-\mathcal{Y}\right) \boldsymbol{q}_i \\
& =\sum_{i=1}^N\left(e^{-\lambda_i t} \boldsymbol{q}_i^T \mathcal{Y}\right) \boldsymbol{q}_i .
\end{aligned}
$$`
Clearly, the network is biased to first learn the target function along the eigendirections of neural tangent kernel with
larger eigenvalues, and then the remaining components corresponding to smaller eigenvalues. For conventional fully-connected
neural networks, the eigenvalues of the NTK shrink monotonically as the frequency of the corresponding eigenfunctions
increases, yielding a significantly lower convergence rate for high-frequency components of the target function.
This indeed reveals the so-called “spectral bias” pathology of deep neural networks.
{{<figure align=center src="images/spectral_bias.png" caption="Figure 1. The learnt function -green- overlayed on the target function -blue- as the training progresses. The target function is a superpositionvof sinusoids equal amplitudes and randomly sampled phases source: On the Spectral Bias of Neural Networks">}}

### Closing remarks
We now conclude our analysis of implicit bias by revising the cornerstones enounced at the beginning of the section:
- Capacity: What's the true complexity measure? We've seen that the number of parameters is usually a poor measure of the capacity of a model because we're not taking into account the implicit regularization added to the loss.
How does the "true" complexity measure control generalization?
- Expressive power: The set of functions that can be represented by a neural network depends in practice on the inductive bias induced by the optimization algorithm and the architecture. 
Does the implicit bias actually match the prior over the distribution that we're trying to infer?
- Computation:  How and where does the optimization bias us? Under what conditions? <br>


## Weight initialization
This is just a reminder to take a look to [weight initialization](https://www.deeplearning.ai/ai-notes/initialization/index.html).
It explains why initializing weights to 0 or just with purely random numbers is a bad choice and how Xavier's initialization allows to have:
- mean of all activations equal to 0
- variance of all activations stays the same across every layer

for a deeper understanding of the math background, we refer to [Glorot et al.](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
<br>

## Invariance and Equivariance
In this section, we will introduce two concepts that will be crucial for the following posts of this series of notes: the concept of invariance and equivariance.We will show then their connection with some architectural bias introduced by pooling and convolution <br>
Given a function `$f: A \rightarrow A $` and a group $G$ acting on $A$
we call $f$:
- **invariant**:  if `$f(gx)=f(x)$` for all `$g \in G$` and `$x \in A$`
- **equivariant**: if $f(gx)=g f(x)$ for all `$g \in G$` and `$x \in A$` <br>

### Pooling and Invariance
This stage is usually performed after a convolutional layer and consists in reducing the dimensionality of the input, by substituting a set of values with a summary statistic of the nearby values. 
If we translate the input by a small amount the pooled output stays the same and for this reason, pooling can provide representations that are approximately invariant by small translations. 
In order to show this property we suppose that the pooling is made over the entire picture to simplify the notation and make it more explicit. 
What is the bias imposed using pooling?<br>
We consider a kernel $K$ and we call $G$ the set of translations that we assume have the structure of a group. We define the i-th element of the output of the convolutional layer with input `$x \in \mathbb{R}^N$` as:
`$$
\left\langle g_i K, x\right\rangle=(x * K)_i 
$$`
We then pool over the entire output:

`$$
f(x)=\sum_i\left\langle g_i K, x\right\rangle \\
$$`

We can see now that:

`$$
f(g x)=\sum_i\left\langle g_i K, g x\right\rangle=\sum_i\left\langle g^T g_i K, x\right\rangle=\sum_i\left\langle g_i^{\prime} K, x\right\rangle
$$`

with `$g^T g_i=g_i^{\prime}$`. But here we have just reshuffled the elements of this sum thus

`$$
f(g x)=f(x) 
$$`

### Convolution and Equivariance
We now consider the convolutional layer and we show that it's equivariant to translation. <br>
We use the same notation as in pooling but here `$f(\cdot)$` is the first convolutional layer and `$\sigma(\cdot)$` is the activation function. We have:
`$$
f(g x)=\left[\begin{array}{c}
\sigma\left\langle g_1 w, g x\right\rangle \\
\vdots \\
\sigma\left\langle g_N w, g x\right\rangle
\end{array}\right]=\left[\begin{array}{c}
\sigma\left\langle w, g_1^T g x\right\rangle \\
\vdots \\
\sigma\left\langle w, g_N^T g x\right\rangle
\end{array}\right]=P_g\left[\begin{array}{c}
\sigma\langle w, g x\rangle \\
\vdots \\
\sigma\langle w, g x\rangle
\end{array}\right]=P_g f(x)
$$`
In other words the translation of the input vector $x$ commutes with the activation function and hence the equivariance:

`$$
f(g x)=P_g f(x)
$$`

**Link between equivariance and convolutionality**

We want to analyze now a very important theorem presented by Kondor and Trivedi in this [Kondor et al.](https://arxiv.org/pdf/1802.03690.pdf)
In this [site](https://dmol.pub/dl/Equivariant.html#groups-on-spaces)
there's a very nice explanation of many concepts of this paper. <br>
G-Equivariant Convolution Theorem:
A neural network layer (linear map) `$\phi$` is G-equivariant if and only if its form is a convolution operator `$*$`
`$$
\psi(f)=(f * \omega)(u)=\sum_{g \in G} f \uparrow^G\left(u g^{-1}\right) \omega \uparrow^G(g)
$$`
where  `$f: U \rightarrow \mathbb{R}^n$` and `$\omega: V \rightarrow \mathbb{R}^n$` are "lifted" functions of the group $U$ and $V$ which are subgroups of $G$ (including possible $G$). On the first layer of a neural network, $f$ is usually defined on the quotient space `$U= \frac{G}{H}$`. If the group $G$ is locally compact, then the convolution operator is 
`$$
\psi(f)=(f * \omega)(u)=\int_G f \uparrow^G\left(u g^{-1}\right) \omega \uparrow^G(g) d \mu(g)
$$` 
where $\mu$ is the Haar measure on $G$. <br>
The blog that is linked above explains exhaustively the above theorem, here I just want to point out a few things that were not so intuitive.<br>
The first confusing thing is the fact that they represent layers in a rather more abstract view, in order to understand it we start by defining the notation: `$L_V(\mathcal{X})$` that denotes the space of function `$\{f: \mathcal{X} \rightarrow V\}$`.
All the activations in any given layer are collected in a function:
`$f^{\ell}: \mathcal{X}_{\ell} \rightarrow V_{\ell}$` where `$\mathcal{X}_{\ell}$` is a set indexing the neurons and `$\mathcal{V}_{\ell}$` is a vector space, in the first layer `$\mathcal{X}_{\ell}$` is the set of inputs. In this way we can represent layers as linear maps $\phi^{\ell}: L_{V_{\ell}}(\mathcal{X}_{\ell}) \rightarrow L_{V_{\ell+1}}(\mathcal{X}_{\ell+1})$. 
So in the first layer we are supposing that of each of our samples of the training set comes from a space `$\mathcal{X}_{0}$` and that there's a function $f_{0}$ that sends this sample to an actual tensor so basically `$f_{0}(x)$` is just one record in the dataset that we fed to the model. 
But in the theorem, $f$ is defined on the quotient space `$U= \frac{G}{H}$`, so how can we reconcile these two things? <br>
The fact is that the authors of the papers wanted to work on a much broader generalization of convolution in which the arguments are functions on a compact group $G$.
In this case, the convolution is defined as:
`$$
(f * g)(u)=\int_G f\left(u v^{-1}\right) g(v) d \mu(v) .
$$`
The problem now is to connect the structure of a group $G$ with its homogeneous space `$\mathcal{X}$` and we want to provide intuition behind it. <br>
Since the `$\mathcal{X}$` is homogeneous if we fix an `$x_{0}$` , for any $x$ we always have a $g$ s.t `$x=g(x_{0})$`. In this way, we can "index" `$\mathcal{X}$` through elements of $G$. Now the set of elements of $G$ that sends $x_{0}$ in itself is a subgroup $H_{0}$ called the "stabilizer" and all the $g_{i}$ that sends $x_{0}$ are coset of this stabilizer we can then chose a coset representative and denote it $\bar{x}$ meaning that it sends $x_{0}$ to $x$.
We can now make an equivalent $f$ defined now on a group this operation is called "lifting" and we obtain a new function:
`$f \uparrow^G(g):\frac{G}{H_{0}} \rightarrow V$`
defined as:
`$$
f \uparrow^G(g)=f\left(g x_0\right)
$$`
As stated in the blog:<br>
"This is a strong theorem. It says there is only one way to achieve equivariance in a neural network. This may seem counter-intuitive since there are many competing approaches to convolutions. These other approaches are actually equivalent to a convolution; just it can be hard to notice."<br>

### Data Augmentation
We have shown that in order to make the neural network learn invariant/ equivariant representations we need to change the architecture to add pooling and convolutional layers.<br>
There's actually a simpler approach very much used in practice that consists in extending the dataset with some of its samples modified. Generally, it means multiplying the original sample by elements of a group or simply injecting noise in it.<br>

We want to show now how data augmentation can be used to construct loss an invariant to the action of a group $G$. To make the result more explicit we suppose that the augmented dataset contains the orbits w.r.t the group of each sample. <br>
Let $G$ be a group acting on the dataset `$\mathcal{X}$`, `$\mathcal{l}(\cdot)$` a generic loss, `$\sigma(\dot)$` and activation function, $w$ the weight of the model. We suppose that `$\forall x \in \mathcal{X}$`
then there exist an augmented dataset `$\mathcal{X}'$` s.t `$g_{\theta}x \in \mathcal{X}'$` for all `$g_{\theta} \in G$`. <br>
`$$
\mathcal{L}(w)=\frac{1}{N} \sum_{i=1}^N \int \ell\left(\sigma\left\langle w, g_\theta x_i\right\rangle ; y_i\right) d \theta
$$`
the loss defined in this way is indeed invariant:
`$$
\begin{aligned}
&\begin{aligned}
\mathcal{L}\left(g_{\bar{\theta}} w\right) & =\frac{1}{N} \sum_{i=1}^N \int \ell\left(\sigma\left\langle w, g^* g_\theta x_i\right\rangle ; y_i\right) d \theta \\
& =\sum_{i=1}^N \int \ell\left(\sigma\left\langle w, g_\theta x_i\right\rangle ; y_i\right) d \theta \\
& =\mathcal{L}(w)
\end{aligned}\\
&g_\theta g_{\theta^{\prime}}=g_{\theta^{\prime \prime}}
\end{aligned}
$$` 

It's interesting to review this approach from a more "harmonic" perspective. <br>
Using the idea above about the loss in general we can construct an invariant function in the following way: <br>
Let `$\bar{f}: \mathbb{R} \rightarrow \mathbb{R}$`  s.t `$\bar{f}(\omega+q)=\bar{f}(\omega) \forall q \in \mathbb{R}$`. `$\bar{f}$` is constant and  `$\widehat{f}(0)$` is the only non zero Fourier transform.

`$$
\underline{\mathrm{f}}(w)=\int_{-\infty}^{\infty} d t f(w-t), \quad f: \mathbb{R} \rightarrow \mathbb{R} .
$$`
Let's relax now the invariance property, as it happens in real applications, and settle for approximate invariance, that is, robustness. To do so we limit to integrate over $[-a,a]$ instead of the whole real line.
`$$
\underline{\mathbf{f}}(w)=\int_{-a}^a d t f(w-t)=\int_{-\infty}^{+\infty} d t \operatorname{Ind}_{[-a, a]}(t) f(w-t) .
$$`
taking the Fourier and usign the property `$\widehat{f(\cdot-t)}=e^{i k t} \hat{f}(k)$`:
`$$
\hat{\mathrm{f}}(k)=\left(\int_{-\infty}^{\infty} d t e^{i k t} \operatorname{Ind} d_{[-a, a]}(t)\right) \hat{f}(k)=2 a \operatorname{sinc}(2 k a) \hat{f}(k) .
$$`
Switching back to our original problem about the loss the above results show that if I do data augmentation the effect on the loss is to modulate the Fourier transform of some coefficient `$2a sin(2ka)$`, this is the implicit bias Because is biasing towards low frequencies!
But having a biased loss means having a biased gradient of the loss and thus the updated of weights using gradient descent are biased.<br>
For instance, if we consider:
`$$
\mathcal{L}\left(g_\theta w ; X, y\right)=\mathcal{L}(w ; X, y), \quad \theta \in R
$$`
where `$g_{\theta} \in G$` which under whose action is invariant, then:
`$$
\frac{d}{d \alpha} \mathcal{L}\left(U_\alpha w\right)=0 \rightarrow\left\langle\nabla_w \mathcal{L}(w), \partial_\alpha U_\alpha w\right\rangle=0
$$`
Thus some directions in the gradient are not allowed by the invariance of the loss. <br>
**Learned weights and symmetry**
As we have seen in the previous section data agumentation bias the loss function and consequently the gradient descent and the learned weights. The effect on the last is remarkably evident as we can see in the below picture:
{{<figure align=center src="images/learned_weights.png" caption="Figure 2. Learned weights in absence of data augmentation (a) and for,respectively, translation (b), rotation (c), scale (d) and horizontal flip (e) data augmentations. source  [Anselmi et al.](https://arts.units.it/retrieve/856df9ab-4e98-4bab-b71e-591336d25db0/)">}}

In  [Anselmi et al.](https://arts.units.it/retrieve/856df9ab-4e98-4bab-b71e-591336d25db0/) they analyze how data augmentation affects the loss in the frequency domain. Using the property:
`$$
\hat{\mathcal{L}}(g W)(K)=|\operatorname{det} g|^{-1} \hat{\mathcal{L}}\left(\left(g^T\right)^{-1} K\right)
$$`
### Permutation Equivariant layer
in the paper [Zaaher et al.](https://arxiv.org/pdf/1703.06114.pdf) the authors provide some theoretical understanding for designing a permutational equivarian layer.<br>
Using the definition of equivariance provided before let's define a neural network layer as:
`$$
\mathrm{f}_{\Theta}(\mathbf{x}) \doteq \sigma(\Theta \mathbf{x}) \quad \Theta \in \mathbb{R}^{M \times M}
$$`
where $\Theta$ is a weight vector and $\sigma$ is a non-linear activation function. The following lemma presented in the paper suggests a sufficient conditions for permutations-equivariance in this type of functions 
**Lemma 3** The function `$\mathbf{f}_{\Theta}: \mathbb{R}^M \rightarrow \mathbb{R}^M$` as defined above is permutation equivariant if and only if all the off-diagonal elements of $\Theta$ are tied together and all the diagonal elements are equal as well:
That is 
`$$
\Theta=\lambda \mathbf{I}+\gamma\left(\mathbf{1 1}^{\boldsymbol{\top}}\right) \quad \lambda, \gamma \in \mathbf{R} \quad \mathbf{1}=[1, \ldots, 1]^{\boldsymbol{\top}} \in \mathbb{R}^M
$$`
where `$\mathbf{I} \in \mathbb{R}^{M \times M}$` is the identity matrix. <br>

The key concept behind the proof consists in noting that <br>
`$$
\sigma(\Theta \Pi x) = \Pi \sigma(\Theta x) \\
\sigma(\Theta \Pi x) - \sigma(\Pi \Theta x) = 0
$$`
So we are asking `$\Theta \Pi = \Pi \Theta $` and then
`$$
\Pi^{top} \Theta \Pi = \Theta
$$`
thus we're looking for `$\Theta$` that is invariant under conjugation <br>
<br>

For a more detailed analysis:

- To see why `$\Theta=\lambda \mathbf{I}+\gamma\left(\mathbf{1} 1^{\top}\right)$` commutes with any permutation matrix first note that commutativity is linear- that is 
`$$
\Theta_1 \pi=\pi \Theta_1 \wedge \Theta_2 \pi=\pi \Theta_2 \quad \Rightarrow \quad\left(a \Theta_1+b \Theta_2\right) \pi=\pi\left(a \Theta_1+b \Theta_2\right)
$$`
Since both Identity matrix `$\boldsymbol{I}$`, and constant matrix `$\boldsymbol{11}^{\top}$`, commute with any permutation
matrix, so does their linear combination `$\Theta=\lambda \mathbf{I}+\gamma\left(\mathbf{1} 1^{\top}\right)$`.
    - We need to show that in a matrix `$\Theta$` that commutes with “all” permutation matrices
    - *All diagonal elements are identical*: Let `$\pi_{k,l}$` for `$1 \leq k, l \leq M$` `$k \neq l$`, be a transposition (i.e. a permutation that only swaps two elements).  The inverse permutation matrix of `$\pi_{k,l}$` l is the permutation matrix of `$\pi_{k,l} = \pi_{k,l}^{\top}$`. We see that commutativity of `$\Theta$` with the transposition `$\pi_{k,l}$` implies that `$\Theta_{k,k} = \Theta_{l,l}$`:

    $$
    \Theta_1 \pi=\pi \Theta_1 \wedge \Theta_2 \pi=\pi \Theta_2 \quad \Rightarrow \quad\left(a \Theta_1+b \Theta_2\right) \pi=\pi\left(a \Theta_1+b \Theta_2\right)
    $$
    Therefore, $\pi$ and $\Theta$ commute for any permutation $\pi$, they also commute for any transposition $\pi_{k,l}$ and therefore $\Theta_{i,i} = \lambda \forall i$.
 
    - All off-diagonal elements are identical: We show that since $\Theta$ commutes with any
    product of transpositions, any choice of two off-diagonal elements should be identical.
    Let $(i,j)$ and $(i',j')$ be the index of two off-diagonal elements (i.e $i \neq j$ and $i' \neq j'$). Moreover, for now assume
    $i \neq j$ and $i' \neq j'$. Application of the transposition $\pi_{i,i'}\Theta$, swaps the rows  $i,i'$ in $\Theta$. Similarly, $\Theta\pi_{j,j'}$ switches the $j^{th}$ column with $j'^{th}$ column. From commutativity property of $\Theta$ and $\pi \in S_n$  (the group of symmetry) we have:
    `$$
    \begin{aligned}
    & \pi_{j^{\prime}, j} \pi_{i, i^{\prime}} \Theta=\Theta \pi_{j^{\prime}, j} \pi_{i, i^{\prime}} \Rightarrow \pi_{j^{\prime}, j} \pi_{i, i^{\prime}} \Theta\left(\pi_{j^{\prime}, j} \pi_{i, i^{\prime}}\right)^{-1}=\Theta \\
    & \pi_{j^{\prime}, j} \pi_{i, i^{\prime}} \Theta \pi_{i^{\prime}, i} \pi_{j, j^{\prime}}=\Theta \Rightarrow\left(\pi_{j^{\prime}, j} \pi_{i, i^{\prime}} \Theta \pi_{i^{\prime}, i} \pi_{j, j^{\prime}}\right)_{i, j}=\Theta_{i, j} \quad \\
    & \Rightarrow \Theta_{i^{\prime}, j^{\prime}}=\Theta_{i, j} 
    \end{aligned} \quad 
    $$`
    where in the last step we used our assumptions that $i \neq i^{\prime}, j \neq j^{\prime}, i \neq j$ and $i' \neq j'$. In the cases where either $i=i^{\prime}$ or $j=j^{\prime}$, we can use the above to show that `$\Theta_{i, j}=\Theta_{i^{\prime \prime}, j^{\prime \prime}}$` and `$\Theta_{i^{\prime}, j^{\prime}}=\Theta_{i^{\prime \prime}, j^{\prime \prime}}$` for some `$i^{\prime \prime} \neq i, i^{\prime}$ and $j^{\prime \prime} \neq j, j^{\prime}$` and conclude `$\Theta_{i,j} = \Theta_{i',j'}$`



## Interpretability
As machine learning-based algorithms are increasingly used in real-world applications, the need for understanding their decisions is becoming more and more important. 
Unfortunately, ANNs are still a blackboxs but some strategies has been devised to try to shed light on what's going on underneath.
In this section, we will highlight some concepts and approaches regarding the interpretability of neural networks. <br>
A very naive approach is to use designed features that are also interpretable, but this is limiting the expressive power of the network. This approach is called *signal processing* and was famous a few decades ago.
This shows one of the main problems with current machine learning: it's a **trade-off between performance and interpretability**. <br>
When the model is agnostic to explicit biases the data "speaks" for themselves but the results are difficult/impossible to interpret. <br>

**Feature Visualization**
Another way to make ANN more interpretable is to devise a way to visualize the features that the network is learning. <br>
On the [page](https://distill.pub/2017/feature-visualization/) there's a beatiful introduction to this topic. <br>
The images below are made creating the input that "excites" the neurons the most per each layer. It's interesting to notice how the pattern to which each layer is more sensible becomes progressively complicated.<br>

During the lessons, we introduced a technique for *feature visualization* using **attention map**. <br>
The idea is that we want to create a map in the Fourier space for the images in order to filter out the frequencies that affect the most the prediciton.
We start by training our net to solve a task, for example, recognizing cats from dogs. 
Then we create a mask that has to be trained to filter the most important frequencies. The training iterate will consist of taking a pic from the dataset, mapping it in the Fourier Space, applying the mask, and feeding this preprocessed image to the initial network.
We then calculate the loss taking into account the error in the prediction and the sparsity of the mask. <br>
The goal is to have the sparser mask that still allows the network to perform well on the task. <br> 
The result of this procedure can be seen below:
{{<figure align=center src="images/feat_viz.png" caption="Figure 3. Feature visualization allows us to see how GoogLeNet, trained on the ImageNet dataset, builds up its understanding of images over many layers. Visualizations of all channels are available in the appendix. source: https://distill.pub/2017/feature-visualization/">}}

As we can see the elements on which the network is focusing are very different from the ones that we would have expected. The majority of the frequencies are killed and only very few are retained, this raises a possible risk of exploitation cause we can fool the network into misclassifying elements by constructing on purpose ambiguous images.<br>

An interesting pathology presented in this [Geirhos et al.](/https://arxiv.org/abs/1811.12231) shows that CNNs classified on Image-Net seem to be biased towards texture rather than shape. It's interesting to notice that the texture is pattern to which the first layers pay more attention as we can see in the previous pic. <br>
**Physical Laws**
Up to now, we have seen examples only for images but there's other contexts in which the nature of the problem allows some simplifications in terms of interpretability. <br>
From a more computer-scientific perspective physical laws can be seen as very efficient algorithms to predict natural phenomena. The key aspects the affect how to compose such rules are:
- select relevant variables
- find the simplest function of the variable that explains the experiments
The last rule is based on the rather romantic idea that the universe is simple and elegant, but as a matter of fact, they reflect just the amount of complexity that our brain can deal with. Another requirement that we need to make sense of physical law is that is to be written in "language" that we can understand, that is through interpretable objects like: 
`$$
\partial_{\mathbf{x}}, \partial_{\mathbf{t}}, \nabla, \mathbf{P}, \mathbf{V}, \ldots
$$`
In the [Rudy et al.](https://www.science.org/doi/10.1126/sciadv.1602614) 
they propose an interesting *dictionary-based* approach to learning physical laws. The idea is to define a dictionary of plausible mathematical symbols that the model will combine linearly to construct formulas.<br>
For example, let's suppose to apply this strategy to harmonic oscillator. We know that this dynamic is governed by:
`$$
\left(\partial_t-\omega^2 \partial_{x x}\right) u \equiv L u=0
$$`
Thus we want to infer the $L$ in the problem using a set of observations: `${u((x, t)_i) \equiv u_i}_{i=1}^N$`.
We proceed to define a dictionary of symbols:
`$$
\partial_t u={1, u, u^2, u_x, u_x^2, u u_x, u_{x x}, u_{x x}^2, u u_{x x}, u_x, u_{x x}} \alpha=D \alpha
$$`
and we want now to minimize the number of active terms in $\alpha$. The minimization problem becomes:
`$$
\alpha^*=\underset{\alpha}{\arg \min }\|\partial_{\mathbf{t}} \mathbf{u}_{\mathbf{i}}-\mathbf{D}_{\mathrm{i}} \alpha\|_2+\|\alpha\|_0
$$`

## References

[1] Weng, Lilian. (Sep 2022). Some math behind neural tangent kernel. Lil’Log. https://lilianweng.github.io/posts/2022-09-08-ntk/.<br>
[2] Rotskoff, Grant. (Jul 2020). Gradient Flows II: Convexity and Connections to Machine Learning https://statmech.stanford.edu/post/gradient_flows_01/ <br>
[3] Gunasekar et al. (2017) Implicit Regularization in Matrix Factorization https://papers.nips.cc/paper_files/paper/2017/hash/58191d2a914c6dae66371c9dcdc91b41-Abstract.html <br>
[4] Srebro, Nathan (Sept 2017) Optimization's Hidden Gift to Learning: Implicit <br>Regularization https://www.youtube.com/watch?v=gh9vrvLx7Mo <br>
[5] Smith, Sam (Apr 2020) On the Origin of Implicit Regularization in Stochastic Gradient Descent https://www.youtube.com/watch?v=pZnZSxOttN0 <br>
[6] Huszár, Ferenc Notes on the Origin of Implicit Regularization in SGD https://www.inference.vc/notes-on-the-origin-of-implicit-regularization-in-stochastic-gradient-descent/<br>
[7] Chablani, Manish (Jun 2017) Batch Normalization https://towardsdatascience.com/batch-normalization-8a2e585775c9 <br>
[8] Santurkar et al. (May 2018) How Does Batch Normalization Help Optimization?
https://arxiv.org/pdf/1805.11604.pdf <br>
[9] Shanahan et al. (May 2019) An Explicitly Relational Neural Network Architecture https://arxiv.org/abs/1905.10307 <br>
[10] Gunasekar et al. (jun 2018) Implicit Bias of Gradient Descent on Linear Convolutional Networks https://arxiv.org/abs/1806.00468 <br>
[11] Wang et al. (jul 2020) When and why PINNs fail to train: A neural tangent kernel perspective  https://arxiv.org/abs/2007.14527 <br>
[12] Wang et al. (oct 2021) On the eigenvector bias of Fourier feature networks: From regression to solving multi-scale PDEs with physics-informed neural networks https://www.sciencedirect.com/science/article/abs/pii/S0045782521002759 <br>
[13] Glorot et al. (2010) Understanding the difficulty of training deep feedforward neural networks 
https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf <br>
[14] Equivariant Neural Networks https://dmol.pub/dl/Equivariant.html#groups-on-spaces<br>
[15] Anselmi et al. (apr 2023) Data Symmetries and Learning in Fully
Connected Neural Networks https://arts.units.it/retrieve/856df9ab-4e98-4bab-b71e-591336d25db0/Data_Symmetries_and_Learning_in_Fully_Connected_Neural_Networks.pdf<br>
[16] Zaheer et al, (apr 2018) DeepSets https://arxiv.org/pdf/1703.06114.pdf<br>
[17] Olah,Chris Mordvintsev,Alexander Schubert,Ludwig (nov 2017) Feature Visualization
https://distill.pub/2017/feature-visualization/<br>
[18] Rahaman et al. On the Spectral Bias of Neural Networks https://arxiv.org/pdf/1806.08734.pdf