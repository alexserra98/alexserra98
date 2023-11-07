---
title: "Advanced Topics in Machine Learning - Notes part 2"
date: 2023-11-06T11:30:03+00:00
weight: 1
mathjax: true
---
Include brain score??

It's surprising how brain is able to recognize objects regardless of their position, scale, rotation, and illumination. The intuitive fact that the brain is able to recognize some persistent or *invariant* characteristics that identify an object is the concpet of at basis of the following notes. The idea is that our visual cortex and up to some degree the ANN used in computer vision are able to recognize objects by learning a set of *invariant representations* of the input. 

# Theory of Invariant Representations

First of all we lay out the mathematical foundation of the theory of invariant representations.
Let's model a data space as a Hilber space `$\mathcal{I}$` and denote by `$\langle\cdot, \cdot\rangle$` and `$\|\cdot\|$` is inner product and norm respectively. We consider a set of transformations over `$\mathcal{I}$` endowed with a group structure and we denoted as 
`$$
\begin{align}
\mathcal{G} \subset\{g \mid g: \mathcal{I} \rightarrow \mathcal{I}\}
\end{align}
$$`
we also define the group action $(g,x) \rightarrow g \cdot x \in \mathcal{I}$ with abuse of notation as  $gx$.

We define the following equivalence relation over $\mathcal{I}$:

`$$
\begin{align}
I \sim I^{\prime} \Leftrightarrow \exists g \in \mathcal{G} \text { such that } g I=I^{\prime}
\end{align}
$$`
i.e.  two elements of $\mathcal{I}$ are equivalent if they belong to the same orbit.
**Invariant Representation**: A representation `$\mu: \mathcal{I} \rightarrow \mathcal{F}$` is invariant under the action of $\mathcal{G}$ if:
`$$
\begin{align}
I \sim I^{\prime} \Rightarrow \mu(I)=\mu\left(I^{\prime}\right),
\end{align}
$$`
for all `$I, I^{\prime} \in \mathcal{I}$`.

In order to exclude trivially invariant representations and to injectivity we also require the other direction of the implication to hold:
**Selective Representation**:  A representation `$\mu: \mathcal{I} \rightarrow \mathcal{F}$` is selective under the action of $\mathcal{G}$ if:
`$$
\begin{align}
\mu(I)=\mu\left(I^{\prime}\right) \Rightarrow I \sim I^{\prime}
\end{align}
$$`
for all `$I, I^{\prime} \in \mathcal{I}$`.
### Compact Group Invartiant Representations
We now restricts our analysis to the cases in which $\mathcal{G}$ is a compact group. Before dive in we need to introduce the concept of *Haar measure* over $\mathcal{G}$.
**Definition .1** Let G be a locally compact group. Then a left invariant *Haar measure* on G is a Borel measure $\mu$ such that for all measurable functions $f: G \rightarrow \mathbb{R}$ and all $g' \in G$ it holds:
`$$
\begin{align}
\int d g f(g)=\int d g f\left(g^{\prime} g\right)
\end{align}
$$`
For example, the Lebesgue measure is an invariant Haar measure on real numbers.
We can now prove the following result: <br>
**Theorem 1.** Let `$\psi: \mathcal{I} \rightarrow \mathbb{R}$` be a possibly non linear, functional on I. Then, the functional defined by
`$$
\begin{align}
\mu: \mathcal{I} \rightarrow \mathbb{R}, \quad \mu(I)=\int \operatorname{dg} \psi(g I), \quad I \in \mathcal{I}
\end{align}
$$`
is invariant in the sense of Definition 1. <br>
**Proof.** We need to prove that `$\mu_{f} = \mu_{f}(e)$` with $e$ the identity element of $\mathcal{G}$. We have:
`$$
\begin{aligned}
\mu_f(\bar{g})= & \int d g f(g \bar{g}) \\
\stackrel{1}= & \int d\left(\hat{g} \bar{g}^{-1}\right) f(\hat{g}) \quad \hat{g}=g \bar{g}, \quad g=\hat{g} \bar{g}^{-1} \\
\stackrel{2}= & \int d \hat{g} f(\hat{g}), \quad d\left(\hat{g} \bar{g}^{-1}\right) =d \hat{g} \\
\stackrel{3}= & \int d \hat{g} f(\hat{g})=\mu_f(e)
\end{aligned}
$$`
where we used: 1. reparametrization of group elements, 2. invariance of Haar measure, 3. group closure under composition.

We now want to apply this result to CNN. Let 
`$$
\begin{align}
\psi: \mathcal{I} \rightarrow \mathbb{R}, \quad \psi(I)=\eta(\langle g I, t\rangle), \quad I \in \mathcal{I}, g \in \mathcal{G}
\end{align}
$$`

where $\psi$ represent a single neuron which is performing the inner product between the input $I$ and its synaptic weights $t$ followed by a pointwise nonlinearity $\eta$ and a pooling layer.
Ok, now that we have a formal description of a neuron we want to make it invariant to the action of $\mathcal{G}$. We can do that using the previous results about Haar measure that is by averaging over the group action. As a matter of fact since the group representation is unitary we have that:
`$$
\begin{align}
\left\langle g I, I^{\prime}\right\rangle=\left\langle I, g^{-1} I^{\prime}\right\rangle, \quad \forall I, I^{\prime} \in \mathcal{I}
\end{align}
$$`
and therefore pluggin in the formulation used before we can rewrite $\psi$ as:
`$$
\begin{align}
\psi_{t}(I)=\int \operatorname{dg} \eta(\langle I, g t\rangle), \quad \forall I \in \mathcal{I}
\end{align}
$$`
We  call the function above **signature**
Now the mathematical of why this is invariant has already been proved, there's a more intuitive argument though that motivates this formulation. In order to compute an invariant representation of an element of the dataset we need to have the orbit of a `$t \in \mathcal{T}$` a template that provides the neuron with a "movie" of an object undergoing a family of transformations.

This formulation provides a way to compute an invariant representation of an element of the dataset. Unofortunately there's no clear way  to ensure that a family of measurements is selective. In the case of compact group the authors of the paper rely on a probabilistic argument. This part is more techincal and we just sketch the idea addressing to the paper for the details. 
The three core steps are:
1. A unique probability distribution can be naturally associated to each orbit.
2. Each such probability distributions can be characterized in terms of onedimensional
projections.
3. One dimensional probability distributions are easy to characterize, e.g. in
terms of their cumulative distribution or their moments.

First we start by a defining a map $P$ to associate a probability distribution to each point:
**Definition 4.**
For all $I \in \mathcal{I}$ define the random variable
`$$
Z_I:(\mathcal{G}, d g) \rightarrow \mathcal{I}, \quad Z_I(g)=g I, \quad \forall g \in \mathcal{G}
$$`
with law 
`$$
\rho_I(A)=\int_{Z_I^{-1}(A)} d g
$$`
for all measurables sets $A \subset \mathcal{I}$. Let
`$$
P: \mathcal{I} \rightarrow \mathcal{P}(\mathcal{I}), \quad P(I)=\rho_I, \quad \forall I \in \mathcal{I}
$$`

So basically the probabilty distribuiton of a point associate to a borel set in $\mathcal{I}$ the measure of the set of group elements that map the point to the set. Now we can prove the following result:
**Theorem 2.** For all $I,I' \in \mathcal{I}$ it holds:
`$$
\begin{align}
I \sim I^{\prime} \Leftrightarrow P(I)=P\left(I^{\prime}\right) .
\end{align}
$$`

Ok now we have identified point belonging to same orbit (i.e. belonging to the same invariant representation) through the probability distribution associated to each point. <br>
In order to avoid the pain of working with high-dimensional distributions we want to characterize the probability distribution in terms of one dimensional projections. The idea of the authors of the paper is to build a representation that they call *Tomographic Probabilistic representation* that is obtained by first mapping each point in the distribution supported on its orbit and then in a (continuous) family of corresponding one-dimensional distributions. 
Thanks to this representation we can get the following result:
**Theorem 3.** Let $\psi$ the TP representation the for all `$I,I' \in \mathcal{I}$` it holds:
`$$
\begin{align}
I \sim I^{\prime} \Leftrightarrow \Psi(I)=\Psi\left(I^{\prime}\right) .
\end{align}
$$`.
The above theorem's proof is based on a known result:
**Theorem 4**
For any `$\rho,\gamma \in \mathcal{P}(\mathbb{I})$` it holds:
`$$
\begin{align}
\rho=\gamma \quad \Leftrightarrow \quad \rho^t=\gamma^t, \quad \forall t \in \mathcal{S} .
\end{align}
$$`.
The gist of this section is that the problem of finding invariant/selective representations
reduces to the study of one dimensional distributions, as we discuss
next.
We can now close our probabilistic argument by bridging the results above with the representation defined in CNN section. The idea is the describe a one-dimensional probability in terms of its (CDF). The authors of the paper define a new CDF representation $\mu$ defined mapping each point to the CDF of a family of one-dimensional distributions. It can be proved that:
**Theorem 5.** 
For all `$I \in \mathcal{I}$` and `$t \in \mathcal{T}$` 
`$$
\begin{align}
\mu^t(I)(b)=\int d g \eta_b(\langle I, g t\rangle), \quad b \in \mathbb{R}
\end{align}
$$`
where we let `$\mu^t(I)=\mu(I)(t)$` and, for all `$b \in \mathbb{R}$`, `$\eta_b : \mathbb{R} \rightarrow \mathbb{R}$` is given by `$\eta_b(a)=H(b-a), \quad a \in \mathbb{R}$`. Moreover, for all `$I,I' \in \mathcal{I}$`:
`$$
\begin{align}
I \sim I^{\prime} \Leftrightarrow \mu(I)=\mu\left(I^{\prime}\right)
\end{align}
$$`

### Selectivity with limited template set
All the previous  discussion were made in the theoretical setting in which we had an infinte set of template avaible, in pratice this is non possible and althoug invariance is preserved we can't get full selectivity. Why? Well the core idea to ensure selectivity in the previous approach consisted in creating a map $\mu$ that was sending images in a family of CDF, indexed, by the template and on th real line: 
`$$
\mu: \mathcal{I} \rightarrow \mathcal{F}(\mathbb{R})^{\mathcal{T}}
$$`
Using additional details of thereom 5 we're able to prove the relation (14):
but this last statement is saying "two images belongs to the same orbit if and only if  ALL of their CDF indexed by $t$ are equal" Having "less" templeate clearly reduce the resolution induced by quotienting the space according to this relation.   
The good news is that in the theorem that we are about to introduce we will set a lower bound to the number of samples needed to get "enough" selectivity. <br>
Let's endow the representation space of metric structure: given `$\rho, \rho^{\prime} \in \mathcal{P}(\mathbb{R})$` two probability distributions  and let `$f_\rho, f_{\rho^{\prime}}$` their cumulative distribution functions, the it's possible to define the **Kolmogorov-Smirnov** metric induced by the uniform distribution. First consider the distance:
`$$
\begin{align}
d_{\infty}\left(f_\rho, f_{\rho^{\prime}}\right)=\sup _{s \in \mathbb{R}}\left|f_\rho(s)-f_{\rho^{\prime}}(s)\right|,
\end{align}
$$`
Using now the representation formulated in the previous section we can deqfine the following metric::
`$$
\begin{align}
\mu^t(I)(b)=\int d g \eta_b(\langle I, g t\rangle), \quad b \in \mathbb{R}
\end{align}
$$`
with $u$ the uniform measure on the sphere $\mathcal{S}$ and the theorems 4 and 5 will ensure that the metric is well defined.

Ok now we have formulation for our metric which is obtaing integrating the distance between two representation varyng the template, unfortunately we don't have in pratice an infinite set of templates but generally a finite `$\mathcal{T}_k=\left\{t_1, \ldots, t_k\right\} \subset \mathcal{S}$` and so we need to rewrite our metric as:
`$$
\begin{align}
\widehat{d}\left(I, I^{\prime}\right)=\frac{1}{k} \sum_{i=1}^k d_{\infty}\left(\mu^{t_i}(I), \mu^{t_i}\left(I^{\prime}\right)\right)
\end{align}
$$`.

Now that we have lay out all the instruments we're ready to enunciate and prove the following theorem:<br>
**Theorem 6** Consider n images `$\mathcal{I}_{n}$` in $\mathcal{I}$. Let v$k \geq \frac{2}{c \epsilon^2} \log \frac{n}{\delta}$` where $c$ is a constant. Then with probability $1-\delta^{2}$ it holds:
`$$
\begin{align}
\left|d\left(I, I^{\prime}\right)-\widehat{d}\left(I, I^{\prime}\right)\right| \leq \epsilon
\end{align}
$$`
for all `$I, I^{\prime} \in \mathcal{I}_{n}$`.
*Proof*:
The proof follows from a direct application of Höeffding's inequality and a Boole's inequalities. Fix `$I, I^{\prime} \in \mathcal{I}_{n}$`. Define the real random variable `$Z: \mathcal{S} \rightarrow[0,1]$`,
`$$
Z\left(t_i\right)=d_{\infty}\left(\mu^{t_i}(I), \mu^{t_i}\left(I^{\prime}\right)\right), \quad i=1, \ldots, k
$$`
From the definitions it follows that $\|Z\| \leq 1$ and `$\mathbb{E}(Z)=d\left(I, I^{\prime}\right)$`. We can now plug the above result in Höeffding's inequality: and get that:
`$$
\mathcal{P}(\left|d\left(I, I^{\prime}\right)-\widehat{d}\left(I, I^{\prime}\right)\right|\geq \epsilon)=\mathcal{P}(\left|\frac{1}{k} \sum_{i=1}^k \mathbb{E}(Z)-Z\left(t_i\right)\right|\geq \epsilon) \leq 2 e^{-\epsilon^2 k}
$$`.
We showed that this bound holds for a fixed pair of images, we can now apply Boole's inequality to get the result for all pairs of images:
`$$
\mathcal{P}(\bigcup_{I,I^{\prime}}\left|d\left(I, I^{\prime}\right)-\widehat{d}\left(I, I^{\prime}\right)\right|\geq \epsilon) \leq \sum_{I,I^{\prime}} \mathcal{P}(\left|d\left(I, I^{\prime}\right)-\widehat{d}\left(I, I^{\prime}\right)\right|\geq \epsilon) \leq n^{2} 2 e^{-\epsilon^2 k}
$$`
Thus the result hold uniformely on the all $\mathcal{I}_{n}$. We conclude the proof setting `$\delta^{2}$` and `$k \geq \frac{2}{c \epsilon^2} \log \frac{n}{\delta}$`

It's interesting to remark that there's other classical results on distance preserving embedding, such as Johnson Linderstrauss Lemma, the above formulation though ensures distance preservation up to a given accuracy which increases with a larger number of projections.

**Brief summary**:
This actually needs to go last
From a more practical point of view the steps to be perfomed are this:
1. Sample a set of templates `$\mathcal{T}_k=\left\{t_1, \ldots, t_k\right\} \subset \mathcal{S}$`
2. Project on transforming templates `$\left\{\left\langle x, g_j t^k\right\rangle\right\}, j=1, \ldots,|G|$`
3. Pooling using non-linear functions: We defined $\mu$ through a sum of non-linear activation function $\eta$ and we proved selectivity for a Heaviside function. In pratice though we can use a more general class of functions such as the sigmoid function.
4. compute the signature from the template set $\mathcal{T}_k$:
`$$
\Phi(x)=\left(\left\{\mu_1^1(x)\right\},\left\{\mu_2^1(x)\right\}, \ldots,\left\{\mu_N^K(x)\right\}\right) \in \mathbb{R}^{N \times|\mathcal{T}|}
$$`
As we proved in the above section we control the selectivity through the number of samples, so for a given $\epsilon$ and $\delta$ we can compute the number of samples needed to get a certain level of selectivity:
`$$
K \geq \frac{2}{c \epsilon^2} \log \frac{|\hat{Y}|}{\delta}
$$`
As a closing remarks we explicitate the connection between the above results and CNN. As a matter of fact we can interpret the representation described in (9) as convolution w.r.t to a group $\mathcal{G}$ followed by a pooling layer, this motivate the following theorem: <br>
**Theorem 7**
Suppose data are generated by $\mathcal{G}$. A CNN with convolutions w.r.t $\mathcal{G}$ is implementing a data representation $\Phi$ that is invariant and selective i.e. `$\mathbf{x} \sim \mathrm{x}^{\prime} \Leftrightarrow \Phi(\mathrm{x})=\boldsymbol{\Phi}\left(\mathrm{x}^{\prime}\right)$` with one layer  `$\Phi: \mathbb{R}^d \rightarrow \mathbb{R}^p$` and:
`$$
\Phi_w(x)=\sum_i\left|\left(x *_{\mathcal{G}} w\right)_i\right|_{+}=\sum_i\left|\left(W^T x\right)_i\right|_{+}=\sum_i\left|\left\langle x, g_i w\right\rangle\right|_{+}
$$`

### Optimal Template
The previous results apply to all groups, in particular to those which are not
compact but only locally compact such as translation and scaling. In this case
it can be proved that invariance holds within an observable window of transformations i.e. we can't observe the full range but just a subset limited for example by the receptive field For maximum range of invariance within the observable window, it is proven in (Anselmi et al. 2014, Anselmi et al. 2013) that the templates must be maximally sparse (minimum size in space and spatial frequency) relative to generic input images. The function that realize this maximum invariance are the Gabor function.
`$$
e^{-\frac{x^2}{2 \sigma^2}} e^{i \omega_0 x}
$$`
## Simple-complex model of visual cortex
The results obtained in the previous section are a rather theorical and seems to be only relevant to interpret the internal mechanics of CNNs. As a matter of fact the mathematical foundation that we layed out seems to be useful to motivate the structures found in the visual cortex. 
### Hubel and Wiesel model
[ADD PIC]   
In the '60 Hubel and Wiesel (HW hence in the notes) proposed a first model of circuits in the visual cortex which introduced the concept of simple and complex cells. The idea is that there's set of simple cells sensible to specific features of the input (such as the orientation) and that there's a complex cell that pools over the output of those simple cells. <br>
It is evident now the bridge between the theory of invariant representations and the model proposed by HW. The signature defined previously is morally equivalent to a simple-complex module
[ADD PIC] 
`$$
\mu_{w}(x) = \sum_{i=1}^{M} \eta \langle x, g_{i} w\rangle
$$`
(I changed 't' to 'w' to make consistent with the pic, but w is the template, while $g_{i}*w$ would be the weights of the CNN layer)
<br>
### Learning the weights
In order to explain the synaptic plasticity and so the adaptation of brain neurons during the learning process Donald Hebb proposed the famouse rule that bears his name: *"Cells that fire together wire together"* that is simultaneous activation of cells leads to pronounced increases in synaptic strength between those cells. In its original formulation the Hebb's rule for the updated of the synaptic weights $w$ is:
`$$
w_n=\alpha y\left(x_n\right) x_n
$$`
where $\alpha$ is the ”learning rate”, $x_{n}$ is the input vector w is the presynaptic weights vector and y is the postsynaptic response. For this dynamical system to actually converge, the weights have to be normalize and there's actuallu biological evidence of this fact as presented in  Turrigiano and Nelson 2004. 
A fundamental modification of this rule, called **Oja's flow** in which the models the update formula as:
`$$
\Delta w_n=w_{n+1}-w_n=\alpha y_n\left(x_n-y_n w_n\right)
$$`

obtained from expanding to the first order Hebb rule normalized to avoid divergence of the weights. 
The remarkable property of this formulation is that  the weights converge to the first principal component of the data, that is **Simple cells weights converge to the first eigenvector of the covariance of the stimuli** .

The algorithm that we provided to learn invariances in a dataset is based on the memorization of a series of "templates" and their transformations. In biological terms the sequence of transformations of one template would correspond to a set of simple cells, each one storing in its tuning a frame of the sequence. In a second learning step a complex cell would be wired to those “simple” cells. However, the idea of a direct storage of sequences of images or image patches is biologically rather implausible. Instead assume that all of the simple cells are exposed while in a plastic state, to a possibly large set of images `$T=\left(t_1, \ldots, t_K\right)$`. A specific cell  is exposed to the set of transformed templates `$g_{\star}T$ where $g_{\star}$` corresponds to the translation and scale, and then the associated covariance matrix will be 
`$$
g_* T T^T g_*^T
$$`

It has been shown (Leibo et al. 2014) that PCA of natural images provides eigenvectors that are maximally invariant for translation and scale and they can serve as "equivalent tempalte" . The Oja's rule will converge to the principal components of the dataset of natural images thus is possible to choose those eigenvector as new templates and both the invariance and selectivity theorem are valid.
The cell is thus exposed to a set of images (columns of $X$) `$X=\left(g_1 T, \ldots, g_{|G|} T\right)$` For the sake of this example, assume that G is the discrete equivalent of a group. Then the resulting covariance matrix  is
`$$
C=X X^T=\sum_{i=1}^{|G|} g_i T T^T g_i^T .
$$`

It is immediate to see that if $\phi$ is an eigenvector of $C$ then `$g_{i}\phi$`  is also an eigenvector
with the same eigenvalue 
`$$
C w=\lambda w \Rightarrow C g w=g C w=\lambda g w, \forall g \in \mathcal{G}, w \in E_\lambda
$$`

Consider for example $G$ to be the discrete rotation group in the plane: then all the (discrete) rotations of an eigenvector are also eigenvectors. The Oja rule will converge to the eigenvectors with the top eigenvalue and thus  to the subspace
spanned by them.
Using the formula above we can conclude that the weights of simple cells converge to linear combinations of elements of an orbit $\mathcal{G}$ 
`$$
\mathbf{E}_{\max }=\operatorname{span}\left(\mathbf{O}_{\mathbf{w}}\right), \quad \forall w \in E_{\max }
$$`

Moving up into the hierarchy of HW modules the principle of "fire together wire together" will ensure that the complex cell will learns to aggregates over simple celss whose weights form an orbit with respect to the usual group $\mathcal{G}$.
<br>

As a final remark two more properties of simple-complex modules:
- Simple cells are permutation-equivariant to $g \in \mathcal{G}$ transformations:
`$$
\sigma\left(W^T g x\right)=P_g \sigma\left(W^T x\right), \quad W=\left(g_1 w, \cdots, g_{|\mathcal{G}|} w\right)
$$`

- Complex cells are invariant to $g \in \mathcal{G}$ transformations:
`$$
\mu_{\mathbf{w}}(\mathbf{x})=\sum \sigma\left\langle x, g_i w\right\rangle=\mu_{\mathbf{w}}(\mathbf{g x}) \quad \forall g \in \mathcal{G}
$$`


    - Definitions 
    - How to build them
    - Why we care? Sample complexity
    
    Practical usage:
    - How many sample do we need?
    - Selectivity! a theorem, general perspective in Rep_learning

    - Does it transfer to test set?
    - A single cell model of simple and complex cells in visual cortex
    


    
    -simple complex model of visual cortex
        - PCA
    - Signature
    - Hierachical model of visual cortex
    - Summary of algorithm

    Compact Group Gabor functions:
    - What is a Gabor function?
    - Explanation from the ebook
    - Mathematical foundation from paper
    - Application: "Two stages in the computation of an invariant signature"
