---
title: "Advanced Topics in Machine Learning - Notes part 1"
date: 2023-09-18T11:30:03+00:00
weight: 1
mathjax: true
editPost:
    URL: "https://alexserra98.github.io/alexserra98/posts/advml_1/"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---


This is the first of a series of posts that collects notes from the course [advaced topics in machine learning]("https://dssc.units.it/advanced-topics-machine-learning"). If you find any mistakes or I've forgotten to cite you feel free to reach out!
## Neural Tangent Kernel
We want to explicit the relationship between kernel methods and neural network: <br>
Shallow learning (using kernel):
- the feature map `$\phi(x)$` is **fixed**
- the model is `$f(x) = \langle w , \phi(x) \rangle$`

Deep learning (using neural network):
- the feature map is **compositional** `$\phi(x)_{L}= \phi_{L} \circ \phi_{L-1} \circ ... \circ \phi_{1} (x) $`
and it is **learned**
- the model is `$f(x) = \langle w , \phi(x)_{L} \rangle$`

The link between the two models is *Neural Tangent Kernel* (NTK), the mathematical details can be found at [ntk](https://lilianweng.github.io/posts/2022-09-08-ntk/), here we limit to provide the intuition behind it. <br> The empirical loss function of a neural network 
`$$
\mathcal{L}: \mathbb{R}^P \rightarrow \mathbb{R}_{+}
$$` 
is defined as follow, using a per-sample cost function 
`$\ell: \mathbb{R}^{n_0} \times \mathbb{R}^{n_L} \rightarrow \mathbb{R}_{+}$` :
`$$
\mathcal{L}(\theta)=\frac{1}{N} \sum_{i=1}^N \ell\left(f\left(\mathbf{x}^{(i)} ; \theta\right), y^{(i)}\right)
$$`,
where `$\theta$` is the parameter of the neural network, `$f(x; \theta)$` is the output of the neural network, `$\ell$` is the loss function, `$x_{i}$` is the input and `$y_{i}$` is the target using the chain rule the gradient is 
`$$
\nabla_\theta \mathcal{L}(\theta)=\frac{1}{N} \sum_{i=1}^N \underbrace{\nabla_\theta f\left(\mathbf{x}^{(i)} ; \theta\right)}_{\text {size } P \times n_L} \underbrace{\nabla_f \ell\left(f, y^{(i)}\right)}_{\text {size } n_L \times 1}
$$` 
When we perform a gradient descent update we introduce only a small increment in the weight for an infinitesimal step size for this reason we can approximately interpret the variation of the weights as:
`$$
\frac{d \theta}{d t}=-\nabla_\theta \mathcal{L}(\theta)=-\frac{1}{N} \sum_{i=1}^N \nabla_\theta f\left(\mathbf{x}^{(i)} ; \theta\right) \nabla_f \ell\left(f, y^{(i)}\right)
$$`
The solution of the ODE in the above expression is known as [*gradient flow*](https://statmech.stanford.edu/post/gradient_flows_01/) 
When can use this result to derive an expression for the evolution of the network output:
`$$
\frac{d f(\mathbf{x} ; \theta)}{d t}=\frac{d f(\mathbf{x} ; \theta)}{d \theta} \frac{d \theta}{d t}=-\frac{1}{N} \sum_{i=1}^N \underbrace{\nabla_\theta f(\mathbf{x} ; \theta)^{\top} \nabla_\theta f\left(\mathbf{x}^{(i)} ; \theta\right)}_{\text {Neural tangent kernel }} \nabla_f \ell\left(f, y^{(i)}\right)
$$` 
Here we find the **Neural Tangent Kernel** which is defined as 
`$$
K\left(\mathbf{x}, \mathbf{x}^{\prime} ; \theta\right)=\nabla_\theta f(\mathbf{x} ; \theta)^{\top} \nabla_\theta f\left(\mathbf{x}^{\prime} ; \theta\right)
$$`
and the associated feature map is `$\phi(x) = \nabla_{\theta} f(x; \theta)$` <br>
The key point is the network is approaching infinite width, the NTK converges to be:

1. deterministic at initialization, meaning that the kernel is irrelevant to the initialization values and only determined by the model architecture;
2. Stay constant during training.

With this setup, we  express the evolution of the network output as:

In order to track the evolution of `$\theta$` time we can simplify our model by linearizing it:
`$$
f(\theta(t)) \approx f^{\operatorname{lin}}(\theta(t))=f(\theta(0))+\underbrace{\nabla_\theta f(\theta(0))}_{\text {formally }\left.\nabla_\theta f(\mathbf{x} ; \theta)\right|_{\theta=\theta(0)}}(\theta(t)-\theta(0))
$$`
and then we can perform the following analysis:
`$$
\begin{aligned}
\theta(t)-\theta(0) & =-\eta \nabla_\theta \mathcal{L}(\theta)=-\eta \nabla_\theta f(\theta)^{\top} \nabla_f \mathcal{L} \\
f^{\operatorname{lin}}(\theta(t))-f(\theta(0)) & =-\eta \nabla_\theta f(\theta(0))^{\top} \nabla_\theta f(\mathcal{X} ; \theta(0)) \nabla_f \mathcal{L} \\
\frac{d f(\theta(t))}{d t} & =-\eta K(\theta(0)) \nabla_f \mathcal{L} \\
\frac{d f(\theta(t))}{d t} & =-\eta K_{\infty} \nabla_f \mathcal{L}
\end{aligned}
$$`
for infinite width network
If the empirical loss is defined as `$\nabla_\theta \mathcal{L}(\theta)=f(\mathcal{X} ; \theta)-\mathcal{Y}$` we can solve the differential equation and obtain the following result: 
`$$
\begin{aligned}
\frac{d f(\theta)}{d t} & =-\eta K_{\infty}(f(\theta)-\mathcal{Y}) \\
\frac{d g(\theta)}{d t} & =-\eta K_{\infty} g(\theta) \quad ; \text { let } g(\theta)=f(\theta)-\mathcal{Y} \\
\int \frac{d g(\theta)}{g(\theta)} & =-\eta \int K_{\infty} d t \\
g(\theta) & =C e^{-\eta K_{\infty} t}
\end{aligned}
$$`
and then get:
`$$
f(\theta)=(f(\theta(0))-\mathcal{Y}) e^{-\eta K_{\infty} t}+\mathcal{Y}=f(\theta(0)) e^{-K_{\infty} t}+\left(I-e^{-\eta K_{\infty} t}\right) \mathcal{Y}
$$`
These results hold for infinite network width, because the change in parameters is infinitesimal, and we can employ the linearization of the model. <br>
In practice we the taylor expansion is accurate in a specific regime called **lazy training** when the net is heavily over-parametrized and we can witness to decent decrease in the loss `$\mathcal{L}$` but minimum change in the jacobian of the matrix is very small
<br>

### Universal approximation theorem

Any continuous function defined in a n-dimensional unit hypercube may be approximated by a finite sum of the type:
`$\sum_{j=1}^N v_j \varphi\left(\vec{\omega}^{(j)} \cdot \vec{x}+b_j\right)$`
wherein `$v_j, b_j \in \mathbb{R}, \vec{\omega}^{(j)} \in \mathbb{R}^n$` and `$\phi$` is continuoi discriminatory function.

## Regularization
A key concept that motivates the usage of regolarization is called: the *manifold hypothesis*. Quoting [wikipedia](https://en.wikipedia.org/wiki/Manifold_hypothesis)
'The **manifold hypothesis** posits that many high-dimensional data sets that occur in the real world actually lie along low-dimensional latent manifolds inside that high-dimensional space. As a consequence of the manifold hypothesis, many data sets that appear to initially require many variables to describes, can actually be described by a comparatively small number of variables, likened to the local coordinate system of the underlying manifold. It is suggested that this principle underpins the effectiveness of machine learning algorithms in describing high-dimensional data sets by considering a few common features...
...Within one of these manifolds, it’s always possible to interpolate between two inputs, that is to say, morph one into another via a continuous path along which all points fall on the manifold.
The ability to interpolate between samples is the key to generalization in deep learning.'

The motivation behind regularization is to enhance the ability of the model to generalize even at discount of the training error. <br>
<br>
### L2 regularization
It is also commonly known as **weight decay** or **ridge regression**, it is a very simple regularization technique that penalizes the model for having large weights by adding the regularization term: `$\Omega(\boldsymbol{\theta})=\frac{1}{2}\|\boldsymbol{w}\|_2^2$`. <br> 
We want to analyze the effect of the regularization term on the loss function and the optimal solution. To simplify the analysis we assume no bias parameter so we can write the total objective function as:
`$$
\tilde{J}(\boldsymbol{w} ; \boldsymbol{X}, \boldsymbol{y})=\frac{\alpha}{2} \boldsymbol{w}^{\top} \boldsymbol{w}+J(\boldsymbol{w} ; \boldsymbol{X}, \boldsymbol{y})
$$`
and the gradient w.r.t parameters:
`$$
\nabla_{\boldsymbol{w}} \tilde{J}(\boldsymbol{w} ; \boldsymbol{X}, \boldsymbol{y})=\alpha \boldsymbol{w}+\nabla_{\boldsymbol{w}} J(\boldsymbol{w} ; \boldsymbol{X}, \boldsymbol{y})
$$`
A single step of gradient descent with learning rate `$\epsilon$` is:
`$$
\boldsymbol{w} \leftarrow \boldsymbol{w}-\epsilon\left(\alpha \boldsymbol{w}+\nabla_{\boldsymbol{w}} J(\boldsymbol{w} ; \boldsymbol{X}, \boldsymbol{y})\right) .
$$`
which can be rewritten as:
`$$
\boldsymbol{w} \leftarrow(1-\epsilon \alpha) \boldsymbol{w}-\epsilon \nabla_{\boldsymbol{w}} J(\boldsymbol{w} ; \boldsymbol{X}, \boldsymbol{y}) .
$$`
Thus we can see that the effect of the regularization term is to rescale the weight vector by a factor of `$(1-\epsilon \alpha)$` on every step. <br>
We want to extend our analysis to the entire course of the training, but first of all, we further simplify the analysis by making a quadratic approximation of the loss function around the optimal unregularized solution `$\boldsymbol{w}^*$`: 
`$$
\hat{J}(\boldsymbol{\theta})=J\left(\boldsymbol{w}^*\right)+\frac{1}{2}\left(\boldsymbol{w}-\boldsymbol{w}^*\right)^{\top} \boldsymbol{H}\left(\boldsymbol{w}-\boldsymbol{w}^*\right)
$$`
where `$\boldsymbol{H}$`is the Hessian matrix of $J$. Remark: the first order term is 0 because we are the minimum and for the same reason `$\boldsymbol{H}$` is positive semidefinite. The minimum of `$\widehat{J}$` occurs when 
`$$
\nabla_{\boldsymbol{w}} \hat{J}(\boldsymbol{w})=\boldsymbol{H}\left(\boldsymbol{w}-\boldsymbol{w}^*\right)
$$`
Now want to perform spectral analysis of the minimum `$\boldsymbol{w}^*$` to explicit the effect of the regularization:

`$$
\begin{array}{r}
\alpha \tilde{\boldsymbol{w}}+\boldsymbol{H}\left(\tilde{\boldsymbol{w}}-\boldsymbol{w}^*\right)=0 \\
(\boldsymbol{H}+\alpha \boldsymbol{I}) \tilde{\boldsymbol{w}}=\boldsymbol{H} \boldsymbol{w}^* \\
\tilde{\boldsymbol{w}}=(\boldsymbol{H}+\alpha \boldsymbol{I})^{-1} \boldsymbol{H} \boldsymbol{w}^* .
\end{array}
$$`
Since `$\boldsymbol{H}$` is real and symmetric we can decompose into a diagonal matrix `$\boldsymbol{\Lambda}$`
and orthonormal basis of eigenvector `$\boldsymbol{\Lambda}$` thus we obtain: 
`$$
\begin{aligned}
\tilde{\boldsymbol{w}} & =\left(\boldsymbol{Q} \boldsymbol{\Lambda} \boldsymbol{Q}^{\top}+\alpha \boldsymbol{I}\right)^{-1} \boldsymbol{Q} \boldsymbol{\Lambda} \boldsymbol{Q}^{\top} \boldsymbol{w}^* \\
& =\left[\boldsymbol{Q}(\boldsymbol{\Lambda}+\alpha \boldsymbol{I}) \boldsymbol{Q}^{\top}\right]^{-1} \boldsymbol{Q} \boldsymbol{\Lambda} \boldsymbol{Q}^{\top} \boldsymbol{w}^* \\
& =\boldsymbol{Q}(\boldsymbol{\Lambda}+\alpha \boldsymbol{I})^{-1} \boldsymbol{\Lambda} \boldsymbol{Q}^{\top} \boldsymbol{w}^*
\end{aligned}
$$`
The effect of the regularization is to rescale the eigenvector by a factor of `$\frac{\lambda}{\lambda+\alpha}$`, thus the regularization has the effect of reducing the magnitude of the eigenvector of the Hessian matrix which represents the curvature of the function
(recall that if `$v^{\intercal}\boldsymbol{H}v > 0$` the function will have positive curvature, and negative curvature if `$v^{\intercal}\boldsymbol{H}v < 0$` ). There are two cases:
- `$\lambda_{i} \gg \alpha$` the regularization has a small effect on the eigenvector
- `$\lambda_{i} \ll \alpha$` the regularization has a large effect on the eigenvector
with `$\lambda_{i}$` the eigenvalue of the Hessian matrix `$\boldsymbol{H}$` . 
In other words we are filtering out the directions that doesn't not contribute much to reducing the loss function. <br>
<br>
<br>
<br>
### Connection with noisy input
Let's suppose to add gaussian noise to the inputs of a MLP model in such a that the variance of the noise is amplified by the square weight before going to the next layer<br>
In the simplified setting of a single layer MLP with a single neuron and without adding non-linearity we have:
Input: `$x_{i} + \mathcal{N}(0,\sigma^{2})$` --> Output: `$y_{i} + \mathcal{N}(0,\sigma^{2}w_{i}^{2})$` <br>
In this way we are adding to the loss a term containing the square of the weights <br>
For the explicit derivation:

`$$
\begin{aligned}
y^{\text {noisy }} & =\sum_i w_i x_i+\sum_i w_i \varepsilon_i \text { where } \varepsilon_i \text { is sampled from } N\left(0, \sigma_i^2\right) \\
\left[\left(y^{\text {noisy }}-t\right)^2\right] & =  E\left[\left(y+\sum_i w_i \varepsilon_i-t\right)^2\right]=E\left[\left((y-t)+\sum_i w_i \varepsilon_i\right)^2\right] \\
& =(y-t)^2+E\left[2(y-t) \sum_i w_i \varepsilon_i\right]+E\left[\left(\sum_i w_i \varepsilon_i\right)^2\right] \\
& =(y-t)^2+E\left[\sum_i w_i^2 \varepsilon_i^2\right] \quad \text { because } \varepsilon_i \text { is independent of } \varepsilon_j \\
& =(y-t)^2+\sum_i w_i^2 \sigma_i^2 \quad \text { So } \sigma_i^2 \text { is equivalent to an L2 penalty }
\end{aligned}
$$`

### Dropout
The regularization technique *dropout* consists in *shutting off* some nodes (input and hidden layer) in a neural network. Since most of neural networks consisits in a series of
affine transformations and nonlinearities, we can effectively remove a unit from a network by multiplying its output value by zero., thus creating a new network architecture out of the parent network. The nodes are dropped by a dropout probability of p.

There are two main reasons why dropout works at reducing overfitting:
- It reduces the co-adaptation of the units, since the network can't rely on the presence of a particular unit, it has to learn a more robust representation. 'Dropout thus regularizes each hidden unit to be
not merely a good feature but a feature that is good in many contexts.'
- We can interpret the application of dropout as *bagging* with a very  large *ensemble* obtained with all the *sub-network* obtained by dropping out some units. Nevertheless these two methods are not identical cause the *sub-network* are not independent since they share the parameters, and the models are not trained to convergence but instead a tiny fraction of the possible
sub-networks are each trained for a single step, and the parameters sharing causes
the remaining sub-networks to arrive at good settings of the parameters
- As we have seen in the before section regularization can be linked to the injection of noise to hidden units. In practice if a specific hidden unit learns a feature to classify an entity, and then we shut off it, the model will have to make another unit learn the same feature (redundant encoding) or another feature that is relevant to classification task, thus the model will be more robust to the loss of a specific unit.
<br>
<br>
## Implicit bias
When implementing a statistical model one has to be careful to leverage the complexity so to avoid underfitting  in which there are too few parameters that can't fit the data and overfitting when the model is too complex and it can't generalize. <br>
A common heuristic in traditional statistics was that model "too" big doesn't provide good generalizationù generally ANN has huge amount of parameters way bigger than the dimension of the sample, and  we would expect the optimization algorithm to get stuck in a local minima over the training set and not achieving good generalization capabilities. We actually observe the opposite, the network seems to biased to reach only certain sets of functions that perform well on the test set, what is going on? <br>
It turned out that the question is quite deep and involves revist the way that we categorize neural net.
Part of the following work is inspired from the following [lesson](https://www.youtube.com/watch?v=gh9vrvLx7Mo) by Nathan Srebro. 

There are three cornerstones to assess the quality of a neural network in terms of generalization:
- Capacity: How training examples do we need so that if we can fit this model we can actually generalize well?
We have a good understanding of this and it's linear in the number of parameters: `$O(|E|)$` (number of edges i.e. parameters)

- Expressive power: We know that neural network are universal approximators but in the general case the require an arbitrarly large neural network. The real question is what can we approximate using a small network? It can be shown that any function that takes a time `$T$` to compute can be approximated by a neural network with ~`$O(T)$` parameters.

- Computation: In the general case finding the best parameters to fit a set of data is NP-hard. Even if consider functions that are perfectly representable by a single hidden layer network, with don't add noise and we larger network for the training, still the problem doesn't admit any poly-time algorithm <br>

{{<figure align=center src="images/modern_ml.png" caption="Figure 1. The test keep decreasing instead manifest sign of overfitting source:Neyshabur et al. https://arxiv.org/abs/1412.6614">}}<br>
We observe a very strange fact at some point the train error is approximately zero and the test error is still decreasing, it means that the model is improving its estimation error even if we are using a model with way more parameters this does not make any sense, it means that we have to give better definition of the complexity of the model ( not just the number of parameters) <br>
<br>

### Matrix Completion
We present now a clarifying example for the previous statements.
**Matrix completion** is the task of filling in the missing entries of a partially observed matrix, without any restrictions on the number of degrees of freedom in the completed matrix this problem is underdetermined since the hidden entries could be assigned arbitrary values. Thus we should require some assumption on the matrix to create a well-posed problem, such as assuming it has maximal determinant, is positive definite, or is low-rank.<br>

We consider the setting in which a neural network has to solve the following task:
`$$
\min _{X \in \mathbb{R}^{n \times n}} \| \text { observed }(X)-y \|_2^2 
$$`
Note that the model is completely **agnostic** to any of the restrictions listed above, thus the problem is nonsensical. <br>
We then rewrite the problem factoring the matrix $X$ as `$X=U V^{T}$`:
`$$
\min _{X \in \mathbb{R}^{n \times n}} \| \text { observed }(X)-y \|_2^2 \equiv \min _{U, V \in \mathbb{R}^{n \times n}}\| \text { observed }\left(U V^{\top}\right)-y \|_2^2
$$`
If we train our model with this setting and using *gradient descent* what we observe is that if we take infinitely small stepsize, and with additional conditions the result to which the algorithm converges is the minimum nuclear norm solution. Besides if the data come from low-rank matrices the model achieves good generalization ability <br>
There's a conjecture associated to this problem:
> With `$U(0) \rightarrow 0$`, the flow `$\dot{U} = - \nabla \| A(UU^{T}) -y \|^{2}$` converge to the minimum nuclear norm solution:
`$$
U(\infty) U(\infty)^{\top} \rightarrow \min _{W \geqslant 0}\|W\|_* \text { s.t. }\left\langle A_i, W\right\rangle=y_i
$$`
[Gunasekar et al.](https://papers.nips.cc/paper_files/paper/2017/hash/58191d2a914c6dae66371c9dcdc91b41-Abstract.html)

### Minimum norm solution in least-square problem
In a simplified setting as the least-square problems we can see more clearly the implicit regularization imposed by the gradient descent.<br>
The following analysis is taken from [Gunasekar et al.](https://arxiv.org/abs/1705.09280) and the [thread](https://math.stackexchange.com/questions/3451272/does-gradient-descent-converge-to-a-minimum-norm-solution-in-least-squares-probl) on stackexchange. <br>
We want to show that choosing gradient descent as the optimization algorithm in an unregularized, underdetermined least squares problem will converge to the minimum Euclidean norm solution. <br>
Given a **fat** matrix `$\mathrm{A} \in \mathbb{R}^{m \times n}(m<n)$` and vector `$\mathrm{b} \in \mathbb{R}^m$`, consider the following linear system `$\mathrm{x} \in \mathbb{R}^n$`:
`$$ 
Ax = b
$$`
where $A$ has full row rank. Let the singular value decomposition ( SVD) of $A$ be as follows: 
`$$
A=U \Sigma V^{\top}=U\left[\begin{array}{ll}\
\Sigma_1 & O
\end{array}\right]\left[\begin{array}{c}
V_1^{\top} \\\
V_2^{\top}
\end{array}\right]=U \Sigma_1 ~V_1^{\top}
$$`
The **least-norm** solution of $Ax = b$ is given by:
`$$
\mathrm{x}_{\mathrm{LN}}:=\mathrm{A}^{\top}\left(\mathrm{AA}^{\top}\right)^{-1} \mathrm{~b}=\cdots=\mathrm{V}_1 \Sigma_1^{-1} \mathrm{U}^{\top} \mathrm{b}
$$`
where the inverse of `$\mathbf{A A}^{\top}$` exists because of $A$ has full row rank. <br>

Let the cost function `$f: \mathbb{R}^n \rightarrow \mathbb{R}$` be defined by:
`$$
f(\mathrm{x}):=\frac{1}{2}\|\mathrm{Ax}-\mathrm{b}\|_2^2
$$`
whose gradient is 
`$$
\nabla f(\mathrm{x})=\mathrm{A}^{\top}(\mathrm{Ax}-\mathrm{b})
$$`
Using gradient descent with steps `$\mu > 0$`,
`$$
\begin{aligned}
\mathrm{x}_{k+1} & =\mathrm{x}_k-\mu \nabla f\left(\mathrm{x}_k\right) \
& =\left(\mathrm{I}-\mu \mathrm{A}^{\top} \mathrm{A}\right) \mathrm{x}_k+\mu \mathrm{A}^{\top} \mathrm{b}
\end{aligned}
$$`
Hence,
`$$
\mathrm{x}_k=\left(\mathrm{I}-\mu \mathrm{A}^{\top} \mathrm{A}\right)^k \mathrm{x}_0+\mu \sum_{\ell=0}^{k-1}\left(\mathrm{I}-\mu \mathrm{A}^{\top} \mathrm{A}\right)^{\ell} \mathrm{A}^{\top} \mathrm{b}
$$`
Letting 
`$$
\mathrm{y}:=\mathrm{V}^{\top} \mathrm{x}
$$`, (so rewriting everything with 
`$A = U\Sigma$`),  we get
`$$
\begin{aligned}
& \mathrm{y}_k=\left(\mathrm{I}-\mu \Sigma^{\top} \Sigma\right)^k \mathrm{y}_0+\mu \sum_{\ell=0}^{k-1}\left(\mathrm{I}-\mu \Sigma^{\top} \Sigma\right)^{\ell} \Sigma^{\top} \mathrm{U}^{\top} \mathrm{b} \\
&=\left[\begin{array}{cc}
\left(\mathrm{I}-\mu \Sigma_1^2\right)^k & \mathrm{O} \\
\mathrm{O} & \mathrm{I}
\end{array}\right] \mathrm{y}_0+\mu \\
& \sum_{\ell=0}^{k-1}\left[\begin{array}{cc}
\left(\mathrm{I}-\mu \Sigma_1^2\right)^{\ell} & \mathrm{O} \\
\mathrm{O} & \mathrm{I}
\end{array}\right]\left[\begin{array}{c}
\Sigma_1 \\
\mathrm{O}
\end{array}\right] \mathrm{U}^{\top} \mathrm{b} \\
&= {\left[\begin{array}{cc}
\left(\mathrm{I}-\mu \Sigma_1^2\right)^k & \mathrm{O} \\
\mathrm{O} & \mathrm{I}
\end{array}\right] \mathrm{y}_0+\mu \sum_{\ell=0}^{k-1}\left[\begin{array}{c}
\left(\mathrm{I}-\mu \Sigma_1^2\right)^{\ell} \Sigma_1 \\
\mathrm{O}
\end{array}\right] \mathrm{U}^{\top} \mathrm{b} }
\end{aligned}
$$`
Choosing `$\mu > 0$` such that all eigenvalues of `$\mathrm{I}-\mu \Sigma_1^2$` are strictly inside the unit circle, then `$\mathrm{y}_k \rightarrow \mathrm{y}_{\infty}$`, where:
`$$
\mathrm{y}_{\infty}=\left[\begin{array}{ll}
\mathrm{O} & \mathrm{O} \\
\mathrm{O} & \mathrm{I}
\end{array}\right] \mathrm{y}_0+\mu \sum_{\ell=0}^{\infty}\left[\begin{array}{c}
\left(\mathrm{I}-\mu \Sigma_1^2\right)^{\ell} \Sigma_1 \\
\mathrm{O}
\end{array}\right] \mathrm{U}^{\top} \mathrm{b}
$$`
where, 
`$$
\mu \sum_{\ell=0}^{\infty}\left(\mathrm{I}-\mu \Sigma_1^2\right)^{\ell} \Sigma_1=\mu\left(\mathrm{I}-\mathrm{I}+\mu \Sigma_1^2\right)^{-1} \Sigma_1=\Sigma_1^{-1}
$$`
and thus,
`$$
\mathrm{y}_{\infty}=\left[\begin{array}{cc}
\mathrm{O} & \mathrm{O} \\
\mathrm{O} & \mathrm{I}
\end{array}\right] \mathrm{y}_0+\left[\begin{array}{c}
\Sigma_1^{-1} \\
\mathrm{O}
\end{array}\right] \mathrm{U}^{\top} \mathrm{b}
$$`
Since `$x := Vy$`
`$$
\mathrm{x}_{\infty}=\mathrm{V}_2 \mathrm{~V}_2^{\top} \mathrm{x}_0+\underbrace{\mathrm{V}_1 \Sigma_1^{-1} \mathrm{U}^{\top} \mathrm{b}}_{=\mathrm{x}_{\mathrm{LN}}}
$$`
Therefore, we conclude that if `$x_{0}$` is orthogonal to the null space of A ( `$V^{2}$` belongs to that null space)
, then gradient descent will converge to the least-norm solution.

### Early stopping
The following calculation comes from Ian Goodfellow's book [Deep Learning](https://www.deeplearningbook.org/contents/regularization.html) <br>

*Early stopping* is a very popular regularization technique that consists in storing the parameter setting reached by the optimization algorithm that realizes the lowest validation set error. <br> The number of steps hyperparameter has U-shaped validation performance curve, thus the algorithm will be at the beginning of the rise.<br>
We can see then that *early stopping* can be interpreted as an hyperparameters selection algorithm that prevent overfitting controlling the capacity of the model through the number of steps. <br>

It has been shown that *early stopping* restricts the optimization procedure to a relatively smaller volume of parameters in the space of the initial parameter value, this is indeed the **regularization effect** of this strategy. <br>
We can actually push our analysis further and show that in the simple setting of quadratic approximation of the loss this technique is equivalent to L2 regularization. <br>
Let's start by approximating the cost function $J$ in the neighborhood of the optimal solution `$w^*$` as we did in *weight decay*:
`$$
\hat{J}(\boldsymbol{\theta})=J\left(\boldsymbol{w}^*\right)+{ }_2^1\left(\boldsymbol{w}-\boldsymbol{w}^*\right)^{\top} \boldsymbol{H}\left(\boldsymbol{w}-\boldsymbol{w}^*\right)
$$`
where `$\boldsymbol{H}$` is the Hessian matrix of $J$. For the same reasons stated in L2 section `$\boldsymbol{H}$` is positive and semidefinite<br>
With this approximation, we can rewrite the gradient descent update rule as:
`$$
\nabla_{\boldsymbol{w}} \hat{J}(\boldsymbol{w})=\boldsymbol{H}\left(\boldsymbol{w}-\boldsymbol{w}^*\right)
$$`
Now we will write the update rule, for simplicity we set the initial parameter vector to the origin `$w_{0} = 0$` and then perform spectral decomposition of the Hessian matrix `$\boldsymbol{H} = \boldsymbol{Q} \Lambda \boldsymbol{Q}^{\top}$`:
`$$
\begin{aligned}
\boldsymbol{w}^{(\tau)} & =\boldsymbol{w}^{(\tau-1)}-\epsilon \nabla_{\boldsymbol{w}} \hat{J}\left(\boldsymbol{w}^{(\tau-1)}\right) \\
& =\boldsymbol{w}^{(\tau-1)}-\epsilon \boldsymbol{H}\left(\boldsymbol{w}^{(\tau-1)}-\boldsymbol{w}^*\right) \\
\boldsymbol{w}^{(\tau)}-\boldsymbol{w}^* & =(\boldsymbol{I}-\epsilon \boldsymbol{H})\left(\boldsymbol{w}^{(\tau-1)}-\boldsymbol{w}^*\right) .
\end{aligned}
$$`
and then:
`$$
\begin{aligned}
\boldsymbol{w}^{(\tau)}-\boldsymbol{w}^* & =\left(\boldsymbol{I}-\epsilon \boldsymbol{Q} \boldsymbol{\Lambda} \boldsymbol{Q}^{\top}\right)\left(\boldsymbol{w}^{(\tau-1)}-\boldsymbol{w}^*\right) \\
\boldsymbol{Q}^{\top}\left(\boldsymbol{w}^{(\tau)}-\boldsymbol{w}^*\right) & =(\boldsymbol{I}-\epsilon \boldsymbol{\Lambda}) \boldsymbol{Q}^{\top}\left(\boldsymbol{w}^{(\tau-1)}-\boldsymbol{w}^*\right)
\end{aligned}
$$`
In order to guarantee convergence (?) we set `$\epsilon$` such that 
`$\left|1-\epsilon \lambda_i\right|<1$` and the update at a time `$\tau$` is:
`$$
\boldsymbol{Q}^{\top} \boldsymbol{w}^{(\tau)}=\left[\boldsymbol{I}-(\boldsymbol{I}-\epsilon \boldsymbol{\Lambda})^\tau\right] \boldsymbol{Q}^{\top} \boldsymbol{w}^*
$$`
We rewrite `$Q^{\top} \tilde{\boldsymbol{w}}$` from !add ref! as:
`$$
\begin{aligned}
& \boldsymbol{Q}^{\top} \tilde{\boldsymbol{w}}=(\boldsymbol{\Lambda}+\alpha \boldsymbol{I})^{-1} \boldsymbol{\Lambda} \boldsymbol{Q}^{\top} \boldsymbol{w}^* \\
& \boldsymbol{Q}^{\top} \tilde{\boldsymbol{w}}=\left[\boldsymbol{I}-(\boldsymbol{\Lambda}+\alpha \boldsymbol{I})^{-1} \alpha\right] \boldsymbol{Q}^{\top} \boldsymbol{w}^*
\end{aligned}
$$`
Now if we compare this update rule of weight decay to the one of a training halted at a time `$\tau$` we can see that they are equivalent if we set::
`$$
(\boldsymbol{I}-\epsilon \boldsymbol{\Lambda})^\tau=(\boldsymbol{\Lambda}+\alpha \boldsymbol{I})^{-1} \alpha
$$`
If then we take the logarithm on both sides and use the Taylor expansion of $log(1+x)$ in the case in which $\epsilon \lambda_{i} \ll 1$ and $\frac{\lambda_{i}}{\alpha} \ll 1 $ we get:
`$$
\begin{aligned}
& \tau \approx \frac{1}{\epsilon \alpha} \\
& \alpha \approx \begin{array}{c}
1 \\
\tau \epsilon
\end{array}
\end{aligned}
$$`
In all the examples of regularization seen so far we have shown that parameter values corresponding to directions of significant curvature (of the
objective function) are affected less than directions of less curvature. In the case of weight decay we have modified the update rule so that we actively rescale $w^{*}$ along the axis of the eigenvectors while in the case of early stopping the parameters that correspond
to directions of significant curvature tend to be learned early relative to parameters
corresponding to directions of less curvature. <br> 

### Stochastic Gradient Descent
The following are the notes from the [lecture](https://www.youtube.com/watch?v=pZnZSxOttN0) by Sam Smith and this [article](https://www.inference.vc/notes-on-the-origin-of-implicit-regularization-in-stochastic-gradient-descent/). <br>
We start with a motivating example (add first graph in the video):
The following performance comes from a 10-1 Wide-ResNet trained on CIFAR-10 with a constant learning rate. As we can see as the learning rate increases the train accuracy decreases but the test accuracy actually increase. Why is this happening? <br>
**Key Claim**: The SGD is  minimizing a modified loss in which there's a regularization term that is proportional to the learning rate.<br>
The first step now is to explicit the relationship between optimization and Numerical Integration. It's well known that gradient descent in the limit of infinitesimal step size is equivalent to a gradient flow problem:
`$$
\begin{gathered}
\omega_{i+1}=\omega_i-\epsilon \nabla C\left(\omega_i\right) \\
\downarrow{\varepsilon \rightarrow 0} \\
\dot{\omega}=-\nabla C(\omega)
\end{gathered}
$$`
But following this interpretation we can see that the gradient descent is actually a numerical integration scheme (*Euler Method*) for a first order ODE. 
`$$
\begin{gathered}
\omega(t+\epsilon) \approx \omega(t)+\epsilon f(\omega(t)) \\
\uparrow{\varepsilon \rightarrow 0} \\
\dot{\omega}=f(\omega)
\end{gathered}
$$`
{{<figure align=center src="images/step_off.png" caption="Figure 2. Discrete update on the original flow stepping of modified flow source:https://www.youtube.com/watch?v=pZnZSxOttN0">}}<br>
<br>
If we follow the steps of Euler update we can see that this approximation scheme introduces a bias at each step and soon the trajectory of the discrete updates will step off from the actual solution.
<br> 
We introduce now an ancillary **modified flow**

`$$
\tilde{f}(\omega)=f(\omega)+\epsilon f_1(\omega)+\epsilon^2 f_2(\omega)+\ldots
$$`

so that the discrete updates on the original flow stay close to the continuous solution to the modified flow for small finite `$\epsilon$`. In loose term we want to find what flow the scheme is actually interpolating, this ancillary flow is constructed adding to the original flow a series of "correction terms" that should account for the bias introduced by the Euler method <br>
We proceed by deriving the first term of the modified flow:
1. First we apply n Euler updates  on the modified flow and Taylor expands the result:
`$$
\begin{aligned}
\omega_{t+n} & =\omega_t+\alpha \tilde{f}\left(\omega_t\right)+\alpha \tilde{f}\left(\omega_{t+1}\right)+\alpha \tilde{f}\left(\omega_{t+2}\right)+\ldots \\
& =\omega_t+\alpha \tilde{f}\left(\omega_t\right)+\alpha \tilde{f}\left(\omega_t+\alpha \tilde{f}\left(\omega_t\right)\right)+\alpha \tilde{f}\left(\omega_t+\alpha \tilde{f}\left(\omega_t\right)+\alpha \tilde{f}\left(\omega_t+\alpha \tilde{f}\left(\omega_t\right)\right)\right)+\ldots \\
& =\omega_t+n \alpha \tilde{f}\left(\omega_t\right)+(n / 2)(n-1) \alpha^2 \nabla \tilde{f}\left(\omega_t\right) \tilde{f}\left(\omega_t\right)+O\left(n^3 \alpha^3\right) .
\end{aligned}
$$`
2. Then we send `$n \rightarrow \infty$` and set `$\alpha=\epsilon / n$`, to obtain the solution of the continuous flow:
`$$
\begin{aligned}
\omega(t+\epsilon) &=\omega(t)+\epsilon \tilde{f}(\omega(t))+\left(\epsilon^2 / 2 \right) \nabla \tilde{f}(\omega(t)) \tilde{f}(\omega(t))+O(\epsilon^3) \\
&=\underbrace{\omega(t)+\epsilon f(\omega(t))}_{\text {Euler step }}+\epsilon^2(\underbrace{.f_1(\omega(t))+(1 / 2) \nabla f(\omega(t)) f(\omega(t))})_{\begin{array}{c}
\text { Set higher order } \\
\text { terms to zero }
\end{array}}+O(\epsilon^3) 
\end{aligned} 
$$`
We want the expression of `$\omega(t+\epsilon)$` equal to the expression of the Euler step (the purpose of this ancillary flow is to stay close to the solution obtained with the Euler updates) and thus we have to set higher order terms to 0. Thus we can now come up with an expression of the first correction term
`$$
f_{1}(\omega(t)) = - \frac{1}{2} \nabla f(\omega(t)) f(\omega(t))
$$`
<br>
We have constructed the modified flow in such a way that SGD solve the following problem:

`$$
\begin{gathered}
\dot{\omega}=\tilde{f}(\omega) \\
\tilde{f}(\omega)=f(\omega)+\epsilon f_1(\omega)+\epsilon^2 f_2(\omega)+\ldots
\end{gathered}
$$`

Where `$f(\omega) = \nabla C(\omega)$` and 

`$$
\begin{aligned}
f_1(\omega(t)) & =-(1 / 2) \nabla \nabla C(\omega) \nabla C(\omega) \\
& =-(1 / 4) \nabla\left(\|\nabla C(\omega)\|^2\right)
\end{aligned}
$$`

Putting everything together we have that for a finite small learning rate:

`$$
\begin{aligned}
& \dot{\omega}=-\nabla \widetilde{C}_{G D}(\omega)+O\left(\epsilon^2\right) \\\
& \widetilde{C}_{G D}(\omega)=\underbrace{C(\omega)}_{\text{Original Loss}}+\underbrace{(\epsilon / 4)\|\nabla C(\omega)\|^2}_{Regularization}
\end{aligned}
$$`
We provide an informal intution on why this is happening:
If we consider the segment between two point `$\omega_{t}$`, `$\omega_{t+1}$` reached by successive iteration of euler method in the above problem we  have that the gradient in `$\omega$` is exactly `$\nabla C(\omega)$` whilist the gradient in `$\omega_{t+1}-\delta$` with `$\delta$` an infintesimum quantiy, the gradient is approximately `$\nabla C(\omega +(\frac{\epsilon}{2}\nabla C))$` thus the gradient of the problem that SGD is solving is on average half a step stale and if we taylor expand it we get:
`$$
\begin{aligned}
\nabla C(\omega+(\epsilon / 2) \nabla C) & \approx \nabla C+(\epsilon / 2) \nabla \nabla C \nabla C \\
& =\nabla C+(\epsilon / 4) \nabla\left(\|\nabla C\|^2\right)
\end{aligned}
$$`
<br>
**Random Shuffling**
We proceed our backward analysis of SGD by considering the effect of random shuffling on the optimization problem. <br>
By random shuffling we mean that we start with a certain order of sample in the dataset, we create batches out of it and at the beginning of every epoch we choose a random permutation of these batches with witch we perform barched gradient descent. <br>

We now want to rewrite the SGD update rule taking into account random shuffling so we introduce some notation:
`$N$` is the number of samples, `$B$` size of the batch, `$m = \frac{N}{B}$`
is the number of updates per epoch, we define the loss over a mini-batch:
`$$
\hat{C}_i=(1 / B) \sum_{j=i B+1}^{(i+1) B} C_j(\omega)
$$`
and get the following update rule:
`$$
\omega_{i+1}=\omega_i-\epsilon \nabla \hat{C}_{i \mbox{mod} m}(\omega)
$$`
(with a shuffle every `$\mathrm{m}$` updates) <br>
The module operation at the index is just to enforce that every example appears exactly once per epoch (no overlap between batches).
So we get:
`$$
\begin{aligned}
C(\omega) & =(1 / N) \sum_{i=1}^N C_i(\omega) \\
& =(1 / m) \sum_{i=0}^{m-1} \hat{C}_i(\omega)
\end{aligned}
$$`
We now have all the instruments to state the main claim of this lesson:
Let `$\omega_{m}$` the value obtained after the m-th iterate of Random Shuffling SGD, then:
`$$
\begin{aligned}
\mathbb{E}\left(\omega_m\right) & =\omega(m \epsilon)+O\left(m^3 \epsilon^3\right) \\
\dot{\omega} & =-\nabla \widetilde{C}_{S G D}(\omega)
\end{aligned}
$$`
Thus the mean iterate lies in the solution of what is still a gradient flow problem, and the expression of the modified loss is:
`$$
\widetilde{C}_{S G D}(\omega)=C(\omega)+\frac{\epsilon}{4 m} \sum_{k=0}^{m-1}\left\|\nabla \hat{C}_k(\omega)\right\|^2
$$`
We remark the expectation is over the order of the mini-batches and not on their composition
Proof:
We start with the Taylor expansion of the updates over:
`$$
\begin{aligned}
\omega_m & =\omega_0-\epsilon \nabla \hat{C}_0\left(\omega_0\right)-\epsilon \nabla \hat{C}_1\left(\omega_1\right)-\epsilon \nabla \hat{C}_2\left(\omega_2\right)-\ldots \\
& =\omega_0-\epsilon \nabla \hat{C}_0\left(\omega_0\right)-\epsilon \nabla \hat{C}_1\left(\omega_0-\epsilon \nabla \hat{C}_0\left(\omega_0\right)\right)-\ldots \\
& =\omega_0-\epsilon \sum_{j=0}^{m-1} \nabla \hat{C}_j\left(\omega_0\right)+\epsilon^2 \sum_{j=0}^{m-1} \sum_{k<j} \nabla \nabla \hat{C}_j\left(\omega_0\right) \nabla \hat{C}_k\left(\omega_0\right)+O\left(m^3 \epsilon^3\right) \\
& =\omega_0-m \epsilon \nabla C\left(\omega_0\right)+\epsilon^2 \xi\left(\omega_0\right)+\dot{O}\left(m^3 \epsilon^3\right)
\end{aligned}
$$`
where `$\xi(\omega)=\sum_{j=0}^{m-1} \sum_{k<j} \nabla \nabla \hat{C}_j(\omega) \nabla \hat{C}_k(\omega)$`
we remark that at `$O(\epsilon)$` there's no variance cause is the full gradient, thus in the limit of vanishing learning rate Random Shuffling follows the same path of SGD. The third term instead at `$O(\epsilon^2)$` cares about the order of the minibatches thus is random variable with non-null mean and variance<br>
If we average over all possible minibatch orders we get:
`$$
\begin{aligned}
\mathbb{E}(\xi(\omega)) & =\frac{1}{2}\left(\sum_{j=0}^{m-1} \sum_{k \neq j} \nabla \nabla \hat{C}_j(\omega) \nabla \hat{C}_k(\omega)\right) \\
& =\frac{m^2}{2} \nabla \nabla C(\omega) \nabla C(\omega)-\frac{1}{2} \sum_{j=0}^{m-1} \nabla \nabla \hat{C}_j \nabla \hat{C}_j \\
& =\frac{m^2}{4} \nabla\left(\|\nabla C(\omega)\|^2-\frac{1}{m^2} \sum_{j=0}^{m-1}\left\|\nabla \hat{C}_j(\omega)\right\|^2\right)
\end{aligned}
$$`
We have now an expression for the mean iterate:
`$$
\begin{aligned}
\mathbb{E}\left(\omega_m\right) & =\omega_0-m \epsilon \nabla C\left(\omega_0\right) \\
& +\frac{m^2 \epsilon^2}{4} \nabla\left(\left\|\nabla C\left(\omega_0\right)\right\|^2-\left(1 / m^2\right) \sum_{j=0}^{m-1}\left\|\nabla \hat{C}_j\left(\omega_0\right)\right\|^2\right)+O\left(m^3 \epsilon^3\right)
\end{aligned}
$$`
We recall now the solution of the modified flow derived earlier where the correction term `$f_{1}$` appears:
`$$
\begin{gathered}
\tilde{f}(\omega)=-\nabla C(\omega)+\epsilon f_1(\omega)+O\left(\epsilon^2\right) \\
\omega(m \epsilon)=\omega_0-m \epsilon \nabla C\left(\omega_0\right)+m^2 \epsilon^2\left(f_1\left(\omega_0\right)+(1 / 4) \nabla\left\|\nabla C\left(\omega_0\right)\right\|^2\right)+O\left(m^3 \epsilon^3\right)
\end{gathered}
$$`
We now set the two expressions equal (the one derived earlier and the mean iterate) and solve for `$f_{1}(\omega_{0})$`, we skip the rather lengthy calculation and QED:
`$$
\begin{gathered}
\mathbb{E}\left(\omega_m\right)=\omega(m \epsilon)+O\left(m^3 \epsilon^3\right) \\
\dot{\omega}=-\nabla \widetilde{C}_{S G D}(\omega) \\
\widetilde{C}_{S G D}(\omega)=C(\omega)+\frac{\epsilon}{4 m} \sum_{k=0}^{m-1}\left\|\nabla \hat{C}_k(\omega)\right\|^2
\end{gathered}
$$`
<br>
It's important to stress the fact that we're making the rather extreme assumption that `$O(m^3\epsilon^3)$` is negligible, this is quite limiting in real applications but it's still a better estimation than the one we get from the previous gradient flow in which we assume negligible the quadratic term. 
Another interesting remark is that most of the analysis on SGD focuses on the variance whilist this revolves around the bias
<br>
We can push our analysis further and explicit the dependence of the modified loss on the batch size. If we take the expectation of the modified loss over the mini batches we get:

`$$
\begin{aligned}
\mathbb{E}\left(\tilde{C}_{S G D}(\omega)\right) & =C(\omega)+(\epsilon / 4) \mathbb{E}\left(\|\nabla \hat{C}(\omega)\|^2\right) \\
& =C(\omega)+(\epsilon / 4)\|\nabla C(\omega)\|^2+\frac{(N-B)}{(N-1)}(\epsilon / 4 B) \Gamma(\omega)
\end{aligned}
$$`

where 

`$$
\Gamma(\omega)=(1 / N) \sum_{i=1}^N\left\|\nabla C_i(\omega)-\nabla C(\omega)\right\|^2
$$`

Let's dissect the above expression:
- The first term is the original loss
- The second term is the regularization term that we have already seen in the previous analysis
- The third term is proportional to the trace of the per example covariance matrix which encodes a preference for the region of parameter spaces in which the gradient of different examples agree. Besides is proportional to `$\frac{\epsilon}{4B}$` which surprisingly motivates the experimental evidence that when you double the batch size the optimal learning rate also doubles  <br>
The lesson now continues with experimental evidence of the above analysis, I omit this section and adress to linked video for all the details.
<br>

**Discretization Error and Stochasticity**
Does the benefit of SGD require finite learning rate?
In order to answer the question we disentangle the effect of the discretization error and the stochasticity of the mini-batches.
The strategy is:
1. Perform n Euler updates on the same minibatch with bare learning rate `$\alpha$`
2. Draw a new minibatch and repeat step 1

In this setting the actual learning rate is 
`$$
\epsilon = \alpha n
$$`
and set the stochasticity,  `$\alpha$` set the discretization error cause it encodes the step size of the Euler method. As `$n \rightarrow \infty$` `$\alpha \rightarrow 0$` thus the discretization error  vanish <br>
<!---
ADD PLOT
(not very clear!! revise)
--->
{{<figure align=center src="images/alpha.png" caption="Figure 3. Training a 16-4 Wide-Res net for 400 epochs">}}

what we see now is that Large bare learning rates  `$\alpha$` are actually beneficial for the optimization procedure, implying that **discretization error** is beneficial, whilist smaller value of n achieves higher accuracy implying that **stochasticity is detrimental**<br>
<br>
### Closing Remarks on gradient descent

The goal of an optimization algorithm is to converge to **good** minima that behaves well in terms of:
- robustness
- generalization
Both the train-set and test-set are sampled from the same distribution, and for this reason, we expect the train loss function to be very close to the test loss function. What we want is to find a shape of the loss that is robust to small perturbations (that comes from training different samples but from the same distribution). There's an ongoing debate regarding the geometry of these shapes, but a generally accepted idea is that a function with a *flat* minima can provide robustness and generalization.  <br>

{{<figure align=center src="images/robust-sgd.png" caption="Figure 4. At each step the SGD is looking to a different sampled function. On average the main effect is dominated by big basins of attraction">}}

<!---
(Not sure rewatch march 22 29.45 Langevin SGD
It didn't talk about this but if I have time I want to add a few remarks about it. <br>)
### Compositional Prior
missing, probably will add it inside geometric deep learning.
--->
An another geometric intuition for SGD is that at each iteration only a subset of the train set is used to fit the loss function, during the optimization procedure the algorithm average over that different fitted functions and this "ensemble" of function get rid of the "sharp" minima leaving only the "flat" ones. <br>




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
[19] Neyshabur et al. (apr 2015) In Search of the Real Inductive Bias: On the Role of Implicit Regularization in Deep Learning https://arxiv.org/abs/1412.6614
<!---

### tmp
possible idea to make the network think more like humans, analyze ECG of patients looking at photos and use a net to extrapolate the frequencies that are most important for our brain then filter images using these frequencies and see what happens

we can do the same with optical illusion, and try to find out which frequencies are responsible for the illusion

unfortunately, lots of priors are not the best choice
what's the invariance version of the kernel?(possible project)
29 march qa
weights are very sparse in fourier
what's the bias that I'm introducing adding convolution and pooling
in the proof of invariance use the fact that the network is invariant to permutation of output
the prior on matrix completion is that the correlation are sparse

pay attention that in equivariance f(x) is just the vector of the covoluted input not the pooling

You integrate over all possible action of the group
if I do data augmentation the effect on the loss is to modulate the fourier transform sin(alpha) this is implicit bias
now we have that to learn the weight we use gradient descent but to do that we use this loss with implicit bias thus also gradient descent has implicit bias

weights that are permutation equivariant are constant under conjugation: $\Pi^{\top}W\Pi = W$

https://www.weizmann.ac.il/math/previous-seminars

Mean Field Analysis of Neural Networks: A Law of Large Numbers
Take a look at this https://en.wikipedia.org/wiki/Capsule_neural_network

## TODO
- [x] Add pictures
- [x] Cite paper with authors et al.
- [x] Add list of references
- [x] Add table of content
- [ ] Add grammarly
- [ ] Add equation numbering
- [ ] Add Data normalization
--->