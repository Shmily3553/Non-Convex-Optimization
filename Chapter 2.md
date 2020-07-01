## Chapter 2 Mathematical Tools

### 2.1 Convex Analysis

**Definition 2.1** (*Convex Combination*) A convex combination of a set of n vectors $x_i \in R^P$, $ i = 1... n$ in an arbitrary real space is a vector $x_\theta := \sum_{i=1}^n\theta_i x_i$ where $ \theta = (\theta_1, \theta_2, \cdots, \theta_n)$, $\theta_i \geq 0$ and $\sum_{i=1}^n\theta_i = 1$.

A set that is closed under arbitrary convex combinations is a convex set. Geometrically speaking, convex sets are those that contain all line segments that join two points inside the set. As a result, they cannot have any inward "bulges".

**Definition 2.2** (*Convex Set*) A set $C\in R^P$ is considered convex if for every $x, y \in C$ and $\lambda \in [0,1]$, we have $(1 - \lambda)\cdot x +\lambda \cdot y \in C$ as well.

This defines the threshold of variable value.

Figure 2.1 gives visual representations of prototypical convex and non-convex sets. 

![](images/Figure2-1.png)

The convex function could be defined as $f: R^P \rightarrow R$, for every $x, y \in R^P$ and $\lambda \in [0,1]$, we have $f((1 - \lambda)\cdot x +\lambda \cdot y) \leq (1 - \lambda)\cdot f(x) +\lambda \cdot f(y))$. 

This means that any value on the line segment that join two points on function $f$, is greater than the value corresponding to the variable value. And this defines the threshold of the value corresponding to the variable value. 

<img src="images/convex function.png" style="zoom:100%;" />

**Definition 2.3** (*Convex Function*) A continuously differentiable function $f: R^P \rightarrow R$ is considered convex if for every  $x, y \in R^P$ we have $f(y) \geq f(x) + \langle\nabla f(x), y - x\rangle$, where $\nabla f(x)$ is the gradient of $f$ at $x$.

![](images/Figure2-2.png)

A more general definition that extends to non-differentiable functions uses the notion of subgradient to replace the gradient in the above expression. A special class of convex functions is the class of *strongly convex* and *strongly smooth* functions. Figure 2.2 provides a handy visual representation of these classes of functions.

**Definition 2.4** (*Strongly Convex / Smooth Function*) A continuously differentiable function $f: R^P \rightarrow R$ is considered $\alpha$-*strongly convex* (**SC**) and $\beta$-*strongly smooth* (**SS**) if for every $x, y \in R^P$ we have

$\frac{\alpha}{2}\parallel x - y \parallel^2_2 \leq f(y) - f(x) - \langle \nabla f(x), y - x\rangle \leq \frac{\beta}{2}\parallel x - y \parallel^2_2$

Given Taylor Theorem, for $\xi\in[x,y]$, we have
$$
f(y) \ge f(x) + f'(x)^T(y-x) + \frac{f''(\xi)}{2}(y-x)^2
$$
Thus, 
$$
f(y)-f(x)-f'(x)^T(y-x) \ge \frac{f''(\xi)}{2}(y-x)^2\\
f(y)-f(x)-\langle \nabla f(x), y - x\rangle \ge \frac{f''(\xi)}{2}(y-x)^2\\
f(y)-f(x)-\langle \nabla f(x), y - x\rangle \ge \frac{f''(\xi)}{2}\parallel x-y\parallel^2_2
$$
The Hessian matrix $H$ of $f$ is a square matrix$_{n*n}$, usually defined as follows:

![](images/hessian matrix.png)

Therefore, we know that the largest eigenvalue (right bottom) of the Hessian of $f$ is uniformly upper bounded by $\beta$, while the smallest eigenvalue (left top) refers to the lower bounded by $\alpha$.

It is useful to note that strong convexity places a quadratic lower bound and let the function grow at least as fast as a quadratic function, with the SC parameter $\alpha$ dictating the lower limit. Similarly, strong smoothness places a quadratic upper bound and does not let the function grow too fast, with the SS parameter $\beta$ dictating the upper limit.

These two properties are extremely useful in forcing optimization algorithms to rapidly converge to optima. 

Whereas strongly convex functions are definitely convex, strong smoothness does not imply convexity. Strongly smooth functions may very well be non-convex.

#### The explanation may could be found in exe 2.1 TRY IT !!!

**Definition 2.5** (*Lipschitz Function*) A function $f: R^P \rightarrow R$ is *B-Lipschitz* if for every $x, y \in R^P$ we have
$$
\mid f(x) - f(y) \mid \leq B \cdot \parallel x - y \parallel_2
$$
Given Lagrange Theorem, we have
$$
\mid f(x) - f(y)\mid = \mid f'(\xi)^T(x-y)\mid \le \parallel f'(\xi)\parallel\cdot\parallel x-y\parallel_2
$$
Similar with $\beta$-*strongly smooth* (**SS**) function, we get  the largest eigenvalue (right bottom) of the Hessian of $f$ is uniformly upper bounded by $B$.

Lipschitzness places a upper bound on the growth of the function that is linear in the perturbation i.e., $\parallel x - y \parallel_2$, whereas strong smoothness places a quadratic upper bound. 

Lipschitz functions need not be differentiable. However, differentiable functions with bounded gradients are always Lipschitz$^2$.

An important property that generalizes the behavior of convex functions on convex combinations is the Jensen's inequality.

**Lemma 2.1** (*Jensen's Inequality*) If $X$ is a random variable taking vales in the domain of a convex function $f$, then $E[f(X)] \geq f(E[X])$

This property will be useful while analyzing iterative algorithms.

### 2.2  Convex Projections

The projected gradient descent technique is a popular method for constrained optimization problems, both convex and non-convex. The projection step plays as an important role in this technique. Given any closed $C \subset R^P$, the projection operator $\prod_C(\cdot)$ is defined as

![](images/projection operator.png)

This definition means that if $z$ is outside $C$, then project $z$ on $C$; whilst $z$ is in $C$, then nothing need to be changed.

In general, $L^2$-norm is not only but the most commonly way to define projection. If $C$ is a convex set, then the above problem reduces to a convex optimization problem.

For instance, if $C = B_2(1)$ i.e., the unit $L_2$ ball, then projection is equivalent to a normalization step

![](images/projection operator exp.png)

For instance, if $C = B_1(1)$, the projection step reduces to the popular *soft thresholding operation*, which finds the minimizer of an objective function that involves data fitting in an $l_2$sense as well as minimization of the $l_1$ norm (i.e., absolute value).  if $\hat{z}:= \prod_{B_1(1)}(z)$, then $\hat{z_i} = max\lbrace z_i - \theta, 0\rbrace$, where $\theta$ is a threshold that can be decided by a sorting operation on the vector.

**Lemma 2.2** (*Projection Property-O*) For any set (convex or not) $C \subset R^P$ and $z \in R^P$, let $\hat{z}:= \prod_{C}(z)$. Then for all $x \in C, \parallel \hat{z} - z \parallel_2 \leq \parallel x - z \parallel_2$.

**Lemma 2.3** (*Projection Property-I*) For any convex set $C \subset R^P$ and $z \in R^P$, let $\hat{z}:= \prod_{C}(z)$. Then for all $x \in C, \langle x - \hat{z}, z - \hat{z}\rangle\leq 0$.

Projection Property-I can be used to prove a very useful contraction property for convex projections. In some sense, a convex projection brings a point closer to all points in the convex set simultaneously.

**Lemma 2.4** (*Projection Property-II*) For any convex set $C \subset R^P$ and $z \in R^P$, let $\hat{z}:= \prod_{C}(z)$. Then for all $x \in C, \parallel \hat{z} - x \parallel_2 \leq \parallel z - x \parallel_2$.

The Definition 2.2 could be used to proof these three above Lemma.

![](images/projection properties.png)

Note that Projection Properties-I and II are also called first order properties, and Projection Property-O is often called zeroth order property.

### 2.3 Projected Gradient Descent

This is an extremely simple and efficient technique that can effortlessly scale to large problems. The projected gradient descent algorithm is stated in Algorithm 1, The procedure generates iterates $x_t$ by taking steps guided by the gradient in an effort to reduce the function value locally. Finally it returns either the final iterate, the average iterate, or the best iterate.

![](images/algorithm1.png)

### 2.4 Convergence Guarantees for PGD

We will analyze PGD for objective functions that either a) convex with bounded gradients, or b) strongly convex and strongly smooth. Let $f^* = \min_{x\in C} f(x)$ be the optimal value of the optimization problem. A point $\hat{x} \in C$ will be said to be an $\epsilon$-optimal solution if $f(\hat{x}) \le f^* + \epsilon$.

#### 2.4.1 Convergence with Bounded Gradient Convex Functions

Consider a convex objective function $f$ with bounded gradients over a convex constraint set $C$. i.e., $\parallel f(x) \parallel_2 \leq G$ for all $x \in C$.

**Theorem 2.5** Let $f$ be a convex objective with bounded gradients and Algorithm 1 be executed for $T$ time steps with step lengths $\eta_t = \eta = \frac{1}{\sqrt{T}}$. Then, for any $\epsilon \gt 0$, if $T = O(\frac{1}{\epsilon^2})$, then $\frac{1}{T}\sum_{t=1}^Tf(x^t) \le f^* + \epsilon$.

##### The proof of Theorem 2.5

Let $x^*\in arg min_{x\in C}f(x)$, and such a point always exists if the constraint set is closed and the objective function continuous.

Let the *potential function* $\Phi_t = f(x^t) - f(x^*)$, which measures the sub-optimality of the $t$-th iterate.

And then, the statement of the theorem is equivalent to claiming that $\frac{1}{T}\sum_{t=1}^T\Phi_t \le \epsilon$.
$$
\frac{1}{T}\sum_{t=1}^Tf(x^t) \le f^* + \epsilon\\

\frac{1}{T}\sum_{t=1}^Tf(x^t) \le \min_{x\in C} f(x) + \epsilon\\

\frac{1}{T}\sum_{t=1}^Tf(x^t) - \min_{x\in C} f(x) \le \epsilon\\

\frac{1}{T}\sum_{t=1}^Tf(x^t) - \frac{1}{T}\sum_{t=1}^Tf(x^*) \le \epsilon\\

\frac{1}{T}\sum_{t=1}^T\Phi_t \le \epsilon
$$
Given the Convexity $f(y) - f(x) \ge \langle\nabla f(x), y - x\rangle$ is a global property and very useful in getting an upper bound on the level of sub-optionality of the current iterate in such analyses. We apply it to upper bound the potential function at every step.
$$
\Phi_t = f(x^t) - f(x^*) \le \langle\nabla f(x^t), x^t - x^*\rangle
$$

And then, given $2ab=a^2 + b^2 - (a - b)^2$,
$$
\langle\nabla f(x^t), x^t - x^*\rangle = \frac{1}{\eta}\langle \eta \cdot \nabla f(x^t), x^t - x^*\rangle \\
=\frac{1}{2\eta}(\parallel x^t - x^*\parallel_2^2 + \parallel \eta \cdot \nabla f(x^t)\parallel_2^2 - \parallel x^t - \eta \cdot \nabla f(x^t) - x^*\parallel_2^2)\\
$$
In the PGD algorithm, we set that $z^{t+1}\leftarrow x^t - \eta \cdot \nabla f(x^t)$, thus
$$
\langle\nabla f(x^t), x^t - x^*\rangle =\frac{1}{2\eta}(\parallel x^t - x^*\parallel_2^2 + \parallel \eta \cdot \nabla f(x^t)\parallel_2^2 - \parallel z^{t + 1} - x^*\parallel_2^2)
$$
And given the fact that the objective function $f$ has bounded gradients as $\parallel f(x) \parallel_2 \leq G$ for all $x \in C$,
$$
\langle\nabla f(x^t), x^t - x^*\rangle \le \frac{1}{2\eta}(\parallel x^t - x^*\parallel_2^2 + \eta^2G^2 - \parallel z^{t + 1} - x^*\parallel_2^2)
$$
Applying Lemma 2.4, $\parallel \hat{z} - x \parallel_2 \leq \parallel z - x \parallel_2$ for all $x \in C$, according to PGD (Line 4), $x^t$ is the result of the projection of $z^t$, we have 
$$
\parallel x^{t} - x^*\parallel_2^2 \le \parallel z^{t} - x^*\parallel_2^2
$$
Which means that
$$
\parallel z^{t + 1} - x^*\parallel_2^2 \ge \parallel x^{t + 1} - x^*\parallel_2^2
$$
Thus,
$$
\Phi_t \le \frac{1}{2\eta}(\parallel x^t - x^*\parallel_2^2 - \parallel x^{t + 1} - x^*\parallel_2^2) + \frac{\eta G^2}{2}
$$
Apart from the $\frac{\eta G^2}{2}$ is small since $\eta = \frac{1}{\sqrt{T}}$, $\Phi_t$ is small if the consecutive iterates $x^t$ and $x^{t+1}$ are close to each other (and hence similar in distance from $x^*$. This observation tells that once PGD stops making a lot of progress, it actually converges to the optimum. In this case, only a vanishing gradient can cause PGD to stop progressing, since the step length is constant. And for convex functions, this only happens at global optima.

Summing up across time steps and performing telescopic cancellations, using $x^1 = 0$, and dividing throughout by $T$ gives us,
$$
\frac{1}{T}\sum_{t=1}^T\Phi_t \\
\le\frac{1}{2\eta T}(\parallel x^1 - x^*\parallel_2^2 - \parallel x^2 - x^*\parallel_2^2 + \parallel x^2 - x^*\parallel_2^2 - \parallel x^3 - x^*\parallel_2^2 + \cdot\cdot\cdot + \parallel x^{T} - x^*\parallel_2^2 - \parallel x^{T + 1} - x^*\parallel_2^2) + \frac{\eta G^2}{2}\\
=\frac{1}{2\eta T}(\parallel x^1 - x^*\parallel_2^2 - \parallel x^{T + 1} - x^*\parallel_2^2) + \frac{\eta G^2}{2}\\
=\frac{1}{2\eta T}(\parallel - x^*\parallel_2^2 - \parallel x^{T + 1} - x^*\parallel_2^2) + \frac{\eta G^2}{2}\\
=\frac{1}{2\eta T}(\parallel x^*\parallel_2^2 - \parallel x^{T + 1} - x^*\parallel_2^2) + \frac{\eta G^2}{2}
$$
Since $\parallel x^{T + 1} - x^*\parallel_2^2 \ge 0$ and $\eta = \frac{1}{\sqrt{T}}$,
$$
\frac{1}{T}\sum_{t=1}^T\Phi_t \le \frac{1}{2\eta T}(\parallel x^*\parallel_2^2 - \parallel x^{T + 1} - x^*\parallel_2^2) + \frac{\eta G^2}{2}\\
= \frac{1}{2\sqrt T}(\parallel x^*\parallel_2^2  - \parallel x^{T + 1} - x^*\parallel_2^2 + G^2)\\
\le \frac{1}{2\sqrt T}(\parallel x^*\parallel_2^2 + G^2)
$$
Till now, we FINALLY get the claimed result $\frac{1}{T}\sum_{t=1}^T\Phi_t \le \epsilon$.

In case we are not sure what the value of $T$ would be, the setting of step length depends on the total number of iterations $T$ for which the PGD algorithm is executed. This is called a *horizon-aware* setting of the step length. 

This setting could ensure the function value of the iterates approaches $f^*$ on an average, which could prove the convergence of the PGD algorithm.

**OPTIOPN 1**: It is the cheapest and doesn't require any additional operations, but it doesn't converge to the optimum for convex functions in general and may oscillate close to the optimum. However it does converge  if the objective function $f$ is strongly smooth, since strongly smooth functions may not grow at a faster-than-quadratic rate.

**OPTION 2**: It is cheaper than than OPTION 3, since we do not have to perform function evaluations to find the best iterate. And it could also prove the convergence of the PGD algorithm by applying Lemma 2.1 Jensen's inequality and Theorem 2.5. For all $t$, we have

$f(\hat{x}_{avg}) = f(\frac{1}{T}\sum_{t=1}^Tx^t) \le \frac{1}{T}\sum_{t=1}^Tf(x^t) \le f^* + \epsilon$ 

And do note that Jensen's inequality may be applied only when the function $f$ is convex.

**OPTION 3**: Could also prove the convergence of the PGD algorithm with the construction of $f(\hat{x}_{best})$ and Theorem 2.5. For all $t$, we have

$f(\hat{x}_{best}) = \arg min_{t\in[T]}f(x^t) \le \frac{1}{T}\sum_{t=1}^Tf(x^t) \le f^* + \epsilon$ 

#### 2.4.2 Convergence with Strongly Convex and Smooth Functions

**Theorem 2.6** Let $f$ be an objective that satisfies the $\alpha$-$SC$ and $\beta$-$SS$ properties. Let Algorithm 1 be executed with step lengths $\eta_t = \eta = \frac{1}{\beta}$. Then after at most $T = O(\frac{\beta}{\alpha}log\frac{\beta}{\epsilon})$ steps, we have $f(x^T) \le f(x^*) + \epsilon$.

When the objective is $SC$/$SS$, it ensures that the final iterate $\hat{x}_{final} = x^T$ converges. And the rate of convergence is accelerated with the properties, since for general convex functions, PGD requires $O(\frac{1}{\epsilon^2})$ iterations to reach an $\epsilon$-optimal solution, whilst it requires only $O(log\frac{1}{\epsilon})$ iterations for $SC$/$SS$ functions.

Notice that the setting of step length, $\eta = \frac{1}{\beta}$, in this case is only to make the computation which presents a problem be more easy. In practice, the step is tuned globally by doing a grid search over several $\eta$ values, or per-iteration using line search mechanisms, to obtain a step length value that assures good convergence rates.

##### The proof of Theorem 2.6

Since $f$ satisfies the $\alpha$-$SC$ and $\beta$-$SS$ properties, for every $x, y \in R^P$ we have
$$
\frac{\alpha}{2}\parallel x - y \parallel^2_2 \leq f(y) - f(x) - \langle \nabla f(x), y - x\rangle \leq \frac{\beta}{2}\parallel x - y \parallel^2_2
$$
Thus,
$$
f(x^T) - f(x^*) - \langle \nabla f(x^*), x^T - x^*\rangle \leq \frac{\beta}{2}\parallel x^* - x^T \parallel^2_2\\
f(x^T) - f(x^*) \le\langle \nabla f(x^*), x^T - x^*\rangle + \frac{\beta}{2}\parallel x^* - x^T \parallel^2_2\\
f(x^T) - f(x^*) \le\langle \nabla f(x^*), x^T - x^*\rangle + \frac{\beta}{2}\parallel x^T - x^* \parallel^2_2
$$
Since  $x^*$ is an optimal point for the constrained optimization problem with a convex constraint set $C$, the first order optimality condition gives us $\langle \nabla f(x^*), x - x^*\rangle \le 0$ for any $x\in C$. Applying this condition with $x=x^T$ gives us
$$
f(x^T) - f(x^*) \le\langle \nabla f(x^*), x^T - x^*\rangle + \frac{\beta}{2}\parallel x^T - x^* \parallel^2_2\le \frac{\beta}{2}\parallel x^T - x^* \parallel^2_2\le\epsilon\\
$$
Therefore, the statement of the theorem is equivalent to claiming that $\parallel x^T - x^* \parallel^2_2\le\frac{2\epsilon}{\beta}$, and $x^T$ is an $\epsilon$-optimal point.

Given the $\beta$-$SS$ property, 
$$
f(x^{t+1}) - f(x^t) \le\langle \nabla f(x^t), x^{t+1} - x^t\rangle + \frac{\beta}{2}\parallel x^t - x^{t+1} \parallel^2_2\\
=\langle \nabla f(x^t), x^{t+1} - x^*\rangle + \langle \nabla f(x^t), x^* - x^t\rangle+ \frac{\beta}{2}\parallel x^t - x^{t+1} \parallel^2_2
$$
In PGD, $z^{t+1} = x^t - \eta\cdot\nabla f(x^t)$, thus
$$
f(x^{t+1}) - f(x^t) \le \frac{1}{\eta} \langle x^t - z^{t+1}, x^{t+1} - x^*\rangle + \langle \nabla f(x^t), x^* - x^t\rangle+ \frac{\beta}{2}\parallel x^t - x^{t+1} \parallel^2_2
$$
The above expression contains an unwieldy term $z^{t+1}$. Since this term only appears during projection steps, we eliminate it by applying Projection Property-I (Lemma 2.3), for all $x \in C, \langle x - \hat{z}, z - \hat{z}\rangle\leq 0$ ,to get
$$
(z^{t+1} - x^{t+1})(x^* - x^{t+1})\le 0\\
(z^{t+1} - x^{t+1})(x^{t+1}-x^*)\ge 0\\
(-x^t + z^{t+1} + x^t - x^{t+1})(x^{t+1}-x^*)\ge 0\\
(x^t - z^{t+1})(x^{t+1} - x^*)\le (x^t - x^{t+1})(x^{t+1} - x^*)\\
\langle x^t - z^{t+1}, x^{t+1} - x^*\rangle \le \langle x^t - x^{t+1}, x^{t+1} - x^*\rangle\\
=\frac{\parallel x^t - x^* \parallel^2_2 - \parallel x^t - x^{t+1} \parallel^2_2 - \parallel x^{t+1} - x^* \parallel^2_2}{2}
$$
Combining the above results and using $\eta=\frac{1}{\beta}$, we have
$$
f(x^{t+1}) - f(x^t) \le \frac{1}{\eta} \langle x^t - z^{t+1}, x^{t+1} - x^*\rangle + \langle \nabla f(x^t), x^* - x^t\rangle+ \frac{\beta}{2}\parallel x^t - x^{t+1} \parallel^2_2\\
\le \frac{1}{\eta} \langle x^t - x^{t+1}, x^{t+1} - x^*\rangle + \langle \nabla f(x^t), x^* - x^t\rangle+ \frac{\beta}{2}\parallel x^t - x^{t+1} \parallel^2_2\\
=\beta(\frac{\parallel x^t - x^* \parallel^2_2 - \parallel x^t - x^{t+1} \parallel^2_2 - \parallel x^{t+1} - x^* \parallel^2_2}{2}) + \langle \nabla f(x^t), x^* - x^t\rangle+ \frac{\beta}{2}\parallel x^t - x^{t+1} \parallel^2_2\\
=\langle \nabla f(x^t), x^* - x^t\rangle+ \frac{\beta}{2}(\parallel x^t - x^* \parallel^2_2 -\parallel x^{t+1} - x^* \parallel^2_2)
$$
The above expression is perfect for a telescoping step but for the inner product term. Given the $\alpha$-$SC$ property, this can be eliminated.
$$
\frac{\alpha}{2}\parallel x^t - x^* \parallel^2_2 \leq f(x^*) - f(x^t) - \langle \nabla f(x^t), x^* - x^t\rangle\\
\langle \nabla f(x^t), x^* - x^t\rangle\le f(x^*) - f(x^t) - \frac{\alpha}{2}\parallel x^t - x^* \parallel^2_2
$$
Thus,
$$
f(x^{t+1}) - f(x^t) \le f(x^*) - f(x^t) - \frac{\alpha}{2}\parallel x^t - x^* \parallel^2_2 + \frac{\beta}{2}(\parallel x^t - x^* \parallel^2_2 -\parallel x^{t+1} - x^* \parallel^2_2)\\
f(x^{t+1}) - f(x^*)\le \frac{\beta-\alpha}{2}\parallel x^t - x^* \parallel^2_2 -\frac{\beta}{2}\parallel x^{t+1} - x^* \parallel^2_2
$$
We have $f(x^{t+1}) \ge f(x^*)$, which means that
$$
\frac{\beta-\alpha}{2}\parallel x^t - x^* \parallel^2_2 -\frac{\beta}{2}\parallel x^{t+1} - x^* \parallel^2_2 \ge 0\\
\frac{\beta}{2}\parallel x^{t+1} - x^* \parallel^2_2 \le \frac{\beta-\alpha}{2}\parallel x^t - x^* \parallel^2_2\\
\parallel x^{t+1} - x^* \parallel^2_2 \le (1-\frac{\alpha}{\beta})\parallel x^t - x^* \parallel^2_2
$$
Which can be written as the following with the fact that $1-x\le exp(-x)$ for all $x\in R$
$$
\Phi_{t+1}\le(1-\frac{\alpha}{\beta})\Phi_t\le exp(-\frac{\alpha}{\beta})\Phi_t
$$
Applying this result recursively gives us
$$
\Phi_{t+1} \le exp(-\frac{\alpha t}{\beta})\Phi_1 = exp(-\frac{\alpha t}{\beta})\parallel x^*\parallel_2^2
$$
Since $x^1 = 0$, we deduce the following expression after at most  $T = O(\frac{\beta}{\alpha}log\frac{\beta}{\epsilon})$ steps which finishes the proof.
$$
\Phi_{t+1}=\parallel x^T-x^*\parallel_2^2\le\frac{2\epsilon}{\beta}
$$
Notice that the convergence of the PGD algorithm is of the form $\parallel x^{t+1}-x^*\parallel_2^2\le exp(-\frac{\alpha t}{\beta})\parallel x^*\parallel_2^2$. The number $\kappa:= \frac{\beta}{\alpha}$ is the *condition number* of the optimization problem, which is central to numerical optimization. The exact numerical form of the condition number will also change depending on the application at hand. In general, all these definitions of condition number will satisfy the following property.

Comparing Theorem 2.5 and 2.6, we may fancy the latter one, because it could converge faster than the former one based $T=O(n)$. 

**Definition 2.6** (*Condition Number - Informal*) The condition number of a function $f: \chi\rightarrow R$ is scalar $\kappa\in R$ that bounds how much the function value can change relative to a perturbation of the input.

Functions with a small condition number are stable and changes to their input do not affect the function output values too much. However, functions with a large condition number can be quite jumpy and experience abrupt changes in output values even if the input is changed slightly. 

Assume that a differentiable function $f$ that is also  $\alpha$-$SC$ and $\beta$-$SS$. And a stationary point for $f$, i.e., a point $x$ such that $\nabla f(x) = 0$. For a general function, such a point can be a local optima or a saddle point. however, since $f$ is strongly convex, $x$ is the (unique) global minima of $f$. Then for any other point $y$ we have
$$
\frac{\alpha}{2}\parallel x - y \parallel^2_2 \leq f(y) - f(x) - \langle \nabla f(x), y - x\rangle \leq \frac{\beta}{2}\parallel x - y \parallel^2_2\\
\frac{f(y)-f(x)}{\frac{\alpha}{2}\parallel x - y \parallel^2_2}\in[1,\frac{\beta}{\alpha}]:=[1,\kappa]
$$
Thus, upon perturbing the input from the global minimum $x$ to a point $\parallel x - y \parallel_2=:\epsilon$ distance away, the function value does change much [$\frac{\alpha\epsilon^2}{2}, \kappa\cdot\frac{\alpha\epsilon^2}{2}$]. Such well behaved response to perturbations is very easy for optimization algorithms to exploit to give fast convergence.

The condition number of the objective function can significantly affect the convergence rate of algorithms. Indeed, if $\kappa=\frac{\beta}{\alpha}$ is small, then $exp(-\frac{\alpha}{\beta}) = exp(-\frac{1}{\kappa})$ would be small, ensuring fast convergence. However, if $\kappa\gg1$ then $exp(-\frac{1}{\kappa})\approx1$ and the procedure might offer slow convergence, which is under *ill condition*. In this case, *pre-condition* is a good technique.