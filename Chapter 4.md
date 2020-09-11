



## Chapter 4 Alternating Minimization

### 4.1 Marginal Convexity and Other Properties

Alternating Minimization is most often adapted into settings where the optimization problem concerns two or more variables.

**Definition 4.1** (*Joint Convexity*) A continuously differentiable function in two variables $f$: $R^p\times R^q\rightarrow R$ is considered jointly convex if for every ($x_1$,$y_1$), ($x_2$,$y_2$) $\in R^p\times R^q$ we have

​										$f(x_2,y_2)\ge f(x_1,y_1)+\langle\nabla f(x_1,y_1),(x_2,y_2)-(x_1,y_1)\rangle$

where $\nabla f(x_1,y_1)$ is the gradient of $f$ at the point $f(x_1,y_1)$.

It seems quite similar with the definition 2.3, if we assume $f$ to be a function of a single variable $z=(x,y)\in R^{p+q}$ instead of these two variables.

**Definition 4.2** (*Marginal Convexity*) A continuously differentiable function of two variables $f$: $R^p\times R^q\rightarrow R$ is considered marginally convex in its first variable if for every value of $y\in R^q$, the function $f(\cdot,y):R^p\rightarrow R$ is convex, i.e., for every $x_1,x_2\in R$, we have

​												$f(x_2,y)\ge f(x_1,y)+\langle\nabla_x f(x_1,y),x_2-x_1\rangle$

where $\nabla_x f(x_1,y)$ is the partial gradient of $f$ with respect to its first variable at the point $(x_1,y)$. A similar condition is imposed for $f$ to be considered marginally convex in its second variable.

Note that not all multivariate functions that arise in applications are jointly convex.  And a marginally convex function is not necessarily (jointly) convex.

**Definition 4.3** (*Marginally Strongly Convex/Smooth Function*) A continuously differentiable function $f$: $R^p\times R^q\rightarrow R$ is considered (uniformly) $\alpha$-marginally strongly convex (MSC) and (uniformly) $\beta$-marginally strongly smooth (MSS) in its first variable if for every value of $y\in R^q$, the function $f(\cdot,y):R^p\rightarrow R$ is $\alpha$-strongly convex and $\beta$-strongly smooth, i.e., for every $x_1,x_2\in R^p$, we have

​							$\frac{\alpha}{2}\parallel x_2-x_1 \parallel^2_2 \leq f(x_2,y) - f(x_1,y) - \langle g, x_2-x_1\rangle \leq \frac{\beta}{2}\parallel x_2-x_1 \parallel^2_2$

where $g=\nabla_x f(x_1,y)$ is the partial gradient of $f$ with respect to its first variable at the point $(x_1,y)$. A similar condition is imposed for $f$ to be considered (uniformly) MSC/MSS in its second variable.

Note that a function that is MSC with respect to all its variables, need not be a convex function.

### 4.2 Generalized Alternating Minimization

**Definition 4.4** (*Marginally Optimum Coordinate*) Let $f$ be a function of two variables constrained to be in the sets $\mathcal{X,Y}$ respectively. For any point $y\in\mathcal{Y}$, we say that $\tilde{x}$ is a marginally optimal coordinate with respect to $y$, and use the shorthand $\tilde{x}\in mOPT_f(y)$, if $f(\tilde{x},y)\le f(x,y)$ for all $x\in\mathcal{X}$. Similarly for any $x\in\mathcal{X}$, we say $\tilde{y}\in mOPT_f(x)$ if $\tilde{y}$ is marginally optimal coordinate with respect to $x$.

**Definition 4.5** (*Bistable Point*) Given a function $f$ over two variables constrained within the sets $\mathcal{X,Y}$ respectively, a point $(x,y)\in\mathcal{X}\times\mathcal{Y}$ is considered a bistable point if $y\in mOPT_f(x)$ and $x\in mOPT_f(y)$ i.e., both coordinates are marginally optimal with respect to each other.

<img src="images\Figure 4-1.png" style="zoom:80%;" />

In the above figure, the green line is randomly set. We pick a point on the green line randomly to run, till we get the bistable point on the intersection of the green line and blue line. Note the right part of this figure shows us a bad loop of searching the bistable point. And in my point of view, it shows the importance of selecting a region in the beginning of gAM.

In case a function taking bounded values possesses multiple bistable points, the bistable point to which a gAM eventually converges depends on where the procedure was initialized. Thus, we often pay special attention to initialize the procedure "close" to the optimum.

Sum up, a biastable point has the following properties:

a) An optimal point must be a bistable point.

b) gAM must stop after it has reached a bistable point.

c) There are multiple bistable points.

d) gAM exhibits rapid convergence to the bistable point.

e) Functions may have multiple bistable points. This may happen even if the marginally optimal coordinates are unique. i.e., for every $x$ there is a unique $\tilde{y}$ s.t. $\tilde{y}\in mOPT_f(x)$ and vice versa.

### 4.3 A Convergence Guarantee for gAM for Convex Problems

In fact, approaches similar to gAM, commonly known as Coordinate Minimization (CM), are extremely popular for large scale convex optimization problems.

<img src="images\algorithm3.png" style="zoom:80%;" />

**Theorem 4.1** Let $f$: $R^p\times R^q\rightarrow R$ be jointly convex, continuously differentiable, satisfy $\beta$-MSS in both its variables, and $f^*=min_{x,y}f(x,y)>-\infty$. Let the region $S_0=\lbrace x,y:f(x,y)\le f(0,0)\rbrace\subset R^{p+q}$ be bounded, i.e., satisfy $S_0\subseteq B_2((0,0),R)$ for some $R>0$. Let Algorithm 3 be executed with the initialization $(x_1,y_1)=(0,0)$. Then after at most $T=O(\frac{1}{\epsilon})$ steps, we have $f(x^T,y^T)\le f^*+\epsilon$.

**Proof of Theorem 4.1** 

Given the first property of the gAM algorithm, monotonicity, we have at all time steps $t$,

​														$f(x^{t+1},y^{t+1})\le f(x^{t+1},y^t)\le f(x^t,y^t)\le f(x^1,y^1)$

The region $S_0$ is the sublevel set of $f$ at the initialization point, we have $(x^t,y^t)\in S_0$ for all $t$. Thus gAM remains restricted to the bounded region $S_0$ and does not diverge.

We assume $\phi_t=\frac{1}{f(x^t,y^t)-f^*}$ as the potential function. And $\Phi_t>0$ for all $t$ and that convergence is equivalent to showing $\Phi_t\rightarrow\infty$, as

​																				$\phi_T=\frac{1}{f(x^T,y^T)-f^*}\rightarrow\infty$

For any time step $t\ge2$, we consider the hypothetical update as follow instead of the marginal minimization step gAM does in step 3.

​																			$\tilde{x}^{t+1}=x^t-\frac{1}{\beta}\nabla_x f(x^t,y^t)$

Applying Marginal Strong Smoothness, we get

​										$f(\tilde{x}^{t+1},y^t)\le f(x^t,y^t)+\langle\nabla_x f(x^t,y^t),\tilde{x}^{t+1}-x^t\rangle+\frac{\beta}{2}\parallel\tilde{x}^{t+1}-x^t\parallel^2_2$

​															$=f(x^t,y^t)+\langle\nabla_x f(x^t,y^t),-\frac{1}{\beta}\nabla_x f(x^t,y^t)\rangle+\frac{\beta}{2}\parallel-\frac{1}{\beta}\nabla_x f(x^t,y^t)\parallel^2_2$

​															$=f(x^t,y^t)-\frac{1}{2\beta}\parallel\nabla_x f(x^t,y^t)\parallel^2_2$

Since $x^{t+1}\in mOPT_f(y^t)$, we have $f(x^{t+1},y^t)\le f(\tilde{x}^{t+1},y^t)$, which gives us 

​							$f(x^{t+1},y^{t+1})\le f(x^{t+1},y^t)\le f(\tilde{x}^{t+1},y^t)\le f(x^t,y^t)-\frac{1}{2\beta}\parallel \nabla_x f(x^t,y^t) \parallel^2_2$

Since $t\ge2$, we must have had $y^t\in min_y f(x^t,y)$. 

Since $f$ is differentiable, we must have $\nabla_y f(x^t,y^t)=0$. Applying the Pythagoras' theorem gives us as a result, $\parallel \nabla f(x^t,y^t) \parallel^2_2=\parallel \nabla_x f(x^t,y^t) \parallel^2_2$.

Since $f$ is jointly convex,  we can state

​								 	$f^*-f(x^t,y^t)\ge\langle\nabla f(x^t,y^t),(x^*,y^*)-(x^t,y^t)\rangle$

​								 	$f(x^t,y^t)-f^*\le\langle\nabla f(x^t,y^t),(x^t,y^t)-(x^*,y^*)\rangle$

Applying Cauchy-Schwartz inequality with the fact that $(x^t,y^t),(x^*,y^*)\in S_0$, we define the radio of $S_0$ to be R, that is $\parallel (x^t,y^t)-(x^*,y^*) \parallel_2\le 2R$, then we have

​								 	$f(x^t,y^t)-f^*\le\parallel \nabla f(x^t,y^t) \parallel_2\cdot\parallel (x^t,y^t)-(x^*,y^*) \parallel_2$

​								 	$f(x^t,y^t)-f^*\le2R\parallel \nabla f(x^t,y^t) \parallel_2$

​						    	   	 	$\frac{f(x^t,y^t)-f^*}{2R}\le\parallel \nabla f(x^t,y^t) \parallel_2$

Putting all these together gives us,

​													$f(x^{t+1},y^{t+1})\le f(x^t,y^t)-\frac{1}{2\beta}\parallel \nabla f(x^t,y^t) \parallel^2_2$

​													$f(x^{t+1},y^{t+1})\le f(x^t,y^t)-\frac{1}{2\beta}(\frac{f(x^t,y^t)-f^*}{2R})^2$

​													$f(x^{t+1},y^{t+1})\le f(x^t,y^t)-\frac{1}{8\beta R^2}(f(x^t,y^t)-f^*)^2$

or in other words,

​														$\frac{1}{\Phi_{t+1}}\le\frac{1}{\Phi_{t}}-\frac{1}{8\beta R^2}\frac{1}{\Phi_t^2}\le\frac{1}{\Phi_{t}}-\frac{1}{8\beta R^2}\frac{1}{\Phi_{t}\Phi_{t+1}}$

where the second step follows from monotonicity. Rearranging gives us

​																			$\Phi_{t+1}-\Phi_t\ge\frac{1}{8\beta R^2}$

which upon telescoping, and using $\Phi_2\ge0$ gives us

​																						$\Phi_T\ge\frac{1}{8\beta R^2}$

which proves the result.

Note that the result holds even if $f$ is jointly convex and satisfies the MSS property only locally in the region $S_0$.

### 4.4 A Convergence Guarantee for gAM under MSC/MSS

**Lemma 4.1** A point (x,y) is bistable with respect to a continuously differentiable function $f$: $R^p\times R^q\rightarrow R$ that is marginally convex in both its variables iff $\nabla f(x,y)=0$.

**Proof of Lemma 4.1** It is easy to see that partial derivatives must vanish at a bistable point since the function is differentiable and thus we get $\nabla f(x,y)=[\nabla_x f(x,y),\nabla_y f(x,y)]=0$. Then, if the gradient, and by extension the partial derivatives, vanish at $(x,y)$, then by marginal convexity, for any $x'$

​												$f(x',y)\ge f(x,y)+\langle\nabla_x f(x,y),x'-x\rangle$

​												$f(x',y)-f(x,y)\ge\langle\nabla_x f(x,y),x'-x\rangle=0$

​													  			$f(x',y)\ge f(x,y)$

Similarly, $f(x,y')\ge f(x,y)$ for any $y'$. Thus, $(x,y)$ is bistable.

Let $f^*=min_{x,y}f(x,y)$ to be the optimum value of the objective function, and $(x^*,y^*)$ to be any point such that $f(x^*,y^*)=f^*$. Also, let $Z^*\subset R^p\times R^q$ to be the set of all bistable points for $f$. Lemma 4.1 tells us that $Z^*$ is also the set of all stationary points of $f$. But not all points in $Z^*$ may be global minima.

**Definition 4.6** (*Robust Bistability Property*) A function $f$: $R^p\times R^q\rightarrow R$ satisfies the C-robust Bistability property if for some C > 0, for every $(x,y)\in R^p\times R^q$,  $\tilde{x}\in mOPT_f(y)$ and $\tilde{y}\in mOPT_f(x)$, we have

​										$f(x,y^*)+f(x^*,y)-2f^*\le C\cdot(2f(x,y)-f(x,\tilde{y})-f(\tilde{x},y))$

The right hand expression captures how much one can reduce the function value locally by performing marginal optimizations. The property suggests that we are close to the optimum if not much local improvement can be made, which means that $f(x,y)\approx f(x,\tilde{y})\approx f(\tilde{x},y)$. And it shows that all bistable points achieve the (globally) optimal function value if it satisfies Robust Bistability Property.

**Lemma 4.2** Let $f$ satisfy the properties mentioned in **Theorem 4.2** (MSC/MSS and robust bistability). Then for any $(x,y)\in R^p\times R^q$,  $\tilde{x}\in mOPT_f(y)$ and $\tilde{y}\in mOPT_f(x)$, 

​											$\parallel x-x^* \parallel^2_2+\parallel y-y^* \parallel^2_2\le\frac{C\beta}{\alpha}(\parallel x-\tilde{x} \parallel^2_2+\parallel y-\tilde{y} \parallel^2_2)$

**Proof of Lemma 4.2**

Applying MSC/MSS repeatedly gives us

$f(x,y^*)+f(x^*,y)\ge f(x,y^*)+f(x^*,y)-\langle g, x-x^*\rangle-\langle g, y-y^*\rangle\ge2f^*+\frac{\alpha}{2}(\parallel x-x^* \parallel^2_2+\parallel y-y^* \parallel^2_2)$

​																	    	$f(x,y^*)+f(x^*,y)-2f^*\ge\frac{\alpha}{2}(\parallel x-x^* \parallel^2_2+\parallel y-y^* \parallel^2_2)$

​														   	 	$\frac{\alpha}{2}(\parallel x-x^* \parallel^2_2+\parallel y-y^* \parallel^2_2)\le f(x,y^*)+f(x^*,y)-2f^*$

and

$2f(x,y)\le f(x,\tilde{y})+f(\tilde{x},y)+\frac{\beta}{2}(\parallel x-\tilde{x} \parallel^2_2+\parallel y-\tilde{y} \parallel^2_2)\le f(x,\tilde{y})+f(\tilde{x},y)+\frac{\beta}{2}(\parallel x-\tilde{x} \parallel^2_2+\parallel y-\tilde{y} \parallel^2_2)+\langle g, x-\tilde{x}\rangle+\langle g, y-\tilde{y}\rangle$

​														$2f(x,y)-f(x,\tilde{y})-f(\tilde{x},y)\le\frac{\beta}{2}(\parallel x-\tilde{x} \parallel^2_2+\parallel y-\tilde{y} \parallel^2_2)$

Applying robust stability then we have

$\frac{\alpha}{2}(\parallel x-x^* \parallel^2_2+\parallel y-y^* \parallel^2_2)\le f(x,y^*)+f(x^*,y)-2f^*\le 2f(x,y)-f(x,\tilde{y})-f(\tilde{x},y)\le\frac{\beta}{2C}(\parallel x-\tilde{x} \parallel^2_2+\parallel y-\tilde{y} \parallel^2_2)$

Thus,

​											$\parallel x-x^* \parallel^2_2+\parallel y-y^* \parallel^2_2\le\frac{C\beta}{\alpha}(\parallel x-\tilde{x} \parallel^2_2+\parallel y-\tilde{y} \parallel^2_2)$

And this lemma relates local convergence to global convergence and assures us that reaching an almost bistable point is similar to converging to the optimum.

**Theorem 4.2** Let $f$: $R^p\times R^q\rightarrow R$ be a continuously differentiable (but possibly non-convex) function that, within the region $S_0=\lbrace x,y:f(x,y)\le f(0,0)\rbrace\subset R^{p+q}$, satisfies the properties of $\alpha$-MSC, $\beta$-MSS in both its variables, and C-robust bistability. Let Algorithm 3 be executed with the initialization $(x_1,y_1)=(0,0)$. Then after at most $T=O(log\frac{1}{\epsilon})$ steps, we have $f(x^T,y^T)\le f^*+\epsilon$.

Note that gAM offers rapid convergence despite the non-convexity of the objective. And  MSC/MSS and RBP need only hold within the sublevel set $S_0$. (This underlies the importance of proper initialization).

**Proof of Theorem 4.2** 

We assume $\phi_t=f(x^t,y^t)-f^*$ as the potential function.

Since $\nabla_x f(x^*,y^*)=0$, applying MSS gives us

​														$f(x^{t+1},y^*)-f(x^*,y^*)\le\frac{\beta}{2}\parallel x^{t+1}-x^*\parallel^2_2$

Given the step 4 in gAM algorithm, we can tell that $y^{t+1}\in mOPT_f(x^{t+1})$ and therefore $f(x^{t+1},y^{t+1})\le f(x^{t+1},y^*)$, which gives

​											$\Phi_{t+1}=f(x^{t+1},y^{t+1})-f^*\le f(x^{t+1},y^*)-f^*\le\frac{\beta}{2}\parallel x^{t+1}-x^*\parallel^2_2$

Since $\nabla_x f(x^{t+1},y^t)=0$, applying MSC gives us

​										$f(x^t,y^t)\ge f(x^{t+1},y^t)+\underbrace{\langle\nabla_x f(x^{t+1},y^t),x^{t+1}-x^t\rangle}_{=0}+\frac{\alpha}{2}\parallel x^{t+1}-x^t \parallel^2_2$

​														$\ge f(x^{t+1},y^{t+1})+\frac{\alpha}{2}\parallel x^{t+1}-x^t \parallel^2_2$

which gives us

​													$f(x^t,y^t)-f^*\ge f(x^{t+1},y^{t+1})-f^*+\frac{\alpha}{2}\parallel x^{t+1}-x^t \parallel^2_2$

​																		$\Phi_t\ge \Phi_{t+1}+\frac{\alpha}{2}\parallel x^{t+1}-x^t \parallel^2_2$

For any $t\ge2$, due to the nature of the gAM updates, we know that $x^{t+1}\in mOPT_f(y^t)$ and $y^t\in mOPT_f(x^t)$. Applying Lemma 4.2, gives us

$\parallel x^t-x^*\parallel^2_2\le\parallel x^t-x^*\parallel^2_2+\parallel y^t-y^*\parallel^2_2\le\frac{C\beta}{\alpha}(\parallel x^t-x^{t+1} \parallel^2_2+\parallel y^t-y^t \parallel^2_2)=\frac{C\beta}{\alpha}\parallel x^t-x^{t+1} \parallel^2_2$

Putting these together and using $(a+b)^2\le2(a^2+b^2)$ gives us

​											$\Phi_{t+1}\le\frac{\beta}{2}\parallel x^{t+1}-x^*\parallel^2_2\le\beta(\parallel x^{t+1}-x^t\parallel^2_2+\parallel x^t-x^*\parallel^2_2)$

​												  	$\le\beta(1+C\kappa)\parallel x^{t+1}-x^t\parallel^2_2$

​													  $\le2\kappa(1+C\kappa)(\Phi_t-\Phi_{t+1})$

where $\kappa=\frac{\beta}{\alpha}$ is the effective condition number of the problem. Rearrange gives us

​																  $\Phi_{t+1}\le2\kappa(1+C\kappa)(\Phi_t-\Phi_{t+1})$

​								  $\kappa(1+C\kappa)\Phi_{t+1}+\Phi_{t+1}\le2\kappa(1+C\kappa)\Phi_t$

​						 										  $\Phi_{t+1}\le\eta_0\Phi_t$

where $\eta_0=\frac{2\kappa(1+C\kappa)}{1+2\kappa(1+C\kappa)}<1$ which proves the result.

Note that small values of $\kappa$ and *C* ensure fast convergence, whereas large value of $\kappa$, *C* promote $\eta_0\rightarrow1$ which slow the procedure down.
