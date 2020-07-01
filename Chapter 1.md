## Chapter 1 Introduction

### 1.1 Non-convex Optimization

The generic form of an analytic optimization problem is the following

![](images\generic form of analytic opt prob.PNG)

where x is the variable of the problem, $$ f: R^P \rightarrow R$$ is the objective function of the problem, and $$C\subseteq R^P$$ is the constraint set of the problem.

A convex optimization problem is  described as the objective is a convex function as well as the constraint set is a convex set. A non-convex optimization problem is described when it violates either one of these conditions, i.e., one that has a non-convex objective, or s non-convex constraint set, or both.

### 1.2 Motivation for Non-convex Optimization

Modern applications frequently require learning algorithms to dealing with extremely high dimensional data, which necessitates the imposition of structural constraints on the learning models and such structural constraints often turn out to be non-convex. In other applications, the natural objective of the learning task is a non-convex function. Although non-convex objectives and constraints allow us to accurately model learning problems, they often present a formidable challenge on designing algorithms.

### 1.3 Examples of Non-convex Optimization Problems

**Sparse Regression** The classical problem of linear regression seeks to recover a linear model which can effectively  predict a response variable as a linear function of covariates.

**Recommendation Systems** Several internet search engines and e-commerce websites utilize recommendation systems to offer items to users that they would benefit from, or like, the most.

### 1.4 The convex Relaxation Approach

 Facing with the challenge of non-convexity and the associated NP-hardness, the *convex relaxation* approach has been widely studied.  This method makes the problems be relaxing, so that they  become convex optimization problems which could be applied with the existing tools.

In general, such modifications change the problem drastically, and the solutions of the relaxed formulation can be poor to the original problem. However, these distortions referred as "relaxation gap" are absent if the problem possesses certain nice structure and then is relaxed carefully.

The most prominent limitation of this approach is being scalability, though it is a popular and successful approach. And this optimization cannot often solve the large-scale problems while it makes the problems are solvable in polynomial time.

### 1.5 The Non-convex Optimization Approach

Given recent studies, it seems that if the problems possess nice structure, convex relaxation-based approaches as well as non-convex techniques both succeed. However, the latter one usually offer more scalable solutions.