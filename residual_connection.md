# Understanding Residual Connections and Vanishing Gradient: From Basics to DeepSeek's Manifold-Constrained Hyper-Connections

*As Deepseek published a very interesting paper on Manifold contrained Hyper-Connections https://arxiv.org/abs/2512.24880, I thought about writing beginner friendly notes on some the basic concepts like Residual connection and Hyper connections that will help understand what this paper is solving. Notes on vanishing gradient is also included as its for audience with no background in deep learning. Note the paper is only briefly discussed towards the end. Also a disclaimer that I have partly used Claude for writing these notes especially the examples and analogy mentioned*

---

## Table of Contents
1. [Residual Connections](#residual-connections)
2. [Understanding Gradients: How Neural Networks Learn](#gradients)
3. [Hyper Connections](#hyperconnection)
4. [DeepSeek's Manifold Constrained Hyper-Connections] (#mhc)

---

## Part 1: Residual Connections {#residual-connections}

### Problem with Neural Networks pre 2016

The world of neural networks before 2016 faced strange problem: **deeper networks (aka stacking more layers) often performed worse than shallow ones.**

This was both counterintuitive and also a fundamental barrier to progress.

### Traditional Neural Networks (Without Residual Connections)

First, let's understand what we had before residual connections.

In a traditional neural network, information flows through layers sequentially:

```
x₁ = F(x₀, W₀)
x₂ = F(x₁, W₁)
x₃ = F(x₂, W₂)
...
```

Where:
- **x₀** is your input (like an image)
- **F** is some transformation (convolution, matrix multiplication, etc.)
- **W** represents the learnable parameters (weights)
- Each layer completely transforms the input from the previous layer

**The Problem:** When you stack many layers (say 50 or 100), two bad things happen:
1. **Vanishing gradients**: The learning signal gets weaker and weaker as it flows backward during training
2. **Degradation**: Surprisingly, deeper networks sometimes perform worse than shallow ones

### What Are Residual Connections?

Residual connections (or "skip connections") were introduced in 2016 with ResNets to address this problem. The core idea is simple: instead of each layer learning the complete transformation of its input, let it learn just the **residual** (the difference or adjustment).

#### The Mathematics

For a single layer with a residual connection:

```
x_{l+1} = x_l + F(x_l, W_l)
```

Let me break this down:
- **x_l**: The input to layer l (your "current state")
- **F(x_l, W_l)**: The residual function—what the layer learns to add or adjust
- **x_{l+1}**: The output, which is the input PLUS the adjustment
- The **"+ x_l"** part is the skip connection (identity mapping)

### Stacking Layers with Residual Connections

When you stack multiple residual layers, something nice happens. Let's trace the flow from layer l to layer L (several layers deeper):

```
x_{l+1} = x_l + F(x_l, W_l)
x_{l+2} = x_{l+1} + F(x_{l+1}, W_{l+1})
x_{l+3} = x_{l+2} + F(x_{l+2}, W_{l+2})
...
```

If we expand this recursively, we get:

```
x_L = x_l + Σ F(x_i, W_i)    [sum from i=l to L-1]
```

**Translation:** The output at layer L equals the input at layer l, plus the sum of all the residual adjustments in between.

```
Final Output = Original Input + All Adjustments
```

### A Concrete Example

Let's work through a simple 3-layer network where x is just a single number:

**Layer 1:**
```
x₁ = x₀ + F(x₀)
   = 5 + 2
   = 7
```

**Layer 2:**
```
x₂ = x₁ + F(x₁)
   = 7 + 1
   = 8
```

**Layer 3:**
```
x₃ = x₂ + F(x₂)
   = 8 + (-1)
   = 7
```

Notice how:
- The original value (5) is preserved in every output: 7 = 5+2, 8 = 5+2+1, 7 = 5+2+1-1
- Each layer only learned small adjustments: +2, +1, -1
- We can trace back to the original input easily

### The Identity Mapping Property

The term **x_l** appears directly in **x_L** without any modification. This is called the "identity mapping" property i.e., the original signal passes through unchanged.

To understand why this is so powerful, we need to dive into gradients.

---

## Part 2: Understanding Gradients - How Neural Networks Learn {#gradients}

### What is a Gradient?

A gradient is simply **how much the output changes when you slightly change the input**. That's it!

#### Real-World Analogy: Hiking

Imagine you're hiking on a hill:
- **Gradient** = the steepness and direction of the slope
- Large gradient → steep hill (small step = big height change)
- Small gradient → gentle slope (small step = tiny height change)
- Zero gradient → flat ground (stepping doesn't change height)

In mathematics this is nothing but derivatives:
```
∂y/∂x = "how much does y change when we change x by a tiny bit?"
```

### How Neural Networks Learn

Neural networks learn through a three-step process:

1. **Make a prediction** (forward pass)
2. **Calculate how wrong the prediction was** (compute the loss)
3. **Adjust the weights to reduce this loss** (backward pass using gradients)

But how do we know which direction to adjust the weights? **We use gradients!**

The gradient tells us:
- Which weights to change
- By how much
- In which direction (increase or decrease)

### Backpropagation: The Chain Rule in Action

Let me walk you through a complete example to show how gradients flow backward through a network.

#### Example Network

```
Input (x) → Layer 1 → Layer 2 → Output (y) → Loss (L)
```

More specifically:
```
x → [+3] → a → [×2] → y → [compare to target] → L
```

Let's say:
- x = 1
- Layer 1 adds 3: `a = x + 3 = 4`
- Layer 2 multiplies by 2: `y = 2a = 8`
- Target is 10, so loss: `L = (y - 10)² = (8 - 10)² = 4`

#### Forward Pass (Making Predictions)

This is straightforward:
```
x = 1 → a = 4 → y = 8 → L = 4
```

#### Backward Pass (Learning)

Now we want to know: **"How should we change x to reduce the loss L?"**

We need to calculate: `∂L/∂x` (how loss changes with respect to x)

**The Chain Rule:**
```
∂L/∂x = (∂L/∂y) × (∂y/∂a) × (∂a/∂x)
```

Think of it like: "the effect of x on L flows through a and y"

Let's calculate each part:

**Step 1: How does L change with y?**
```
L = (y - 10)²
∂L/∂y = 2(y - 10) = 2(8 - 10) = -4
```
*Interpretation: If we increase y by 1, L decreases by 4.*

**Step 2: How does y change with a?**
```
y = 2a
∂y/∂a = 2
```
*Interpretation: If we increase a by 1, y increases by 2.*

**Step 3: How does a change with x?**
```
a = x + 3
∂a/∂x = 1
```
*Interpretation: If we increase x by 1, a increases by 1.*

**Step 4: Multiply them together (Chain Rule)**
```
∂L/∂x = (-4) × (2) × (1) = -8
```

**What this means:** If we increase x by 0.1, the loss will decrease by about 0.8. So we should increase x!

### The Vanishing Gradient Problem

Now let's see what happens with many layers and why it's a disaster.

#### Example: Deep Network Without Residual Connections

```
x → [×0.5] → [×0.5] → [×0.5] → [×0.5] → [×0.5] → y
```

Each layer multiplies by 0.5.

**Forward pass:**
```
x = 1 → 0.5 → 0.25 → 0.125 → 0.0625 → 0.03125 = y
```

**Backward pass (calculating gradients):**

At each layer, `∂output/∂input = 0.5`

By the chain rule:
```
∂y/∂x = 0.5 × 0.5 × 0.5 × 0.5 × 0.5 = 0.03125
```

**The Problem:** The gradient becomes very small! If we have 50 layers:
```
0.5^50 ≈ 0.00000000000001
```

This is **vanishing gradient**—the gradient becomes so tiny that:
1. Early layers barely learn anything. We need to make MASSIVE changes to early layers to have any effect.
2. Training becomes extremely slow. The network can't effectively learn, so learning basically stops.

This is why deeper networks performed worse—they couldn't learn!

### How Residual Connections Fix Everything

#### The Mathematics

Recall that with residual connections:
```
x_{l+1} = x_l + F(x_l, W_l)
```

When we stack multiple layers (from layer l to layer L):
```
x_L = x_l + Σ F(x_i, W_i)
```

#### Taking the Gradient

Let's calculate how x_L changes with respect to x_l:

```
∂x_L/∂x_l = ∂/∂x_l [x_l + Σ F(x_i, W_i)]
```

Using the sum rule of derivatives:
```
∂x_L/∂x_l = ∂x_l/∂x_l + ∂/∂x_l[Σ F(x_i, W_i)]
```

Since the derivative of something with respect to itself is always 1:

```
∂x_L/∂x_l = 1 + ∂/∂x_l[Σ F(x_i, W_i)]
```

### Why This "1" is Magical

Let's break down what this equation means:

```
∂x_L/∂x_l = 1 + [other stuff]
           ↑     ↑
      identity   residual
       path      path
```

**The identity path (the "1"):**
- Always present
- Always equals 1
- Never vanishes
- Provides a **guaranteed highway** for gradients to flow back

**The residual path (the "other stuff"):**
- Can be small
- Can even be zero
- But it doesn't matter because we still have the "1"!


This is why residual connections revolutionized deep learning. They provide **guaranteed gradient flow** that never vanishes, allowing us to train very deep networks successfully!

---

## Part 3: Hyper Connections {#hyperconnection}

Now that we understand residual connections and gradients, we can appreciate DeepSeek's innovation and why it matters. Before going further we will introduce another concept named Hyper Connections.

### The Evolution: From ResNets to Hyper-Connections

Recently, a new approach called **Hyper-Connections (HC)** emerged, and it showed promising results. The key insight was: **"Can we make the architecture more expressive without adding computational cost?"**. Think of it as an upgrade where instead of having just one highway lane of information in ResNets, it has Multiple parallel highway lanes (say, 4 lanes instead of 1).

**Standard ResNet (1 stream):**
```
═══════════════════════> (dimension C)
```

**Hyper-Connections (n streams, e.g., n=4):**
```
Stream 1: ═══════════════════════>
Stream 2: ═══════════════════════>
Stream 3: ═══════════════════════>
Stream 4: ═══════════════════════>
          (dimension expanded to n×C = 4C)
```

#### The Mathematics of Hyper-Connections

Instead of the simple residual formula:
```
x_{l+1} = x_l + F(x_l, W_l)
```

Hyper-Connections use:
```
x_{l+1} = H^res_l · x_l + H^post_l^T · F(H^pre_l · x_l, W_l)
```

Breaking this down:
- **x_l** is now **n×C dimensional** instead of just C (multiple streams)
- **H^res_l**: An **n×n learnable matrix** that mixes information between the n streams
- **H^pre_l**: A **1×n matrix** that aggregates features from all n streams into a C-dimensional input
- **H^post_l**: A **1×n matrix** that distributes the layer output back across the n streams

#### Visual Flow Through One Layer

Here's what happens in a single Hyper-Connection layer:

```
INPUT (n×C dimensional, e.g., 4 streams)
    ↓
    ├─→ Stream 1: [f₁, f₂, ..., fC]
    ├─→ Stream 2: [f₁, f₂, ..., fC]
    ├─→ Stream 3: [f₁, f₂, ..., fC]
    └─→ Stream 4: [f₁, f₂, ..., fC]
    ↓
    [H^res_l mixes between streams]
    ↓
    ├─→ Stream 1 (mixed)
    ├─→ Stream 2 (mixed)
    ├─→ Stream 3 (mixed)
    └─→ Stream 4 (mixed)
    ↓
    [H^pre_l aggregates all streams]
    ↓
    Single C-dimensional vector
    ↓
    [F processes it (convolution, attention, etc.)]
    ↓
    Single C-dimensional output
    ↓
    [H^post_l distributes back to streams]
    ↓
    ├─→ Stream 1 (updated)
    ├─→ Stream 2 (updated)
    ├─→ Stream 3 (updated)
    └─→ Stream 4 (updated)
    ↓
OUTPUT (n×C dimensional)
```

At each layer, information can:
1. Mix between streams (H^res)
2. Combine for processing (H^pre)
3. Distribute back to streams (H^post)

#### Advantages of Hyper-Connections

**Why was this exciting?**

✅ **Increased capacity without FLOPs overhead**: n×C features but same computation cost
✅ **Richer feature representations**: multiple streams can specialize in different aspects
✅ **Enhanced topological complexity**: more sophisticated information routing
✅ **Better empirical performance**: improved results on benchmarks

#### Disadvantages of Hyper-Connections

Now here's where things get problematic, especially at scale.

##### 1. Broken Identity Mapping (The Critical Flaw)

Remember our magical equation from ResNets?
```
∂x_L/∂x_l = 1 + [other stuff]
```

That "1" guaranteed stable gradient flow. **Hyper-Connections break this!**

Let's see why. When we stack multiple HC layers:

**ResNet (good):**
```
x_L = x_l + Σ F(x_i, W_i)
      ↑
  Pure identity - signals pass unchanged
```

**Hyper-Connections (problematic):**
```
x_L = (∏ H^res_{L-i}) · x_l + Σ [...]
      ↑
  Product of matrices - NOT identity!
```

The product of matrices `(∏ H^res_{L-i})` means:
```
H^res_{L-1} · H^res_{L-2} · H^res_{L-3} · ... · H^res_l
```

Due to product of matrices we go back to the same problem we faced pre 2016 i.e., with deeper networks orignal signal can either amplify (**signal explosion**) or it can vanish (**signal attenuation**)

##### 2. Loss of Signal Conservation

In standard ResNets, the identity mapping has a beautiful property: **the average signal strength is conserved**.

If you input a signal with mean value μ:
```
x_l = [3, 5, 7, 9]  → mean = 6
```

After many layers:
```
x_L = [4, 7, 5, 8]  → mean ≈ 6  (still around 6!)
```

In Hyper-Connections, the product of unconstrained H^res matrices does **NOT** preserve the mean:
```
x_l = [3, 5, 7, 9]  → mean = 6

After many layers:
x_L = [0.1, 0.3, 0.2, 0.5]  → mean = 0.275  (drastically changed!)
```

or

```
x_L = [300, 500, 700, 900]  → mean = 600  (exploded!)
```

##### 3. Memory Inefficiency

While FLOPs stay the same, **memory usage increases**:

**ResNet:**
- Residual stream: C dimensions
- Memory for one layer: O(C)

**Hyper-Connections:**
- Residual stream: n×C dimensions
- Memory for one layer: O(n×C)

With n=4, you're using **4× more memory** for the residual stream!

Hyper-Connections are a clever idea that worked well in small scale experiments, but they have a fundamental flaw: **they sacrificed the stability guarantees that made ResNets successful in the first place**.

This is exactly what DeepSeek's paper addresses next.

## Part 4: DeepSeek's Manifold Constrained Hyper-Connections (mHC) {#mhc}

As we saw in the previous section, Hyper-Connections have a critical flaw: they break the identity mapping property that made ResNets so successful. The unconstrained mixing matrices cause signals to explode or vanish as they flow through many layers.

### DeepSeek's Solution: Manifold-Constrained Hyper-Connections (mHC)

DeepSeek's key insight is brilliant: **constrain the mixing matrices to a special mathematical space where they preserve signal properties.**

#### The Mathematical Constraint

They use something called **doubly stochastic matrices** (matrices in the Birkhoff polytope). These are matrices where:
- Every row sums to 1
- Every column sums to 1

For example:
```
[0.3  0.7]     [0.25  0.25  0.25  0.25]
[0.7  0.3]     [0.2   0.3   0.3   0.2 ]
               [0.3   0.2   0.2   0.3 ]
               [0.25  0.25  0.25  0.25]
```

#### Why This Fixes Everything

When H^res is doubly stochastic:

1. **Signal conservation**: The operation `H^res · x` is a weighted average of inputs
   - No amplification (signals can't grow)
   - No attenuation (signals can't shrink)
   - Average strength is preserved

2. **Closure property**: The product of doubly stochastic matrices is also doubly stochastic
   - So `(∏ H^res_{L-i})` maintains the same properties
   - This works no matter how many layers you stack!

3. **Gradient stability**: Since signals don't explode or vanish, neither do gradients

### How They Enforce the Constraint

DeepSeek uses the **Sinkhorn-Knopp algorithm**, which is an iterative method that projects any matrix onto the space of doubly stochastic matrices.

Think of it like this:
1. Start with any mixing matrix H^res
2. Repeatedly normalize rows and columns
3. After a few iterations, you get a doubly stochastic matrix
4. This happens automatically during training

The beauty is that this is differentiable, so gradients can still flow through it!

### Additional Optimizations

DeepSeek didn't stop at mathematical elegance. They also addressed practical concerns:

**1. Memory Efficiency**
- Multiple streams (n×C dimensions) use more memory
- They use selective recomputing and kernel fusion
- Carefully overlap communication in distributed training

**2. Hardware Optimization**
- Developed custom kernels using TileLang
- Mixed precision computation
- Result: Only 6.7% additional time overhead when using 4 streams

### The Complete Picture

Let's visualize the full architecture:

**Standard ResNet:**
```
Input ─────┬─────> Layer ────┬────> Output
           │                  │
           └──────────────────┘
           (identity, gradient = 1)
```

**Hyper-Connections (HC):**
```
Stream 1 ──┬──> mix ──> Layer ──> mix ──┬──> 
Stream 2 ──┤                              ├──>
Stream 3 ──┤                              ├──>
Stream 4 ──┴──────────────────────────────┘
           (unconstrained mixing, unstable)
```

**Manifold-Constrained HC (mHC):**
```
Stream 1 ──┬──> [constrained mix] ──> Layer ──> [constrained mix] ──┬──> 
Stream 2 ──┤         (doubly                         (doubly          ├──>
Stream 3 ──┤         stochastic)                     stochastic)      ├──>
Stream 4 ──┴────────────────────────────────────────────────────────┘
           (preserves signal strength and gradient flow)
```

### Why This Matters

1. **Best of both worlds**: Multiple streams (capacity) + stable training (identity mapping)
2. **Scalability**: Works reliably at large scale (billions of parameters)
3. **Practical efficiency**: Only 6.7% overhead with 4× the representational capacity
4. **Theoretical grounding**: Mathematical guarantees about stability

### Experimental Results

DeepSeek tested mHC on large language model pretraining and found:
- Maintains the performance advantages of Hyper-Connections
- No training instabilities, even at scale
- Successfully trains models with billions of parameters
- Gradient flow remains healthy throughout training

### The Broader Impact

This work represents an important principle in deep learning research:

**Mathematical constraints can enable practical innovations.**

By understanding the fundamental properties that make residual connections work (identity mapping, gradient flow), and applying rigorous mathematical constraints (doubly stochastic matrices), DeepSeek has created an architecture that:
- Extends beyond simple residual connections
- Maintains the stability that made ResNets successful
- Opens new possibilities for even deeper, more capable models

---

## Conclusion: The Journey from ResNets to mHC

Let's trace the complete arc of this story:

**2016 - ResNets:** 
- Problem: Deep networks couldn't train
- Solution: Add identity mappings (skip connections)
- Result: Stable gradients, deep networks work

**2024 - Hyper-Connections:**
- Idea: Expand to multiple parallel streams
- Benefit: More capacity and flexibility
- Problem: Breaks identity mapping, causes instability

**2025 - DeepSeek's mHC:**
- Innovation: Constrain to doubly stochastic matrices
- Result: Multiple streams + stable identity mapping
- Impact: Scalable, efficient, theoretically grounded

The "1" in the equation `∂x_L/∂x_l = 1 + ...` was the key to deep learning's success. DeepSeek's contribution is showing how to preserve that essential property while pushing the boundaries of what neural network architectures can do.

This is deep learning research at its best: taking something that works (residual connections), understanding deeply why it works (identity mapping and gradient flow), and carefully extending it with mathematical rigor to create something better.

---

## Further Reading

If you want to dive deeper:
- **ResNets**: "Deep Residual Learning for Image Recognition" (He et al., 2016)
- **Gradient Flow**: "Understanding the difficulty of training deep feedforward neural networks" (Glorot & Bengio, 2010)
- **Hyper-Connections**: The original HC paper (Zhu et al., 2024)
- **DeepSeek mHC**: "Manifold-Constrained Hyper-Connections" (DeepSeek, 2025)

---

*This blog post was written to make complex deep learning concepts accessible to everyone. If you found it helpful, consider sharing it with others who are learning about neural networks!*
