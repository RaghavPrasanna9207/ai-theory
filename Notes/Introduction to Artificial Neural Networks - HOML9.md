# Introduction to Artificial Neural Networks

## Why Neural Networks, and Why Now

*   **Artificial Neural Networks (ANNs)** are machine learning models loosely inspired by the networks of neurons in animal brains. The inspiration is loose on purpose: just like airplanes don't flap their wings even though birds inspired flight, ANNs have drifted quite far from biological accuracy in exchange for things that actually work well computationally.
*   ANNs are the foundation of **deep learning**. They power large-scale, complex tasks: image classification (Google Images), speech recognition (Siri, Google Assistant), chatbots (ChatGPT, Claude), recommendation systems (YouTube), and scientific tools (AlphaFold's protein folding).
*   **A brief history, because it explains why the field looks the way it does today:**
    *   1943 — McCulloch and Pitts published the first artificial neuron model, showing that networks of simple binary units could compute logical propositions.
    *   1960s — Early hype ("we'll be talking to intelligent machines soon") faded once it became clear the promise wouldn't be delivered quickly. Funding dried up — the first *AI winter*.
    *   Early 1980s — New architectures and better training methods revived interest (*connectionism* — the study of neural networks — as a field name comes from this era).
    *   1990s — Other techniques (like support vector machines) seemed to offer better results with stronger theoretical grounding, so neural nets were sidelined again.
    *   Today — A third wave, and there are concrete reasons to think it's different this time, not just hype:
        *   **Much more data** is available to train on, and ANNs tend to outperform other techniques specifically when data is abundant and the problem is complex.
        *   **Much more compute.** This comes from decades of hardware improvement (Moore's law) plus, crucially, **GPUs** (graphical processing units). GPUs were built to accelerate graphics rendering, which involves huge amounts of matrix multiplication — it turns out neural network training is also mostly matrix multiplication, so GPUs accelerate it too, almost by accident. Cloud platforms then made this compute rentable by anyone.
        *   **Better training algorithms** — mostly small refinements of 1990s techniques, but the refinements added up.
        *   **Theoretical fears turned out to be less scary in practice.** Researchers worried training would get permanently stuck in bad *local optima* (points that look like a minimum of the error but aren't the true best possible minimum — the *global optimum*). In practice, especially for large networks, the local optima found are nearly as good as the global one.
        *   **The Transformer architecture (2017)** — covered in a later chapter — was a genuine breakthrough: one architecture that handles text, images, and audio, scales extremely well, and enabled today's giant "foundation models" (models pretrained once and then reused/adapted for many tasks via fine-tuning or prompting).
        *   **A self-reinforcing cycle**: visible AI products attract funding and talent, which produces more visible products, and so on.

## Biological Neurons (the inspiration)

Understanding the biological neuron is not strictly necessary to use ANNs, but it's useful context for why some of the terminology (and some historical dead ends) exist.

*   A biological neuron is a cell with a **cell body** (containing the nucleus), many short branching extensions called **dendrites** that receive signals, and one long extension called the **axon** that sends signals out.
*   Near its tip, the axon splits into branches ending in **synaptic terminals** (synapses), which sit close to — but don't physically touch — the dendrites of other neurons.
*   Neurons communicate with short electrical pulses called **action potentials**. When a neuron receives enough chemical signal (**neurotransmitters**) from other neurons within a short time window, it fires its own action potential; some neurotransmitters instead inhibit firing.
*   Individually, a neuron's behavior is simple. But the brain wires billions of them together (each connected to thousands of others), and complex computation emerges from that scale — similar to how a colony of simple ants can build a complex anthill through collective behavior.
*   Neurons in the brain, especially in the cerebral cortex (the brain's outer layer, responsible for most higher-order thinking), are organized in **layers** — this is the one structural feature that carried over most directly into artificial neural network design.

## Logical Computations with Neurons (McCulloch-Pitts, 1943)

The very first artificial neuron model was much simpler than what's used today, but it's worth understanding because it shows *why* connecting simple units together is powerful at all.

*   A **McCulloch-Pitts neuron** (this is what people mean by "artificial neuron" in this historical context) takes one or more binary inputs (on/off) and produces one binary output. It fires (outputs 1) once more than a certain number of its inputs are active.
*   With this rule alone, you can wire together small networks of these neurons to compute any logical proposition — for example (assuming a neuron fires once at least 2 of its inputs are active):
    *   **Identity**: one neuron feeding a second neuron through two duplicated connections. If the first neuron is on, the second always sees 2 active inputs and fires; if the first is off, the second sees 0 and stays off. Net effect: output equals input.
    *   **AND**: a neuron receiving one connection from each of two source neurons fires only when *both* are active (a single active input isn't enough to reach the firing threshold of 2).
    *   **OR**: give the same neuron *two* connections from each source instead of one. Now a single active source neuron already supplies 2 active inputs, so the neuron fires if *either* source is active.
    *   **NOT** (via inhibition): if a connection is allowed to *inhibit* (block) firing rather than encourage it, you can build "neuron C fires only if A is on AND B is off." If A is wired to always be on, this reduces to plain NOT B.
*   The takeaway that matters for everything that follows: **simple binary units, wired together correctly, are enough to build up arbitrarily complex logic.** This is the seed idea behind stacking layers of neurons — complexity comes from composition, not from making any single neuron smarter.

## The Perceptron (Rosenblatt, 1957)

The **perceptron** is the first practical, *trainable* neural network architecture. It's built from a different, more flexible kind of artificial neuron than the McCulloch-Pitts one.

### The Threshold Logic Unit (TLU)

*   A **Threshold Logic Unit (TLU)**, also called a **Linear Threshold Unit (LTU)**, is the building block of a perceptron. Unlike the McCulloch-Pitts neuron, its inputs and output are real numbers, not just on/off.
*   Each input connection has an associated **weight**. The TLU does two things in sequence:
    1.  Compute a **weighted sum** of the inputs plus a **bias** term: $z = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b$. The bias is a constant offset, not tied to any input — think of it as letting the neuron have a baseline tendency to fire or not fire even when all inputs are zero.
    2.  Apply a **step function** to $z$: if $z$ crosses a threshold, output one value (e.g., 1); otherwise output another (e.g., 0). This step is what makes the whole thing a *classifier* rather than a plain linear equation.
*   This is almost identical to **logistic regression** (covered in the Training Models chapter) — the difference is that logistic regression squashes $z$ through a smooth *sigmoid* curve to get a probability, while the TLU applies a hard, abrupt step. This difference turns out to matter a lot for how each one is trained (see below).
*   The two most common step functions:
    *   **Heaviside step function**: outputs 0 for $z < 0$, 1 for $z \geq 0$.
    *   **Sign function**: outputs $-1$ for $z<0$, $0$ for $z=0$, $+1$ for $z>0$.
*   A single TLU can do simple linear binary classification — e.g., classifying iris flowers by petal length/width. "Training" it means finding good values for the weights and bias.

### Perceptron architecture

*   A **perceptron** is one or more TLUs arranged side by side in a single layer, where *every* TLU receives *every* input. A layer wired this way — every neuron connected to every input — is called a **fully connected layer** or **dense layer**.
*   The raw inputs form the **input layer**. The layer of TLUs that produces the final predictions is the **output layer**.
*   Because a perceptron can have multiple output TLUs, it can predict several independent binary labels at once for the same input — this is a **multilabel classifier**. It can also be adapted for multiclass classification (one class among several, not several labels at once — the distinction matters and comes back later in this chapter).
*   **Computing outputs for a whole batch at once (Equation 9-2):**
    $$\hat{\mathbf{Y}} = \phi(\mathbf{X}\mathbf{W} + \mathbf{b})$$
    *   $\mathbf{X}$: the input matrix — one row per training instance, one column per input feature.
    *   $\mathbf{W}$: the weight matrix — one row per input feature, one column per neuron in the layer.
    *   $\mathbf{b}$: the bias vector — one value per neuron.
    *   $\phi$ (phi): the **activation function** applied to every entry of the result — for a TLU, this is the step function.
    *   The subtlety worth calling out explicitly: mathematically you can't add a vector to a matrix. Data science gets around this with **broadcasting** — the bias vector $\mathbf{b}$ is conceptually copied and added to *every row* of $\mathbf{X}\mathbf{W}$, so each neuron's bias gets applied once per instance in the batch.

### Training a perceptron: Hebb's rule and the perceptron learning rule

*   Rosenblatt's training method was inspired by **Hebb's rule** (Donald Hebb, 1949): biological synapses between two neurons that fire together get stronger over time — summarized by the phrase "cells that fire together, wire together."
*   The **perceptron learning rule** adapts this idea but adds error-correction: instead of just strengthening co-active connections blindly, it strengthens connections *in proportion to how much they would have helped make the correct prediction*.
*   **The weight update rule (Equation 9-3):**
    $$w_{i,j}^{(\text{next step})} = w_{i,j} + \eta(y_j - \hat{y}_j)x_i$$
    *   $w_{i,j}$: the weight connecting input $i$ to neuron (output) $j$.
    *   $x_i$: the value of input $i$ for the current training instance.
    *   $\hat{y}_j$: what neuron $j$ actually predicted for this instance.
    *   $y_j$: what neuron $j$ *should* have predicted (the true label).
    *   $\eta$ (eta): the **learning rate** — how big a step to take.
    *   Intuition: if the prediction was already correct, $(y_j - \hat{y}_j) = 0$ and nothing changes. If the neuron under-fired, this pushes the weight up in the direction of the active inputs that should have triggered it; if it over-fired, it pushes weights down.
*   The perceptron is fed one training instance at a time (not the whole dataset at once), and only the weights feeding into *wrongly predicted* outputs get adjusted.
*   **Convergence guarantee**: if the training data is **linearly separable** (a straight line, or hyperplane in higher dimensions, can perfectly divide the classes), Rosenblatt proved this algorithm is guaranteed to eventually find a solution. This is the **perceptron convergence theorem**. Note that the solution found isn't unique — when data is linearly separable, there are infinitely many valid separating boundaries, and which one you land on depends on initialization and the order instances are seen in.
*   Scikit-Learn's `Perceptron` class implements this directly. It turns out to be mathematically identical to Scikit-Learn's `SGDClassifier` (stochastic gradient descent classifier, from the Training Models chapter) with the specific settings `loss="perceptron"`, `learning_rate="constant"`, `eta0=1`, and `penalty=None` (no regularization) — the perceptron learning rule is really just a special case of stochastic gradient descent.

### Limitations of the perceptron

*   Unlike logistic regression, a plain perceptron does **not** output a probability — just a hard class prediction. This alone is often reason enough to prefer logistic regression.
*   It uses **no regularization** by default, and training simply stops the moment there are zero errors on the training set — so it tends to generalize worse than logistic regression or a linear SVM, which both keep optimizing for a better-separated boundary even after the training points are correctly classified.
*   **The fatal flaw, highlighted by Minsky and Papert (1969):** a perceptron (and any purely linear classifier, logistic regression included) cannot solve problems that aren't linearly separable — the canonical example being **XOR** (exclusive or: true when exactly one of two inputs is true, false otherwise). No single straight line can separate XOR's true cases from its false cases on a 2D plot. This criticism was disproportionately damaging to the field's reputation and contributed to funding drying up, even though the same limitation applies to other linear models too.
*   **The fix**: stack multiple layers of these units. A network with a hidden layer *can* solve XOR — this is exactly the **Multilayer Perceptron**.

## The Multilayer Perceptron (MLP) and Backpropagation

### Architecture

*   An **MLP** stacks: one **input layer**, one or more **hidden layers** of artificial neurons in between, and one final **output layer**. Layers near the input are called **lower layers**; layers near the output are **upper layers**.
*   Because signal only ever flows forward — input toward output, never backward — this architecture is called a **feedforward neural network (FNN)**.
*   When an MLP has a *deep* stack of hidden layers, it's called a **deep neural network (DNN)**. ("Deep" used to specifically mean "more than 2 hidden layers" in the 1990s; today networks routinely have dozens or hundreds of layers, so the term has become fuzzy. In casual usage, "deep learning" often just means "neural networks," even shallow ones.)
*   You can verify by hand that a small MLP solves XOR: with a hidden layer of two TLUs computing intermediate logic and an output TLU combining them, the network correctly outputs 0 for (0,0) and (1,1), and 1 for (0,1) and (1,0).

### Why MLPs were hard to train, and how backpropagation solved it

*   Training a perceptron is straightforward because there's only one layer of weights to blame for any mistake. With multiple layers, it wasn't obvious for decades how to figure out *how much each individual weight, buried deep inside the network, contributed to the final error* — you need the **gradient** (the direction and rate the error changes) of the error with respect to every single weight and bias in order to run gradient descent, but computing that directly for a deep stack of layers is extremely expensive if done naively.
*   **Reverse-mode automatic differentiation** ("reverse-mode autodiff"), introduced by Seppo Linnainmaa in 1970 (not originally about neural networks at all — it's a general calculus/computation technique), solved this: it computes the gradient of *one* output (like a total error) with respect to *every* parameter in a computation, using just two passes through the whole computation graph — one forward, one backward. This efficiency is what makes training large networks with millions of parameters computationally feasible at all.
*   **Backpropagation** ("backprop") = reverse-mode autodiff + gradient descent, applied specifically to train neural networks. It was popularized by Rumelhart, Hinton, and Williams's 1985 paper and remains, 40+ years later, the dominant way neural networks are trained.
*   **An intuition-building analogy**: learning to shoot a basketball. You throw the ball (**forward pass**), see it went too far right (**error/loss**), and reason backward through your body about what to adjust — rotate your arm a bit, which means your shoulder should turn, which means your feet should turn too (**backward pass**, working from "output" error back toward the earliest cause). You then actually adjust your stance (**gradient descent step**). Small errors produce small corrections; you repeat many times until you consistently score.
*   **Backpropagation, step by step:**
    1.  Work on one **mini-batch** at a time (a small subset of the training data, e.g. 32 instances), and repeat over the whole training set multiple times. One full pass through the entire training set is called an **epoch**.
    2.  **Forward pass**: feed the mini-batch through the network layer by layer using Equation 9-2 above, computing and *keeping* every intermediate layer's output (not just the final one) — these intermediate values are needed again in the backward pass.
    3.  **Compute the error**: compare the network's actual output to the desired output using a **loss function**, which produces a single number summarizing how wrong the network was.
    4.  **Backward pass**: starting at the output layer, use the **chain rule** (the calculus rule for differentiating a function that's built by composing other functions — exactly what a multi-layer network is) to work out how much each output-layer weight contributed to the error, and produce one gradient value per parameter. Then repeat this layer by layer moving *backward* toward the input, at each step figuring out how much each connection in the layer below contributed to the error one level up. This backward flow of error-attribution is where the algorithm gets its name.
    5.  **Gradient descent step**: nudge every weight and bias slightly in the direction that reduces the error, using the gradients just computed.
*   **Why random weight initialization is essential, not optional**: if every weight in a layer starts at the exact same value (e.g., all zero), then every neuron in that layer computes the exact same output and receives the exact same gradient update during backprop — they stay identical forever. A layer of 500 identically-behaving neurons is functionally no better than a layer of 1. Randomly initializing weights **breaks this symmetry**, letting different neurons specialize in different things.
*   **The other essential fix Rumelhart's team made**: replace the TLU's hard step function with a *smooth* one. A step function is flat everywhere except at one point, so its derivative is 0 almost everywhere — gradient descent literally has nothing to follow ("no slope to walk down"). Swapping in a smooth curve like the **sigmoid function** gives a well-defined, nonzero gradient at (almost) every point, so gradient descent can always make some progress.

### Activation functions

Why do we even need a nonlinear activation function between layers, instead of just letting each layer output a raw weighted sum? Because **stacking purely linear functions collapses back into a single linear function.** Example: if $f(x) = 2x+3$ and $g(x)=5x-1$, then $f(g(x)) = 2(5x-1)+3 = 10x+1$ — still just one linear function. So without nonlinearity inserted between layers, a 100-layer network would be mathematically no more powerful than a single-layer one. Add nonlinear activations, though, and a large enough network can, in principle, approximate essentially any continuous function — this is why "depth + nonlinearity" is the whole trick.

Common activation functions:

*   **Sigmoid / logistic function**: $\sigma(z) = \frac{1}{1+e^{-z}}$. S-shaped, output squashed to (0, 1), smooth and differentiable everywhere. This was the historical default that made backprop work at all.
*   **Hyperbolic tangent (tanh)**: $\tanh(z) = 2\sigma(2z) - 1$. Also S-shaped and smooth, but output ranges from $-1$ to $1$ instead of 0 to 1. Because its output is centered around 0, it tends to help the network's internal signals stay balanced early in training, which often speeds up convergence compared to sigmoid.
*   **ReLU (Rectified Linear Unit)**: $\text{ReLU}(z) = \max(0, z)$. Simple, extremely cheap to compute, and it's the default choice for hidden layers in most modern architectures (the Transformer is a notable exception). Its downsides: it isn't differentiable exactly at $z=0$ (a small technical wrinkle that doesn't cause real problems in practice), and its derivative is flat 0 for any negative input, meaning a neuron that's stuck outputting negative values contributes zero gradient and stops learning (this "dying ReLU" issue and its fixes are explored further in the Training Deep Neural Networks chapter). Its big practical advantage: it has no upper bound, which helps avoid a specific gradient problem that recurs in deep networks (also covered in that later chapter).
*   Historically, researchers stuck with sigmoid because biological neurons seem to behave in a roughly sigmoid way — an example of the biological analogy actually being *misleading*, since ReLU performs better in practice despite being biologically less "realistic."

## Building and Training MLPs with Scikit-Learn

Scikit-Learn's neural network support (`MLPRegressor`, `MLPClassifier`) is convenient for getting started, but it has real limits: no GPU acceleration, and a fairly fixed set of options. This is exactly why the next chapter moves to PyTorch, which is far more flexible and can exploit GPU hardware. Still, Scikit-Learn is a fast way to get a working MLP in a few lines of code.

### Regression MLPs

*   **Output layer sizing**: one output neuron per value you're predicting. Predicting a single number (house price) needs 1 output neuron; predicting 2D coordinates needs 2; a full bounding box (position + width + height) needs 4.
*   Worked example from the chapter: an `MLPRegressor` with 3 hidden layers of 50 neurons each, trained on the California housing dataset (a simpler, fully-numeric version of the dataset used in the End to End Project chapter — no categorical `ocean_proximity` feature, no missing values, target scaled so 1 unit = \$100,000).
*   **Input scaling matters just as much here as for plain gradient descent** (see the Training Models chapter) — features need to be standardized (e.g., with `StandardScaler`) before training, or gradient descent converges poorly.
*   **Early stopping** (`early_stopping=True`): the model automatically carves off 10% of the training data as an internal validation set (adjustable via `validation_fraction`), checks performance on it after every epoch, and halts training once that score stops improving for a set number of epochs (`n_iter_no_change`, default 10). This guards against overfitting without you having to babysit the training run.
*   Internally, `MLPRegressor` optimizes with **Adam** (a gradient descent variant covered in the Training Deep Neural Networks chapter) and applies a small amount of $\ell_2$ (Ridge-style) regularization by default, controlled by the `alpha` hyperparameter.
*   In the chapter's run: training stopped itself after 45 epochs, reached a validation $R^2$ score of about 0.79 (i.e., the model explains ~79% of the variance in the validation targets), and a final test RMSE around 0.53 — comparable to a random forest on the same task, which is a solid result for a first attempt with no tuning.
*   **Choosing an output activation** (`MLPRegressor` in Scikit-Learn doesn't actually support this — but it's a design decision worth understanding for later, more flexible libraries):
    *   No activation at all (the default) → output can be any real number, positive or negative. Fine for most regression targets.
    *   ReLU or **softplus** ($\text{softplus}(z) = \log(1+e^z)$, a smooth version of ReLU that's close to 0 for negative $z$ and close to $z$ for positive $z$) → guarantees a positive-only output.
    *   Sigmoid or tanh, combined with rescaling your targets to match their range (0 to 1, or $-1$ to 1) → guarantees output stays within a fixed range.
*   **Choosing a loss function**: Mean Squared Error (MSE) is the standard default. If your data has a lot of outliers, Mean Absolute Error (MAE) is more robust, or better yet the **Huber loss** — a hybrid that behaves like MSE for small errors (below some threshold $\delta$, typically 1) and like MAE for large errors, getting MSE's fast/precise convergence on typical points without MSE's oversensitivity to extreme ones. (`MLPRegressor` only supports MSE.)
*   **Typical regression MLP recipe**: 1–5 hidden layers, 10–100 neurons per hidden layer (problem-dependent), 1 output neuron per target dimension, ReLU activation in hidden layers, MSE (or Huber if outlier-heavy) as the loss.

### Classification MLPs

*   **Binary classification**: 1 output neuron with a **sigmoid** activation — its output is directly interpretable as "probability of the positive class" (probability of the negative class is just 1 minus that).
*   **Multilabel binary classification** (each instance can independently belong to any number of classes at once — e.g., an email can be simultaneously spam/ham *and* urgent/non-urgent): one output neuron *per label*, each with its own sigmoid. Because each is computed independently, the probabilities do **not** need to sum to 1 — a model could, in principle, predict "high probability of both spam and urgent" simultaneously.
*   **Multiclass classification** (each instance belongs to exactly one of 3+ mutually exclusive classes, e.g. digit recognition): one output neuron per class, with a single **softmax** activation shared across the whole output layer. Softmax (covered in the Training Models chapter) forces all the outputs to be between 0 and 1 *and* to sum to exactly 1, which correctly models "these are mutually exclusive possibilities."
*   **Loss function**: cross-entropy (also called *log loss* or *x-entropy*) is the standard choice whenever the model is outputting a probability distribution, i.e. for all three cases above.
*   **Worked example: Fashion MNIST.** This dataset is a drop-in replacement for the classic MNIST handwritten-digit dataset — same format (70,000 grayscale 28×28 images, 10 classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot), but meaningfully harder, since each clothing category is visually far more diverse than a handwritten digit is. A plain linear model gets ~92% on MNIST but only ~83% on Fashion MNIST — enough of a gap to make it a genuinely useful benchmark for testing whether a fancier model (like an MLP) actually helps.
*   Pixel values arrive as raw integers 0–255. For image data specifically, `MinMaxScaler` (rescaling to a fixed 0–1 range) tends to work better than `StandardScaler` (rescaling by mean/variance): many pixels — especially near image edges — barely vary across the dataset, so standardizing them would inflate their importance to match highly-variable pixels, which isn't what you want.
*   With two hidden layers (200 and 100 neurons in the accompanying notebook) and early stopping, this setup reaches roughly 89–90% validation accuracy and a comparable test accuracy — a clear improvement over the 83% linear baseline, though later chapters (convolutional neural networks) do noticeably better still on image tasks like this one.
*   **A caution about `predict_proba()`**: neural nets tend to be *overconfident*. In the chapter's run, out of 10,000 test images, only 16 got a top-class probability below 99.9% — despite overall accuracy being only ~89%. In other words, the model is very often extremely (100%) confident even when it's flat wrong. Treat predicted probabilities as a rough signal, not a calibrated truth, especially for a model trained for a long time.
*   **Two ways targets can be represented**: as plain class indices (e.g., `3`), or as **one-hot vectors** (a vector that's all 0s except a single 1 marking the true class, e.g. `[0,0,0,1,0,0,0,0,0,0]`). If overconfidence is a problem, **label smoothing** softens the one-hot target slightly — e.g. instead of `[0,...,1,...,0]`, use `[0.1/9, ..., 0.9, ..., 0.1/9]`, spreading a small amount of probability mass onto the wrong classes on purpose, which discourages the model from driving its confidence all the way to the extremes.
*   **Typical classification MLP recipe**: 1–5 hidden layers; output layer has 1 neuron (binary), 1 per label (multilabel), or 1 per class (multiclass); output activation is sigmoid, sigmoid, or softmax respectively; loss is cross-entropy in all three cases.

## Hyperparameter Tuning Guidelines

Neural nets have a lot of dials to turn, which is both their strength (flexibility) and their weakness (nothing prescribes the "right" architecture up front). General guidance:

*   **Number of hidden layers.** A single hidden layer can, in theory, represent almost any function if it has enough neurons — but in practice, *deeper* networks reach the same accuracy with dramatically fewer total neurons than *wider, shallower* ones. This is because layers build on each other hierarchically: e.g. in a face-recognition network, early layers might learn to detect edges and simple textures, the next layer combines edges into shapes like circles or curves, the next combines shapes into eyes/noses/mouths, and the final layers combine those into whole faces. This reuse of lower-level features is also what makes **transfer learning** possible: you can take the early layers of a network already trained on one task (say, general face recognition) and reuse them as a head-start for a related task (say, hairstyle recognition), since those early layers already learned generally-useful low-level structure. Practical rule of thumb: start with 1–2 hidden layers for most problems (one hidden layer with a few hundred neurons gets >97% on plain MNIST; two hidden layers with the same total neuron count gets >98%, in similar training time); ramp up depth only if you're underfitting and can tolerate the overfitting risk that comes with more capacity. Very hard problems (large-scale image or speech tasks) use networks dozens or hundreds of layers deep — but you'll rarely train those from scratch; reusing a pretrained network (again, transfer learning) is far more common and needs much less data.
*   **Number of neurons per hidden layer.** Input/output layer sizes are fixed by the problem itself (e.g. 784 inputs for a 28×28 image, 10 outputs for 10 classes). For hidden layers, the old "pyramid" convention (each layer smaller than the last, tapering down) has mostly fallen out of favor — using the *same* neuron count in every hidden layer tends to work just as well or better, and it's simpler to tune (one number instead of one per layer). A useful mental model here is Vincent Vanhoucke's **"stretch pants" approach**: rather than hunting for the exact right size, deliberately build the network a bit larger than you think you need, then lean on early stopping and other regularization to prevent it from overfitting — this avoids accidentally creating a **bottleneck layer** (a layer too small to carry all the useful information forward, which caps the whole network's performance no matter how good the other layers are). As a sanity check for what "too small" looks like: PCA analysis (a dimensionality-reduction technique) shows Fashion MNIST needs about 187 dimensions to retain 95% of its information, so a hidden layer meaningfully smaller than that risks losing signal. That said, bottlenecks aren't always bad — deliberately narrow layers can filter out noise, or force the network to learn a compact, information-dense internal representation of the data (useful in its own right — this is called **representation learning**, explored further in a later chapter). General rule of thumb: **you generally get more improvement from adding layers than from adding neurons within a layer.**
*   **Learning rate.** Probably the single most consequential hyperparameter. As a rule of thumb, the best learning rate tends to be roughly half of the *maximum* learning rate — the threshold above which training visibly diverges (loss explodes instead of shrinking; see the Training Models chapter). A practical way to find it: train for a few hundred iterations while ramping the learning rate smoothly from something tiny (like $10^{-5}$) up to something large (like $10$), plot loss against learning rate (log scale on the rate axis), and look for where the loss stops dropping and starts climbing again — the optimal rate is typically about 10x lower than that turning point. Once found, reinitialize the model and train normally at that fixed rate.
*   **Batch size.** Larger batches let GPUs churn through more training instances per second, so many practitioners default to "the largest batch size that fits in **VRAM**" (the GPU's onboard memory). But large batches can also cause training instability (especially early in training or with smaller models) and sometimes generalize worse than smaller ones — Yann LeCun has (informally) argued for batches no larger than 32, citing research showing small batches (2–32) reaching better results in less wall-clock time. Other research shows the opposite is achievable: batches up to 8,192 work fine *if* combined with techniques like **learning rate warmup** (starting training with a small learning rate and gradually ramping it up, rather than using the full rate from step one). Practical strategy: start large (with warmup) for speed; if training is unstable or the final model underperforms, fall back to a smaller batch size.
*   **Other hyperparameters worth tuning if you have the budget**: the choice of **optimizer** (an alternative to plain gradient descent that can converge faster or to a better solution — Adam, seen above, is one example; more are covered in the Training Deep Neural Networks chapter), and the **activation function** (ReLU is a strong default for hidden layers, but variants sometimes help — again, more in the next chapters).
*   **Important interaction to remember**: the ideal learning rate depends on the other hyperparameters — batch size especially. If you change the batch size (or really any other hyperparameter), re-tune the learning rate rather than assuming the old value still holds.

## Typical Architecture Cheat Sheet

| Hyperparameter | Regression | Binary classification | Multilabel classification | Multiclass classification |
|:---|:---|:---|:---|:---|
| Hidden layers | 1–5, problem-dependent | 1–5, problem-dependent | 1–5, problem-dependent | 1–5, problem-dependent |
| Neurons/hidden layer | 10–100, problem-dependent | — | — | — |
| Output neurons | 1 per target dimension | 1 | 1 per label | 1 per class |
| Hidden activation | ReLU | ReLU | ReLU | ReLU |
| Output activation | None, or ReLU/softplus (positive-only) or sigmoid/tanh (bounded) | Sigmoid | Sigmoid | Softmax |
| Loss function | MSE (or Huber if outliers) | Cross-entropy | Cross-entropy | Cross-entropy |

## Formulas

**TLU linear function, then step function:**
$$z = \mathbf{w}^T \mathbf{x} + b, \qquad h_{\mathbf{w},b}(\mathbf{x}) = \text{step}(z)$$

**Heaviside and sign step functions:**
$$\text{heaviside}(z) = \begin{cases} 0 & \text{if } z < 0 \\ 1 & \text{if } z \geq 0 \end{cases} \qquad \text{sgn}(z) = \begin{cases} -1 & \text{if } z < 0 \\ 0 & \text{if } z = 0 \\ +1 & \text{if } z > 0 \end{cases}$$

**Computing the output of a whole fully connected layer at once (broadcasting the bias across every row):**
$$\hat{\mathbf{Y}} = \phi(\mathbf{X}\mathbf{W} + \mathbf{b})$$

**Perceptron learning rule (weight update after each mistake):**
$$w_{i,j}^{(\text{next step})} = w_{i,j} + \eta(y_j - \hat{y}_j)x_i$$

**Sigmoid (logistic) activation function:**
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Hyperbolic tangent activation function (in terms of sigmoid):**
$$\tanh(z) = 2\sigma(2z) - 1$$

**ReLU activation function:**
$$\text{ReLU}(z) = \max(0, z)$$

**Softplus activation function (smooth ReLU, guarantees positive output):**
$$\text{softplus}(z) = \log(1 + e^z)$$

## Questions and Answers

### 1. What does the TensorFlow/neural-network Playground exercise teach you about how MLPs behave?

Playing with the interactive playground (train small networks on 2D toy datasets and watch them learn live) surfaces several intuitions that are hard to get from equations alone:

*   Early hidden-layer neurons learn simple patterns (e.g. rough linear splits); later hidden layers combine those simple patterns into more complex ones — depth builds complexity out of simplicity, concretely visible.
*   Swapping the tanh activation for ReLU makes training converge faster, but the resulting decision boundary becomes visibly made of straight line segments rather than smooth curves — a direct visual consequence of ReLU's own piecewise-linear shape.
*   Very small networks (e.g. one hidden layer with just 3 neurons) train inconsistently — sometimes fast, sometimes stuck for a long time in a local minimum, depending on random initialization.
*   Networks that are too small (e.g. only 2 neurons total in the hidden layer) can't find a good solution *at all*, no matter how many times you retry — this is **underfitting**: too few parameters to represent the pattern.
*   Networks that are "large enough" (e.g. 8 neurons for the same problem) train fast and reliably every time — confirming that larger networks rarely get permanently trapped in bad local minima, though they can still stall on flat plateaus for a while.
*   On harder datasets (e.g. a spiral pattern) with several hidden layers, training slows down and stalls on plateaus more, and — notably — the neurons *closest to the output* learn faster than the neurons *closest to the input*. This is a first hands-on look at the **vanishing gradients problem**: in deep networks, the error signal flowing backward through many layers can shrink to almost nothing by the time it reaches the earliest layers, so those layers barely update. (Fixes — better weight initialization, better optimizers, batch normalization — are covered in the Training Deep Neural Networks chapter.)

### 2. Draw an ANN (using McCulloch-Pitts-style neurons) that computes A ⊕ B (XOR).

Use the identity $A \oplus B = (A \land \lnot B) \lor (\lnot A \land B)$. Build two neurons in a hidden layer: one computes $A \land \lnot B$ (an AND-style neuron where B's connection inhibits instead of excites), the other computes $\lnot A \land B$ (mirrored). A final output neuron performs an OR over those two hidden neurons. Since exactly one of the two hidden neurons fires whenever A and B differ, and neither fires when A and B match, the OR output correctly reproduces XOR. (Other equally valid constructions exist, e.g. built from $(A \lor B) \land \lnot(A \land B)$.)

### 3. Why prefer logistic regression over a classic perceptron? How would you turn a perceptron into a logistic regression classifier?

A plain perceptron only converges to a valid answer when the data is linearly separable, and even then it only outputs a hard yes/no — never a probability. Logistic regression converges to a reasonable solution even when the classes overlap somewhat (it's minimizing a smooth loss via gradient descent, not just chasing zero training errors), and it naturally outputs calibrated-ish probabilities. To turn a perceptron into a logistic regression classifier: replace its step activation function with the sigmoid function (or softmax, if there are multiple output neurons for multiple classes), and train it with gradient descent minimizing cross-entropy loss rather than the perceptron learning rule.

### 4. Why was the sigmoid activation function essential to training the first MLPs?

Because gradient descent needs a nonzero *slope* to know which direction to move weights in. The step function used in the original TLU is flat everywhere except at a single discontinuous jump — its derivative is 0 (or undefined) almost everywhere, so gradient descent has no signal to follow. The sigmoid function has a smooth, nonzero derivative at essentially every point, so gradient descent can always make at least a small amount of progress, however small the network's current error.

### 5. Name three popular activation functions.

Sigmoid ($\sigma(z) = \frac{1}{1+e^{-z}}$, S-shaped, output in (0,1)), hyperbolic tangent ($\tanh(z)$, S-shaped, output in (-1,1), centered at 0), and ReLU ($\max(0,z)$, flat at 0 for negative input then rises linearly, cheap to compute and the default for most modern hidden layers).

### 6. For an MLP with 10 passthrough input neurons, one hidden layer of 50 ReLU neurons, and an output layer of 3 ReLU neurons, what are the shapes of everything?

*   Input matrix $\mathbf{X}$: shape $m \times 10$, where $m$ is however many instances are in the current batch.
*   Hidden layer weight matrix $\mathbf{W}_h$: shape $10 \times 50$ (one row per input feature, one column per hidden neuron). Hidden bias vector $\mathbf{b}_h$: length 50.
*   Output layer weight matrix $\mathbf{W}_o$: shape $50 \times 3$ (one row per hidden neuron feeding in, one column per output neuron). Output bias vector $\mathbf{b}_o$: length 3.
*   Output matrix $\mathbf{Y}$: shape $m \times 3$.
*   Full equation: $\mathbf{Y} = \text{ReLU}(\text{ReLU}(\mathbf{X}\mathbf{W}_h + \mathbf{b}_h)\mathbf{W}_o + \mathbf{b}_o)$. The bias vectors are added to every row via broadcasting, and ReLU is applied entrywise (every negative value in the resulting matrix gets clipped to 0).

### 7. How many output neurons and what activation for: spam detection, MNIST, and housing price prediction?

*   **Spam detection** (binary): 1 output neuron, sigmoid activation, interpreted as "probability this email is spam."
*   **MNIST** (10 mutually-exclusive digit classes): 10 output neurons, softmax activation, so the 10 probabilities sum to 1.
*   **Housing prices** (regression): 1 output neuron, no activation function at all (so the network is free to output any real number). Note: if the target values span a very wide range of magnitudes, it's often easier to have the network predict $\log(\text{price})$ instead of price directly, and then exponentiate the network's output to recover the actual predicted price.

### 8. What is backpropagation, and how does it differ from reverse-mode autodiff?

**Backpropagation** is the *whole training procedure*: repeatedly (a) run a batch forward through the network, (b) compute the error, (c) compute the gradient of that error with respect to every weight and bias, and (d) take a gradient descent step using those gradients — done over and over across many batches until the parameters converge. **Reverse-mode autodiff** is specifically the *technique used in step (c)* — an efficient way to compute all those gradients at once via one forward pass (computing every intermediate value) followed by one backward pass (propagating the error's contribution back through the chain rule). In short: reverse-mode autodiff is the gradient-computation engine; backpropagation is the full training loop that uses that engine.

### 9. List the hyperparameters you can tweak in a basic MLP. If it's overfitting, how would you adjust them?

Tweakable hyperparameters: number of hidden layers, number of neurons per hidden layer, the activation function used in each hidden layer, the activation function used in the output layer, the loss function, the learning rate, the batch size, the optimizer, and (in Scikit-Learn's `MLPRegressor`/`MLPClassifier`) the `alpha` regularization strength and early-stopping settings.

If the model is overfitting (training performance much better than validation performance), the general fix is to *reduce capacity or add regularization*: shrink the number of hidden layers, shrink the number of neurons per layer, turn on/strengthen early stopping, increase $\ell_2$ regularization (`alpha`), or get more training data if that's an option.

### 10. Train a deep MLP on the CoverType dataset — can you exceed 93% test accuracy?

Yes. Using `sklearn.datasets.fetch_covtype()`, a `StandardScaler` → `MLPClassifier` pipeline with 3 hidden layers (200, 100, 50 neurons) and `early_stopping=True` reaches about **93.3% accuracy on the held-out test set** — training stopped itself automatically once the internal validation score plateaued (around 58 epochs in this run), so no manual epoch-count tuning was needed to clear the 93% bar. This confirms the earlier guidance in practice: a moderately deep, tapering-width network combined with early stopping and proper feature scaling is often enough to get solid results without an extensive hyperparameter search.
