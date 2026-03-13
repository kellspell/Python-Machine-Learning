## Understanding integrals and their applications on ML
    * what are integrals ?
        * Compute the area under the curve, representing accumutation
        * Defined as integral of f(x) from A to B
    * Applications on ML
        * Probability distribution
        * Cost Functions 
    * Optimization Concepts
        Think of training a machine learning model like finding the lowest point in a mountain range while blindfolded. The "lowest point" represents the best possible model performance (minimum error/loss).

        * Local vs Global Minima
            * Local minimum
                A local minimum is the lowest point in a specific neighborhood or region, but not necessarily the absolute lowest point overall.

            * Global minimum
                The global minimum is the absolute lowest point across the entire domain of the function - the best possible solution.    

        * Convex Function 
            A convex function is one where the line segment between any two points lies above or on the graph of the function. This creates a single, bowl-shaped valley with only one global minimum and no local minima. In convex optimization, if you find a minimum, you can be guaranteed it's the global minimum.

        * Non-Convex Function 
            A non-convex function violates this property—the line segment between points can go below the graph, creating a landscape with multiple valleys, peaks, and irregular terrain. This results in many local minima, saddle points, and plateaus, making it impossible to guarantee that a found minimum is global.

            * Key Distinction *
                    * Convex: One basin, guaranteed global minimum, easier to optimize

                    * Non-Convex: Complex landscape with multiple basins, no guarantees, harder to optimize

                Most real-world deep learning problems involve non-convex loss functions, which is why optimization is challenging and finding the true global minimum is practically impossible.

## Stochastic Gradient Descent(SGD) and its variants
    * Whats is Stochastic Gradient Descent?
        * Optimization algorithm that uses random subset(mini batches) of the data to cumpute gradients and updates parameters 
    * Why use SGD:

        * SGD is a general-purpose optimization algorithm that works for any differentiable loss    function, regardless of:

            * The type of data (images, text, audio, tabular)

            * The model architecture (CNNs, RNNs, Transformers, MLPs)

            * The task (classification, regression, generation, clustering)

        Think of SGD like a hammer—you can use it with many different types of nails across many different projects, not just one specific task! 
    
    * Variants for SGD 
        * Mini-batch SGD
            A compromise between full batch GD (uses all data) and pure SGD (uses one sample). Processes a small random subset of data (e.g., 32, 64, 128 samples) per iteration, computing gradient on the batch then updating parameters. Balances computational efficiency with stable gradient estimates.

        * Momentum
            Accelerates SGD by accumulating a velocity vector in directions of persistent gradient reduction. Like a ball rolling downhill—gains speed in consistent directions and smooths through noisy updates. Helps escape shallow local minima and speeds up convergence.

        * Adam Optimazer 
            Adaptive Moment Estimation combines Momentum with per-parameter learning rates. Maintains both:

            * First moment: Average of past gradients (like momentum)

            * Second moment: Average of squared gradients (adapts learning rate per parameter)

        Adapts learning rates individually for each parameter, handles sparse gradients well, and requires little hyperparameter tuning. Currently one of the most popular optimizers in deep learning.                                