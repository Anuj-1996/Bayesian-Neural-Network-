# Bayesian-Neural-Network


A Bayesian neural network is a type of neural network that is trained using Bayesian inference. This means that instead of just learning the weights and biases of the network based on a training dataset, the network also learns a distribution over those weights and biases.

In a traditional neural network, the weights and biases are typically chosen to minimize the error between the predicted output of the network and the true output on the training dataset. This is done using an optimization algorithm, such as gradient descent.

However, in a Bayesian neural network, the weights and biases are treated as random variables, and a probability distribution is learned over them. This distribution reflects the uncertainty of the network about the optimal weights and biases for the given task.One advantage of using a Bayesian neural network is that it can provide uncertainty estimates for its predictions. This can be useful in applications where the uncertainty of the prediction is important, such as in medical diagnosis or self-driving cars. Overall, Bayesian neural networks offer a different approach to training neural networks that can be useful in certain situations where traditional neural networks may not be ideal.

In this, I have used the DenseReparameterization layer from the TensorFlow Probability (TFP) library to implement a fully-connected BNN. I have used the negative log likelihood loss function for the likelihood and the Kullback-Leibler divergence loss function for the prior, and optimize the sum of these two losses using the Adam optimizer. Finally, I evaluate the accuracy of the model on the training and test sets.

Additional Information:

Kullback-Leibler (KL) divergence, also known as relative entropy, is a measure of the difference between two probability distributions. It is a non-symmetric measure, meaning that the KL divergence between distribution A and B is not necessarily the same as the KL divergence between distribution B and A.

In the context of machine learning, KL divergence is often used as a measure of the difference between the distribution of the model's weights (the posterior distribution) and the distribution of the weights specified by the prior. The KL divergence loss is then used as a regularization term in the loss function, encouraging the posterior distribution to be similar to the prior distribution.

KL divergence is defined as:

KL(p || q) = ∑p(x)log(p(x)/q(x))

where p and q are the two probability distributions being compared. The KL divergence is always non-negative and is equal to zero if and only if p and q are equal.

In TensorFlow, KL divergence can be calculated using the tfp.distributions.kl_divergence function.
