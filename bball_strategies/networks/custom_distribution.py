import tensorflow as tf
tfd = tf.contrib.distributions

# TensorFlow's default implementation of the KL divergence between two
# tf.contrib.distributions.MultivariateNormalDiag instances sometimes results
# in NaN values in the gradients (not in the forward pass). Until the default
# implementation is fixed, we use our own KL implementation.


class CustomKLDiagNormal(tfd.MultivariateNormalDiag):
    """Multivariate Normal with diagonal covariance and our custom KL code."""
    pass


@tfd.RegisterKL(CustomKLDiagNormal, CustomKLDiagNormal)
def _custom_diag_normal_kl(lhs, rhs, name=None):  # pylint: disable=unused-argument
    """Empirical KL divergence of two normals with diagonal covariance.

    Args:
      lhs: Diagonal Normal distribution.
      rhs: Diagonal Normal distribution.
      name: Name scope for the op.

    Returns:
      KL divergence from lhs to rhs.
    """
    with tf.name_scope(name or 'kl_divergence'):
        mean0 = lhs.mean()
        mean1 = rhs.mean()
        logstd0 = tf.log(lhs.stddev())
        logstd1 = tf.log(rhs.stddev())
        logstd0_2, logstd1_2 = 2 * logstd0, 2 * logstd1
        return 0.5 * (
            tf.reduce_sum(tf.exp(logstd0_2 - logstd1_2), -1) +
            tf.reduce_sum((mean1 - mean0) ** 2 / tf.exp(logstd1_2), -1) +
            tf.reduce_sum(logstd1_2, -1) - tf.reduce_sum(logstd0_2, -1) -
            mean0.shape[-1].value)
