# Deep BSDE solver in Pytorch

## Usage

This package provides a solver for the following FBSDE:

$$
 \left\{
    \begin{aligned}
        X_{t} & = x + \int_{0}^{t}b(s,X_{s},Y_{s})\mathop{}\!\mathrm{d} s+ \int_{0}^{t}\langle\sigma(s,X_{s}),\mathop{}\!\mathrm{d} W_{s}\rangle \\
        Y_{t} & =g(X_{T})+\int_{t}^{T}f(s,X_{s},Y_{s},Z_{s})\mathop{}\!\mathrm{d} s-\int_{t}^{T}\langle Z_{s},\mathop{}\!\mathrm{d} W_{s}\rangle.
    \end{aligned}
    \right.
$$

`solver.py` contains the neural network based on the Deep BSDE method [1].

`cir_bond.py` and `multi_cir_bond.py` are codes for the examples in [2].

One can easily apply this solver to other FBSDEs by redefining the coefficients and parameters.

It is important to notice that the output of the coefficient should be defined in an appropriate size.
Here, we list the corresponding size as follows.
| Coefficient |        Output Size         |
| :---------: | :------------------------: |
|     $b$     |    [batch_size, dim_x]     |
|  $\sigma$   | [batch_size, dim_x, dim_d] |
|     $f$     |    [batch_size, dim_y]     |
|     $g$     |    [batch_size, dim_y]     |

The training data will be saved in `loss_data.npy` and `y0_data.npy` which record the loss $\mathbb{E}[g(X_{T})-Y_{T}]^{2}$ and the initial value $y_0$ after each iteration.

## Dependencies

* Pytorch >= 1.4.0
* Numpy

## References

[1] E, W., Han, J., and Jentzen, A. Deep learning-based numerical methods for high-dimensional parabolic partial differential equations and backward stochastic differential equations, Communications in Mathematics and Statistics, 5, 349–380 (2017).

[2] Jiang,Y., Li, J. Convergence of the deep bsde method for fbsdes with non-lipschitz coefficients.
