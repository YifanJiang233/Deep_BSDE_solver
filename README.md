# Deep BSDE solver in Pytorch

## Usage

This package provides a solver for the following FBSDE:

![FBSDE](https://render.githubusercontent.com/render/math?math=%5Cleft%5C%7B+%5Cbegin%7Baligned%7D+X_%7Bt%7D+%26+%3D+x+%2B+%5Cint_%7B0%7D%5E%7Bt%7Db%28s%2CX_%7Bs%7D%2CY_%7Bs%7D%29%5C%2C%5Cmathrm%7Bd%7D+s%2B+%5Cint_%7B0%7D%5E%7Bt%7D%5Clangle%5Csigma%28s%2CX_%7Bs%7D%29%2C%5Cmathrm%7Bd%7D+W_%7Bs%7D%5Crangle+%5C%5C+Y_%7Bt%7D+%26+%3Dg%28X_%7BT%7D%29%2B%5Cint_%7Bt%7D%5E%7BT%7Df%28s%2CX_%7Bs%7D%2CY_%7Bs%7D%2CZ_%7Bs%7D%29%5C%2C%5Cmathrm%7Bd%7D+s-%5Cint_%7Bt%7D%5E%7BT%7D%5Clangle+Z_%7Bs%7D%2C%5Cmathrm%7Bd%7D+W_%7Bs%7D%5Crangle+%5Cend%7Baligned%7D+%5Cright.)

`solver.py` contains the neural network based on the Deep BSDE method [1].

`cir_bond.py` and `multi_cir_bond.py` are codes for the examples in [2].

One can easily apply this solver to other FBSDEs by redefining the coefficients and parameters.

It is important to notice that the output of the coefficient should be defined in an appropriate size.
Here, we list the corresponding size as follows.

| Coefficient |        Output Size         |
| :---------: | :------------------------: |
|     *b*     |    [batch_size, dim_x]     |
|     *σ*     | [batch_size, dim_x, dim_d] |
|     *f*     |    [batch_size, dim_y]     |
|     *g*     |    [batch_size, dim_y]     |

The training data will be saved in `loss_data.npy` and `y0_data.npy` which record the loss ![loss](https://render.githubusercontent.com/render/math?math=%24%5Cmathbb%7BE%7D%5Bg%28X_%7BT%7D%29-Y_%7BT%7D%5D%5E%7B2%7D%24) and the initial value ![initial_value](https://render.githubusercontent.com/render/math?math=%24y_0%24) after each iteration.

## Dependencies

* Pytorch >= 1.4.0
* Numpy

## References

[1] E, W., Han, J., and Jentzen, A. Deep learning-based numerical methods for high-dimensional parabolic partial differential equations and backward stochastic differential equations, Communications in Mathematics and Statistics, 5, 349–380 (2017).

[2] Jiang,Y., Li, J. Convergence of the deep bsde method for fbsdes with non-lipschitz coefficients.
