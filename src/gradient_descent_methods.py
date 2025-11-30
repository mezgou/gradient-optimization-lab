import numpy as np

from typing import Callable


Function2D = Callable[[float, float], float]
Grad2D = Callable[[float, float], tuple[float, float]]
AlphaRule = Callable[[Function2D, Grad2D, int, float, float, float, float], float]


def alpha_const(f: Function2D,
                grad_f: Grad2D,
                k: int,
                x: float, y: float,
                gx: float,
                gy: float,
                alpha0: float = 0.05
) -> float:
    return alpha0


def alpha_diminishing(f: Function2D,
                      grad_f: Grad2D,
                      k: int,
                      x: float,
                      y: float,
                      gx: float,
                      gy: float,
                      alpha0: float = 0.1
) -> float:
    return alpha0 / (1 + k)


def alpha_backtracking(f: Function2D,
                       grad_f: Grad2D,
                       k: int,
                       x: float,
                       y: float,
                       gx: float,
                       gy: float,
                       alpha0: float = 1.0,
                       beta: float = 0.5,
                       c: float = 1e-4
) -> float:
    alpha = alpha0
    fx = f(x, y)
    grad_norm_sq = gx**2 + gy**2

    if grad_norm_sq == 0:
        return 0.0
    
    while True:
        x_new = x - alpha * gx
        y_new = y - alpha * gy
        f_new = f(x_new, y_new)

        if f_new <= fx - c * alpha * grad_norm_sq:
            break

        alpha *= beta

        if alpha <= 1e-10:
            break

    return alpha


def alpha_backtracking_dir(f: Function2D,
                           grad_f: Grad2D,
                           x: float,
                           y: float,
                           gx: float,
                           gy: float,
                           dx: float,
                           dy: float,
                           alpha0: float = 1.0,
                           beta: float = 0.5,
                           c: float = 1e-4
) -> float:
    alpha = alpha0
    fx = f(x, y)
    grad_dot_d = gx * dx + gy * dy

    if grad_dot_d >= 0:
        return 0.0
    
    while True:
        x_new = x + alpha * dx
        y_new = y + alpha * dy
        f_new = f(x_new, y_new)

        if f_new <= fx + c * alpha * grad_dot_d:
            break

        alpha *= beta
        if alpha < 1e-10:
            break

    return alpha


def gradient_descent(f: Function2D,
                     grad_f: Grad2D,
                     x0: float,
                     y0: float,
                     alpha_rule: AlphaRule,
                     eps: float = 1e-6,
                     max_iter: int = 100_000
) -> tuple[float, float, list[tuple[float, float, float]]]:
    x: float = x0
    y: float = y0
    history: list[tuple[float, float, float]] = [(x, y, f(x, y))]

    for k in range(max_iter):
        gx, gy = grad_f(x, y)
        grad_norm = np.sqrt(gx**2 + gy**2)
        if grad_norm < eps:
            break

        alpha = alpha_rule(f, grad_f, k, x, y, gx, gy)

        x = x - alpha * gx
        y = y - alpha * gy

        history.append((x, y, f(x, y)))

    return x, y, history


def conjugate_gradient_fr(f: Function2D,
                          grad_f: Grad2D,
                          x0: float,
                          y0: float,
                          eps: float = 1e-6,
                          max_iter: int = 100_000
) -> tuple[float, float, list[tuple[float, float, float]]]:
    x: float = x0
    y: float = y0

    gx, gy = grad_f(x, y)
    dx, dy = -gx, -gy

    history: list[tuple[float, float, float]] = [(x, y, f(x, y))]

    for k in range(max_iter):
        grad_norm = np.sqrt(gx**2 + gy**2)
        if grad_norm < eps:
            break

        alpha = alpha_backtracking_dir(f, grad_f, x, y, gx, gy, dx, dy)

        x_new = x + alpha * dx
        y_new = y + alpha * dy

        gx_new, gy_new = grad_f(x_new, y_new)
        grad_norm_new_sq = gx_new**2 + gy_new**2
        grad_norm_sq = gx**2 + gy**2

        if grad_norm_sq == 0:
            x, y = x_new, y_new
            history.append((x, y, f(x, y)))
            break

        beta = grad_norm_new_sq / grad_norm_sq

        dx = -gx_new + beta * dx
        dy = -gy_new + beta * dy

        x, y = x_new, y_new
        gx, gy = gx_new, gy_new
        history.append((x, y, f(x, y)))

    return x, y, history
