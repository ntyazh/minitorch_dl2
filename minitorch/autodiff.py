from dataclasses import dataclass
from typing import Any, Iterable, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    vals_list, eps_vals = list(vals), list(vals)
    eps_vals[arg] = eps_vals[arg] + epsilon
    return (f(*eps_vals) - f(*vals_list)) / epsilon


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    def visit(node):
        if node.unique_id in visited_nodes.keys() or node.is_constant():
            return

        if not node.is_leaf():
            for neigh in node.parents:
                visit(neigh)

        visited_nodes[node.unique_id] = node

    # TODO: Implement for Task 1.4.
    visited_nodes = dict()
    visit(variable)
    topological_vars = list(visited_nodes.values())
    topological_vars.reverse()
    return topological_vars


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    queue = topological_sort(variable)
    var_derivs = {variable.unique_id: deriv}
    for scalar in queue:
        deriv = var_derivs[scalar.unique_id]
        if scalar.is_leaf():
            scalar.accumulate_derivative(deriv)
            # var_derivs.update({scalar.unique_id: scalar.grad})
        else:
            chain_derivs = scalar.chain_rule(deriv)
            for var, chain_deriv in chain_derivs:
                if var.unique_id in var_derivs.keys():
                    var_derivs[var.unique_id] += chain_deriv
                else:
                    var_derivs[var.unique_id] = chain_deriv


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
