"""
Math Computation OpenAPI server for Open-WebUI tools usage.

Provides math computations like evaluate, solve, dsolve, derivate, integrate and others (using SymPy).

Author: bgeneto
Date: 2025-04-16
Version: 1.2.8
Last Modified: 2025-05-04
"""

import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)
from fastapi import FastAPI, HTTPException, Request, APIRouter, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional, Dict, Any, Tuple
import logging
import traceback
import signal
import time
import uuid


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("math-computation-tool")

app = FastAPI(
    title="Math Computation LLM Tool",
    version="1.2.8",
    description="Provides math computations like evaluate, solve, dsolve, derivate, integrate, and others (using SymPy).",
)

# Enable CORS (allowing all origins for simplicity)
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define valid operations for better validation
VALID_OPERATIONS = Literal[
    "simplify",
    "evaluate",
    "solve",
    "derivate",
    "integrate",
    "factor",
    "expand",
    "dsolve",
]


# Base model with common fields for all math operations
class ExpressionBase(BaseModel):
    """
    Base model for mathematical expression inputs.

    Attributes:
        expression (str): The mathematical expression to process using SymPy syntax.
    """

    expression: str = Field(
        ...,
        description="The mathematical expression to process using SymPy syntax",
        min_length=1,
        max_length=1000,
    )

    @field_validator("expression")
    @classmethod
    def validate_expression(cls, v):
        """Validate that the expression is not empty."""
        if not v.strip():
            raise ValueError("Expression cannot be empty")
        return v


# For operations that need simple variable substitution (simplify, evaluate, factor, expand)
class SubstitutionInput(ExpressionBase):
    """
    Input model for operations that support simple variable substitution.

    Attributes:
        expression (str): The mathematical expression to process.
        variables (Dict[str, float], optional): Variable substitutions.
    """

    variables: Optional[Dict[str, float]] = Field(
        None, description="Optional variable substitutions, e.g. {'x': 2, 'y': 3}"
    )


# For derivative calculations
class DerivativeInput(ExpressionBase):
    """
    Input model for derivative operations.

    Attributes:
        expression (str): The mathematical expression to differentiate.
        eval_point (float, optional): Point to evaluate the derivative at.
        variable (str, optional): Variable to differentiate with respect to.
    """

    eval_point: Optional[float] = Field(
        None, description="Optional point to evaluate the derivative at"
    )
    variable: Optional[str] = Field(
        None,
        description="Variable to differentiate with respect to (defaults to auto-detect)",
    )


# For integration
class Bounds(BaseModel):
    lower: float = Field(..., description="Lower bound for definite integration")
    upper: float = Field(..., description="Upper bound for definite integration")


class IntegrationInput(ExpressionBase):
    """
    Input model for indefinite and definite integration operations.

    Attributes:
        expression (str): The mathematical expression to integrate.
        bounds (Bounds, optional): Lower and upper bounds for definite integration.
        variable (str, optional): Variable to integrate with respect to.
    """

    bounds: Optional[Bounds] = Field(
        None,
        description="Optional lower and upper bounds for definite integration. Provide an object with 'lower' and 'upper'.",
    )
    variable: Optional[str] = Field(
        None,
        description="Variable to integrate with respect to (defaults to auto-detect)",
    )


# For solving equations
class EquationInput(ExpressionBase):
    """
    Input model for equation solving operations.

    Attributes:
        expression (str): The equation to solve (must contain '=').
        solve_for (str, optional): Variable to solve for.
    """

    solve_for: Optional[str] = Field(
        None, description="Variable to solve for (defaults to auto-detect)"
    )


# For differential equations
class DifferentialEquationInput(ExpressionBase):
    """
    Input model for differential equation operations.

    Attributes:
        expression (str): The differential equation to solve.
        initial_conditions (Dict[str, float], optional): Initial conditions for the differential equation,
                                                         e.g. {'y(0)': 1.0, 'y\'(0)': 0.0}
    """

    initial_conditions: Optional[Dict[str, float]] = Field(
        None, description="Initial conditions for the differential equation"
    )


class Math:
    def __init__(self):
        # Define common symbols for reuse
        self.x, self.z, self.t = sp.symbols("x z t")
        self.y_sym = sp.Symbol("y")
        self.y_func = sp.Function("y")
        self.common_symbols_sym = {
            "x": self.x,
            "y": self.y_sym,
            "z": self.z,
            "t": self.t,
        }
        self.common_symbols_func = {
            "x": self.x,
            "y": self.y_func,
            "z": self.z,
            "t": self.t,
            "Derivative": sp.Derivative,
            "Eq": sp.Eq,
        }
        self.common_symbols = self.common_symbols_sym
        # enable '^' to '**' and implicit multiplication parsing
        self.transformations = standard_transformations + (
            convert_xor,
            implicit_multiplication_application,
        )
        # Set default timeout in seconds
        self.default_timeout = 10

    def _validate_expression(self, expression: str) -> str:
        """Validate and sanitize the input expression."""
        # Check for potentially dangerous operations
        dangerous_patterns = ["__", "os.", "sys.", "eval", "exec", "import", "open"]
        for pattern in dangerous_patterns:
            if pattern in expression:
                raise ValueError(f"Expression contains forbidden pattern: {pattern}")

        # Limit expression length
        if len(expression) > 1000:
            raise ValueError("Expression too long")

        return expression.strip()

    def _parse_expression(self, expression: str) -> sp.Expr:
        """Safely parse a mathematical expression using sympy parser."""
        try:
            expression = self._validate_expression(expression)
            return parse_expr(
                expression,
                local_dict=self.common_symbols,
                transformations=self.transformations,
            )
        except Exception as e:
            logger.error(f"Error parsing expression '{expression}': {str(e)}")
            raise ValueError(f"Invalid expression syntax: {str(e)}")

    def _substitute_variables(
        self, expr: sp.Expr, variables: Dict[str, float]
    ) -> sp.Expr:
        """Substitute variable values into an expression."""
        if not variables:
            return expr

        subs_dict = {}
        for var_name, value in variables.items():
            if var_name in self.common_symbols:
                subs_dict[self.common_symbols[var_name]] = value
            else:
                # Create symbol on the fly if not common
                subs_dict[sp.Symbol(var_name)] = value

        return expr.subs(subs_dict)

    def _detect_variable(self, expr: sp.Expr) -> sp.Symbol:
        """Detect the variable in an expression. If multiple variables exist,
        prioritize t, x, y, and z or return the first one found."""
        symbols = list(expr.free_symbols)

        if not symbols:
            # If no variables found, default to x
            return self.x

        # Check for common variables first
        for var in ["t", "x", "y", "z"]:
            for symbol in symbols:
                if symbol.name == var:
                    return symbol

        # If no common variables found, return the first one
        return symbols[0]

    def _parse_diff_equation(self, expression: str):
        """Parse a differential equation and identify the independent variable,
        dependent function, and any initial conditions."""
        # Normalize the expression
        # Remove newlines, trim, and sanitize
        expression = expression.replace("\n", " ").strip()
        expression = self._validate_expression(expression)

        # Dynamically add any functions found in the expression to common_symbols_func
        import re

        func_names = set(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", expression))
        # also detect functions via prime notation without parentheses
        simple_prime_names = {
            m.group(1)
            for m in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)(?P<primes>'+)", expression)
        }
        func_names |= simple_prime_names

        for name in func_names:
            if name not in self.common_symbols_func and name not in (
                "Derivative",
                "Eq",
            ):
                self.common_symbols_func[name] = sp.Function(name)

        # Determine default variable from function calls if present
        var_match = re.search(r"[A-Za-z]+\( *([A-Za-z_][A-Za-z0-9_]*) *\)", expression)
        default_var = var_match.group(1) if var_match else "x"

        # Preprocess the differential equation to handle prime notation and other formats
        def preprocess_diff_notation(expr_str):
            import re

            seen_funcs = set()

            # 1. Process .diff notation
            diff_pattern = r"(\w+\([^)]+\))\.diff\(([^,]+)(?:,\s*(\d+))?\)"

            def replace_diff(match):
                func_expr = match.group(1)
                var = match.group(2)
                order = match.group(3) if match.group(3) else "1"

                # Extract function name for seen_funcs
                func_name_match = re.match(r"(\w+)\(", func_expr)
                if func_name_match:
                    seen_funcs.add(func_name_match.group(1))

                return f"Derivative({func_expr}, {var}, {order})"

            expr_str = re.sub(diff_pattern, replace_diff, expr_str)

            # Continue with the original prime notation processing
            prime_pattern = (
                r"(?P<func>[A-Za-z_][A-Za-z0-9_]*)"
                + r"(?P<primes>'+)\((?P<arg>[A-Za-z_][A-Za-z0-9_]*)\)"
            )

            def replace_prime(match):
                func_name = match.group("func")
                seen_funcs.add(func_name)
                primes = len(match.group("primes"))
                arg = match.group("arg")
                return f"Derivative({func_name}({arg}), {arg}, {primes})"

            expr_str = re.sub(prime_pattern, replace_prime, expr_str)

            simple_prime_pattern = (
                r"(?P<func>[A-Za-z_][A-Za-z0-9_]*)(?P<primes>'+)(?!\()"
            )

            def replace_simple(m):
                func_name = m.group("func")
                seen_funcs.add(func_name)
                primes = len(m.group("primes"))
                return (
                    f"Derivative({func_name}({default_var}), {default_var}, {primes})"
                )

            expr_str = re.sub(simple_prime_pattern, replace_simple, expr_str)

            # wrap bare function occurrences into calls
            for func_name in seen_funcs:
                pattern = rf"\b{func_name}\b(?!\s*\()"
                expr_str = re.sub(pattern, f"{func_name}({default_var})", expr_str)
            return expr_str

        # Split input at first comma: equation and initial conditions
        orig_parts = expression.split(",", 1)
        eq_str = orig_parts[0].strip()
        ic_strings = orig_parts[1] if len(orig_parts) > 1 else ""

        # Preprocess the differential equation part with enhanced function
        diff_eq_part = preprocess_diff_notation(eq_str)

        # Prepare initial condition parts (if any), splitting further commas
        ic_parts = [s.strip() for s in ic_strings.split(",")] if ic_strings else []

        # Also preprocess any initial conditions
        processed_ic_parts = []
        for part in ic_parts:
            processed_ic_parts.append(preprocess_diff_notation(part))

        ic_parts = processed_ic_parts

        # Initialize variables for the result
        independent_var = None
        dependent_func = None
        eq_sides = None
        ics = []

        # Parse differential equation: split LHS and RHS
        if "=" in diff_eq_part:
            lhs_str, rhs_str = diff_eq_part.split("=", 1)
            lhs = self._parse_expression(lhs_str)
            rhs = self._parse_expression(rhs_str)
            eq_sides = lhs - rhs
        else:
            eq_sides = self._parse_expression(diff_eq_part)

        # Analyze the equation to find the dependent function and independent variable
        dependent_func = None
        independent_var = None
        # Find all undefined functions in the equation
        funcs = list(eq_sides.atoms(sp.Function))
        if funcs:
            func_expr = list(funcs)[0]
            if isinstance(func_expr, AppliedUndef):
                dependent_func = func_expr.func
                if func_expr.args:
                    independent_var = func_expr.args[0]
        # If not found, try to infer from Derivative terms
        if not dependent_func or not independent_var:
            derivs = list(eq_sides.atoms(sp.Derivative))
            for d in derivs:
                if isinstance(d.expr, AppliedUndef):
                    dependent_func = d.expr.func
                    if d.expr.args:
                        independent_var = d.expr.args[0]
                    break
        # Fallback: if still not found, use free symbols
        if not independent_var:
            free_symbols = list(eq_sides.free_symbols)
            for sym in free_symbols:
                if isinstance(sym, sp.Symbol):
                    independent_var = sym
                    break
        # Always ensure dependent_func is a SymPy Function, not a Symbol
        if dependent_func is None or isinstance(dependent_func, sp.Symbol):
            dependent_func = sp.Function("y")
        # Now parse initial conditions if provided
        for part in ic_parts:
            if "=" in part:
                ic_lhs_str, ic_rhs_str = part.split("=", 1)
                ic_lhs_str = ic_lhs_str.strip()
                ic_rhs_str = ic_rhs_str.strip()
                # Parse initial condition left side (e.g., y(0), y'(0))
                try:
                    ic_lhs = self._parse_expression(ic_lhs_str)
                except Exception:
                    # Try to parse as function call manually
                    if "'" in ic_lhs_str:
                        # Derivative initial condition, e.g., y'(0)
                        import re

                        match = re.match(r"([a-zA-Z]+)'*\((.*)\)", ic_lhs_str)
                        if match:
                            func_name = match.group(1)
                            arg = match.group(2)
                            order = ic_lhs_str.count("'")
                            var = sp.Symbol(arg)
                            func = sp.Function(func_name)
                            ic_lhs = sp.Derivative(func(var), var, order)
                        else:
                            raise
                    else:
                        # Function initial condition, e.g., y(0)
                        match = re.match(r"([a-zA-Z]+)\((.*)\)", ic_lhs_str)
                        if match:
                            func_name = match.group(1)
                            arg = match.group(2)
                            var = sp.Symbol(arg)
                            func = sp.Function(func_name)
                            ic_lhs = func(var)
                        else:
                            raise
                ic_rhs = self._parse_expression(ic_rhs_str)
                ics.append((ic_lhs, ic_rhs))

        return {
            "equation": eq_sides,
            "independent_var": independent_var,
            "dependent_func": dependent_func,
            "initial_conditions": ics,
        }

    def _verify_diff_solution(self, diff_eq, solution, independent_var, dependent_func):
        """
        Verify the solution to a differential equation by substituting it back.

        Parameters:
        -----------
        diff_eq : sympy.Expr
            The differential equation to verify (should be in the form LHS - RHS = 0)
        solution : sympy.Expr or list
            The solution or list of solutions to verify
        independent_var : sympy.Symbol
            The independent variable in the differential equation
        dependent_func : sympy.Function
            The dependent function in the differential equation

        Returns:
        --------
        dict
            A dictionary containing verification results
        """
        try:
            if not isinstance(solution, list):
                solution = [solution]

            verification_results = []
            verification_successful = True

            for sol in solution:
                # Extract the right-hand side if the solution is an equation
                if isinstance(sol, sp.Eq):
                    sol_func = sol.rhs
                else:
                    sol_func = sol

                # Create the function expression
                y_func = dependent_func(independent_var)

                # Substitute the solution into the differential equation
                subs_dict = {y_func: sol_func}
                # Substitute derivatives up to order 5 (arbitrary, should be enough for most ODEs)
                for i in range(1, 6):
                    y_deriv = sp.Derivative(y_func, (independent_var, i))
                    try:
                        sol_deriv = sp.diff(sol_func, independent_var, i)
                        subs_dict[y_deriv] = sol_deriv
                    except Exception:
                        break

                try:
                    verification = diff_eq.subs(subs_dict)
                    # Try to fully evaluate and simplify
                    verification = verification.doit().simplify()
                    # Use .equals(0) for robust symbolic check
                    is_zero = verification.equals(0)
                    verification_results.append(verification)
                    if not is_zero:
                        verification_successful = False
                except Exception as e:
                    logger.error(f"Error in substitution: {str(e)}")
                    verification_successful = False
                    verification_results.append("Substitution failed")

            all_zero = verification_successful

            return {
                "verified": all_zero,
                "residuals": verification_results,
                "symbolic_verifications": [str(r) for r in verification_results],
                "all_zero": all_zero,
                "verification_successful": verification_successful,
            }
        except Exception as e:
            logger.error(f"Error in verification: {str(e)}")
            return {
                "verified": False,
                "error": str(e),
                "residuals": None,
                "symbolic_verifications": None,
                "all_zero": False,
                "verification_successful": False,
            }

    def _verify_equation_solution(self, equation, solutions, variable=None):
        """
        Verify the solutions to an algebraic equation by substituting them back.

        Parameters:
        -----------
        equation : sympy.Eq
            The equation to verify
        solutions : list or sympy.Expr
            The solutions to verify
        variable : sympy.Symbol, optional
            The variable that was solved for

        Returns:
        --------
        dict
            A dictionary containing verification results
        """
        try:
            if not isinstance(solutions, list):
                solutions = [solutions]

            # Extract LHS and RHS of the equation
            lhs = equation.lhs
            rhs = equation.rhs

            # The equation to verify is LHS - RHS = 0
            eq_to_verify = lhs - rhs

            # If no variable provided, use the first free symbol in the equation
            if variable is None and eq_to_verify.free_symbols:
                variable = list(eq_to_verify.free_symbols)[0]

            verification_results = []
            overall_verified = True

            for sol in solutions:
                # Substitute the solution
                subs_dict = {variable: sol}
                verification = eq_to_verify.subs(subs_dict)

                # Try multiple simplification strategies
                try:
                    # First try basic simplify
                    residual = sp.simplify(verification)

                    # If not zero, try expansion
                    if residual != 0:
                        residual_expanded = sp.expand(residual)
                        if residual_expanded == 0:
                            residual = residual_expanded

                    # If still not zero, try factoring
                    if residual != 0:
                        residual_factored = sp.factor(residual)
                        if residual_factored == 0:
                            residual = residual_factored

                    verification_results.append(residual)

                    # Update overall verification status
                    if residual != 0:
                        overall_verified = False

                except Exception as e:
                    logger.error(f"Error in simplification: {str(e)}")
                    verification_results.append(verification)  # Use unsimplified result
                    overall_verified = False

            return {
                "verified": overall_verified,
                "residuals": verification_results,
                "symbolic_verifications": [str(r) for r in verification_results],
                "all_zero": all(sp.simplify(r) == 0 for r in verification_results),
            }

        except Exception as e:
            logger.error(f"Error in verification: {str(e)}")
            return {
                "verified": False,
                "error": str(e),
                "residuals": None,
                "symbolic_verifications": None,
                "all_zero": False,
            }

    def _safe_computation(self, func, *args, timeout=None, **kwargs):
        """
        Execute computation with resource limits to prevent DoS attacks.

        Parameters:
        -----------
        func : callable
            The function to execute safely
        *args :
            Arguments to pass to the function
        timeout : int, optional
            Timeout in seconds (defaults to self.default_timeout)
        **kwargs :
            Keyword arguments to pass to the function

        Returns:
        --------
        The result of the function call

        Raises:
        -------
        TimeoutException
            If the computation exceeds the time limit
        """
        if timeout is None:
            timeout = self.default_timeout

        # Define the timeout handler
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Computation timed out after {timeout} seconds")

        # Set the timeout using SIGALRM
        original_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)

        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.debug(f"Computation completed in {elapsed:.3f} seconds")
            return result
        finally:
            # Restore the original signal handler and disable the alarm
            signal.alarm(0)
            signal.signal(signal.SIGALRM, original_handler)

    def simple_math_computation(
        self,
        expression: str,
        operation: str,
        variables: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Perform simple mathematical operations like evaluate, simplify, factor and expand with the given expression.
        Returns a dictionary with result (including a LaTeX formatted one) and additional information.
        """
        try:
            # Validate and sanitize input
            if not expression or not expression.strip():
                return {"success": False, "error": "Expression cannot be empty"}

            # Handle other operations
            expr = self._parse_expression(expression)

            try:
                if operation == "simplify":
                    result = self._safe_computation(sp.simplify, expr)
                elif operation == "evaluate":
                    # Apply variable substitutions if provided
                    if variables:
                        expr = self._substitute_variables(expr, variables)
                    result = self._safe_computation(lambda e: e.evalf(), expr)
                elif operation == "factor":
                    result = self._safe_computation(sp.factor, expr)
                elif operation == "expand":
                    result = self._safe_computation(sp.expand, expr)
                else:
                    return {
                        "success": False,
                        "error": f"Unsupported operation: {operation}",
                    }
            except TimeoutError as e:
                return {"success": False, "error": str(e), "error_type": "timeout"}

            return {
                "success": True,
                "result": {
                    "text_output": str(result),
                    "latex_output": sp.latex(result),
                },
            }

        except ValueError as e:
            logger.warning(f"Invalid input: {str(e)}")
            return {"success": False, "error": str(e), "error_type": "input_error"}
        except MemoryError:
            logger.error(f"Memory limit exceeded for expression: {expression}")
            return {
                "success": False,
                "error": "Computation exceeded memory limits",
                "error_type": "resource_error",
            }
        except Exception as e:
            error_id = uuid.uuid4()
            logger.error(
                f"Error ID: {error_id}, Unexpected error: {str(e)}\n{traceback.format_exc()}"
            )
            return {
                "success": False,
                "error": "An unexpected error occurred",
                "error_type": "system_error",
                "error_id": str(error_id),
            }

    def compute_derivative(
        self,
        expression: str,
        eval_point: Optional[float] = None,
        variable: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compute the derivative of an expression.

        Args:
            expression (str): The expression to differentiate.
            eval_point (float, optional): Point to evaluate the derivative at.
            variable (str, optional): Variable to differentiate with respect to.

        Returns:
            Dict[str, Any]: Dictionary containing the result and metadata.
        """
        try:
            # Use symbol mapping for derivatives
            self.common_symbols = self.common_symbols_sym

            # Parse the expression
            expr = self._parse_expression(expression)

            # Determine the variable to differentiate with respect to
            if variable:
                # Use the specified variable
                sym_var = self.common_symbols.get(variable, sp.Symbol(variable))
            else:
                # Auto-detect
                sym_var = self._detect_variable(expr)

            # Calculate the derivative
            derivative = self._safe_computation(sp.diff, expr, sym_var)

            # Evaluate at a specific point if provided
            if eval_point is not None:
                result = derivative.subs(sym_var, eval_point)
                evaluated_at = eval_point
            else:
                result = derivative
                evaluated_at = None

            return {
                "success": True,
                "with_respect_to": str(sym_var),
                "evaluated_at": evaluated_at,
                "result": {
                    "text_output": str(result),
                    "latex_output": sp.latex(result),
                },
            }
        except TimeoutError as e:
            return {"success": False, "error": str(e), "error_type": "timeout"}
        except ValueError as e:
            logger.warning(f"Invalid input: {str(e)}")
            return {"success": False, "error": str(e), "error_type": "input_error"}
        except Exception as e:
            error_id = uuid.uuid4()
            logger.error(
                f"Error ID: {error_id}, Error computing derivative: {str(e)}\n{traceback.format_exc()}"
            )
            return {
                "success": False,
                "error": f"Error computing derivative: {str(e)}",
                "error_type": "computation_error",
                "error_id": str(error_id),
            }

    def compute_integral(
        self,
        expression: str,
        bounds: Optional[Tuple[float, float]] = None,
        variable: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compute the integral of an expression.

        Args:
            expression (str): The expression to integrate.
            bounds (Tuple[float, float], optional): Lower and upper bounds for definite integration.
            variable (str, optional): Variable to integrate with respect to.

        Returns:
            Dict[str, Any]: Dictionary containing the result and metadata.
        """
        try:
            # Use symbol mapping for integrals
            self.common_symbols = self.common_symbols_sym

            # Parse the expression
            expr = self._parse_expression(expression)

            # Determine the variable to integrate with respect to
            if variable:
                # Use the specified variable
                sym_var = self.common_symbols.get(variable, sp.Symbol(variable))
            else:
                # Auto-detect
                sym_var = self._detect_variable(expr)

            if bounds:
                # Definite integration with specific bounds
                lower, upper = bounds
                result = self._safe_computation(
                    sp.integrate, expr, (sym_var, lower, upper)
                )
                return {
                    "success": True,
                    "with_respect_to": str(sym_var),
                    "bounds": {"lower": lower, "upper": upper},
                    "definite": True,
                    "result": {
                        "text_output": str(result),
                        "latex_output": sp.latex(result),
                    },
                }
            else:
                # Indefinite integration (no bounds)
                result = self._safe_computation(sp.integrate, expr, sym_var)
                return {
                    "success": True,
                    "with_respect_to": str(sym_var),
                    "definite": False,
                    "result": {
                        "text_output": str(result),
                        "latex_output": sp.latex(result),
                    },
                }
        except TimeoutError as e:
            return {"success": False, "error": str(e), "error_type": "timeout"}
        except ValueError as e:
            logger.warning(f"Invalid input: {str(e)}")
            return {"success": False, "error": str(e), "error_type": "input_error"}
        except Exception as e:
            error_id = uuid.uuid4()
            logger.error(
                f"Error ID: {error_id}, Error computing integral: {str(e)}\n{traceback.format_exc()}"
            )
            return {
                "success": False,
                "error": f"Error computing integral: {str(e)}",
                "error_type": "computation_error",
                "error_id": str(error_id),
            }

    def solve_equation(
        self, expression: str, solve_for: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Solve an algebraic equation.

        Args:
            expression (str): The equation to solve (must contain '=').
            solve_for (str, optional): Variable to solve for.

        Returns:
            Dict[str, Any]: Dictionary containing the result and metadata.
        """
        try:
            # Use symbol mapping for equations
            self.common_symbols = self.common_symbols_sym

            if "=" not in expression:
                return {
                    "success": False,
                    "error": "Solve operation requires an equation with '=' symbol",
                }

            lhs_str, rhs_str = expression.split("=", 1)
            lhs = self._parse_expression(lhs_str)
            rhs = self._parse_expression(rhs_str)
            equation = sp.Eq(lhs, rhs)

            # Find all symbols in the equation
            symbols_in_eq = list(equation.free_symbols)
            if not symbols_in_eq:
                return {
                    "success": False,
                    "error": "No variables found in equation to solve for",
                }

            # Determine which variable to solve for
            if solve_for:
                # Use the specified variable if provided
                solve_sym = self.common_symbols.get(solve_for, sp.Symbol(solve_for))
                if solve_sym not in symbols_in_eq:
                    return {
                        "success": False,
                        "error": f"Variable '{solve_for}' not found in the equation",
                    }
            else:
                # Default to solving for x if present, otherwise use the first symbol
                solve_sym = self.x if self.x in symbols_in_eq else symbols_in_eq[0]

            # Use safe computation for potentially expensive solve operation
            result = self._safe_computation(sp.solve, equation, solve_sym)

            # Verify the solution
            verification_result = self._verify_equation_solution(
                equation, result, solve_sym
            )

            return {
                "success": True,
                "solved_for": str(solve_sym),
                "verified": verification_result.get("verified", False),
                "verification_details": verification_result.get(
                    "symbolic_verifications", None
                ),
                "result": {
                    "text_output": str(result),
                    "latex_output": sp.latex(result),
                },
            }
        except TimeoutError as e:
            return {"success": False, "error": str(e), "error_type": "timeout"}
        except ValueError as e:
            logger.warning(f"Invalid input: {str(e)}")
            return {"success": False, "error": str(e), "error_type": "input_error"}
        except Exception as e:
            error_id = uuid.uuid4()
            logger.error(
                f"Error ID: {error_id}, Error solving equation: {str(e)}\n{traceback.format_exc()}"
            )
            return {
                "success": False,
                "error": f"Error solving equation: {str(e)}",
                "error_type": "computation_error",
                "error_id": str(error_id),
            }

    def solve_differential_equation(
        self, expression: str, initial_conditions: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Solve a differential equation.

        Args:
            expression (str): The differential equation to solve.
            initial_conditions (Dict[str, float], optional): Initial conditions in the form
                                                            {'y(0)': 1, 'y\'(0)': 0}.

        Returns:
            Dict[str, Any]: Dictionary containing the result and metadata.
        """
        try:
            self.common_symbols = self.common_symbols_func

            # --- Improved preprocessing for initial_conditions ---
            def parse_ic_key(key: str):
                import re

                # split key into inner expression and numeric value
                m = re.match(r"^(.*)\(\s*([0-9]+(?:\.[0-9]*)?)\s*\)$", key)
                if not m:
                    raise ValueError(f"Invalid initial condition format: '{key}'")
                inner, val = m.group(1).strip(), float(m.group(2))
                # handle explicit Derivative notation
                if inner.startswith("Derivative"):
                    expr = parse_expr(inner, local_dict=self.common_symbols)
                # handle D(func) shorthand for first derivative
                elif re.match(r"^D\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)$", inner):
                    func_name = re.match(
                        r"^D\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)$", inner
                    ).group(1)
                    expr = sp.Derivative(
                        sp.Function(func_name)(self.common_symbols["x"]),
                        self.common_symbols["x"],
                    )
                # handle prime notation (e.g., y', y'')
                elif "'" in inner:
                    qm = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)(\'+)$", inner)
                    if not qm:
                        raise ValueError(f"Invalid prime initial condition: '{key}'")
                    func_name, primes = qm.group(1), qm.group(2)
                    order = len(primes)
                    expr = sp.Derivative(
                        sp.Function(func_name)(self.common_symbols["x"]),
                        self.common_symbols["x"],
                        order,
                    )
                # handle function value (e.g., y)
                else:
                    fm = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)$", inner)
                    if not fm:
                        raise ValueError(f"Invalid function initial condition: '{key}'")
                    expr = sp.Function(fm.group(1))(self.common_symbols["x"])
                # substitute numeric value for independent var
                return expr.subs(self.common_symbols["x"], val)

            preprocessed_ics = None
            if initial_conditions:
                preprocessed_ics = {}
                for k, v in initial_conditions.items():
                    sympy_key = parse_ic_key(k)
                    preprocessed_ics[sympy_key] = v

            parsed = self._parse_diff_equation(expression)
            diff_eq = parsed["equation"]
            independent_var = parsed["independent_var"]
            dependent_func = parsed["dependent_func"]
            ics = parsed["initial_conditions"]

            # Check if we have enough information to proceed
            if independent_var is None or dependent_func is None:
                return {
                    "success": False,
                    "error": "Could not identify independent variable or dependent function in the differential equation",
                }

            # Create the function expression (e.g., y(x))
            if callable(dependent_func):
                func_expr = dependent_func(independent_var)
            else:
                func_expr = dependent_func

            # Build ics dict in SymPy's expected format
            ics_dict = None
            if preprocessed_ics:
                ics_dict = preprocessed_ics
            elif ics:
                ics_dict = {}
                for ic_expr, ic_val in ics:
                    # Handle function value: y(x0)
                    if isinstance(ic_expr, sp.Function) or ic_expr.is_Function:
                        # e.g., y(0)
                        x0 = ic_expr.args[0]
                        ics_dict[func_expr.subs(independent_var, x0)] = ic_val
                    # Handle derivatives: Derivative(y(x), x, n) at x0
                    elif isinstance(ic_expr, sp.Derivative):
                        # e.g., Derivative(y(x), x, n)
                        y_call = ic_expr.expr
                        x0 = y_call.args[0]
                        order = sum([o[1] for o in ic_expr.variable_count])
                        ics_dict[
                            func_expr.diff(independent_var, order).subs(
                                independent_var, x0
                            )
                        ] = ic_val
                    else:
                        # fallback: try to substitute
                        ics_dict[ic_expr] = ic_val

            # Solve the ODE
            if ics_dict:
                solution = self._safe_computation(
                    sp.dsolve, diff_eq, func_expr, ics=ics_dict
                )
            else:
                solution = self._safe_computation(sp.dsolve, diff_eq, func_expr)

            # Verify the solution
            verification_result = self._verify_diff_solution(
                diff_eq, solution, independent_var, dependent_func
            )

            # Get verification status from result
            verified = verification_result.get("verified", False)

            # Format the response
            return {
                "success": True,
                "differential_equation": str(diff_eq) + " = 0",
                "dependent_function": str(dependent_func),
                "independent_variable": str(independent_var),
                "verified": verified,
                "verification_details": verification_result.get(
                    "symbolic_verifications", None
                ),
                "result": {
                    "text_output": str(solution),
                    "latex_output": sp.latex(solution),
                },
            }
        except TimeoutError as e:
            return {"success": False, "error": str(e), "error_type": "timeout"}
        except ValueError as e:
            logger.warning(f"Invalid input: {str(e)}")
            return {"success": False, "error": str(e), "error_type": "input_error"}
        except Exception as e:
            error_id = uuid.uuid4()
            logger.error(
                f"Error ID: {error_id}, Error solving differential equation: {str(e)}\n{traceback.format_exc()}"
            )
            return {
                "success": False,
                "error": f"Error solving differential equation: {str(e)}",
                "error_type": "computation_error",
                "error_id": str(error_id),
            }


# Create math router
math_router = APIRouter(prefix="/math", tags=["math"])


# Simplify endpoint
@math_router.post(
    "/simplify",
    response_model_exclude_none=True,
    summary="Simplify an expression",
    operation_id="math_simplify",
    responses={
        200: {
            "description": "Successful Response",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "result": {"text_output": "9", "latex_output": "9"},
                    }
                }
            },
        },
        400: {
            "description": "Bad Request",
            "content": {
                "application/json": {
                    "example": {"success": False, "error": "Expression cannot be empty"}
                }
            },
        },
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "example": {"success": False, "error": "Internal server error: ..."}
                }
            },
        },
    },
)
async def simplify(
    data: SubstitutionInput = Body(
        ..., example={"expression": "x**2 + 2*x + 1", "variables": {"x": 2}}
    )
):
    """
    Simplify a mathematical expression.

    This endpoint uses SymPy's simplify function to reduce an expression to a simpler form.

    Args:
        data (SubstitutionInput): The expression to simplify and optional variable values.

    Returns:
        Dict[str, Any]: The simplified expression and related metadata.

    Raises:
        HTTPException: If the expression cannot be parsed or simplified.
    """
    math = Math()
    result = math.simple_math_computation(data.expression, "simplify", data.variables)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result


# Evaluate endpoint
@math_router.post(
    "/evaluate",
    response_model_exclude_none=True,
    summary="Evaluate an expression with optional substitutions",
    operation_id="math_evaluate",
    responses={
        200: {
            "description": "Successful Response",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "result": {"text_output": "3.0", "latex_output": "3.0"},
                    }
                }
            },
        },
        400: {
            "description": "Bad Request",
            "content": {
                "application/json": {
                    "example": {"success": False, "error": "Expression cannot be empty"}
                }
            },
        },
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "example": {"success": False, "error": "Internal server error: ..."}
                }
            },
        },
    },
)
async def evaluate(
    data: SubstitutionInput = Body(
        ..., example={"expression": "x**2 + 1", "variables": {"x": 1.0}}
    )
):
    """
    Evaluate a mathematical expression.

    This endpoint uses SymPy's evalf function to compute the numerical value of an expression.

    Args:
        data (SubstitutionInput): The expression to evaluate and optional variable values.

    Returns:
        Dict[str, Any]: The evaluated expression and related metadata.

    Raises:
        HTTPException: If the expression cannot be parsed or evaluated.
    """
    math = Math()
    result = math.simple_math_computation(data.expression, "evaluate", data.variables)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result


# Solve endpoint
@math_router.post(
    "/solve",
    response_model_exclude_none=True,
    summary="Solve an equation",
    operation_id="math_solve",
    responses={
        200: {
            "description": "Successful Response",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "solved_for": "x",
                        "verified": True,
                        "verification_details": ["0"],
                        "result": {
                            "text_output": "2.00000000000000",
                            "latex_output": "2.0",
                        },
                    }
                }
            },
        },
        400: {
            "description": "Bad Request",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "error": "Solve operation requires an equation with '=' symbol",
                    }
                }
            },
        },
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "example": {"success": False, "error": "Internal server error: ..."}
                }
            },
        },
    },
)
async def solve(
    data: EquationInput = Body(
        ..., example={"expression": "2*x + 3 = 7", "solve_for": "x"}
    )
):
    """
    Solve an algebraic equation.

    This endpoint uses SymPy's solve function to find the solutions to an equation.

    Args:
        data (EquationInput): The equation to solve and optional variable to solve for.

    Returns:
        Dict[str, Any]: The solutions and related metadata.

    Raises:
        HTTPException: If the equation cannot be parsed or solved.
    """
    math = Math()
    result = math.solve_equation(data.expression, data.solve_for)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result


# Derivate endpoint
@math_router.post(
    "/derivate",
    response_model_exclude_none=True,
    summary="Differentiate an expression",
    operation_id="math_derivate",
    responses={
        200: {
            "description": "Successful Response",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "with_respect_to": "x",
                        "evaluated_at": None,
                        "result": {
                            "text_output": "4.00000000000000",
                            "latex_output": "4",
                        },
                    }
                }
            },
        },
        400: {
            "description": "Bad Request",
            "content": {
                "application/json": {
                    "example": {"success": False, "error": "Expression cannot be empty"}
                }
            },
        },
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "example": {"success": False, "error": "Internal server error: ..."}
                }
            },
        },
    },
)
async def derivate(
    data: DerivativeInput = Body(
        ..., example={"expression": "x**2", "eval_point": 2, "variable": "x"}
    )
):
    """
    Compute the derivative of a mathematical expression.

    This endpoint uses SymPy's diff function to compute the derivative of an expression.

    Args:
        data (DerivativeInput): The expression to differentiate and optional evaluation point.

    Returns:
        Dict[str, Any]: The derivative and related metadata.

    Raises:
        HTTPException: If the expression cannot be parsed or differentiated.
    """
    math = Math()
    result = math.compute_derivative(data.expression, data.eval_point, data.variable)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result


# Integrate endpoint
@math_router.post(
    "/integrate",
    response_model_exclude_none=True,
    summary="Integrate an expression",
    operation_id="math_integrate",
    responses={
        200: {
            "description": "Successful Response",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "with_respect_to": "x",
                        "definite": False,
                        "result": {
                            "text_output": "x**3/3",
                            "latex_output": "\\frac{x^{3}}{3}",
                        },
                    }
                }
            },
        },
        400: {
            "description": "Bad Request",
            "content": {
                "application/json": {
                    "example": {"success": False, "error": "Expression cannot be empty"}
                }
            },
        },
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "example": {"success": False, "error": "Internal server error: ..."}
                }
            },
        },
    },
)
async def integrate(
    data: IntegrationInput = Body(
        ...,
        example={
            "expression": "x**2",
            "bounds": {"lower": 0, "upper": 2},
            "variable": "x",
        },
    )
):
    """
    Compute the integral of a mathematical expression.

    This endpoint uses SymPy's integrate function to compute the integral of an expression.

    Args:
        data (IntegrationInput): The expression to integrate and optional bounds.

    Returns:
        Dict[str, Any]: The integral and related metadata.

    Raises:
        HTTPException: If the expression cannot be parsed or integrated.
    """
    math = Math()
    result = math.compute_integral(data.expression, data.bounds, data.variable)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result


# Factor endpoint
@math_router.post(
    "/factor",
    response_model_exclude_none=True,
    summary="Factor an expression",
    operation_id="math_factor",
    responses={
        200: {
            "description": "Successful Response",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "result": {
                            "text_output": "(x + 1)*(x - 1)",
                            "latex_output": "\\left(x + 1\\right) \\left(x - 1\\right)",
                        },
                    }
                }
            },
        },
        400: {
            "description": "Bad Request",
            "content": {
                "application/json": {
                    "example": {"success": False, "error": "Expression cannot be empty"}
                }
            },
        },
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "example": {"success": False, "error": "Internal server error: ..."}
                }
            },
        },
    },
)
async def factor(
    data: SubstitutionInput = Body(..., example={"expression": "x**2 - 1"})
):
    """
    Factor a mathematical expression.

    This endpoint uses SymPy's factor function to factorize an expression.

    Args:
        data (SubstitutionInput): The expression to factor.

    Returns:
        Dict[str, Any]: The factored expression and related metadata.

    Raises:
        HTTPException: If the expression cannot be parsed or factored.
    """
    math = Math()
    result = math.simple_math_computation(data.expression, "factor", data.variables)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result


# Expand endpoint
@math_router.post(
    "/expand",
    response_model_exclude_none=True,
    summary="Expand an expression",
    operation_id="math_expand",
    responses={
        200: {
            "description": "Successful Response",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "result": {
                            "text_output": "x**2 + 2*x + 1",
                            "latex_output": "x^{2} + 2 x + 1",
                        },
                    }
                }
            },
        },
        400: {
            "description": "Bad Request",
            "content": {
                "application/json": {
                    "example": {"success": False, "error": "Expression cannot be empty"}
                }
            },
        },
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "example": {"success": False, "error": "Internal server error: ..."}
                }
            },
        },
    },
)
async def expand(
    data: SubstitutionInput = Body(..., example={"expression": "(x + 1)**2"})
):
    """
    Expand a mathematical expression.

    This endpoint uses SymPy's expand function to expand an expression.

    Args:
        data (SubstitutionInput): The expression to expand.

    Returns:
        Dict[str, Any]: The expanded expression and related metadata.

    Raises:
        HTTPException: If the expression cannot be parsed or expanded.
    """
    math = Math()
    result = math.simple_math_computation(data.expression, "expand", data.variables)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result


# Dsolve endpoint
@math_router.post(
    "/dsolve",
    response_model_exclude_none=True,
    summary="Solve a differential equation",
    operation_id="math_dsolve",
    responses={
        200: {
            "description": "Successful Response",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "differential_equation": "y(x) + Derivative(y(x), x) + Derivative(y(x), (x, 2)) = 0",
                        "dependent_function": "y",
                        "independent_variable": "x",
                        "verified": True,
                        "verification_details": ["0"],
                        "result": {
                            "text_output": "Eq(y(x), (C1*sin(sqrt(3)*x/2) + C2*cos(sqrt(3)*x/2))*exp(-x/2))",
                            "latex_output": "y{\\left(x \\right)} = \\left(C_{1} \\sin{\\left(\\frac{\\sqrt{3} x}{2} \\right)} + C_{2} \\cos{\\left(\\frac{\\sqrt{3} x}{2} \\right)}\\right) e^{- \\frac{x}{2}}",
                        },
                    }
                }
            },
        },
        400: {
            "description": "Bad Request",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "error": "Could not identify independent variable or dependent function in the differential equation",
                    }
                }
            },
        },
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "example": {"success": False, "error": "Internal server error: ..."}
                }
            },
        },
    },
)
async def dsolve(
    data: DifferentialEquationInput = Body(
        ...,
        example={
            "expression": "y''(x) + y'(x) + y(x) = 0",
            "initial_conditions": {"y(0)": 1, "y'(0)": 0},
        },
    )
):
    """
    Solve a differential equation.

    This endpoint uses SymPy's dsolve function to find the solutions to a differential equation.

    Args:
        data (DifferentialEquationInput): The differential equation to solve and optional initial conditions.

    Returns:
        Dict[str, Any]: The solutions and related metadata.

    Raises:
        HTTPException: If the differential equation cannot be parsed or solved.
    """
    math = Math()
    result = math.solve_differential_equation(data.expression, data.initial_conditions)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result


# Include math router in the app
app.include_router(math_router)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": f"Internal server error: {str(exc)}"},
    )
