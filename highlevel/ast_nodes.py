# MyLand/highlevel/ast_nodes.py
from typing import List, Optional, Any

class ASTNode:
    """Base class for all AST nodes."""
    def __init__(self, line: int = 0, col: int = 0):
        self.line = line
        self.col = col


class ProgramNode(ASTNode):
    """Root node representing the whole program."""
    def __init__(self, statements: List[ASTNode], line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.statements = statements


# ---------------- Statements ----------------

class SetStmtNode(ASTNode):
    """Node for variable assignment: set x = expr
    Also handles array-element assignment: set arr[i] = expr, in which case
    index_expr is not None and name refers to the array variable."""
    def __init__(self, name: str, expr: ASTNode, index_expr: Optional[ASTNode] = None, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.name = name
        self.expr = expr
        self.index_expr = index_expr


class SayStmtNode(ASTNode):
    """Node for output: say expr"""
    def __init__(self, expr: ASTNode, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.expr = expr


class AskStmtNode(ASTNode):
    """Node for user input: ask var or input_num var"""
    def __init__(self, var_name: str, as_number: bool = False, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.var_name = var_name
        self.as_number = as_number


class RepeatStmtNode(ASTNode):
    """Node for fixed loops: repeat N ... end_repeat"""
    def __init__(self, count_expr: ASTNode, body: List[ASTNode], line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.count_expr = count_expr
        self.body = body


class LoopIfStmtNode(ASTNode):
    """Node for conditional loops: loop_if COND ... end_loop"""
    def __init__(self, condition: ASTNode, body: List[ASTNode], line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.condition = condition
        self.body = body


class BreakLoopStmtNode(ASTNode):
    """Node for exiting the nearest enclosing loop_if/repeat early: break_loop"""
    def __init__(self, line: int = 0, col: int = 0):
        super().__init__(line, col)


class CheckStmtNode(ASTNode):
    """Node for conditionals: check COND ... otherwise_check COND ... otherwise ... end_check
    elif_branches is a list of (condition, body) tuples for otherwise_check clauses."""
    def __init__(self, condition: ASTNode, then_body: List[ASTNode], elif_branches: Optional[List] = None, else_body: Optional[List[ASTNode]] = None, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.condition = condition
        self.then_body = then_body
        self.elif_branches = elif_branches or []   # [(CondNode/LogicalNode, [stmts]), ...]
        self.else_body = else_body or []


class ForStmtNode(ASTNode):
    """Node for range-based iteration: for i in START..END ... end_for
    Iterates i from START up to (but not including) END, i.e. [start, end)."""
    def __init__(self, var_name: str, start_expr: ASTNode, end_expr: ASTNode, body: List[ASTNode], line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.var_name = var_name
        self.start_expr = start_expr
        self.end_expr = end_expr
        self.body = body


class BlockDefNode(ASTNode):
    """Node for block subroutine definition: block name(p1, p2) ... end_block
    params is empty list for the old no-arg form (fully backward compatible)."""
    def __init__(self, name: str, body: List[ASTNode], params: Optional[List[str]] = None, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.name = name
        self.body = body
        self.params = params or []


class RunStmtNode(ASTNode):
    """Node to call a subroutine as a statement (return value discarded):
    run name  OR  run name(arg1, arg2)"""
    def __init__(self, name: str, args: Optional[List[ASTNode]] = None, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.name = name
        self.args = args or []


class CallExprNode(ASTNode):
    """Node for calling a block as an EXPRESSION (its return value is used):
    set x = run total(a, b)"""
    def __init__(self, name: str, args: Optional[List[ASTNode]] = None, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.name = name
        self.args = args or []


class ReturnStmtNode(ASTNode):
    """Node for returning a value from a block: return expr"""
    def __init__(self, expr: Optional[ASTNode] = None, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.expr = expr


class GotoStmtNode(ASTNode):
    """Node for jumps: goto ~label"""
    def __init__(self, label: str, condition: Optional[ASTNode] = None, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.label = label
        self.condition = condition


class LabelDefNode(ASTNode):
    """Node for jump targets: ~label"""
    def __init__(self, name: str, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.name = name


class FileOpNode(ASTNode):
    """Node for file operations: file:open, file:write, file:read, file:close"""
    def __init__(self, op_type: str, args: List[ASTNode], line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.op_type = op_type
        self.args = args


class StopStmtNode(ASTNode):
    """Node for exit: stop or stop:error"""
    def __init__(self, is_error: bool = False, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.is_error = is_error


# ---------------- Expressions ----------------

class NumberLitNode(ASTNode):
    """Node for integer numbers"""
    def __init__(self, value: int, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.value = value


class StringLitNode(ASTNode):
    """Node for string literals"""
    def __init__(self, value: str, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.value = value


class VarRefNode(ASTNode):
    """Node for referencing variables"""
    def __init__(self, name: str, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.name = name


class BinOpNode(ASTNode):
    """Node for math operations: +, -, *, /, %
    '+' also doubles as STRING CONCATENATION when either operand is a string
    (resolved at codegen time using compile-time-tracked string types)."""
    def __init__(self, left: ASTNode, op: str, right: ASTNode, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.left = left
        self.op = op
        self.right = right


class ArrayLitNode(ASTNode):
    """Node for array constructor: array N  (creates an N-element, zeroed,
    heap-allocated array of 8-byte slots)."""
    def __init__(self, size_expr: ASTNode, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.size_expr = size_expr


class IndexExprNode(ASTNode):
    """Node for reading an array element: arr[i]"""
    def __init__(self, array_name: str, index_expr: ASTNode, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.array_name = array_name
        self.index_expr = index_expr


class CondNode(ASTNode):
    """Node for comparisons: ==, !=, >, <, >=, <="""
    def __init__(self, left: ASTNode, op: str, right: ASTNode, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.left = left
        self.op = op
        self.right = right


class LogicalNode(ASTNode):
    """Node for compound boolean conditions: cond AND cond, cond OR cond.
    Operands are themselves CondNode/LogicalNode/NotNode (a condition tree,
    not a value expression) -- evaluated via short-circuit jumps in codegen."""
    def __init__(self, left: ASTNode, op: str, right: ASTNode, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.left = left
        self.op = op   # "and" or "or"
        self.right = right


class NotNode(ASTNode):
    """Node for negating a condition: not COND"""
    def __init__(self, cond: ASTNode, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.cond = cond


class BoolLitNode(ASTNode):
    """Node for boolean literals: true / false (sugar for 1 / 0 as a VALUE
    expression, e.g. 'set flag = true' -- distinct from CondNode, which is
    used only inside check/loop_if/for conditions)."""
    def __init__(self, value: bool, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.value = value


class ToStringNode(ASTNode):
    """Node for to_string(expr): converts a number to its decimal string
    representation. Result is always treated as a string-typed value."""
    def __init__(self, expr: ASTNode, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.expr = expr


class ToNumberNode(ASTNode):
    """Node for to_number(expr): parses a string's leading digits (with
    optional '-') into an integer. Non-numeric/empty input yields 0."""
    def __init__(self, expr: ASTNode, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.expr = expr


class StrLengthNode(ASTNode):
    """Node for str:length(expr): returns the byte length of a string."""
    def __init__(self, expr: ASTNode, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.expr = expr


class StrCharAtNode(ASTNode):
    """Node for str:char_at(expr, idx): returns the character at idx as a
    single-character string. Out-of-range idx returns an empty string."""
    def __init__(self, expr: ASTNode, index_expr: ASTNode, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.expr = expr
        self.index_expr = index_expr


class StrSliceNode(ASTNode):
    """Node for str:slice(expr, start, end): returns the substring
    [start, end) as a new heap string. Clamps out-of-range indices."""
    def __init__(self, expr: ASTNode, start_expr: ASTNode, end_expr: ASTNode, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.expr = expr
        self.start_expr = start_expr
        self.end_expr = end_expr


class RandomNode(ASTNode):
    """Node for random(min, max): returns a pseudo-random integer in the
    inclusive range [min, max]."""
    def __init__(self, min_expr: ASTNode, max_expr: ASTNode, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.min_expr = min_expr
        self.max_expr = max_expr


class ArrayLengthNode(ASTNode):
    """Node for array:length(expr): returns the element count of an array
    created via 'array N' (the N it was allocated with)."""
    def __init__(self, expr: ASTNode, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.expr = expr


# ---------------- Dictionaries ----------------

class DictLitNode(ASTNode):
    """Node for dict constructor: dict (creates an empty heap-allocated
    key-value store). Grows dynamically as dict:set adds new keys."""
    def __init__(self, line: int = 0, col: int = 0):
        super().__init__(line, col)


class DictGetNode(ASTNode):
    """Node for dict:get(d, key): returns the value stored under key, or 0
    if the key is not present."""
    def __init__(self, dict_expr: ASTNode, key_expr: ASTNode, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.dict_expr = dict_expr
        self.key_expr = key_expr


class DictSetStmtNode(ASTNode):
    """Node for the statement form dict:set d, key = expr: inserts or
    updates the value stored under key."""
    def __init__(self, dict_expr: ASTNode, key_expr: ASTNode, value_expr: ASTNode, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.dict_expr = dict_expr
        self.key_expr = key_expr
        self.value_expr = value_expr


class DictHasNode(ASTNode):
    """Node for dict:has(d, key): returns 1 if key is present, else 0."""
    def __init__(self, dict_expr: ASTNode, key_expr: ASTNode, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.dict_expr = dict_expr
        self.key_expr = key_expr


class DictDeleteStmtNode(ASTNode):
    """Node for the statement form dict:delete d, key: removes key (and its
    value) if present; a no-op if the key isn't there."""
    def __init__(self, dict_expr: ASTNode, key_expr: ASTNode, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.dict_expr = dict_expr
        self.key_expr = key_expr


class DictLengthNode(ASTNode):
    """Node for dict:length(d): returns the number of key-value pairs
    currently stored."""
    def __init__(self, expr: ASTNode, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.expr = expr


class DictKeyAtNode(ASTNode):
    """Node for dict:key_at(d, i): returns the i-th key (insertion order)
    as a string, for iterating over a dict's contents alongside
    dict:length. Out-of-range i returns an empty string."""
    def __init__(self, dict_expr: ASTNode, index_expr: ASTNode, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.dict_expr = dict_expr
        self.index_expr = index_expr


# ---------------- Error handling ----------------

class TryStmtNode(ASTNode):
    """Node for try ... catch err ... end_try. Runtime errors raised inside
    'body' (currently: division/modulo by zero, and explicit 'raise expr')
    transfer control to 'catch_body' with the error message/code bound to
    err_var as a string. If no error occurs, catch_body is skipped."""
    def __init__(self, body: List[ASTNode], err_var: str, catch_body: List[ASTNode], line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.body = body
        self.err_var = err_var
        self.catch_body = catch_body


class RaiseStmtNode(ASTNode):
    """Node for raise expr: raises a user-defined error with the given
    (string) message, unwinding to the nearest enclosing try/catch. Raising
    with no enclosing try is a fatal error (program exits with code 1,
    message printed to stdout) -- same behavior as an uncaught built-in
    error like division by zero."""
    def __init__(self, expr: ASTNode, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.expr = expr
