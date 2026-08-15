# MyLand/highlevel/semantic.py
from ast_nodes import (
    ProgramNode, SetStmtNode, SayStmtNode, AskStmtNode, RepeatStmtNode,
    LoopIfStmtNode, CheckStmtNode, BlockDefNode, RunStmtNode, GotoStmtNode,
    LabelDefNode, FileOpNode, StopStmtNode, BinOpNode, CondNode, VarRefNode,
    NumberLitNode, StringLitNode, ASTNode, BreakLoopStmtNode,
    CallExprNode, ReturnStmtNode, ArrayLitNode, IndexExprNode,
    ForStmtNode, LogicalNode, NotNode, BoolLitNode, ToStringNode,
    ToNumberNode, StrLengthNode, StrCharAtNode, StrSliceNode,
    RandomNode, ArrayLengthNode, DictLitNode, DictGetNode,
    DictSetStmtNode, DictHasNode, DictDeleteStmtNode, DictLengthNode,
    DictKeyAtNode, TryStmtNode, RaiseStmtNode
)

class SemanticAnalyzer:
    """
    Analyzes the AST before code generation to catch scope errors,
    undefined variables, missing subroutines, and invalid jumps.
    """

    def __init__(self, filename: str = "<source>"):
        self.filename = filename
        self.declared_vars = set()
        self.defined_blocks = set()          # name -> handled separately below (block_params, block_returns)
        self.block_params = {}               # name -> [param names]
        self.block_returns = {}              # name -> True if it has at least one 'return expr'
        self.defined_labels = set()
        self.errors = []
        self.loop_depth = 0
        self.current_block_name = None       # for validating 'return' is inside a block
        # 'set x = "..."', 'ask x' (non-numeric), 'file:read h x', string
        # concatenation results, aur 'run f(...)' jo string return karta hai
        # -- in sab se x ek string-typed variable ban jaata hai. Arithmetic
        # operators (-,*,/,%) sirf numbers ke liye valid hain; '+' strings
        # pe bhi valid hai (concatenation).
        self.string_vars = set()
        # 'set arr = array N' se arr ek array-typed variable ban jaata hai.
        # Indexing (arr[i]) sirf array-typed variables pe allowed hai.
        self.array_vars = set()
        # 'set d = dict' se d ek dict-typed variable ban jaata hai.
        self.dict_vars = set()
        # name -> True agar us block ka koi 'return expr' string return
        # karta hai. 'set x = run f(...)' mein x ka type isi se decide hota
        # hai (taaki agar f() string return karta hai toh x bhi string-typed
        # maana jaaye, sirf arithmetic type check ke liye).
        self.block_return_is_string = set()

    def _sub_bodies(self, stmt):
        """Yields every nested statement list a node can carry (body,
        then_body, else_body, and elif branch bodies) -- single place that
        knows about CheckStmtNode's elif_branches shape so every recursive
        walker below doesn't need to special-case it separately."""
        for attr in ("body", "then_body", "else_body"):
            if hasattr(stmt, attr):
                yield getattr(stmt, attr)
        if isinstance(stmt, CheckStmtNode):
            for _cond, ebody in stmt.elif_branches:
                yield ebody
        if isinstance(stmt, TryStmtNode):
            yield stmt.catch_body

    def log_error(self, msg: str, line: int, col: int):
        self.errors.append(f"Semantic Error [{self.filename}:{line}:{col}]: {msg}")

    def analyze(self, program: ProgramNode) -> bool:
        """Pass 1: collect block/label definitions + params.
        Pass 2: infer which variables are strings/arrays (needs block
        params/returns from pass 1, since a param can be a string if the
        block is ever called with a string argument).
        Pass 3: full validation."""
        for stmt in program.statements:
            if isinstance(stmt, BlockDefNode):
                if stmt.name in self.defined_blocks:
                    self.log_error(f"Redefinition of block '{stmt.name}'", stmt.line, stmt.col)
                self.defined_blocks.add(stmt.name)
                self.block_params[stmt.name] = stmt.params
                self.block_returns[stmt.name] = self._body_has_return(stmt.body)
            elif isinstance(stmt, LabelDefNode):
                if stmt.name in self.defined_labels:
                    self.log_error(f"Redefinition of label '~{stmt.name}'", stmt.line, stmt.col)
                self.defined_labels.add(stmt.name)

        self._collect_array_vars(program.statements)

        # Fixed point: string_vars, param types, aur return types ek doosre
        # pe depend karte hain (ek var CallExprNode se string ban sakta hai,
        # jiska return type khud kisi param par depend karta ho). Jab tak
        # naya kuch add na ho, dobara scan karte raho.
        while True:
            before = len(self.string_vars) + len(self.block_return_is_string)
            self._collect_string_vars(program.statements)
            self._collect_param_string_types(program.statements)
            self._collect_block_return_types(program.statements)
            if len(self.string_vars) + len(self.block_return_is_string) == before:
                break

        for stmt in program.statements:
            self.visit_node(stmt)

        if self.errors:
            for err in self.errors:
                print(err)
            return False
        return True

    def _body_has_return(self, stmts) -> bool:
        for stmt in stmts:
            if isinstance(stmt, ReturnStmtNode) and stmt.expr is not None:
                return True
            for attr in ("body", "then_body", "else_body"):
                if hasattr(stmt, attr) and self._body_has_return(getattr(stmt, attr)):
                    return True
            if isinstance(stmt, CheckStmtNode):
                for _cond, ebody in stmt.elif_branches:
                    if self._body_has_return(ebody):
                        return True
        return False

    def _collect_string_vars(self, stmts):
        # Ek single top-to-bottom pass forward-chained string concatenation
        # ko poora nahi pakad payega (jaise: set a = "x"; set b = a + "y";
        # set c = b + "z" -- 'c' string hai yeh sirf 'b' ke resolve hone ke
        # baad pata chalta hai). Isliye fixed-point tak repeat karte hain:
        # jab tak koi naya string_var add na ho.
        while True:
            before = len(self.string_vars)
            self._collect_string_vars_pass(stmts)
            if len(self.string_vars) == before:
                break

    def _collect_string_vars_pass(self, stmts):
        for stmt in stmts:
            if isinstance(stmt, SetStmtNode) and stmt.index_expr is None:
                if self._is_string_expr(stmt.expr):
                    self.string_vars.add(stmt.name)
            elif isinstance(stmt, AskStmtNode) and not stmt.as_number:
                self.string_vars.add(stmt.var_name)
            elif isinstance(stmt, FileOpNode) and stmt.op_type == "read" and len(stmt.args) > 1:
                dest = stmt.args[1]
                if isinstance(dest, VarRefNode):
                    self.string_vars.add(dest.name)
            for attr in ("body", "then_body", "else_body"):
                if hasattr(stmt, attr):
                    self._collect_string_vars_pass(getattr(stmt, attr))
            if isinstance(stmt, CheckStmtNode):
                for _cond, ebody in stmt.elif_branches:
                    self._collect_string_vars_pass(ebody)
            if isinstance(stmt, TryStmtNode):
                self._collect_string_vars_pass(stmt.catch_body)
                self.string_vars.add(stmt.err_var)   # caught error is always bound as a string message

    def _collect_block_return_types(self, stmts):
        """Ek block ka return type (string vs number) uske 'return expr'
        statements ke expr type se determine hota hai."""
        def find_returns(ss, out):
            for s in ss:
                if isinstance(s, ReturnStmtNode) and s.expr is not None:
                    out.append(s.expr)
                for attr in ("body", "then_body", "else_body"):
                    if hasattr(s, attr):
                        find_returns(getattr(s, attr), out)
                if isinstance(s, CheckStmtNode):
                    for _cond, ebody in s.elif_branches:
                        find_returns(ebody, out)

        for stmt in stmts:
            if isinstance(stmt, BlockDefNode):
                returns = []
                find_returns(stmt.body, returns)
                if any(self._is_string_expr(r) for r in returns):
                    self.block_return_is_string.add(stmt.name)

    def _is_string_expr(self, node: ASTNode, context_stmts=None) -> bool:
        if isinstance(node, StringLitNode):
            return True
        if isinstance(node, VarRefNode):
            return node.name in self.string_vars
        if isinstance(node, BinOpNode) and node.op == "+":
            return self._is_string_expr(node.left) or self._is_string_expr(node.right)
        if isinstance(node, CallExprNode):
            return node.name in self.block_return_is_string
        if isinstance(node, (ToStringNode, StrSliceNode, StrCharAtNode, DictKeyAtNode)):
            return True
        return False

    def _collect_array_vars(self, stmts):
        for stmt in stmts:
            if isinstance(stmt, SetStmtNode) and stmt.index_expr is None and isinstance(stmt.expr, ArrayLitNode):
                self.array_vars.add(stmt.name)
            if isinstance(stmt, SetStmtNode) and stmt.index_expr is None and isinstance(stmt.expr, DictLitNode):
                self.dict_vars.add(stmt.name)
            for attr in ("body", "then_body", "else_body"):
                if hasattr(stmt, attr):
                    self._collect_array_vars(getattr(stmt, attr))
            if isinstance(stmt, CheckStmtNode):
                for _cond, ebody in stmt.elif_branches:
                    self._collect_array_vars(ebody)
            if isinstance(stmt, TryStmtNode):
                self._collect_array_vars(stmt.catch_body)

    def _collect_param_string_types(self, stmts):
        """Agar koi block kabhi bhi ek string argument ke saath call hota
        hai, us position ka param bhi string-typed maana jaata hai. Isse
        function ke andar 'say param' ya 'param + "..."' sahi codegen paata
        hai. Best-effort static inference hai (ek hi param do alag types se
        call ho sakta hai in theory, lekin us case mein bhi hum 'string'
        priority dete hain taaki galat number-as-pointer read na ho)."""
        block_body_map = {}
        def find_blocks(ss):
            for s in ss:
                if isinstance(s, BlockDefNode):
                    block_body_map[s.name] = s
                for attr in ("body", "then_body", "else_body"):
                    if hasattr(s, attr):
                        find_blocks(getattr(s, attr))
                if isinstance(s, CheckStmtNode):
                    for _cond, ebody in s.elif_branches:
                        find_blocks(ebody)
        find_blocks(stmts)

        def scan_expr(e):
            if isinstance(e, CallExprNode):
                params = self.block_params.get(e.name, [])
                for i, arg in enumerate(e.args):
                    if i < len(params) and self._is_string_expr(arg):
                        self.string_vars.add(params[i])
                    scan_expr(arg)
            elif isinstance(e, BinOpNode):
                scan_expr(e.left)
                scan_expr(e.right)
            elif isinstance(e, LogicalNode):
                scan_expr(e.left)
                scan_expr(e.right)
            elif isinstance(e, NotNode):
                scan_expr(e.cond)
            elif isinstance(e, CondNode):
                scan_expr(e.left)
                scan_expr(e.right)
            elif isinstance(e, (ToStringNode, ToNumberNode, StrLengthNode)):
                scan_expr(e.expr)
            elif isinstance(e, StrCharAtNode):
                scan_expr(e.expr)
                scan_expr(e.index_expr)
            elif isinstance(e, StrSliceNode):
                scan_expr(e.expr)
                scan_expr(e.start_expr)
                scan_expr(e.end_expr)

        def scan_calls(ss):
            for s in ss:
                if isinstance(s, RunStmtNode):
                    for a in s.args:
                        scan_expr(a)
                    scan_expr(CallExprNode(s.name, s.args, s.line, s.col))
                elif isinstance(s, SetStmtNode):
                    scan_expr(s.expr)
                    if s.index_expr is not None:
                        scan_expr(s.index_expr)
                elif isinstance(s, SayStmtNode):
                    scan_expr(s.expr)
                elif isinstance(s, CheckStmtNode):
                    scan_expr(s.condition)
                    for elif_cond, _ in s.elif_branches:
                        scan_expr(elif_cond)
                elif isinstance(s, LoopIfStmtNode):
                    scan_expr(s.condition)
                elif isinstance(s, ForStmtNode):
                    scan_expr(s.start_expr)
                    scan_expr(s.end_expr)
                elif isinstance(s, RepeatStmtNode):
                    scan_expr(s.count_expr)
                elif isinstance(s, ReturnStmtNode) and s.expr is not None:
                    scan_expr(s.expr)

                for attr in ("body", "then_body", "else_body"):
                    if hasattr(s, attr):
                        scan_calls(getattr(s, attr))
                if isinstance(s, CheckStmtNode):
                    for _cond, ebody in s.elif_branches:
                        scan_calls(ebody)
        # Fixed-point: ek block ke andar dusre block ko call karna bhi ho
        # sakta hai jiske apne params bhi string ban jaayein iteratively.
        while True:
            before = len(self.string_vars)
            scan_calls(stmts)
            for blk in block_body_map.values():
                scan_calls(blk.body)
            if len(self.string_vars) == before:
                break

    def visit_node(self, node: ASTNode):
        if isinstance(node, SetStmtNode):
            if node.index_expr is not None:
                # set arr[i] = expr -- arr already declared aur array-typed honi chahiye
                if node.name not in self.declared_vars:
                    self.log_error(f"Variable '{node.name}' used before declaration", node.line, node.col)
                elif node.name not in self.array_vars:
                    self.log_error(f"'{node.name}' is not an array (use 'set {node.name} = array N' to create one first)", node.line, node.col)
                self.visit_expr(node.index_expr)
                self.visit_expr(node.expr)
            else:
                self.visit_expr(node.expr)
                self.declared_vars.add(node.name)

        elif isinstance(node, SayStmtNode):
            self.visit_expr(node.expr)

        elif isinstance(node, AskStmtNode):
            self.declared_vars.add(node.var_name)

        elif isinstance(node, RepeatStmtNode):
            self.visit_expr(node.count_expr)
            self.loop_depth += 1
            for stmt in node.body:
                self.visit_node(stmt)
            self.loop_depth -= 1

        elif isinstance(node, LoopIfStmtNode):
            self.visit_cond(node.condition)
            self.loop_depth += 1
            for stmt in node.body:
                self.visit_node(stmt)
            self.loop_depth -= 1

        elif isinstance(node, ForStmtNode):
            self.visit_expr(node.start_expr)
            self.visit_expr(node.end_expr)
            self.declared_vars.add(node.var_name)
            self.loop_depth += 1
            for stmt in node.body:
                self.visit_node(stmt)
            self.loop_depth -= 1

        elif isinstance(node, BreakLoopStmtNode):
            if self.loop_depth <= 0:
                self.log_error("'break_loop' used outside of any loop (loop_if/repeat)", node.line, node.col)

        elif isinstance(node, CheckStmtNode):
            self.visit_cond(node.condition)
            for stmt in node.then_body:
                self.visit_node(stmt)
            for elif_cond, elif_body in node.elif_branches:
                self.visit_cond(elif_cond)
                for stmt in elif_body:
                    self.visit_node(stmt)
            for stmt in node.else_body:
                self.visit_node(stmt)

        elif isinstance(node, BlockDefNode):
            # Function body ek naya local scope hai: current declared_vars
            # ko save/restore karo taaki outer scope ke variables block ke
            # andar accidentally visible na ho jaayein (aur vice versa) --
            # sirf global vars aur is block ke apne params visible hote hain.
            # (Globals continue to be visible for backward-compat with the
            # old no-scoping behaviour of simple scripts.)
            saved_declared = set(self.declared_vars)
            saved_block_name = self.current_block_name
            self.current_block_name = node.name
            for p in node.params:
                self.declared_vars.add(p)
            for stmt in node.body:
                self.visit_node(stmt)
            self.current_block_name = saved_block_name
            self.declared_vars = saved_declared

        elif isinstance(node, RunStmtNode):
            self._check_call(node.name, node.args, node.line, node.col)

        elif isinstance(node, ReturnStmtNode):
            if self.current_block_name is None:
                self.log_error("'return' used outside of any block (return is only valid inside 'block ... end_block')", node.line, node.col)
            if node.expr is not None:
                self.visit_expr(node.expr)

        elif isinstance(node, GotoStmtNode):
            if node.condition:
                self.visit_cond(node.condition)
            if node.label not in self.defined_labels:
                self.log_error(f"Undefined jump label '~{node.label}'", node.line, node.col)

        elif isinstance(node, FileOpNode):
            if node.op_type == "open" and len(node.args) > 0:
                # file:open ka pehla argument handle_var hota hai, isey declared mark karo
                handle_arg = node.args[0]
                if isinstance(handle_arg, VarRefNode):
                    self.declared_vars.add(handle_arg.name)
                # baki ke arguments (path, mode) evaluate karo
                for arg in node.args[1:]:
                    self.visit_expr(arg)
            elif node.op_type == "read":
                # file:read handle dest_var -- pehla arg handle hai (already
                # declared honi chahiye), dusra dest_var hai jo YAHIN declare
                # ho raha hai (yeh usko likhta hai, use nahi karta), isliye
                # "used before declaration" nahi lagana chahiye.
                if len(node.args) > 0:
                    self.visit_expr(node.args[0])
                if len(node.args) > 1 and isinstance(node.args[1], VarRefNode):
                    self.declared_vars.add(node.args[1].name)
                elif len(node.args) > 1:
                    self.visit_expr(node.args[1])
            else:
                # write, close ke saare arguments check karo
                for arg in node.args:
                    self.visit_expr(arg)

        elif isinstance(node, (LabelDefNode, StopStmtNode)):
            pass

        elif isinstance(node, DictSetStmtNode):
            self.visit_expr(node.dict_expr)
            self.visit_expr(node.key_expr)
            self.visit_expr(node.value_expr)

        elif isinstance(node, DictDeleteStmtNode):
            self.visit_expr(node.dict_expr)
            self.visit_expr(node.key_expr)

        elif isinstance(node, TryStmtNode):
            for stmt in node.body:
                self.visit_node(stmt)
            self.declared_vars.add(node.err_var)
            for stmt in node.catch_body:
                self.visit_node(stmt)

        elif isinstance(node, RaiseStmtNode):
            self.visit_expr(node.expr)

    def _check_call(self, name: str, args, line: int, col: int):
        if name not in self.defined_blocks:
            self.log_error(f"Undefined block '{name}' (Define it using 'block {name} ... end_block')", line, col)
            return
        expected = len(self.block_params.get(name, []))
        got = len(args)
        if expected != got:
            self.log_error(
                f"Block '{name}' expects {expected} argument(s) but got {got}",
                line, col
            )
        for arg in args:
            self.visit_expr(arg)

    def visit_expr(self, node: ASTNode):
        if isinstance(node, VarRefNode):
            if node.name not in self.declared_vars:
                self.log_error(f"Variable '{node.name}' used before declaration", node.line, node.col)
        elif isinstance(node, BinOpNode):
            self.visit_expr(node.left)
            self.visit_expr(node.right)
            self._check_operand_types(node)
        elif isinstance(node, ArrayLitNode):
            self.visit_expr(node.size_expr)
        elif isinstance(node, IndexExprNode):
            if node.array_name not in self.declared_vars:
                self.log_error(f"Variable '{node.array_name}' used before declaration", node.line, node.col)
            elif node.array_name not in self.array_vars:
                self.log_error(f"'{node.array_name}' is not an array (use 'set {node.array_name} = array N' to create one first)", node.line, node.col)
            self.visit_expr(node.index_expr)
        elif isinstance(node, CallExprNode):
            self._check_call(node.name, node.args, node.line, node.col)
            if node.name in self.defined_blocks and not self.block_returns.get(node.name, False):
                self.log_error(
                    f"Block '{node.name}' is used as an expression (its return value is needed) "
                    f"but it never executes 'return <value>'",
                    node.line, node.col
                )
        elif isinstance(node, BoolLitNode):
            pass
        elif isinstance(node, ToStringNode):
            self.visit_expr(node.expr)
        elif isinstance(node, ToNumberNode):
            self.visit_expr(node.expr)
        elif isinstance(node, StrLengthNode):
            self.visit_expr(node.expr)
        elif isinstance(node, StrCharAtNode):
            self.visit_expr(node.expr)
            self.visit_expr(node.index_expr)
        elif isinstance(node, StrSliceNode):
            self.visit_expr(node.expr)
            self.visit_expr(node.start_expr)
            self.visit_expr(node.end_expr)
        elif isinstance(node, RandomNode):
            self.visit_expr(node.min_expr)
            self.visit_expr(node.max_expr)
        elif isinstance(node, ArrayLengthNode):
            if isinstance(node.expr, VarRefNode) and node.expr.name not in self.array_vars:
                self.log_error(f"'{node.expr.name}' is not an array (array:length only applies to variables created with 'array N')", node.line, node.col)
            self.visit_expr(node.expr)

        elif isinstance(node, DictLitNode):
            pass

        elif isinstance(node, DictGetNode):
            self.visit_expr(node.dict_expr)
            self.visit_expr(node.key_expr)

        elif isinstance(node, DictHasNode):
            self.visit_expr(node.dict_expr)
            self.visit_expr(node.key_expr)

        elif isinstance(node, DictLengthNode):
            self.visit_expr(node.expr)

        elif isinstance(node, DictKeyAtNode):
            self.visit_expr(node.dict_expr)
            self.visit_expr(node.index_expr)

    def _check_operand_types(self, node: BinOpNode):
        """'+' ab string concatenation ke liye bhi valid hai. Baaki
        operators (-, *, /, %) sirf numbers ke liye valid hain -- ek string
        (pointer) ko subtract/multiply/divide karna silently garbage
        result dega, isliye yahin pakadte hain."""
        left_is_str = self._is_string_expr(node.left)
        right_is_str = self._is_string_expr(node.right)
        if node.op != "+" and (left_is_str or right_is_str):
            self.log_error(
                f"Operator '{node.op}' cannot be used with a string operand "
                f"(only '+' supports string concatenation).",
                node.line, node.col
            )
        elif node.op == "+" and left_is_str != right_is_str:
            # Ek side string hai, doosri number -- yeh clearly not what the
            # user meant, though we *could* silently coerce. Erroring here
            # avoids a whole class of "why is my string garbled" bugs.
            self.log_error(
                "Cannot mix a string and a number with '+'. Convert the "
                "number to text first, or use a separate 'say' statement.",
                node.line, node.col
            )

    def visit_cond(self, node: ASTNode):
        if isinstance(node, LogicalNode):
            self.visit_cond(node.left)
            self.visit_cond(node.right)
        elif isinstance(node, NotNode):
            self.visit_cond(node.cond)
        elif isinstance(node, CondNode):
            self.visit_expr(node.left)
            self.visit_expr(node.right)
