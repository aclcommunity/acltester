# MyLand/highlevel/codegen.py
from typing import List, Dict, Tuple, Set, Optional
from ast_nodes import (
    ProgramNode, SetStmtNode, SayStmtNode, AskStmtNode, RepeatStmtNode,
    LoopIfStmtNode, CheckStmtNode, BlockDefNode, RunStmtNode, GotoStmtNode,
    LabelDefNode, FileOpNode, StopStmtNode, NumberLitNode, StringLitNode,
    VarRefNode, BinOpNode, CondNode, ASTNode, BreakLoopStmtNode,
    CallExprNode, ReturnStmtNode, ArrayLitNode, IndexExprNode,
    ForStmtNode, LogicalNode, NotNode, BoolLitNode, ToStringNode,
    ToNumberNode, StrLengthNode, StrCharAtNode, StrSliceNode,
    RandomNode, ArrayLengthNode, DictLitNode, DictGetNode,
    DictSetStmtNode, DictHasNode, DictDeleteStmtNode, DictLengthNode,
    DictKeyAtNode, TryStmtNode, RaiseStmtNode
)

SYS_READ = 0
SYS_WRITE = 1
SYS_OPEN = 2
SYS_CLOSE = 3
SYS_MMAP = 9
SYS_EXIT = 60
SYS_GETRANDOM = 318

STDOUT = 1
STDIN = 0

# mmap flags for the bump-allocator heap (anonymous, private, read+write)
PROT_READ = 1
PROT_WRITE = 2
MAP_PRIVATE = 2
MAP_ANONYMOUS = 0x20

class CodeGenerator:
    """
    Generates NASM x86_64 Linux assembly from High-Level AST nodes.
    No libc, no dynamic dependencies -- pure native syscall execution.

    Runtime memory model:
    - Global variables: fixed .bss slots (as before).
    - Function locals/params: stack-relative via rbp (standard C-like frame).
      Params pushed right-to-left by the caller, callee accesses them at
      positive offsets from rbp; the callee's own 'set' locals get their
      own .bss slots per call-site-unique name (simple and correct, though
      not reentrant/recursive-safe -- see NOTE in gen_block_def).
    - Strings/arrays: heap-allocated via a simple bump allocator backed by
      one big anonymous mmap (see __heap_alloc). This is what makes real
      string concatenation and dynamic arrays possible.
    """

    def __init__(self, filename: str = "<source>"):
        self.filename = filename
        self.text: List[str] = []
        self.data: List[str] = []
        self.bss: List[str] = []

        self.variables: Dict[str, str] = {}
        self.string_pool: Dict[str, str] = {}
        self.string_vars: Set[str] = set()
        self.array_vars: Set[str] = set()
        self.dict_vars: Set[str] = set()
        self.block_return_is_string: Set[str] = set()

        self.label_counter = 0
        self.str_counter = 0
        self.num_buf_declared = False
        self.heap_helpers_declared = False
        self.rand_buf_declared = False
        self.rand_seed_declared = False

        self.NUMBUF_LABEL = "__numbuf"
        self.NUMBUF_LEN = 24
        self.READBUF_LEN = 256
        self.HEAP_SIZE = 16 * 1024 * 1024  # 16MB bump-allocated heap
        self.MAX_TRY_DEPTH = 64            # max nested try/catch depth

        self.loop_end_stack: List[str] = []

        # try/catch codegen state: a stack of (catch_label, err_var_bss_label)
        # so a runtime error (raise, or a built-in trap like divide-by-zero)
        # inside a try block can jump straight to the nearest enclosing
        # catch handler. Empty stack means "no active try" -- an uncaught
        # error falls through to __fatal_error instead.
        # NOTE: error-handler nesting is tracked entirely at RUNTIME via
        # __try_stack (see gen_runtime_helpers/gen_try) rather than here,
        # because a function body is generated once but can be invoked
        # from inside a try, outside any try, or from different try blocks
        # depending on the call site -- only a runtime stack can know
        # which one actually applies at any given call.
        self.needs_error_handling = False
        self.needs_dict_helpers = False

        # Function-call codegen state
        self.block_params: Dict[str, List[str]] = {}   # block name -> param names
        self.current_params: Dict[str, int] = {}        # param name -> stack offset from rbp (while inside that block)
        self.in_block = False
        self.current_block_end_label: Optional[str] = None  # for 'return' to jump to

    def new_label(self, prefix="L") -> str:
        self.label_counter += 1
        return f"__{prefix}_{self.label_counter}"

    def emit(self, line: str):
        self.text.append(line)

    def var_label(self, name: str) -> str:
        """Resolve a variable name to its storage location.
        - If it's a current function's parameter, returns an [rbp+N] operand
          string directly usable inside mov [ ... ] (caller wraps in brackets
          for globals too, so we normalize: this always returns something
          usable inside 'mov X, [<result>]').
        - Otherwise falls back to a global .bss slot (existing behaviour)."""
        if name in self.current_params:
            return f"rbp+{self.current_params[name]}"
        if name not in self.variables:
            label = f"__var_{name}"
            self.variables[name] = label
            self.bss.append(f"{label}: resq 1")
        return self.variables[name]

    def string_label(self, text: str) -> Tuple[str, int]:
        if text in self.string_pool:
            label = self.string_pool[text]
        else:
            self.str_counter += 1
            label = f"__str_{self.str_counter}"
            self.string_pool[text] = label
            escaped_bytes = ", ".join(str(b) for b in text.encode("utf-8"))
            if escaped_bytes:
                self.data.append(f"{label}: db {escaped_bytes}, 0")
            else:
                self.data.append(f"{label}: db 0")
        return label, len(text.encode("utf-8"))

    def ensure_numbuf(self):
        if not self.num_buf_declared:
            self.bss.append(f"{self.NUMBUF_LABEL}: resb {self.NUMBUF_LEN}")
            self.num_buf_declared = True

    def ensure_random_buf(self):
        if not self.rand_buf_declared:
            self.bss.append("__randbuf: resq 1")
            self.rand_buf_declared = True

    def ensure_heap_helpers(self):
        """Just marks that __heap_alloc/__heap_init/__memcopy are needed;
        they're unconditionally emitted in gen_runtime_helpers anyway (the
        program always calls __heap_init at _start), so this is currently
        a no-op placeholder kept for clarity/future conditional emission."""
        self.heap_helpers_declared = True

    def ensure_dict_helpers(self):
        """Marks that the __dict_* runtime functions are needed; actually
        emitted in gen_runtime_helpers only when this flag is set, since
        (unlike heap helpers) most programs won't use dicts at all."""
        self.needs_dict_helpers = True

    def generate(self, program: ProgramNode) -> str:
        # Pre-scan string/array assignments to keep track of variable types.
        # These interact (a var can become 'string' because it holds a call
        # result, whose return type depends on params, which depend on
        # call-site argument types...), so we run everything to a combined
        # fixed point rather than a fixed sequence of one-shot passes.
        self.scan_array_vars(program.statements)
        self.scan_block_params(program.statements)
        while True:
            before = len(self.string_vars)
            self._scan_string_vars_pass(program.statements)
            self.scan_param_string_types(program.statements)
            self.scan_block_return_types(program.statements)
            if len(self.string_vars) == before:
                break

        # Collect the set of TOP-LEVEL (truly global) variable names. A
        # 'set x = ...' inside a block body only becomes a genuine
        # function-local (stack-allocated) variable if 'x' is NOT also a
        # global -- otherwise it's the existing common pattern of a block
        # reading/writing a shared global (like the original guessing-game
        # sample's 'won' variable), and must keep using the single shared
        # .bss slot exactly as before.
        self.global_var_names: Set[str] = set()
        self._collect_top_level_names(program.statements)

        self.emit("global _start")
        self.emit("")
        self.emit("section .text")
        self.emit("_start:")
        self.emit("    call __heap_init")

        # Function (block) definitions are emitted separately from the main
        # top-level statement stream so 'call __block_X' can appear before
        # its textual definition without issue (labels resolve at link time
        # regardless of emission order, but we still jump over their bodies
        # from the linear entry flow to avoid falling into a function body).
        top_level = [s for s in program.statements if not isinstance(s, BlockDefNode)]
        block_defs = [s for s in program.statements if isinstance(s, BlockDefNode)]

        for stmt in top_level:
            self.gen_stmt(stmt)

        # Default exit(0)
        self.emit("    mov rax, 60")
        self.emit("    xor rdi, rdi")
        self.emit("    syscall")

        for stmt in block_defs:
            self.gen_stmt(stmt)

        self.gen_runtime_helpers()

        out = [
            "; ============================================================",
            f"; Generated by MyLand High-Level Compiler ({self.filename})",
            "; Target: Linux x86_64 ELF (Native Syscalls)",
            "; ============================================================",
            "",
            "\n".join(self.text),
            "",
            "section .data",
            "\n".join(self.data) if self.data else "    ; empty data section",
            "",
            "section .bss",
            "\n".join(self.bss) if self.bss else "    ; empty bss section",
            ""
        ]
        return "\n".join(out)

    def _collect_top_level_names(self, stmts: List[ASTNode]):
        """Sirf program ke top-level (block ke bahar) 'set'/'ask'/file-op
        destination variable names collect karta hai -- yeh 'true globals'
        hain jinke naam agar kisi function ke andar dubara 'set' se milte
        hain, toh wo shared global hi maana jaayega (naya local nahi).
        NOTE: BlockDefNode bodies are deliberately skipped entirely -- a
        block's own locals must never leak into global_var_names."""
        for stmt in stmts:
            if isinstance(stmt, BlockDefNode):
                continue
            if isinstance(stmt, SetStmtNode) and stmt.index_expr is None:
                self.global_var_names.add(stmt.name)
            elif isinstance(stmt, AskStmtNode):
                self.global_var_names.add(stmt.var_name)
            elif isinstance(stmt, FileOpNode):
                if stmt.op_type == "open" and stmt.args and isinstance(stmt.args[0], VarRefNode):
                    self.global_var_names.add(stmt.args[0].name)
                elif stmt.op_type == "read" and len(stmt.args) > 1 and isinstance(stmt.args[1], VarRefNode):
                    self.global_var_names.add(stmt.args[1].name)
            elif isinstance(stmt, TryStmtNode):
                self.global_var_names.add(stmt.err_var)
            # Recurse into control-flow bodies (repeat/loop_if/check) but
            # NOT into BlockDefNode bodies (handled by the 'continue' above).
            for attr in ("body", "then_body", "else_body"):
                if hasattr(stmt, attr) and not isinstance(stmt, BlockDefNode):
                    self._collect_top_level_names(getattr(stmt, attr))
            if isinstance(stmt, CheckStmtNode):
                for _cond, ebody in stmt.elif_branches:
                    self._collect_top_level_names(ebody)
            if isinstance(stmt, TryStmtNode):
                self._collect_top_level_names(stmt.catch_body)

    def _scan_string_vars_pass(self, stmts: List[ASTNode]):
        for stmt in stmts:
            if isinstance(stmt, SetStmtNode) and stmt.index_expr is None:
                if self._expr_is_string(stmt.expr):
                    self.string_vars.add(stmt.name)
            elif isinstance(stmt, AskStmtNode) and not stmt.as_number:
                self.string_vars.add(stmt.var_name)
            elif isinstance(stmt, FileOpNode) and stmt.op_type == "read" and len(stmt.args) > 1:
                dest = stmt.args[1]
                if isinstance(dest, VarRefNode):
                    self.string_vars.add(dest.name)
            elif isinstance(stmt, TryStmtNode):
                self.string_vars.add(stmt.err_var)
            for attr in ("body", "then_body", "else_body"):
                if hasattr(stmt, attr):
                    self._scan_string_vars_pass(getattr(stmt, attr))
            if isinstance(stmt, CheckStmtNode):
                for _cond, ebody in stmt.elif_branches:
                    self._scan_string_vars_pass(ebody)
            if isinstance(stmt, TryStmtNode):
                self._scan_string_vars_pass(stmt.catch_body)

    def scan_array_vars(self, stmts: List[ASTNode]):
        for stmt in stmts:
            if isinstance(stmt, SetStmtNode) and stmt.index_expr is None and isinstance(stmt.expr, ArrayLitNode):
                self.array_vars.add(stmt.name)
            if isinstance(stmt, SetStmtNode) and stmt.index_expr is None and isinstance(stmt.expr, DictLitNode):
                self.dict_vars.add(stmt.name)
            for attr in ("body", "then_body", "else_body"):
                if hasattr(stmt, attr):
                    self.scan_array_vars(getattr(stmt, attr))
            if isinstance(stmt, CheckStmtNode):
                for _cond, ebody in stmt.elif_branches:
                    self.scan_array_vars(ebody)
            if isinstance(stmt, TryStmtNode):
                self.scan_array_vars(stmt.catch_body)

    def scan_block_params(self, stmts: List[ASTNode]):
        for stmt in stmts:
            if isinstance(stmt, BlockDefNode):
                self.block_params[stmt.name] = stmt.params

    def scan_block_return_types(self, stmts: List[ASTNode]):
        """Ek block ka return type (string vs number) uske 'return expr'
        statements ke expr type se determine hota hai. Caller (generate())
        isko ek fixed-point loop ke andar baar-baar call karta hai kyunki
        return type params/other calls pe depend kar sakta hai."""
        block_bodies = {s.name: s.body for s in stmts if isinstance(s, BlockDefNode)}

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
                if isinstance(s, TryStmtNode):
                    find_returns(s.catch_body, out)

        for name, body in block_bodies.items():
            returns = []
            find_returns(body, returns)
            if any(self._expr_is_string(r) for r in returns):
                self.block_return_is_string.add(name)

    def scan_param_string_types(self, stmts: List[ASTNode]):
        """Same best-effort inference as semantic.py: if a block is ever
        called with a string argument, treat that parameter as a string
        inside the function body too. Walks every expression position a
        CallExprNode can appear in -- not just 'run f(...)' as a
        standalone statement, but also inside set/say expressions,
        condition trees (check/loop_if, including compound and/or/not and
        the implicit 'expr != 0' truthiness wrapper), and for-loop bounds.
        Missing a position here doesn't just mis-infer a type: it makes
        the generated code call the wrong print/concat routine on a
        string parameter (e.g. printing its pointer as a raw integer)."""
        block_body_map = {s.name: s for s in stmts if isinstance(s, BlockDefNode)}

        def scan_expr(e):
            if isinstance(e, CallExprNode):
                params = self.block_params.get(e.name, [])
                for i, arg in enumerate(e.args):
                    if i < len(params) and self._expr_is_string(arg):
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
                if isinstance(s, TryStmtNode):
                    scan_calls(s.catch_body)

        while True:
            before = len(self.string_vars)
            scan_calls(stmts)
            for blk in block_body_map.values():
                scan_calls(blk.body)
            if len(self.string_vars) == before:
                break

    def _expr_is_string(self, node: ASTNode) -> bool:
        if isinstance(node, StringLitNode):
            return True
        if isinstance(node, VarRefNode):
            return node.name in self.string_vars
        if isinstance(node, BinOpNode) and node.op == "+":
            return self._expr_is_string(node.left) or self._expr_is_string(node.right)
        if isinstance(node, CallExprNode):
            return node.name in self.block_return_is_string
        if isinstance(node, (ToStringNode, StrSliceNode, StrCharAtNode, DictKeyAtNode)):
            # All four always produce a heap string pointer (to_string
            # converts a number to text; str:slice/str:char_at always
            # return a new heap string, even the empty-string fallback
            # case in str:char_at; dict:key_at likewise always returns a
            # string pointer, including its own empty-string fallback).
            return True
        return False

    def gen_stmt(self, stmt: ASTNode):
        if isinstance(stmt, SetStmtNode):
            if stmt.index_expr is not None:
                # set arr[i] = expr -- array-element assignment
                self.gen_expr_to_rax(stmt.expr)
                self.emit("    push rax")                    # save value
                self.gen_expr_to_rax(stmt.index_expr)
                self.emit("    mov rbx, rax")                 # rbx = index
                arr_lbl = self.var_label(stmt.name)
                self.emit(f"    mov rax, [{arr_lbl}]")        # rax = array base ptr
                self.emit("    pop rcx")                      # rcx = value
                self.emit("    mov [rax + rbx*8], rcx")
                return

            label = self.var_label(stmt.name)
            if isinstance(stmt.expr, StringLitNode):
                slabel, _ = self.string_label(stmt.expr.value)
                self.emit(f"    mov rax, {slabel}")
                self.emit(f"    mov [{label}], rax")
            elif isinstance(stmt.expr, ArrayLitNode):
                self.gen_array_new(stmt.expr)   # -> rax = heap pointer to array
                self.emit(f"    mov [{label}], rax")
            else:
                self.gen_expr_to_rax(stmt.expr)
                self.emit(f"    mov [{label}], rax")

        elif isinstance(stmt, SayStmtNode):
            expr = stmt.expr
            if isinstance(expr, StringLitNode):
                slabel, slen = self.string_label(expr.value)
                self.emit(f"    mov rax, {SYS_WRITE}")
                self.emit(f"    mov rdi, {STDOUT}")
                self.emit(f"    mov rsi, {slabel}")
                self.emit(f"    mov rdx, {slen}")
                self.emit("    syscall")
            elif self._expr_is_string(expr):
                # String variable OR string-concat expression -- evaluate
                # to get a heap/data pointer in rax, then print it as text.
                self.gen_expr_to_rax(expr)
                self.emit("    mov rsi, rax")
                self.emit("    call __strlen")
                self.emit(f"    mov rax, {SYS_WRITE}")
                self.emit(f"    mov rdi, {STDOUT}")
                self.emit("    syscall")
            else:
                self.gen_expr_to_rax(expr)
                self.ensure_numbuf()
                self.emit("    call __print_int_rax")

            # Automatic newline output after every 'say'
            self.emit(f"    mov rax, {SYS_WRITE}")
            self.emit(f"    mov rdi, {STDOUT}")
            slabel, slen = self.string_label("\n")
            self.emit(f"    mov rsi, {slabel}")
            self.emit(f"    mov rdx, {slen}")
            self.emit("    syscall")

        elif isinstance(stmt, AskStmtNode):
            buf_label = f"__askbuf_{self.label_counter}"
            self.label_counter += 1
            self.bss.append(f"{buf_label}: resb {self.READBUF_LEN}")
            var_lbl = self.var_label(stmt.var_name)

            loop_l = self.new_label("ask_loop")
            done_l = self.new_label("ask_done")

            self.emit(f"    lea r12, [{buf_label}]")
            self.emit("    xor r13, r13")
            self.emit(f"{loop_l}:")
            self.emit(f"    cmp r13, {self.READBUF_LEN - 1}")
            self.emit(f"    jge {done_l}")
            self.emit(f"    mov rax, {SYS_READ}")
            self.emit(f"    mov rdi, {STDIN}")
            self.emit("    mov rsi, r12")
            self.emit("    mov rdx, 1")
            self.emit("    syscall")
            self.emit("    cmp rax, 1")
            self.emit(f"    jne {done_l}")
            self.emit("    cmp byte [r12], 10")  # newline check
            self.emit(f"    je {done_l}")
            self.emit("    inc r12")
            self.emit("    inc r13")
            self.emit(f"    jmp {loop_l}")
            self.emit(f"{done_l}:")
            self.emit("    mov byte [r12], 0")

            if stmt.as_number:
                self.emit("    mov rbx, r13")
                self.emit(f"    lea rsi, [{buf_label}]")
                self.emit("    call __atoi")
                self.emit(f"    mov [{var_lbl}], rax")
            else:
                self.emit(f"    lea rax, [{buf_label}]")
                self.emit(f"    mov [{var_lbl}], rax")

        elif isinstance(stmt, RepeatStmtNode):
            counter_var = self.new_label("rep_cnt")
            top = self.new_label("rep_top")
            end = self.new_label("rep_end")
            self.bss.append(f"{counter_var}: resq 1")

            self.gen_expr_to_rax(stmt.count_expr)
            self.emit(f"    mov [{counter_var}], rax")
            self.emit(f"{top}:")
            self.emit(f"    mov rax, [{counter_var}]")
            self.emit("    cmp rax, 0")
            self.emit(f"    jle {end}")
            self.loop_end_stack.append(end)
            for s in stmt.body:
                self.gen_stmt(s)
            self.loop_end_stack.pop()
            self.emit(f"    mov rax, [{counter_var}]")
            self.emit("    dec rax")
            self.emit(f"    mov [{counter_var}], rax")
            self.emit(f"    jmp {top}")
            self.emit(f"{end}:")

        elif isinstance(stmt, LoopIfStmtNode):
            top = self.new_label("loop_top")
            end = self.new_label("loop_end")
            self.emit(f"{top}:")
            self.gen_cond_jump(stmt.condition, end, jump_if_false=True)
            self.loop_end_stack.append(end)
            for s in stmt.body:
                self.gen_stmt(s)
            self.loop_end_stack.pop()
            self.emit(f"    jmp {top}")
            self.emit(f"{end}:")

        elif isinstance(stmt, ForStmtNode):
            # for i in start..end ... end_for
            # i lives in a .bss/stack slot like any other variable (via
            # var_label), so it's readable/writable from inside the body
            # exactly like a normal 'set' variable would be.
            var_lbl = self.var_label(stmt.var_name)
            end_var = self.new_label("for_end_val")
            self.bss.append(f"{end_var}: resq 1")
            top = self.new_label("for_top")
            end = self.new_label("for_end")

            self.gen_expr_to_rax(stmt.start_expr)
            self.emit(f"    mov [{var_lbl}], rax")
            self.gen_expr_to_rax(stmt.end_expr)
            self.emit(f"    mov [{end_var}], rax")

            self.emit(f"{top}:")
            self.emit(f"    mov rax, [{var_lbl}]")
            self.emit(f"    mov rbx, [{end_var}]")
            self.emit("    cmp rax, rbx")
            self.emit(f"    jge {end}")   # exclusive end: stop when i >= end
            self.loop_end_stack.append(end)
            for s in stmt.body:
                self.gen_stmt(s)
            self.loop_end_stack.pop()
            self.emit(f"    mov rax, [{var_lbl}]")
            self.emit("    inc rax")
            self.emit(f"    mov [{var_lbl}], rax")
            self.emit(f"    jmp {top}")
            self.emit(f"{end}:")

        elif isinstance(stmt, BreakLoopStmtNode):
            if self.loop_end_stack:
                self.emit(f"    jmp {self.loop_end_stack[-1]}")

        elif isinstance(stmt, CheckStmtNode):
            # check COND ... [otherwise_check COND ...]* [otherwise ...] end_check
            # Each branch's condition-false path falls through to the next
            # otherwise_check (or otherwise, or end) in source order.
            end_l = self.new_label("check_end")
            num_elifs = len(stmt.elif_branches)
            has_else = bool(stmt.else_body)

            # Pre-allocate one 'next branch' label per elif + one for the
            # final else/end target, so gen_cond_jump can always jump
            # forward to "wherever the next check happens".
            elif_labels = [self.new_label("check_elif") for _ in range(num_elifs)]
            else_l = self.new_label("check_else") if has_else else None

            first_false_target = elif_labels[0] if num_elifs else (else_l if has_else else end_l)
            self.gen_cond_jump(stmt.condition, first_false_target, jump_if_false=True)
            for s in stmt.then_body:
                self.gen_stmt(s)
            self.emit(f"    jmp {end_l}")

            for i, (elif_cond, elif_body) in enumerate(stmt.elif_branches):
                self.emit(f"{elif_labels[i]}:")
                next_target = elif_labels[i + 1] if i + 1 < num_elifs else (else_l if has_else else end_l)
                self.gen_cond_jump(elif_cond, next_target, jump_if_false=True)
                for s in elif_body:
                    self.gen_stmt(s)
                self.emit(f"    jmp {end_l}")

            if has_else:
                self.emit(f"{else_l}:")
                for s in stmt.else_body:
                    self.gen_stmt(s)

            self.emit(f"{end_l}:")


        elif isinstance(stmt, BlockDefNode):
            self.gen_block_def(stmt)

        elif isinstance(stmt, RunStmtNode):
            self.gen_call(stmt.name, stmt.args)
            # Statement-form 'run' discards the return value (if any) --
            # nothing else to do, rax is simply not used further.

        elif isinstance(stmt, ReturnStmtNode):
            if stmt.expr is not None:
                self.gen_expr_to_rax(stmt.expr)
            # else: rax left as-is (undefined) -- caller shouldn't use the
            # return value of a block that has bare 'return' with no expr;
            # semantic.py already flags CallExprNode usage of such blocks.
            if self.current_block_end_label:
                self.emit(f"    jmp {self.current_block_end_label}")

        elif isinstance(stmt, LabelDefNode):
            self.emit(f"__lbl_{stmt.name}:")

        elif isinstance(stmt, GotoStmtNode):
            if stmt.condition:
                self.gen_cond_jump(stmt.condition, f"__lbl_{stmt.label}", jump_if_false=False)
            else:
                self.emit(f"    jmp __lbl_{stmt.label}")

        elif isinstance(stmt, StopStmtNode):
            self.emit(f"    mov rax, {SYS_EXIT}")
            self.emit(f"    mov rdi, {1 if stmt.is_error else 0}")
            self.emit("    syscall")

        elif isinstance(stmt, FileOpNode):
            self.gen_file_op(stmt)

        elif isinstance(stmt, DictSetStmtNode):
            self.gen_dict_set(stmt)

        elif isinstance(stmt, DictDeleteStmtNode):
            self.gen_dict_delete(stmt)

        elif isinstance(stmt, TryStmtNode):
            self.gen_try(stmt)

        elif isinstance(stmt, RaiseStmtNode):
            self.gen_raise(stmt)

    def gen_checked_divisor(self, is_mod: bool):
        """Emits the div/mod sequence with a zero-divisor check in front of
        it: rax = dividend, rbx = divisor are already loaded by the caller.
        x86 raises a hardware #DE exception (killing the whole process,
        uncatchable) on integer division by zero -- this check intercepts
        that case first and turns it into a normal raise/catch-able error
        instead, matching how 'raise' from user code behaves."""
        self.needs_error_handling = True
        ok_label = self.new_label("divok")
        self.emit("    cmp rbx, 0")
        self.emit(f"    jne {ok_label}")
        msg_label, msg_len = self.string_label("division by zero")
        self.emit(f"    mov rax, {msg_label}")
        self.emit("    mov [__err_message], rax")
        self.emit("    call __raise_to_handler")
        self.emit(f"{ok_label}:")
        self.emit("    cqo")
        self.emit("    idiv rbx")
        if is_mod:
            self.emit("    mov rax, rdx")

    def gen_try(self, node: TryStmtNode):
        """try body catch err ... end_try.
        A runtime error (raise, or a built-in trap like divide-by-zero)
        anywhere inside 'body' -- including inside a function this try
        calls into, at any call depth -- jumps straight to this try's
        catch label, skipping any remaining statements in body. The error
        message is left in the shared __err_message slot by whatever
        raised it; the catch handler loads it into err_var before running
        catch_body.

        Unlike a compile-time label stack, the set of "which try is
        currently active" has to be tracked at RUNTIME (in __try_stack),
        because a function's body is only generated once but can be
        called from inside a try, outside any try, or from several
        different try blocks depending on the call site -- a compile-time
        stack can't know which one applies at any given call. So entering
        a try pushes this try's catch-label address onto a runtime stack,
        and raise / a built-in trap always jumps through whatever's
        currently on top of that stack, however deep the call chain that
        led there.

        Nesting: a try's own catch_body runs with that try already popped
        off __try_stack (an error while handling the error is not
        re-caught by the same try -- it propagates to whatever try
        encloses this one, or is fatal)."""
        catch_label = self.new_label("catch")
        end_label = self.new_label("try_end")
        err_var_label = self.var_label(node.err_var)
        self.string_vars.add(node.err_var)

        self.needs_error_handling = True
        self.emit(f"    lea rax, [{catch_label}]")
        self.emit("    call __try_push")
        for s in node.body:
            self.gen_stmt(s)
        self.emit("    call __try_pop")
        self.emit(f"    jmp {end_label}")   # body finished with no error -- skip the catch handler

        self.emit(f"{catch_label}:")
        # A raise/trap that lands here has already popped this try off
        # __try_stack itself (see __raise_to_handler), so catch_body runs
        # with the *enclosing* try (if any) on top, exactly like the body
        # finishing normally above.
        self.emit(f"    mov rax, [__err_message]")
        self.emit(f"    mov [{err_var_label}], rax")
        for s in node.catch_body:
            self.gen_stmt(s)

        self.emit(f"{end_label}:")

    def gen_raise(self, node: RaiseStmtNode):
        """raise expr: stores expr (as a string -- non-string values are
        converted first) into the shared __err_message slot, then hands
        off to __raise_to_handler, which jumps to whatever try is
        currently on top of the runtime __try_stack (or __fatal_error if
        the stack is empty)."""
        self.needs_error_handling = True
        if self._expr_is_string(node.expr):
            self.gen_expr_to_rax(node.expr)
        else:
            # Non-string raise value: convert the number to a heap string
            # the same way to_string(...) does, so err_var always ends up
            # holding text regardless of what was raised.
            self.ensure_numbuf()
            self.ensure_heap_helpers()
            self.gen_expr_to_rax(node.expr)
            self.emit("    call __int_to_buf")
            self.emit("    push rsi\n    push rdx")
            self.emit("    lea rdi, [rdx + 1]")
            self.emit("    call __heap_alloc")
            self.emit("    pop rcx\n    pop rsi")
            self.emit("    mov rdi, rax")
            self.emit("    push rax")
            self.emit("    call __memcopy")
            self.emit("    pop rax")
            self.emit("    mov byte [rax + rcx], 0")
        self.emit("    mov [__err_message], rax")
        self.emit("    call __raise_to_handler")

    def gen_file_op(self, node: FileOpNode):
        if node.op_type == "open":
            handle_var = node.args[0].name
            path_arg = node.args[1]
            mode = node.args[2].value if len(node.args) > 2 and isinstance(node.args[2], StringLitNode) else "r"
            flags = {"r": 0, "w": 0o1101, "a": 0o2001}.get(mode, 0)

            if isinstance(path_arg, StringLitNode):
                plabel, _ = self.string_label(path_arg.value)
                self.emit(f"    mov rdi, {plabel}")
            else:
                self.emit(f"    mov rdi, [{self.var_label(path_arg.name)}]")

            self.emit(f"    mov rax, {SYS_OPEN}")
            self.emit(f"    mov rsi, {flags}")
            self.emit("    mov rdx, 0o644")
            self.emit("    syscall")
            self.emit(f"    mov [{self.var_label(handle_var)}], rax")

        elif node.op_type == "write":
            handle_var = node.args[0].name
            content = node.args[1]
            if isinstance(content, StringLitNode):
                slabel, slen = self.string_label(content.value)
                self.emit(f"    mov rax, {SYS_WRITE}")
                self.emit(f"    mov rdi, [{self.var_label(handle_var)}]")
                self.emit(f"    mov rsi, {slabel}")
                self.emit(f"    mov rdx, {slen}")
                self.emit("    syscall")
            elif isinstance(content, VarRefNode):
                # Content ek variable hai (string ya number). Agar wo ek
                # string variable hai toh uska runtime length __strlen se
                # nikalo (compile-time pe pata nahi hota). Number variable
                # hone par __print_int_to_buf se pehle decimal string mein
                # convert karke likhte hain.
                var_lbl = self.var_label(content.name)
                if content.name in self.string_vars:
                    self.emit(f"    mov rsi, [{var_lbl}]")
                    self.emit("    call __strlen")   # rdx = strlen(rsi)
                    self.emit(f"    mov rdi, [{self.var_label(handle_var)}]")
                    self.emit(f"    mov rax, {SYS_WRITE}")
                    self.emit("    syscall")
                else:
                    self.ensure_numbuf()
                    self.emit(f"    mov rax, [{var_lbl}]")
                    self.emit("    call __int_to_buf")   # -> rsi=buf ptr, rdx=len
                    self.emit(f"    mov rdi, [{self.var_label(handle_var)}]")
                    self.emit(f"    mov rax, {SYS_WRITE}")
                    self.emit("    syscall")
            else:
                # Number literal seedha likha gaya (file:write h 42)
                self.ensure_numbuf()
                self.gen_expr_to_rax(content)
                self.emit("    call __int_to_buf")
                self.emit(f"    mov rdi, [{self.var_label(handle_var)}]")
                self.emit(f"    mov rax, {SYS_WRITE}")
                self.emit("    syscall")

        elif node.op_type == "read":
            # file:read handle dest_var
            # Poori file ko heap pe padhta hai, ek fixed-size stack chunk
            # buffer se baar-baar read karke aur __heap_alloc se growing
            # result buffer mein copy karke -- ab file-size par koi
            # (256-byte jaisa) artificial cap nahi hai, EOF tak read karta
            # hai.
            self.ensure_heap_helpers()
            handle_var = node.args[0].name
            dest_name = node.args[1].name
            chunk_label = f"__readchunk_{self.label_counter}"
            self.label_counter += 1
            CHUNK = 4096
            self.bss.append(f"{chunk_label}: resb {CHUNK}")

            read_loop = self.new_label("fread_loop")
            read_done = self.new_label("fread_done")

            self.emit("    push r12")   # r12 = growing result buffer ptr (heap)
            self.emit("    push r13")   # r13 = total bytes accumulated so far
            self.emit("    push r14")   # r14 = current capacity of result buffer
            self.emit("    push r15")   # r15 = bytes read in the current chunk

            self.emit("    mov rdi, 4096")
            self.emit("    call __heap_alloc")
            self.emit("    mov r12, rax")
            self.emit("    xor r13, r13")
            self.emit("    mov r14, 4096")

            self.emit(f"{read_loop}:")
            self.emit(f"    mov rax, {SYS_READ}")
            self.emit(f"    mov rdi, [{self.var_label(handle_var)}]")
            self.emit(f"    lea rsi, [{chunk_label}]")
            self.emit(f"    mov rdx, {CHUNK}")
            self.emit("    syscall")
            self.emit("    cmp rax, 0")
            self.emit(f"    jle {read_done}")   # EOF (0) or error (negative)
            self.emit("    mov r15, rax")       # r15 = bytes just read this chunk (callee-saved, survives calls below)

            # Grow result buffer if this chunk won't fit (need r13+r15+1 <= r14).
            self.emit("    lea rbx, [r13 + r15 + 1]")
            self.emit("    cmp rbx, r14")
            after_grow_l = self.new_label("fread_after_grow")
            self.emit(f"    jle {after_grow_l}")
            self.emit("    lea r14, [r14*2 + r15 + 4096]")   # new capacity: comfortably bigger
            self.emit("    mov rdi, r14")
            self.emit("    call __heap_alloc")               # rax = new bigger buffer (zeroed)
            self.emit("    mov rdi, rax")                    # copy-dest = new buffer
            self.emit("    mov rsi, r12")                    # copy-src = old buffer
            self.emit("    mov rcx, r13")                    # copy-len = bytes accumulated so far
            self.emit("    call __memcopy")
            self.emit("    mov r12, rax")                    # NOTE: rax preserved by __memcopy (see helper)
            self.emit(f"{after_grow_l}:")

            # Append this chunk to the result buffer.
            self.emit("    lea rdi, [r12 + r13]")
            self.emit(f"    lea rsi, [{chunk_label}]")
            self.emit("    mov rcx, r15")
            self.emit("    call __memcopy")
            self.emit("    add r13, r15")
            self.emit(f"    jmp {read_loop}")

            self.emit(f"{read_done}:")
            self.emit("    mov byte [r12 + r13], 0")   # null-terminate
            self.emit("    mov rax, r12")
            self.emit(f"    mov [{self.var_label(dest_name)}], rax")

            self.emit("    pop r15")
            self.emit("    pop r14")
            self.emit("    pop r13")
            self.emit("    pop r12")

        elif node.op_type == "close":
            handle_var = node.args[0].name
            self.emit(f"    mov rax, {SYS_CLOSE}")
            self.emit(f"    mov rdi, [{self.var_label(handle_var)}]")
            self.emit("    syscall")

    def _collect_locals(self, stmts: List[ASTNode], params: Set[str], out: List[str], seen: Set[str]):
        """Function body ke andar 'set x = ...' se banne wale saare naye
        local variable names collect karta hai (jo params mein nahi hain
        AUR jo top-level global bhi nahi hain -- agar naam ek global se
        match karta hai, wo global hi treat hota hai, taaki ek block ke
        andar se ek shared global variable set/read karna purane behavior
        jaisa hi kaam kare). Nested blocks (check/repeat/loop_if ke andar)
        ke locals bhi isi function-level frame mein aate hain."""
        for stmt in stmts:
            if isinstance(stmt, SetStmtNode) and stmt.index_expr is None:
                if stmt.name not in params and stmt.name not in seen and stmt.name not in self.global_var_names:
                    seen.add(stmt.name)
                    out.append(stmt.name)
            elif isinstance(stmt, AskStmtNode):
                if stmt.var_name not in params and stmt.var_name not in seen and stmt.var_name not in self.global_var_names:
                    seen.add(stmt.var_name)
                    out.append(stmt.var_name)
            elif isinstance(stmt, FileOpNode):
                # file:open handle_var, file:read ... dest_var -- pehla arg
                # (open) ya doosra arg (read) ek naya local ho sakta hai.
                candidates = []
                if stmt.op_type == "open" and stmt.args:
                    candidates.append(stmt.args[0])
                elif stmt.op_type == "read" and len(stmt.args) > 1:
                    candidates.append(stmt.args[1])
                for c in candidates:
                    if isinstance(c, VarRefNode) and c.name not in params and c.name not in seen and c.name not in self.global_var_names:
                        seen.add(c.name)
                        out.append(c.name)
            for attr in ("body", "then_body", "else_body"):
                if hasattr(stmt, attr):
                    self._collect_locals(getattr(stmt, attr), params, out, seen)

    def gen_block_def(self, stmt: BlockDefNode):
        """Emits a callable function with a standard rbp-based stack frame.
        Both parameters AND locals ('set x = ...' declared inside the body)
        live on the stack, at negative/positive offsets from rbp
        respectively -- this makes the function correctly reentrant, so
        recursive calls (e.g. fibonacci) don't clobber each other's local
        variables. Globals declared at the top level of the program are
        unaffected and still use fixed .bss slots."""
        after_l = self.new_label(f"after_block_{stmt.name}")
        end_l = self.new_label(f"end_block_{stmt.name}")
        self.emit(f"    jmp {after_l}")
        self.emit(f"__block_{stmt.name}:")
        self.emit("    push rbp")
        self.emit("    mov rbp, rsp")

        saved_params = self.current_params
        saved_end_label = self.current_block_end_label
        self.current_params = {}
        for i, pname in enumerate(stmt.params):
            self.current_params[pname] = 16 + i * 8

        # Collect this function's own locals and give each an [rbp-N] slot.
        local_names: List[str] = []
        self._collect_locals(stmt.body, set(stmt.params), local_names, set())
        frame_size = 8 * len(local_names)
        for i, lname in enumerate(local_names):
            self.current_params[lname] = -(8 * (i + 1))
        if frame_size:
            # Round up to 16 for stack alignment hygiene (not strictly
            # required here since we don't call any external ABI that
            # demands 16-byte alignment, but it's good practice).
            aligned = frame_size + (frame_size % 16)
            self.emit(f"    sub rsp, {aligned}")

        self.current_block_end_label = end_l

        for s in stmt.body:
            self.gen_stmt(s)

        self.emit(f"{end_l}:")
        self.emit("    mov rsp, rbp")
        self.emit("    pop rbp")
        self.emit("    ret")
        self.emit(f"{after_l}:")

        self.current_params = saved_params
        self.current_block_end_label = saved_end_label

    def gen_call(self, name: str, args: List[ASTNode]):
        """Calling convention (caller side):
        Args are pushed onto the stack in REVERSE order (rightmost/last
        argument pushed FIRST, leftmost/first argument pushed LAST). This
        means immediately before 'call', the first argument sits on top of
        the stack (lowest address of the arg block). After 'call' pushes
        the return address, rbp is set to rsp in the callee, so:
            rbp+8  = return address
            rbp+16 = first argument   (last thing pushed by caller)
            rbp+24 = second argument
            ... etc.
        This matches gen_block_def's offset assignment (16 + i*8 for the
        i-th param in declaration order). Stack is caller-cleaned after
        the call returns."""
        for arg in reversed(args):
            self.gen_expr_to_rax(arg)
            self.emit("    push rax")
        self.emit(f"    call __block_{name}")
        if args:
            self.emit(f"    add rsp, {8 * len(args)}")   # caller cleans up
        # Return value convention: rax. Callers that need it read rax right
        # after this call; statement-form 'run' simply ignores it.

    def gen_array_new(self, node: ArrayLitNode):
        """Allocates an N-element (8 bytes each) array on the heap, with an
        8-byte length header immediately before the element data. Layout:
        [header: N][elem 0][elem 1]...[elem N-1]. Result (rax) points at
        elem 0, same as before -- the header is invisible to normal
        indexing (arr[i] still means [arr + i*8]) and only read by
        array:length via [arr - 8]. This keeps every existing array op
        (allocation, indexing, string/array type inference) unchanged
        while making the length recoverable at runtime."""
        self.ensure_heap_helpers()
        self.gen_expr_to_rax(node.size_expr)
        self.emit("    push rax")            # save element count for the header write
        self.emit("    imul rax, rax, 8")
        self.emit("    add rax, 8")          # +8 for the header word itself
        self.emit("    mov rdi, rax")
        self.emit("    call __heap_alloc")   # rax = pointer to header (zero-filled)
        self.emit("    pop rbx")             # rbx = element count
        self.emit("    mov [rax], rbx")      # write header
        self.emit("    add rax, 8")          # rax = pointer to elem 0 (what callers expect)

    # ---------------- Dictionaries ----------------
    #
    # Runtime layout (all on the heap, pointer returned to callers points
    # at the header, matching how strings/arrays already work):
    #   [0]  count     -- number of key-value pairs currently stored
    #   [8]  capacity  -- number of (key,val) slots currently allocated
    #   [16] key0_ptr  [24] val0
    #   [32] key1_ptr  [40] val1
    #   ...
    # Keys are always heap string pointers, compared by content via
    # __streq (never by pointer identity -- two different allocations can
    # hold the same key text). Values are opaque 8-byte slots, same
    # convention as array elements: a raw number, or a pointer (string/
    # array/another dict) depending on what the caller stored there.
    # Lookup is linear scan -- simple and correct, and plenty fast for the
    # dict sizes actual scripts use (dozens to low hundreds of keys).
    DICT_INITIAL_CAPACITY = 8

    def gen_dict_new(self):
        """dict -> a new, empty heap dict with room for
        DICT_INITIAL_CAPACITY pairs before its first resize."""
        self.ensure_heap_helpers()
        self.ensure_dict_helpers()
        self.emit("    call __dict_new")   # rax = pointer to new dict header

    def gen_dict_get(self, node: DictGetNode):
        self.ensure_heap_helpers()
        self.ensure_dict_helpers()
        self.gen_expr_to_rax(node.key_expr)
        self.emit("    push rax")             # save key ptr
        self.gen_expr_to_rax(node.dict_expr)
        self.emit("    mov rdi, rax")         # rdi = dict ptr
        self.emit("    pop rsi")              # rsi = key ptr
        self.emit("    call __dict_get")      # rax = value, or 0 if not found

    def gen_dict_has(self, node: DictHasNode):
        self.ensure_heap_helpers()
        self.ensure_dict_helpers()
        self.gen_expr_to_rax(node.key_expr)
        self.emit("    push rax")
        self.gen_expr_to_rax(node.dict_expr)
        self.emit("    mov rdi, rax")
        self.emit("    pop rsi")
        self.emit("    call __dict_find_slot")   # rax = slot ptr, or 0 if not found
        self.emit("    cmp rax, 0")
        self.emit("    setne al")
        self.emit("    movzx rax, al")

    def gen_dict_key_at(self, node: DictKeyAtNode):
        self.ensure_heap_helpers()
        self.ensure_dict_helpers()
        self.gen_expr_to_rax(node.index_expr)
        self.emit("    push rax")             # save requested index
        self.gen_expr_to_rax(node.dict_expr)
        self.emit("    mov rdi, rax")         # rdi = dict ptr
        self.emit("    pop rsi")              # rsi = index
        self.emit("    call __dict_key_at")   # rax = key string ptr, or empty string if out of range

    def gen_dict_set(self, node: DictSetStmtNode):
        self.ensure_heap_helpers()
        self.ensure_dict_helpers()
        self.gen_expr_to_rax(node.value_expr)
        self.emit("    push rax")             # save value
        self.gen_expr_to_rax(node.key_expr)
        self.emit("    push rax")             # save key ptr
        self.gen_expr_to_rax(node.dict_expr)
        self.emit("    mov rdi, rax")         # rdi = dict ptr
        self.emit("    pop rsi")              # rsi = key ptr
        self.emit("    pop rdx")              # rdx = value
        self.emit("    call __dict_set")      # rax = (possibly reallocated) dict ptr
        # dict:set can trigger a resize, which reallocates the backing
        # buffer -- write the (possibly new) pointer back into the
        # variable so later uses of this dict see the current storage.
        if isinstance(node.dict_expr, VarRefNode):
            self.emit(f"    mov [{self.var_label(node.dict_expr.name)}], rax")

    def gen_dict_delete(self, node: DictDeleteStmtNode):
        self.ensure_heap_helpers()
        self.ensure_dict_helpers()
        self.gen_expr_to_rax(node.key_expr)
        self.emit("    push rax")
        self.gen_expr_to_rax(node.dict_expr)
        self.emit("    mov rdi, rax")
        self.emit("    pop rsi")
        self.emit("    call __dict_delete")

    def gen_random(self, node: RandomNode):
        """random(min, max): fills __randbuf with 8 fresh random bytes via
        the getrandom(2) syscall (no manual seeding needed, unlike an LCG --
        the kernel's CSPRNG handles that), then folds it into the inclusive
        range [min, max] as min + (unsigned(rand) mod (max - min + 1)).
        Uses unsigned mod (div, not idiv) since the raw random bytes are
        treated as an arbitrary bit pattern, not a signed quantity -- an
        idiv here would bias the result and could raise #DE on edge
        patterns that look like INT_MIN / -1 in two's complement."""
        self.ensure_random_buf()
        self.gen_expr_to_rax(node.min_expr)
        self.emit("    push rax")            # save min
        self.gen_expr_to_rax(node.max_expr)
        self.emit("    push rax")            # save max

        self.emit("    push rdi\n    push rsi\n    push rdx")
        self.emit("    lea rdi, [__randbuf]")
        self.emit("    mov rsi, 8")
        self.emit("    xor rdx, rdx")
        self.emit(f"    mov rax, {SYS_GETRANDOM}")
        self.emit("    syscall")
        self.emit("    pop rdx\n    pop rsi\n    pop rdi")

        self.emit("    mov rax, [__randbuf]")
        self.emit("    pop rcx")             # rcx = max
        self.emit("    pop rbx")             # rbx = min
        self.emit("    mov r11, rcx")
        self.emit("    sub r11, rbx")
        self.emit("    inc r11")             # r11 = range size = (max - min + 1)
        self.emit("    xor rdx, rdx")        # zero-extend for unsigned div (not sign-extend)
        self.emit("    div r11")             # rax = quotient (discarded), rdx = remainder
        self.emit("    mov rax, rdx")
        self.emit("    add rax, rbx")        # rax = min + (rand mod range)

    def gen_expr_to_rax(self, expr: ASTNode):
        if isinstance(expr, NumberLitNode):
            self.emit(f"    mov rax, {expr.value}")

        elif isinstance(expr, StringLitNode):
            slabel, _ = self.string_label(expr.value)
            self.emit(f"    mov rax, {slabel}")

        elif isinstance(expr, VarRefNode):
            self.emit(f"    mov rax, [{self.var_label(expr.name)}]")

        elif isinstance(expr, ArrayLengthNode):
            self.gen_expr_to_rax(expr.expr)
            self.emit("    mov rax, [rax - 8]")   # read the header word

        elif isinstance(expr, RandomNode):
            self.gen_random(expr)

        elif isinstance(expr, IndexExprNode):
            self.gen_expr_to_rax(expr.index_expr)
            self.emit("    mov rbx, rax")
            arr_lbl = self.var_label(expr.array_name)
            self.emit(f"    mov rax, [{arr_lbl}]")
            self.emit("    mov rax, [rax + rbx*8]")

        elif isinstance(expr, ArrayLitNode):
            self.gen_array_new(expr)

        elif isinstance(expr, DictLitNode):
            self.gen_dict_new()

        elif isinstance(expr, DictGetNode):
            self.gen_dict_get(expr)

        elif isinstance(expr, DictHasNode):
            self.gen_dict_has(expr)

        elif isinstance(expr, DictLengthNode):
            self.gen_expr_to_rax(expr.expr)
            self.emit("    mov rax, [rax]")   # header word 0 = pair count

        elif isinstance(expr, DictKeyAtNode):
            self.gen_dict_key_at(expr)

        elif isinstance(expr, CallExprNode):
            self.gen_call(expr.name, expr.args)
            # rax already holds the return value after gen_call

        elif isinstance(expr, BoolLitNode):
            self.emit(f"    mov rax, {1 if expr.value else 0}")

        elif isinstance(expr, ToStringNode):
            self.ensure_numbuf()
            self.ensure_heap_helpers()
            self.gen_expr_to_rax(expr.expr)
            self.emit("    call __int_to_buf")   # -> rsi=buf ptr, rdx=len (writes into shared NUMBUF)
            # __int_to_buf writes into the single shared NUMBUF scratch
            # buffer -- the text must be copied off it immediately onto the
            # heap so it stays valid independently of any later codegen
            # (concatenation, another to_string call, etc. would otherwise
            # overwrite NUMBUF before this value gets used).
            self.emit("    push rsi")               # save src ptr
            self.emit("    push rdx")               # save len
            self.emit("    lea rdi, [rdx + 1]")
            self.emit("    call __heap_alloc")      # rax = new zeroed buffer
            self.emit("    pop rcx")                # rcx = len (preserved across __memcopy)
            self.emit("    pop rsi")                # rsi = src ptr (NUMBUF digits)
            self.emit("    mov rdi, rax")            # rdi = dest cursor
            self.emit("    push rax")                # save result ptr (return value)
            self.emit("    call __memcopy")          # preserves rax/rdx/rcx
            self.emit("    pop rax")                 # rax = pointer to the new heap string
            self.emit("    mov byte [rax + rcx], 0")

        elif isinstance(expr, ToNumberNode):
            self.ensure_numbuf()
            self.gen_expr_to_rax(expr.expr)         # rax = string pointer
            self.emit("    mov rsi, rax")
            self.emit("    call __strlen")          # rdx = strlen(rsi)
            self.emit("    mov rbx, rdx")
            self.emit("    call __atoi")            # rsi + rbx (len) -> rax = parsed int

        elif isinstance(expr, StrLengthNode):
            self.gen_expr_to_rax(expr.expr)
            self.emit("    mov rsi, rax")
            self.emit("    call __strlen")
            self.emit("    mov rax, rdx")

        elif isinstance(expr, StrCharAtNode):
            self.ensure_heap_helpers()
            self.emit("    push r12")
            self.gen_expr_to_rax(expr.expr)
            self.emit("    mov r12, rax")           # r12 = string ptr
            self.emit("    mov rsi, r12")
            self.emit("    call __strlen")          # rdx = strlen
            self.emit("    push rdx")               # save length
            self.gen_expr_to_rax(expr.index_expr)   # rax = requested index
            self.emit("    pop rdx")
            self.emit("    cmp rax, 0")
            oob_l = self.new_label("charat_oob")
            ok_l = self.new_label("charat_ok")
            end_l = self.new_label("charat_end")
            self.emit(f"    jl {oob_l}")
            self.emit("    cmp rax, rdx")
            self.emit(f"    jge {oob_l}")
            self.emit(f"    jmp {ok_l}")
            self.emit(f"{oob_l}:")
            # Out-of-range index -> empty string (single NUL byte on the heap)
            self.emit("    mov rdi, 1")
            self.emit("    call __heap_alloc")
            self.emit(f"    jmp {end_l}")
            self.emit(f"{ok_l}:")
            self.emit("    mov rbx, rax")           # rbx = index
            self.emit("    mov rdi, 2")
            self.emit("    call __heap_alloc")      # rax = 2-byte buffer (char + NUL, zeroed)
            self.emit("    mov cl, [r12 + rbx]")
            self.emit("    mov [rax], cl")
            self.emit(f"{end_l}:")
            self.emit("    pop r12")

        elif isinstance(expr, StrSliceNode):
            self.ensure_heap_helpers()
            self.emit("    push r12")
            self.emit("    push r13")
            self.emit("    push r14")
            self.gen_expr_to_rax(expr.expr)
            self.emit("    mov r12, rax")           # r12 = source string ptr
            self.emit("    mov rsi, r12")
            self.emit("    call __strlen")          # rdx = source length
            self.emit("    mov r13, rdx")           # r13 = source length

            self.gen_expr_to_rax(expr.start_expr)
            self.emit("    push rax")               # stash start
            self.gen_expr_to_rax(expr.end_expr)
            self.emit("    mov r14, rax")           # r14 = end (pre-clamp)
            self.emit("    pop rax")                # rax = start (pre-clamp)

            # Clamp start to [0, source_len], end to [start, source_len].
            self.emit("    cmp rax, 0")
            clamp1 = self.new_label("slice_c1")
            self.emit(f"    jge {clamp1}")
            self.emit("    xor rax, rax")
            self.emit(f"{clamp1}:")
            self.emit("    cmp rax, r13")
            clamp2 = self.new_label("slice_c2")
            self.emit(f"    jle {clamp2}")
            self.emit("    mov rax, r13")
            self.emit(f"{clamp2}:")
            self.emit("    mov rbx, rax")           # rbx = clamped start

            self.emit("    cmp r14, rbx")
            clamp3 = self.new_label("slice_c3")
            self.emit(f"    jge {clamp3}")
            self.emit("    mov r14, rbx")
            self.emit(f"{clamp3}:")
            self.emit("    cmp r14, r13")
            clamp4 = self.new_label("slice_c4")
            self.emit(f"    jle {clamp4}")
            self.emit("    mov r14, r13")
            self.emit(f"{clamp4}:")

            self.emit("    mov rcx, r14")
            self.emit("    sub rcx, rbx")           # rcx = slice length (end - start)
            self.emit("    push rcx")
            self.emit("    lea rdi, [rcx + 1]")
            self.emit("    call __heap_alloc")      # rax = new zeroed buffer
            self.emit("    pop rcx")
            self.emit("    push rax")               # save result buffer ptr
            self.emit("    lea rsi, [r12 + rbx]")   # source cursor = ptr + start
            self.emit("    mov rdi, rax")
            self.emit("    call __memcopy")
            self.emit("    pop rax")                # rax = result buffer ptr (final answer)
            self.emit("    pop r14")
            self.emit("    pop r13")
            self.emit("    pop r12")

        elif isinstance(expr, BinOpNode):
            if expr.op == "+" and (self._expr_is_string(expr.left) or self._expr_is_string(expr.right)):
                self.gen_string_concat(expr)
                return
            self.gen_expr_to_rax(expr.right)
            self.emit("    push rax")
            self.gen_expr_to_rax(expr.left)
            self.emit("    pop rbx")
            if expr.op == "+": self.emit("    add rax, rbx")
            elif expr.op == "-": self.emit("    sub rax, rbx")
            elif expr.op == "*": self.emit("    imul rax, rbx")
            elif expr.op == "/":
                self.gen_checked_divisor(is_mod=False)
            elif expr.op == "%":
                self.gen_checked_divisor(is_mod=True)

    def gen_string_concat(self, expr: BinOpNode):
        """Real runtime string concatenation: evaluates both sides to
        C-string pointers, computes combined length, heap-allocates a new
        buffer, and copies both strings into it back-to-back with a null
        terminator. Result: rax = pointer to the new concatenated string.

        Uses r12/r13/r14 (callee-saved) to hold the two source pointers and
        their lengths across the __strlen/__heap_alloc calls -- this is
        simpler and less error-prone than juggling everything through the
        stack with push/pop."""
        self.ensure_heap_helpers()
        self.emit("    push r12")
        self.emit("    push r13")
        self.emit("    push r14")

        self.gen_expr_to_rax(expr.left)
        self.emit("    mov r12, rax")          # r12 = left ptr
        self.emit("    mov rsi, r12")
        self.emit("    call __strlen")
        self.emit("    mov r13, rdx")          # r13 = left len

        self.gen_expr_to_rax(expr.right)
        self.emit("    mov r14, rax")          # r14 = right ptr
        self.emit("    mov rsi, r14")
        self.emit("    call __strlen")
        # rdx = right len

        self.emit("    lea rdi, [r13 + rdx + 1]")   # total bytes needed (+1 for NUL)
        self.emit("    push rdx")                    # save right len across the alloc call
        self.emit("    call __heap_alloc")            # rax = new zeroed buffer
        self.emit("    pop rdx")                      # restore right len

        # Copy left string into the new buffer.
        self.emit("    mov rdi, rax")          # rdi = dest cursor
        self.emit("    push rax")              # save buffer start (this is our final result)
        self.emit("    mov rsi, r12")          # rsi = left ptr
        self.emit("    mov rcx, r13")          # rcx = left len
        self.emit("    call __memcopy")        # copies rcx bytes rsi->rdi, advances both

        # Copy right string right after it.
        self.emit("    mov rsi, r14")          # rsi = right ptr
        self.emit("    mov rcx, rdx")          # rcx = right len
        self.emit("    call __memcopy")
        self.emit("    mov byte [rdi], 0")     # null terminator

        self.emit("    pop rax")               # rax = pointer to the concatenated string
        self.emit("    pop r14")
        self.emit("    pop r13")
        self.emit("    pop r12")

    def gen_cond_jump(self, cond: ASTNode, target: str, jump_if_false: bool):
        """Emits code that jumps to 'target' when the condition evaluates
        to the polarity requested by jump_if_false (True = jump when the
        condition is FALSE, False = jump when it's TRUE).

        Handles compound conditions (LogicalNode/NotNode) with proper
        short-circuit evaluation: for 'and', a false left operand skips
        evaluating the right operand entirely (jumps straight past it);
        for 'or', a true left operand does the same. This mirrors how
        every C-like language evaluates &&/|| -- important not just for
        performance but for correctness when the right operand has side
        effects (e.g. 'x != 0 and (10 / x) > 1' must not evaluate the
        division when x is 0)."""
        if isinstance(cond, LogicalNode):
            if cond.op == "and":
                if jump_if_false:
                    # Jump to target (false-target) as soon as EITHER side
                    # is false -- that's exactly what evaluating left with
                    # jump_if_false=True does, then falling through to
                    # evaluate right the same way.
                    self.gen_cond_jump(cond.left, target, jump_if_false=True)
                    self.gen_cond_jump(cond.right, target, jump_if_false=True)
                else:
                    # Jump to target (true-target) only if BOTH are true:
                    # if left is false, skip past right entirely to a local
                    # 'skip' label; otherwise fall through and test right.
                    skip_l = self.new_label("and_skip")
                    self.gen_cond_jump(cond.left, skip_l, jump_if_false=True)
                    self.gen_cond_jump(cond.right, target, jump_if_false=False)
                    self.emit(f"{skip_l}:")
            else:  # "or"
                if jump_if_false:
                    # Jump to target (false-target) only if BOTH are false:
                    # if left is true, skip past right to a local label.
                    skip_l = self.new_label("or_skip")
                    self.gen_cond_jump(cond.left, skip_l, jump_if_false=False)
                    self.gen_cond_jump(cond.right, target, jump_if_false=True)
                    self.emit(f"{skip_l}:")
                else:
                    # Jump to target (true-target) as soon as EITHER is true.
                    self.gen_cond_jump(cond.left, target, jump_if_false=False)
                    self.gen_cond_jump(cond.right, target, jump_if_false=False)
            return

        if isinstance(cond, NotNode):
            # 'not X' simply flips which polarity we ask the inner
            # condition to jump on.
            self.gen_cond_jump(cond.cond, target, jump_if_false=not jump_if_false)
            return

        if isinstance(cond, BoolLitNode):
            # A bare 'true'/'false' used directly as a condition.
            is_true = cond.value
            should_jump = (not is_true) if jump_if_false else is_true
            if should_jump:
                self.emit(f"    jmp {target}")
            return

        # Base case: a plain comparison (CondNode).
        if cond.op in ("==", "!=") and (self._expr_is_string(cond.left) or self._expr_is_string(cond.right)):
            # String equality must compare CONTENT, not pointers -- two
            # different heap allocations (e.g. a literal vs. a str:slice
            # result) can hold identical text but never have the same
            # address. __streq does a byte-for-byte compare and returns
            # 1/0 in rax, which we then treat as an ordinary boolean.
            self.ensure_heap_helpers()
            self.gen_expr_to_rax(cond.right)
            self.emit("    push rax")
            self.gen_expr_to_rax(cond.left)
            self.emit("    pop rbx")
            self.emit("    mov rsi, rax")
            self.emit("    mov rdi, rbx")
            self.emit("    call __streq")     # rax = 1 if equal, 0 if not
            self.emit("    cmp rax, 1")
            jmap = {"==": ("jne", "je"), "!=": ("je", "jne")}
            false_j, true_j = jmap[cond.op]
            self.emit(f"    {false_j if jump_if_false else true_j} {target}")
            return

        self.gen_expr_to_rax(cond.right)
        self.emit("    push rax")
        self.gen_expr_to_rax(cond.left)
        self.emit("    pop rbx")
        self.emit("    cmp rax, rbx")

        false_jumps = {"==": "jne", "!=": "je", ">": "jle", "<": "jge", ">=": "jl", "<=": "jg"}
        true_jumps = {"==": "je", "!=": "jne", ">": "jg", "<": "jl", ">=": "jge", "<=": "jle"}

        jmap = false_jumps if jump_if_false else true_jumps
        self.emit(f"    {jmap[cond.op]} {target}")

    def gen_runtime_helpers(self):
        self.emit("")
        self.emit("; --- Runtime Helpers ---")

        if self.needs_error_handling:
            self.bss.append("__err_message: resq 1")   # pointer to the current error's text (set by raise / a built-in trap)
            self.bss.append(f"__try_stack: resq {self.MAX_TRY_DEPTH}")   # runtime stack of catch-label addresses
            self.bss.append("__try_sp: resq 1")          # number of entries currently on __try_stack

            # __try_push: rax = catch label address -> pushes it onto
            # __try_stack. Called at the start of every try block, before
            # its body runs, so any raise/trap during the body (even deep
            # inside a called function) knows where to jump.
            self.emit("__try_push:")
            self.emit("    push rbx")
            self.emit("    mov rbx, [__try_sp]")
            self.emit(f"    cmp rbx, {self.MAX_TRY_DEPTH}")
            self.emit("    jl .tp_ok")
            # Nested too deep -- extremely unlikely for hand-written
            # scripts, but fail loudly rather than silently corrupting
            # unrelated memory past the end of __try_stack.
            oflow_label, oflow_len = self.string_label("MyLand runtime error: try nesting too deep\n")
            self.emit(f"    mov rax, {SYS_WRITE}\n    mov rdi, {STDOUT}")
            self.emit(f"    mov rsi, {oflow_label}\n    mov rdx, {oflow_len}\n    syscall")
            self.emit("    mov rax, 60\n    mov rdi, 1\n    syscall")
            self.emit(".tp_ok:")
            self.emit("    mov [__try_stack + rbx*8], rax")
            self.emit("    inc qword [__try_sp]")
            self.emit("    pop rbx")
            self.emit("    ret")

            # __try_pop: removes the most recently pushed catch label
            # (called when a try's body finishes without raising, right
            # before falling through to skip its own catch handler).
            self.emit("__try_pop:")
            self.emit("    cmp qword [__try_sp], 0")
            self.emit("    je .tpp_done")   # defensive: never pop an empty stack
            self.emit("    dec qword [__try_sp]")
            self.emit(".tpp_done:")
            self.emit("    ret")

            # __raise_to_handler: __err_message is already set by the
            # caller (raise, or a built-in trap like divide-by-zero) --
            # pop the top-of-stack try (it's about to handle this error,
            # so it's no longer "active" for anything raised from within
            # its own catch_body) and jump to its catch label. An empty
            # stack means no enclosing try anywhere in the call chain, so
            # fall through to __fatal_error instead.
            self.emit("__raise_to_handler:")
            self.emit("    mov rax, [__try_sp]")
            self.emit("    cmp rax, 0")
            self.emit("    je __fatal_error")
            self.emit("    dec rax")
            self.emit("    mov [__try_sp], rax")
            self.emit("    mov rax, [__try_stack + rax*8]")
            self.emit("    jmp rax")

            # __fatal_error: an error was raised with no enclosing try to
            # catch it. Print "Uncaught error: <message>\n" to stdout and
            # exit(1) -- same exit code as stop:error, since an unhandled
            # runtime error is exactly that: the program couldn't continue.
            self.emit("__fatal_error:")
            prefix_label, prefix_len = self.string_label("Uncaught error: ")
            self.emit(f"    mov rax, {SYS_WRITE}\n    mov rdi, {STDOUT}")
            self.emit(f"    mov rsi, {prefix_label}\n    mov rdx, {prefix_len}")
            self.emit("    syscall")
            self.emit("    mov rsi, [__err_message]")
            self.emit("    call __strlen")            # rdx = length
            self.emit("    push rdx")
            self.emit(f"    mov rax, {SYS_WRITE}\n    mov rdi, {STDOUT}")
            self.emit("    mov rsi, [__err_message]")
            self.emit("    pop rdx")
            self.emit("    syscall")
            nl_label, nl_len = self.string_label("\n")
            self.emit(f"    mov rax, {SYS_WRITE}\n    mov rdi, {STDOUT}")
            self.emit(f"    mov rsi, {nl_label}\n    mov rdx, {nl_len}")
            self.emit("    syscall")
            self.emit("    mov rax, 60\n    mov rdi, 1\n    syscall")   # exit(1)

        self.emit("__strlen:")
        self.emit("    push rax\n    push rsi\n    xor rdx, rdx")
        self.emit(".loop:\n    mov al, [rsi + rdx]\n    cmp al, 0\n    je .done\n    inc rdx\n    jmp .loop")
        self.emit(".done:\n    pop rsi\n    pop rax\n    ret")

        # __streq: rdi = ptr A, rsi = ptr B -> rax = 1 if the two
        # NUL-terminated strings have identical content, 0 otherwise.
        # Byte-for-byte compare, stops at the first NUL (both strings end
        # there simultaneously only if every prior byte matched).
        self.emit("__streq:")
        self.emit("    push rcx\n    push rdx")
        self.emit("    xor rcx, rcx")
        self.emit(".seq_loop:")
        self.emit("    mov dl, [rdi + rcx]")
        self.emit("    cmp dl, [rsi + rcx]")
        self.emit("    jne .seq_false")
        self.emit("    cmp dl, 0")
        self.emit("    je .seq_true")
        self.emit("    inc rcx")
        self.emit("    jmp .seq_loop")
        self.emit(".seq_true:\n    mov rax, 1\n    jmp .seq_done")
        self.emit(".seq_false:\n    xor rax, rax")
        self.emit(".seq_done:\n    pop rdx\n    pop rcx\n    ret")

        self.ensure_numbuf()

        # __int_to_buf: rax (signed int) -> writes decimal text into NUMBUF,
        # returns rsi = pointer to first digit, rdx = length. Shared by
        # __print_int_rax (stdout) and file:write with a numeric variable.
        self.emit("__int_to_buf:")
        self.emit("    push rbx\n    push rcx\n    push rdi\n    push r11")
        self.emit(f"    lea rdi, [{self.NUMBUF_LABEL} + {self.NUMBUF_LEN - 1}]")
        self.emit("    mov byte [rdi], 0\n    mov rbx, 10\n    mov rcx, 0\n    mov r11, 0")
        self.emit("    cmp rax, 0\n    jge .ib_pos\n    mov r11, 1\n    neg rax")
        self.emit(".ib_pos:\n    cmp rax, 0\n    jne .ib_loop\n    dec rdi\n    mov byte [rdi], '0'\n    inc rcx\n    jmp .ib_sign")
        self.emit(".ib_loop:\n    cmp rax, 0\n    je .ib_sign\n    xor rdx, rdx\n    div rbx\n    add dl, '0'\n    dec rdi\n    mov [rdi], dl\n    inc rcx\n    jmp .ib_loop")
        self.emit(".ib_sign:\n    cmp r11, 1\n    jne .ib_write\n    dec rdi\n    mov byte [rdi], '-'\n    inc rcx")
        self.emit(".ib_write:\n    mov rsi, rdi\n    mov rdx, rcx")
        self.emit("    pop r11\n    pop rdi\n    pop rcx\n    pop rbx\n    ret")

        self.emit("__print_int_rax:")
        self.emit("    push rdi\n    push rsi\n    push rdx")
        self.emit("    call __int_to_buf")
        self.emit(f"    mov rax, {SYS_WRITE}\n    mov rdi, {STDOUT}\n    syscall")
        self.emit("    pop rdx\n    pop rsi\n    pop rdi\n    ret")

        self.emit("__atoi:")
        self.emit("    push rcx\n    push rdx\n    push rdi\n    xor rax, rax\n    xor rcx, rcx\n    xor rdx, rdx")
        self.emit("    cmp rbx, 0\n    je .done\n    mov dl, [rsi]\n    cmp dl, '-'\n    jne .loop_init\n    mov rdx, 1\n    mov rcx, 1\n    jmp .loop")
        self.emit(".loop_init:\n    xor rdx, rdx")
        self.emit(".loop:\n    cmp rcx, rbx\n    jge .neg_check\n    movzx rdi, byte [rsi + rcx]")
        self.emit("    cmp rdi, '0'\n    jl .neg_check\n    cmp rdi, '9'\n    jg .neg_check")
        self.emit("    imul rax, rax, 10\n    sub rdi, '0'\n    add rax, rdi\n    inc rcx\n    jmp .loop")
        self.emit(".neg_check:\n    cmp rdx, 1\n    jne .done\n    neg rax")
        self.emit(".done:\n    pop rdi\n    pop rdx\n    pop rcx\n    ret")

        self.gen_heap_helpers()

        if self.needs_dict_helpers:
            self.gen_dict_helpers()

    def gen_dict_helpers(self):
        """
        Dict layout (heap pointer points at the header):
          [0]  count     -- number of key-value pairs currently stored
          [8]  capacity  -- number of (key,val) slots currently allocated
          [16] key0_ptr  [24] val0
          [32] key1_ptr  [40] val1
          ...
        Keys are always heap string pointers, compared by content (never
        pointer identity) via __streq. Lookup is linear scan -- simple and
        correct, plenty fast for the dict sizes real scripts use.
        """
        PAIR = 16   # 8 bytes key ptr + 8 bytes value, per slot
        HDR = 16    # count (8) + capacity (8)

        # __dict_new: (no args) -> rax = pointer to a fresh empty dict.
        self.emit("__dict_new:")
        self.emit("    push rdi")
        self.emit(f"    mov rdi, {HDR + self.DICT_INITIAL_CAPACITY * PAIR}")
        self.emit("    call __heap_alloc")
        self.emit("    mov qword [rax], 0")                                  # count = 0
        self.emit(f"    mov qword [rax + 8], {self.DICT_INITIAL_CAPACITY}")  # capacity
        self.emit("    pop rdi")
        self.emit("    ret")

        # __dict_find_slot: rdi = dict ptr, rsi = key ptr
        #                -> rax = &key_i (matching slot), or 0 if not found.
        # Keeps the dict ptr and search key in r8/r9 (registers __streq
        # never touches) so nothing needs saving/restoring around the call,
        # apart from the slot address and loop index which do get clobbered.
        self.emit("__dict_find_slot:")
        self.emit("    push rcx\n    push rdx\n    push r8\n    push r9\n    push r10")
        self.emit("    mov r8, rdi")                    # r8 = dict ptr
        self.emit("    mov r9, rsi")                    # r9 = search key ptr
        self.emit("    mov rcx, [r8]")                   # rcx = count
        self.emit("    xor rdx, rdx")                    # rdx = index
        self.emit(".dfs_loop:")
        self.emit("    cmp rdx, rcx")
        self.emit("    jge .dfs_notfound")
        self.emit("    mov rax, rdx")
        self.emit(f"    imul rax, rax, {PAIR}")
        self.emit(f"    lea rax, [r8 + rax + {HDR}]")    # rax = &key_index for this slot
        self.emit("    mov rdi, [rax]")                   # candidate key ptr -> __streq operand A
        self.emit("    mov rsi, r9")                       # search key ptr    -> __streq operand B
        self.emit("    push rax")                           # save slot addr (call clobbers rax/rdi/rsi)
        self.emit("    push rdx")                           # save loop index
        self.emit("    call __streq")                        # rax = 1 if equal, 0 if not
        self.emit("    mov r10, rax")                         # stash result before popping back over rax
        self.emit("    pop rdx")                              # restore loop index
        self.emit("    pop rax")                              # restore slot addr
        self.emit("    cmp r10, 1")
        self.emit("    je .dfs_found")
        self.emit("    inc rdx")
        self.emit("    jmp .dfs_loop")
        self.emit(".dfs_found:")
        self.emit("    pop r10\n    pop r9\n    pop r8\n    pop rdx\n    pop rcx")
        self.emit("    ret")
        self.emit(".dfs_notfound:")
        self.emit("    xor rax, rax")
        self.emit("    pop r10\n    pop r9\n    pop r8\n    pop rdx\n    pop rcx")
        self.emit("    ret")

        # __dict_get: rdi = dict ptr, rsi = key ptr -> rax = value, or 0.
        self.emit("__dict_get:")
        self.emit("    call __dict_find_slot")
        self.emit("    cmp rax, 0")
        self.emit("    je .dg_notfound")
        self.emit("    mov rax, [rax + 8]")   # value sits 8 bytes after the key in its slot
        self.emit("    ret")
        self.emit(".dg_notfound:")
        self.emit("    xor rax, rax")
        self.emit("    ret")

        # __dict_set: rdi = dict ptr, rsi = key ptr, rdx = value
        #          -> rax = dict ptr (possibly reallocated by a resize --
        #             callers must write this back into their dict variable).
        self.emit("__dict_set:")
        self.emit("    push rbx\n    push r12\n    push r13\n    push rcx")
        self.emit("    mov rbx, rdi")     # rbx = dict ptr   (survives calls below)
        self.emit("    mov r12, rsi")     # r12 = key ptr    (survives calls below)
        self.emit("    mov r13, rdx")     # r13 = value      (survives calls below)
        self.emit("    call __dict_find_slot")     # rdi/rsi already hold dict/key from caller
        self.emit("    cmp rax, 0")
        self.emit("    je .ds_new_key")
        self.emit("    mov [rax + 8], r13")         # key exists -- overwrite value in place
        self.emit("    mov rax, rbx")
        self.emit("    pop rcx\n    pop r13\n    pop r12\n    pop rbx")
        self.emit("    ret")
        self.emit(".ds_new_key:")
        self.emit("    mov rcx, [rbx]")              # rcx = count
        self.emit("    cmp rcx, [rbx + 8]")           # count vs capacity
        self.emit("    jl .ds_has_room")
        self.emit("    mov rdi, rbx")
        self.emit("    call __dict_grow")             # rax = new dict ptr, doubled capacity, pairs copied
        self.emit("    mov rbx, rax")
        self.emit(".ds_has_room:")
        self.emit("    mov rcx, [rbx]")               # count (unchanged by grow, re-read for clarity)
        self.emit(f"    imul rax, rcx, {PAIR}")
        self.emit(f"    lea rax, [rbx + rax + {HDR}]") # rax = &key_count (first free slot)
        self.emit("    mov [rax], r12")                 # store key ptr
        self.emit("    mov [rax + 8], r13")              # store value
        self.emit("    inc qword [rbx]")                 # count += 1
        self.emit("    mov rax, rbx")
        self.emit("    pop rcx\n    pop r13\n    pop r12\n    pop rbx")
        self.emit("    ret")

        # __dict_grow: rdi = old dict ptr -> rax = new dict ptr with
        # doubled capacity, existing pairs copied over. Internal to
        # __dict_set; not called from generated program code directly.
        self.emit("__dict_grow:")
        self.emit("    push rbx\n    push r12\n    push rcx\n    push rdx")
        self.emit("    mov rbx, rdi")             # rbx = old dict ptr
        self.emit("    mov r12, [rbx]")            # r12 = old count (need it again after __heap_alloc)
        self.emit("    mov rcx, [rbx + 8]")         # rcx = old capacity
        self.emit("    imul rcx, rcx, 2")           # new capacity = old * 2
        self.emit("    push rcx")
        self.emit(f"    imul rdi, rcx, {PAIR}")
        self.emit(f"    add rdi, {HDR}")
        self.emit("    call __heap_alloc")           # rax = new dict ptr, zero-filled
        self.emit("    pop rcx")                      # rcx = new capacity
        self.emit("    mov [rax], r12")               # copy count
        self.emit("    mov [rax + 8], rcx")            # store new capacity
        self.emit("    push rax")
        self.emit(f"    imul rcx, r12, {PAIR}")          # rcx = byte length of existing pairs to copy
        self.emit(f"    lea rdi, [rax + {HDR}]")
        self.emit(f"    lea rsi, [rbx + {HDR}]")
        self.emit("    call __memcopy")
        self.emit("    pop rax")
        self.emit("    pop rdx\n    pop rcx\n    pop r12\n    pop rbx")
        self.emit("    ret")

        # __dict_delete: rdi = dict ptr, rsi = key ptr -> removes the pair
        # if present, shifting every later pair down one slot to keep
        # storage dense. No-op if the key isn't found.
        self.emit("__dict_delete:")
        self.emit("    push rbx\n    push r12\n    push rcx\n    push rdx")
        self.emit("    mov rbx, rdi")     # rbx = dict ptr
        self.emit("    call __dict_find_slot")     # rdi/rsi already hold dict/key from caller
        self.emit("    cmp rax, 0")
        self.emit("    je .dd_done")
        self.emit("    mov r12, rax")                # r12 = &deleted key slot
        # Number of trailing bytes to shift down = (bytes from just after
        # this slot's value, to the end of the last live pair).
        self.emit("    mov rcx, [rbx]")               # rcx = count
        self.emit(f"    imul rdx, rcx, {PAIR}")
        self.emit(f"    lea rdx, [rbx + rdx + {HDR}]") # rdx = end of live pairs (one-past-last)
        self.emit("    lea rdi, [r12]")                 # dest = the deleted slot
        self.emit(f"    lea rsi, [r12 + {PAIR}]")         # src  = the slot right after it
        self.emit("    sub rdx, rsi")                     # rdx = byte count to shift (0 if it was the last pair)
        self.emit("    cmp rdx, 0")
        self.emit("    jle .dd_no_shift")
        self.emit("    mov rcx, rdx")
        self.emit("    call __memcopy")
        self.emit(".dd_no_shift:")
        self.emit("    dec qword [rbx]")                  # count -= 1
        self.emit(".dd_done:")
        self.emit("    pop rdx\n    pop rcx\n    pop r12\n    pop rbx")
        self.emit("    ret")

        # __dict_key_at: rdi = dict ptr, rsi = index -> rax = the index-th
        # key (insertion order; note __dict_delete shifts later indices
        # down), or a pointer to an empty string if index is out of range.
        self.emit("__dict_key_at:")
        self.emit("    push rcx")
        self.emit("    mov rcx, [rdi]")     # rcx = count
        self.emit("    cmp rsi, 0")
        self.emit("    jl .dka_oob")
        self.emit("    cmp rsi, rcx")
        self.emit("    jge .dka_oob")
        self.emit(f"    imul rax, rsi, {PAIR}")
        self.emit(f"    lea rax, [rdi + rax + {HDR}]")
        self.emit("    mov rax, [rax]")      # rax = key ptr at this index
        self.emit("    pop rcx")
        self.emit("    ret")
        self.emit(".dka_oob:")
        empty_label, _ = self.string_label("")
        self.emit(f"    mov rax, {empty_label}")
        self.emit("    pop rcx")
        self.emit("    ret")

    def gen_heap_helpers(self):
        """
        __heap_init: reserves one big anonymous mmap region (HEAP_SIZE
        bytes) and stores its base pointer + current bump offset in .bss.
        Called once at program start.

        __heap_alloc: rdi = number of bytes requested -> rax = pointer to
        a zeroed block of at least that many bytes. Simple bump allocator
        (never frees individual blocks -- fine for a scripting-style
        language where programs are short-lived processes; the whole heap
        is reclaimed by the OS on exit). 8-byte aligned. Preserves all
        registers except rax (and flags), so callers don't need to save
        rdi/rsi/etc. around it purely because of this call.

        __memcopy: rdi = dest, rsi = src, rcx = length in bytes -> copies
        byte-by-byte, advances rdi and rsi past the copied region (so
        sequential calls can build up a buffer), preserves rax.
        """
        self.bss.append("__heap_base: resq 1")
        self.bss.append("__heap_off: resq 1")

        self.emit("__heap_init:")
        self.emit("    push rax\n    push rdi\n    push rsi\n    push rdx\n    push r10\n    push r8\n    push r9")
        self.emit("    xor rdi, rdi")                       # addr = NULL (let kernel choose)
        self.emit(f"    mov rsi, {self.HEAP_SIZE}")         # length
        self.emit(f"    mov rdx, {PROT_READ | PROT_WRITE}") # prot
        self.emit(f"    mov r10, {MAP_PRIVATE | MAP_ANONYMOUS}")  # flags
        self.emit("    mov r8, -1")                          # fd (ignored for anon)
        self.emit("    xor r9, r9")                          # offset
        self.emit(f"    mov rax, {SYS_MMAP}")
        self.emit("    syscall")
        self.emit("    mov [__heap_base], rax")
        self.emit("    mov qword [__heap_off], 0")
        self.emit("    pop r9\n    pop r8\n    pop r10\n    pop rdx\n    pop rsi\n    pop rdi\n    pop rax")
        self.emit("    ret")

        self.emit("__heap_alloc:")
        self.emit("    push rbx\n    push rcx\n    push rdx\n    push rdi")
        self.emit("    mov rbx, rdi")                # rbx = requested size
        self.emit("    add rbx, 7")
        self.emit("    and rbx, ~7")                 # round up to 8-byte alignment
        self.emit("    mov rax, [__heap_off]")
        self.emit("    mov rdx, [__heap_base]")
        self.emit("    add rax, rdx")                # rax = pointer to return (base + old offset)
        self.emit("    mov rcx, [__heap_off]")
        self.emit("    add rcx, rbx")
        self.emit(f"    cmp rcx, {self.HEAP_SIZE}")
        self.emit("    jle .ha_ok")
        # Heap exhausted -- rather than silently returning a bad pointer
        # (which would corrupt unrelated memory), fail loudly. A 16MB heap
        # is generous for a scripting-style program; hitting this usually
        # means an unbounded loop is allocating (e.g. string-concatenating
        # inside a large 'repeat').
        self.emit("    mov rax, 1\n    mov rdi, 2")   # write(stderr, ...)
        oom_label, oom_len = self.string_label("MyLand runtime error: out of heap memory\n")
        self.emit(f"    mov rsi, {oom_label}\n    mov rdx, {oom_len}\n    syscall")
        self.emit("    mov rax, 60\n    mov rdi, 1\n    syscall")  # exit(1)
        self.emit(".ha_ok:")
        self.emit("    mov [__heap_off], rcx")
        # Zero-fill the newly allocated block (rax=start, rbx=aligned size).
        self.emit("    push rax\n    push rdi\n    push rsi")
        self.emit("    mov rdi, rax\n    xor rsi, rsi\n    mov rcx, rbx")
        self.emit(".ha_zero:\n    cmp rsi, rcx\n    jge .ha_zero_done\n    mov byte [rdi + rsi], 0\n    inc rsi\n    jmp .ha_zero")
        self.emit(".ha_zero_done:")
        self.emit("    pop rsi\n    pop rdi\n    pop rax")
        self.emit("    pop rdi\n    pop rdx\n    pop rcx\n    pop rbx")
        self.emit("    ret")

        self.emit("__memcopy:")
        self.emit("    push rax\n    push rdx")
        self.emit("    xor rdx, rdx")
        self.emit(".mc_loop:\n    cmp rdx, rcx\n    jge .mc_done")
        self.emit("    mov al, [rsi + rdx]\n    mov [rdi + rdx], al\n    inc rdx\n    jmp .mc_loop")
        self.emit(".mc_done:")
        self.emit("    add rdi, rcx\n    add rsi, rcx")   # advance past the copied region
        self.emit("    pop rdx\n    pop rax")
        self.emit("    ret")
