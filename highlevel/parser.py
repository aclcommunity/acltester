# MyLand/highlevel/parser.py
from typing import List, Optional, Tuple
from tokens import TokenType, Token
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

class Parser:
    """
    Recursive-descent parser producing a Program AST Node from Tokens.
    Enforces strict block-ending semantics (end_repeat, end_loop, end_check, end_block).
    """

    def __init__(self, tokens: List[Token], filename: str = "<source>"):
        self.toks = tokens
        self.pos = 0
        self.filename = filename

    def cur(self) -> Token:
        return self.toks[self.pos]

    def at(self, kind: TokenType) -> bool:
        return self.cur().type == kind

    def advance(self) -> Token:
        t = self.cur()
        if self.pos < len(self.toks) - 1:
            self.pos += 1
        return t

    def expect(self, kind: TokenType, err_msg: str = "") -> Token:
        if not self.at(kind):
            t = self.cur()
            msg = err_msg if err_msg else f"Expected token {kind.name}, but got {t.type.name}"
            raise SyntaxError(f"Parser Error [{self.filename}:{t.line}:{t.col}]: {msg}")
        return self.advance()

    def skip_newlines(self):
        while self.at(TokenType.NEWLINE):
            self.advance()

    def parse(self) -> ProgramNode:
        stmts = []
        self.skip_newlines()
        while not self.at(TokenType.EOF):
            stmts.append(self.parse_statement())
            self.skip_newlines()
        return ProgramNode(statements=stmts)

    def parse_block_body(self, terminators: Tuple[TokenType, ...]) -> List[ASTNode]:
        body = []
        self.skip_newlines()
        while not any(self.at(t) for t in terminators) and not self.at(TokenType.EOF):
            body.append(self.parse_statement())
            self.skip_newlines()
        return body

    # ---------------- Statement Parsers ----------------

    def parse_statement(self) -> ASTNode:
        t = self.cur()

        if t.type == TokenType.LABEL:
            self.advance()
            return LabelDefNode(name=t.value, line=t.line, col=t.col)

        if t.type == TokenType.SET:
            return self.parse_set()
        if t.type == TokenType.SAY:
            return self.parse_say()
        if t.type in (TokenType.ASK, TokenType.INPUT_NUM):
            return self.parse_ask()
        if t.type == TokenType.REPEAT:
            return self.parse_repeat()
        if t.type == TokenType.LOOP_IF:
            return self.parse_loop_if()
        if t.type == TokenType.FOR:
            return self.parse_for()
        if t.type == TokenType.BREAK_LOOP:
            self.advance()
            return BreakLoopStmtNode(line=t.line, col=t.col)
        if t.type == TokenType.RETURN:
            return self.parse_return()
        if t.type == TokenType.CHECK:
            return self.parse_check()
        if t.type == TokenType.BLOCK:
            return self.parse_block_def()
        if t.type == TokenType.RUN:
            return self.parse_run()
        if t.type == TokenType.GOTO:
            return self.parse_goto()
        if t.type in (TokenType.STOP, TokenType.STOP_ERROR):
            return self.parse_stop()
        if t.type in (TokenType.FILE_OPEN, TokenType.FILE_WRITE, TokenType.FILE_READ, TokenType.FILE_CLOSE):
            return self.parse_file_op()
        if t.type == TokenType.DICT_SET:
            return self.parse_dict_set()
        if t.type == TokenType.DICT_DELETE:
            return self.parse_dict_delete()
        if t.type == TokenType.TRY:
            return self.parse_try()
        if t.type == TokenType.RAISE:
            return self.parse_raise()

        raise SyntaxError(f"Parser Error [{self.filename}:{t.line}:{t.col}]: Unknown statement starting with '{t.value}'")

    def parse_set(self) -> SetStmtNode:
        t = self.advance()  # set
        name_tok = self.expect(TokenType.IDENTIFIER, "Expected variable name after 'set'")

        index_expr = None
        if self.at(TokenType.LBRACKET):
            # Array-element assignment: set arr[i] = expr
            self.advance()  # [
            index_expr = self.parse_expr()
            self.expect(TokenType.RBRACKET, "Expected ']' after array index")

        compound_ops = {
            TokenType.PLUS_ASSIGN: "+", TokenType.MINUS_ASSIGN: "-",
            TokenType.STAR_ASSIGN: "*", TokenType.SLASH_ASSIGN: "/",
        }
        if self.cur().type in compound_ops:
            # Desugar 'x += expr' into 'x = x + expr' (and the array form
            # 'arr[i] += expr' into 'arr[i] = arr[i] + expr') right here at
            # parse time, so every later pass (semantic, codegen) only ever
            # sees plain SetStmtNode + BinOpNode -- no new node type, no new
            # code path to keep in sync with the rest of the compiler.
            op_tok = self.advance()
            op = compound_ops[op_tok.type]
            rhs = self.parse_expr()
            if index_expr is not None:
                current = IndexExprNode(array_name=name_tok.value, index_expr=index_expr, line=name_tok.line, col=name_tok.col)
            else:
                current = VarRefNode(name=name_tok.value, line=name_tok.line, col=name_tok.col)
            expr = BinOpNode(left=current, op=op, right=rhs, line=op_tok.line, col=op_tok.col)
            return SetStmtNode(name=name_tok.value, expr=expr, index_expr=index_expr, line=t.line, col=t.col)

        self.expect(TokenType.ASSIGN, "Expected '=' in set statement")
        expr = self.parse_expr()
        return SetStmtNode(name=name_tok.value, expr=expr, index_expr=index_expr, line=t.line, col=t.col)

    def parse_say(self) -> SayStmtNode:
        t = self.advance()  # say
        expr = self.parse_expr()
        return SayStmtNode(expr=expr, line=t.line, col=t.col)

    def parse_ask(self) -> AskStmtNode:
        t = self.advance()  # ask or input_num
        as_num = (t.type == TokenType.INPUT_NUM)
        var_tok = self.expect(TokenType.IDENTIFIER, "Expected variable name for input")
        return AskStmtNode(var_name=var_tok.value, as_number=as_num, line=t.line, col=t.col)

    def parse_repeat(self) -> RepeatStmtNode:
        t = self.advance()  # repeat
        count_expr = self.parse_expr()
        self.skip_newlines()
        body = self.parse_block_body((TokenType.END_REPEAT,))
        self.expect(TokenType.END_REPEAT, "Expected 'end_repeat' at the end of repeat block")
        return RepeatStmtNode(count_expr=count_expr, body=body, line=t.line, col=t.col)

    def parse_loop_if(self) -> LoopIfStmtNode:
        t = self.advance()  # loop_if
        cond = self.parse_cond()
        self.skip_newlines()
        body = self.parse_block_body((TokenType.END_LOOP,))
        self.expect(TokenType.END_LOOP, "Expected 'end_loop' at the end of loop_if block")
        return LoopIfStmtNode(condition=cond, body=body, line=t.line, col=t.col)

    def parse_for(self) -> ForStmtNode:
        t = self.advance()  # for
        var_tok = self.expect(TokenType.IDENTIFIER, "Expected loop variable name after 'for'")
        self.expect(TokenType.IN, "Expected 'in' after loop variable (e.g. for i in 0..10)")
        start_expr = self.parse_expr()
        self.expect(TokenType.DOTDOT, "Expected '..' between range start and end (e.g. for i in 0..10)")
        end_expr = self.parse_expr()
        self.skip_newlines()
        body = self.parse_block_body((TokenType.END_FOR,))
        self.expect(TokenType.END_FOR, "Expected 'end_for' at the end of for block")
        return ForStmtNode(var_name=var_tok.value, start_expr=start_expr, end_expr=end_expr, body=body, line=t.line, col=t.col)

    def parse_check(self) -> ASTNode:
        t = self.advance()  # check
        cond = self.parse_cond()

        # Inline jump form: check x == 10 -> goto ~label
        if self.at(TokenType.ARROW):
            self.advance()  # ->
            self.expect(TokenType.GOTO, "Expected 'goto' after '->'")
            lbl_tok = self.expect(TokenType.LABEL, "Expected '~label' after goto")
            return GotoStmtNode(label=lbl_tok.value, condition=cond, line=t.line, col=t.col)

        # Block form
        self.skip_newlines()
        then_body = self.parse_block_body((TokenType.OTHERWISE_CHECK, TokenType.OTHERWISE, TokenType.END_CHECK))

        elif_branches = []
        while self.at(TokenType.OTHERWISE_CHECK):
            self.advance()  # otherwise_check
            elif_cond = self.parse_cond()
            self.skip_newlines()
            elif_body = self.parse_block_body((TokenType.OTHERWISE_CHECK, TokenType.OTHERWISE, TokenType.END_CHECK))
            elif_branches.append((elif_cond, elif_body))

        else_body = []
        if self.at(TokenType.OTHERWISE):
            self.advance()
            self.skip_newlines()
            else_body = self.parse_block_body((TokenType.END_CHECK,))
        self.expect(TokenType.END_CHECK, "Expected 'end_check' at the end of check statement")
        return CheckStmtNode(condition=cond, then_body=then_body, elif_branches=elif_branches, else_body=else_body, line=t.line, col=t.col)

    def parse_block_def(self) -> BlockDefNode:
        t = self.advance()  # block
        name_tok = self.expect(TokenType.IDENTIFIER, "Expected block subroutine name")

        params = []
        if self.at(TokenType.LPAREN):
            # New form: block name(p1, p2) ... end_block
            self.advance()  # (
            if not self.at(TokenType.RPAREN):
                params.append(self.expect(TokenType.IDENTIFIER, "Expected parameter name").value)
                while self.at(TokenType.COMMA):
                    self.advance()
                    params.append(self.expect(TokenType.IDENTIFIER, "Expected parameter name").value)
            self.expect(TokenType.RPAREN, "Expected ')' after parameter list")

        self.skip_newlines()
        body = self.parse_block_body((TokenType.END_BLOCK,))
        self.expect(TokenType.END_BLOCK, "Expected 'end_block' at the end of block definition")
        return BlockDefNode(name=name_tok.value, body=body, params=params, line=t.line, col=t.col)

    def parse_run(self) -> RunStmtNode:
        t = self.advance()  # run
        name_tok = self.expect(TokenType.IDENTIFIER, "Expected block name to run")

        args = []
        if self.at(TokenType.LPAREN):
            # New form: run name(arg1, arg2)
            self.advance()  # (
            if not self.at(TokenType.RPAREN):
                args.append(self.parse_expr())
                while self.at(TokenType.COMMA):
                    self.advance()
                    args.append(self.parse_expr())
            self.expect(TokenType.RPAREN, "Expected ')' after argument list")

        return RunStmtNode(name=name_tok.value, args=args, line=t.line, col=t.col)

    def parse_return(self) -> ReturnStmtNode:
        t = self.advance()  # return
        expr = None
        if not self.at(TokenType.NEWLINE) and not self.at(TokenType.EOF):
            expr = self.parse_expr()
        return ReturnStmtNode(expr=expr, line=t.line, col=t.col)

    def parse_goto(self) -> GotoStmtNode:
        t = self.advance()  # goto
        lbl_tok = self.expect(TokenType.LABEL, "Expected label starting with '~' after goto")
        return GotoStmtNode(label=lbl_tok.value, line=t.line, col=t.col)

    def parse_stop(self) -> StopStmtNode:
        t = self.advance()  # stop / stop:error
        return StopStmtNode(is_error=(t.type == TokenType.STOP_ERROR), line=t.line, col=t.col)

    def parse_file_op(self) -> FileOpNode:
        t = self.advance()
        op_map = {
            TokenType.FILE_OPEN: "open",
            TokenType.FILE_WRITE: "write",
            TokenType.FILE_READ: "read",
            TokenType.FILE_CLOSE: "close"
        }
        args = []
        while not self.at(TokenType.NEWLINE) and not self.at(TokenType.EOF):
            args.append(self.parse_primary())
        return FileOpNode(op_type=op_map[t.type], args=args, line=t.line, col=t.col)

    def parse_dict_set(self) -> DictSetStmtNode:
        # dict:set d, key = expr
        t = self.advance()  # dict:set
        dict_expr = self.parse_primary()
        self.expect(TokenType.COMMA, "Expected ',' after dict in 'dict:set d, key = expr'")
        key_expr = self.parse_expr()
        self.expect(TokenType.ASSIGN, "Expected '=' in 'dict:set d, key = expr'")
        value_expr = self.parse_expr()
        return DictSetStmtNode(dict_expr=dict_expr, key_expr=key_expr, value_expr=value_expr, line=t.line, col=t.col)

    def parse_dict_delete(self) -> DictDeleteStmtNode:
        # dict:delete d, key
        t = self.advance()  # dict:delete
        dict_expr = self.parse_primary()
        self.expect(TokenType.COMMA, "Expected ',' after dict in 'dict:delete d, key'")
        key_expr = self.parse_expr()
        return DictDeleteStmtNode(dict_expr=dict_expr, key_expr=key_expr, line=t.line, col=t.col)

    def parse_try(self) -> TryStmtNode:
        # try ... catch err ... end_try
        t = self.advance()  # try
        self.skip_newlines()
        body = self.parse_block_body((TokenType.CATCH,))
        self.expect(TokenType.CATCH, "Expected 'catch err' after try block")
        err_tok = self.expect(TokenType.IDENTIFIER, "Expected an error variable name after 'catch'")
        self.skip_newlines()
        catch_body = self.parse_block_body((TokenType.END_TRY,))
        self.expect(TokenType.END_TRY, "Expected 'end_try' at the end of try/catch")
        return TryStmtNode(body=body, err_var=err_tok.value, catch_body=catch_body, line=t.line, col=t.col)

    def parse_raise(self) -> RaiseStmtNode:
        t = self.advance()  # raise
        expr = self.parse_expr()
        return RaiseStmtNode(expr=expr, line=t.line, col=t.col)

    # ---------------- Expression Parsers ----------------

    def parse_expr(self) -> ASTNode:
        return self.parse_add_sub()

    def parse_add_sub(self) -> ASTNode:
        left = self.parse_mul_div()
        while self.at(TokenType.PLUS) or self.at(TokenType.MINUS):
            op_tok = self.advance()
            right = self.parse_mul_div()
            left = BinOpNode(left=left, op=op_tok.value, right=right, line=op_tok.line, col=op_tok.col)
        return left

    def parse_mul_div(self) -> ASTNode:
        left = self.parse_primary()
        while self.at(TokenType.STAR) or self.at(TokenType.SLASH) or self.at(TokenType.PERCENT):
            op_tok = self.advance()
            right = self.parse_primary()
            left = BinOpNode(left=left, op=op_tok.value, right=right, line=op_tok.line, col=op_tok.col)
        return left

    def parse_primary(self) -> ASTNode:
        t = self.cur()
        if t.type == TokenType.NUMBER:
            self.advance()
            return NumberLitNode(value=int(t.value), line=t.line, col=t.col)
        if t.type == TokenType.STRING:
            self.advance()
            return StringLitNode(value=t.value, line=t.line, col=t.col)
        if t.type == TokenType.ARRAY:
            self.advance()  # array
            size_expr = self.parse_expr()
            return ArrayLitNode(size_expr=size_expr, line=t.line, col=t.col)
        if t.type == TokenType.TRUE:
            self.advance()
            return BoolLitNode(value=True, line=t.line, col=t.col)
        if t.type == TokenType.FALSE:
            self.advance()
            return BoolLitNode(value=False, line=t.line, col=t.col)
        if t.type == TokenType.TO_STRING:
            self.advance()
            self.expect(TokenType.LPAREN, "Expected '(' after 'to_string'")
            expr = self.parse_expr()
            self.expect(TokenType.RPAREN, "Expected ')' after to_string argument")
            return ToStringNode(expr=expr, line=t.line, col=t.col)
        if t.type == TokenType.TO_NUMBER:
            self.advance()
            self.expect(TokenType.LPAREN, "Expected '(' after 'to_number'")
            expr = self.parse_expr()
            self.expect(TokenType.RPAREN, "Expected ')' after to_number argument")
            return ToNumberNode(expr=expr, line=t.line, col=t.col)
        if t.type == TokenType.STR_LENGTH:
            self.advance()
            self.expect(TokenType.LPAREN, "Expected '(' after 'str:length'")
            expr = self.parse_expr()
            self.expect(TokenType.RPAREN, "Expected ')' after str:length argument")
            return StrLengthNode(expr=expr, line=t.line, col=t.col)
        if t.type == TokenType.STR_CHAR_AT:
            self.advance()
            self.expect(TokenType.LPAREN, "Expected '(' after 'str:char_at'")
            expr = self.parse_expr()
            self.expect(TokenType.COMMA, "Expected ',' between str:char_at arguments")
            idx = self.parse_expr()
            self.expect(TokenType.RPAREN, "Expected ')' after str:char_at arguments")
            return StrCharAtNode(expr=expr, index_expr=idx, line=t.line, col=t.col)
        if t.type == TokenType.STR_SLICE:
            self.advance()
            self.expect(TokenType.LPAREN, "Expected '(' after 'str:slice'")
            expr = self.parse_expr()
            self.expect(TokenType.COMMA, "Expected ',' after str:slice string argument")
            start = self.parse_expr()
            self.expect(TokenType.COMMA, "Expected ',' between str:slice start and end")
            end = self.parse_expr()
            self.expect(TokenType.RPAREN, "Expected ')' after str:slice arguments")
            return StrSliceNode(expr=expr, start_expr=start, end_expr=end, line=t.line, col=t.col)
        if t.type == TokenType.RANDOM:
            self.advance()
            self.expect(TokenType.LPAREN, "Expected '(' after 'random'")
            min_expr = self.parse_expr()
            self.expect(TokenType.COMMA, "Expected ',' between random min and max")
            max_expr = self.parse_expr()
            self.expect(TokenType.RPAREN, "Expected ')' after random arguments")
            return RandomNode(min_expr=min_expr, max_expr=max_expr, line=t.line, col=t.col)
        if t.type == TokenType.ARRAY_LENGTH:
            self.advance()
            self.expect(TokenType.LPAREN, "Expected '(' after 'array:length'")
            expr = self.parse_expr()
            self.expect(TokenType.RPAREN, "Expected ')' after array:length argument")
            return ArrayLengthNode(expr=expr, line=t.line, col=t.col)
        if t.type == TokenType.DICT:
            self.advance()
            return DictLitNode(line=t.line, col=t.col)
        if t.type == TokenType.DICT_GET:
            self.advance()
            self.expect(TokenType.LPAREN, "Expected '(' after 'dict:get'")
            d = self.parse_expr()
            self.expect(TokenType.COMMA, "Expected ',' between dict:get arguments")
            key = self.parse_expr()
            self.expect(TokenType.RPAREN, "Expected ')' after dict:get arguments")
            return DictGetNode(dict_expr=d, key_expr=key, line=t.line, col=t.col)
        if t.type == TokenType.DICT_HAS:
            self.advance()
            self.expect(TokenType.LPAREN, "Expected '(' after 'dict:has'")
            d = self.parse_expr()
            self.expect(TokenType.COMMA, "Expected ',' between dict:has arguments")
            key = self.parse_expr()
            self.expect(TokenType.RPAREN, "Expected ')' after dict:has arguments")
            return DictHasNode(dict_expr=d, key_expr=key, line=t.line, col=t.col)
        if t.type == TokenType.DICT_LENGTH:
            self.advance()
            self.expect(TokenType.LPAREN, "Expected '(' after 'dict:length'")
            expr = self.parse_expr()
            self.expect(TokenType.RPAREN, "Expected ')' after dict:length argument")
            return DictLengthNode(expr=expr, line=t.line, col=t.col)
        if t.type == TokenType.DICT_KEY_AT:
            self.advance()
            self.expect(TokenType.LPAREN, "Expected '(' after 'dict:key_at'")
            d = self.parse_expr()
            self.expect(TokenType.COMMA, "Expected ',' between dict:key_at arguments")
            idx = self.parse_expr()
            self.expect(TokenType.RPAREN, "Expected ')' after dict:key_at arguments")
            return DictKeyAtNode(dict_expr=d, index_expr=idx, line=t.line, col=t.col)
        if t.type == TokenType.RUN:
            # Function-call-as-expression: set x = run total(a, b)
            self.advance()  # run
            name_tok = self.expect(TokenType.IDENTIFIER, "Expected block name to call")
            args = []
            self.expect(TokenType.LPAREN, "Expected '(' after block name in a call expression (use 'run name(...)' inside an expression)")
            if not self.at(TokenType.RPAREN):
                args.append(self.parse_expr())
                while self.at(TokenType.COMMA):
                    self.advance()
                    args.append(self.parse_expr())
            self.expect(TokenType.RPAREN, "Expected ')' after argument list")
            return CallExprNode(name=name_tok.value, args=args, line=t.line, col=t.col)
        if t.type == TokenType.IDENTIFIER:
            self.advance()
            if self.at(TokenType.LBRACKET):
                # Array indexing: arr[i]
                self.advance()  # [
                index_expr = self.parse_expr()
                self.expect(TokenType.RBRACKET, "Expected ']' after array index")
                return IndexExprNode(array_name=t.value, index_expr=index_expr, line=t.line, col=t.col)
            return VarRefNode(name=t.value, line=t.line, col=t.col)
        if t.type == TokenType.LPAREN:
            self.advance()  # (
            expr = self.parse_expr()
            self.expect(TokenType.RPAREN, "Expected ')' to close '('")
            return expr
        raise SyntaxError(f"Parser Error [{self.filename}:{t.line}:{t.col}]: Unexpected expression token '{t.value}'")

    def parse_cond(self) -> ASTNode:
        """Grammar (lowest to highest precedence):
        cond_or  := cond_and (OR cond_and)*
        cond_and := cond_not (AND cond_not)*
        cond_not := NOT cond_not | cond_atom
        cond_atom := '(' cond_or ')' | comparison
        """
        return self.parse_cond_or()

    def parse_cond_or(self) -> ASTNode:
        left = self.parse_cond_and()
        while self.at(TokenType.OR):
            op_tok = self.advance()
            right = self.parse_cond_and()
            left = LogicalNode(left=left, op="or", right=right, line=op_tok.line, col=op_tok.col)
        return left

    def parse_cond_and(self) -> ASTNode:
        left = self.parse_cond_not()
        while self.at(TokenType.AND):
            op_tok = self.advance()
            right = self.parse_cond_not()
            left = LogicalNode(left=left, op="and", right=right, line=op_tok.line, col=op_tok.col)
        return left

    def parse_cond_not(self) -> ASTNode:
        if self.at(TokenType.NOT):
            t = self.advance()
            inner = self.parse_cond_not()
            return NotNode(cond=inner, line=t.line, col=t.col)
        return self.parse_cond_atom()

    def parse_cond_atom(self) -> ASTNode:
        if self.at(TokenType.TRUE) or self.at(TokenType.FALSE):
            # A bare boolean literal used directly as a condition (e.g.
            # 'check true' or 'loop_if false') -- not part of a comparison.
            t = self.advance()
            return BoolLitNode(value=(t.type == TokenType.TRUE), line=t.line, col=t.col)
        if self.at(TokenType.LPAREN):
            # Could be a parenthesized compound condition '(a == 1 and b == 2)'
            # or just a parenthesized arithmetic sub-expression inside a
            # plain comparison ((a+1) == 2). We speculatively try parsing
            # as a compound condition first; if what follows the matching
            # ')' isn't a comparison operator, it's just a normal expression
            # and parse_comparison (which calls parse_expr) will re-parse it.
            save_pos = self.pos
            self.advance()  # (
            inner = self.parse_cond_or()
            if self.at(TokenType.RPAREN):
                closing_pos = self.pos
                self.advance()  # )
                # If a comparison operator follows, this parenthesized group
                # was actually just a sub-expression of a larger comparison
                # (e.g. '(a+1) == 2') -- rewind and let parse_comparison
                # handle it via the normal expression grammar instead.
                cmp_types = (TokenType.EQ, TokenType.NEQ, TokenType.GT, TokenType.LT, TokenType.GTE, TokenType.LTE)
                if self.cur().type in cmp_types:
                    self.pos = save_pos
                    return self.parse_comparison()
                return inner
            self.pos = save_pos
            return self.parse_comparison()
        return self.parse_comparison()

    def parse_comparison(self) -> ASTNode:
        left = self.parse_expr()
        cmp_types = (TokenType.EQ, TokenType.NEQ, TokenType.GT, TokenType.LT, TokenType.GTE, TokenType.LTE)
        if self.cur().type not in cmp_types:
            # No comparison operator follows -- treat the bare expression as
            # an implicit truthiness check (same convention as C/Python:
            # 'check flag' means 'check flag != 0'). This covers bare
            # variables, function calls, and arithmetic expressions used
            # directly as a condition without an explicit '== 0' etc.
            return CondNode(left=left, op="!=", right=NumberLitNode(value=0, line=left.line, col=left.col), line=left.line, col=left.col)
        op_tok = self.advance()
        right = self.parse_expr()
        return CondNode(left=left, op=op_tok.value, right=right, line=op_tok.line, col=op_tok.col)
