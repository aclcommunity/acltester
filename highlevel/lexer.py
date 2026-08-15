# MyLand/highlevel/lexer.py
from tokens import TokenType, Token

KEYWORDS = {
    "set": TokenType.SET,
    "say": TokenType.SAY,
    "ask": TokenType.ASK,
    "input_num": TokenType.INPUT_NUM,
    "repeat": TokenType.REPEAT,
    "end_repeat": TokenType.END_REPEAT,
    "loop_if": TokenType.LOOP_IF,
    "end_loop": TokenType.END_LOOP,
    "break_loop": TokenType.BREAK_LOOP,
    "check": TokenType.CHECK,
    "otherwise_check": TokenType.OTHERWISE_CHECK,
    "otherwise": TokenType.OTHERWISE,
    "end_check": TokenType.END_CHECK,
    "and": TokenType.AND,
    "or": TokenType.OR,
    "not": TokenType.NOT,
    "true": TokenType.TRUE,
    "false": TokenType.FALSE,
    "for": TokenType.FOR,
    "in": TokenType.IN,
    "end_for": TokenType.END_FOR,
    "block": TokenType.BLOCK,
    "end_block": TokenType.END_BLOCK,
    "run": TokenType.RUN,
    "goto": TokenType.GOTO,
    "stop": TokenType.STOP,
    "stop:error": TokenType.STOP_ERROR,
    "file:open": TokenType.FILE_OPEN,
    "file:write": TokenType.FILE_WRITE,
    "file:read": TokenType.FILE_READ,
    "file:close": TokenType.FILE_CLOSE,
    "return": TokenType.RETURN,
    "array": TokenType.ARRAY,
    "to_string": TokenType.TO_STRING,
    "to_number": TokenType.TO_NUMBER,
    "str:length": TokenType.STR_LENGTH,
    "str:char_at": TokenType.STR_CHAR_AT,
    "str:slice": TokenType.STR_SLICE,
    "random": TokenType.RANDOM,
    "array:length": TokenType.ARRAY_LENGTH,
    "dict": TokenType.DICT,
    "dict:get": TokenType.DICT_GET,
    "dict:set": TokenType.DICT_SET,
    "dict:has": TokenType.DICT_HAS,
    "dict:delete": TokenType.DICT_DELETE,
    "dict:length": TokenType.DICT_LENGTH,
    "dict:key_at": TokenType.DICT_KEY_AT,
    "try": TokenType.TRY,
    "catch": TokenType.CATCH,
    "end_try": TokenType.END_TRY,
    "raise": TokenType.RAISE,
}

class Lexer:
    def __init__(self, source: str, filename: str = "<source>"):
        self.src = source
        self.filename = filename
        self.pos = 0
        self.line = 1
        self.col = 1
        self.length = len(source)
        self.tokens = []

    def error(self, msg: str):
        raise SyntaxError(f"Lexer Error [{self.filename}:{self.line}:{self.col}]: {msg}")

    def peek(self, offset: int = 0) -> str:
        p = self.pos + offset
        if p < self.length:
            return self.src[p]
        return ""

    def advance(self) -> str:
        ch = self.src[self.pos]
        self.pos += 1
        if ch == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def tokenize(self):
        while self.pos < self.length:
            ch = self.peek()

            # Newline token
            if ch == "\n":
                start_line, start_col = self.line, self.col
                self.advance()
                self.tokens.append(Token(TokenType.NEWLINE, "\n", start_line, start_col))
                continue

            # Skip spaces & tabs
            if ch in " \t\r":
                self.advance()
                continue

            # Skip comments (# or //)
            if ch == "#" or (ch == "/" and self.peek(1) == "/"):
                while self.pos < self.length and self.peek() != "\n":
                    self.advance()
                continue

            # Strings
            if ch == '"':
                self.tokens.append(self._read_string())
                continue

            # Labels: ~label_name
            if ch == "~":
                self.tokens.append(self._read_label())
                continue

            # Numbers
            if ch.isdigit() or (ch == "-" and self.peek(1).isdigit() and self._is_unary_minus()):
                self.tokens.append(self._read_number())
                continue

            # Directives: @mode
            if ch == "@":
                self.tokens.append(self._read_directive())
                continue

            # Identifiers / Keywords / Namespaces
            if ch.isalpha() or ch == "_":
                self.tokens.append(self._read_ident_or_keyword())
                continue

            # Operators
            if self.src.startswith("..", self.pos):
                sl, sc = self.line, self.col
                self.advance(); self.advance()
                self.tokens.append(Token(TokenType.DOTDOT, "..", sl, sc))
                continue
            if self.src.startswith("->", self.pos):
                sl, sc = self.line, self.col
                self.advance(); self.advance()
                self.tokens.append(Token(TokenType.ARROW, "->", sl, sc))
                continue
            if self.src.startswith("==", self.pos):
                sl, sc = self.line, self.col
                self.advance(); self.advance()
                self.tokens.append(Token(TokenType.EQ, "==", sl, sc))
                continue
            if self.src.startswith("!=", self.pos):
                sl, sc = self.line, self.col
                self.advance(); self.advance()
                self.tokens.append(Token(TokenType.NEQ, "!=", sl, sc))
                continue
            if self.src.startswith(">=", self.pos):
                sl, sc = self.line, self.col
                self.advance(); self.advance()
                self.tokens.append(Token(TokenType.GTE, ">=", sl, sc))
                continue
            if self.src.startswith("<=", self.pos):
                sl, sc = self.line, self.col
                self.advance(); self.advance()
                self.tokens.append(Token(TokenType.LTE, "<=", sl, sc))
                continue
            if self.src.startswith("+=", self.pos):
                sl, sc = self.line, self.col
                self.advance(); self.advance()
                self.tokens.append(Token(TokenType.PLUS_ASSIGN, "+=", sl, sc))
                continue
            if self.src.startswith("-=", self.pos):
                sl, sc = self.line, self.col
                self.advance(); self.advance()
                self.tokens.append(Token(TokenType.MINUS_ASSIGN, "-=", sl, sc))
                continue
            if self.src.startswith("*=", self.pos):
                sl, sc = self.line, self.col
                self.advance(); self.advance()
                self.tokens.append(Token(TokenType.STAR_ASSIGN, "*=", sl, sc))
                continue
            if self.src.startswith("/=", self.pos):
                sl, sc = self.line, self.col
                self.advance(); self.advance()
                self.tokens.append(Token(TokenType.SLASH_ASSIGN, "/=", sl, sc))
                continue

            # Single character operators
            single_ops = {
                "=": TokenType.ASSIGN, "+": TokenType.PLUS, "-": TokenType.MINUS,
                "*": TokenType.STAR, "/": TokenType.SLASH, "%": TokenType.PERCENT,
                ">": TokenType.GT, "<": TokenType.LT,
                "[": TokenType.LBRACKET, "]": TokenType.RBRACKET,
                "(": TokenType.LPAREN, ")": TokenType.RPAREN,
                ",": TokenType.COMMA,
            }
            if ch in single_ops:
                sl, sc = self.line, self.col
                self.advance()
                self.tokens.append(Token(single_ops[ch], ch, sl, sc))
                continue

            self.error(f"Unexpected character '{ch}'")

        self.tokens.append(Token(TokenType.EOF, "", self.line, self.col))
        return self.tokens

    def _is_unary_minus(self) -> bool:
        if not self.tokens:
            return True
        last = self.tokens[-1].type
        return last in (
            TokenType.NEWLINE, TokenType.ASSIGN, TokenType.PLUS, TokenType.MINUS,
            TokenType.STAR, TokenType.SLASH, TokenType.IN, TokenType.DOTDOT,
            TokenType.LPAREN, TokenType.COMMA, TokenType.EQ, TokenType.NEQ,
            TokenType.GT, TokenType.LT, TokenType.GTE, TokenType.LTE,
            TokenType.AND, TokenType.OR, TokenType.NOT,
        )

    def _read_string(self) -> Token:
        sl, sc = self.line, self.col
        self.advance()
        out = []
        while True:
            if self.pos >= self.length:
                self.error("Unterminated string literal")
            ch = self.advance()
            if ch == '"':
                break
            if ch == "\\":
                esc = self.advance()
                mapping = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"'}
                out.append(mapping.get(esc, esc))
            else:
                out.append(ch)
        return Token(TokenType.STRING, "".join(out), sl, sc)

    def _read_label(self) -> Token:
        sl, sc = self.line, self.col
        self.advance()
        out = []
        while self.pos < self.length and (self.peek().isalnum() or self.peek() == "_"):
            out.append(self.advance())
        if not out:
            self.error("Expected label name after '~'")
        return Token(TokenType.LABEL, "".join(out), sl, sc)

    def _read_number(self) -> Token:
        sl, sc = self.line, self.col
        out = []
        if self.peek() == "-":
            out.append(self.advance())
        while self.pos < self.length and self.peek().isdigit():
            out.append(self.advance())
        return Token(TokenType.NUMBER, "".join(out), sl, sc)

    def _read_directive(self) -> Token:
        sl, sc = self.line, self.col
        self.advance()
        out = ["@"]
        while self.pos < self.length and (self.peek().isalnum() or self.peek() == "_"):
            out.append(self.advance())
        return Token(TokenType.MODE, "".join(out), sl, sc)

    def _read_ident_or_keyword(self) -> Token:
        sl, sc = self.line, self.col
        out = []
        while self.pos < self.length and (self.peek().isalnum() or self.peek() == "_"):
            out.append(self.advance())
        
        # Handle namespaces like file:open
        if self.peek() == ":" and (self.peek(1).isalpha() or self.peek(1) == "_"):
            out.append(self.advance())
            while self.pos < self.length and (self.peek().isalnum() or self.peek() == "_"):
                out.append(self.advance())

        val = "".join(out)
        ttype = KEYWORDS.get(val, TokenType.IDENTIFIER)
        return Token(ttype, val, sl, sc)
