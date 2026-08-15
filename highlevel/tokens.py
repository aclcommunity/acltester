# MyLand/highlevel/tokens.py
from enum import Enum, auto

class TokenType(Enum):
    # Directives
    MODE = auto()          # @mode

    # Unique Keywords
    SET = auto()           # set
    SAY = auto()           # say
    ASK = auto()           # ask
    INPUT_NUM = auto()     # input_num
    
    # Loops
    REPEAT = auto()        # repeat
    END_REPEAT = auto()    # end_repeat
    LOOP_IF = auto()       # loop_if
    END_LOOP = auto()      # end_loop
    BREAK_LOOP = auto()    # break_loop
    FOR = auto()           # for
    IN = auto()            # in
    END_FOR = auto()       # end_for
    DOTDOT = auto()        # ..
    
    # Conditionals
    CHECK = auto()         # check
    OTHERWISE_CHECK = auto()  # otherwise_check (elif)
    OTHERWISE = auto()     # otherwise
    END_CHECK = auto()     # end_check

    # Logical operators
    AND = auto()           # and
    OR = auto()            # or
    NOT = auto()           # not
    TRUE = auto()          # true
    FALSE = auto()         # false
    
    # Subroutines / Blocks
    BLOCK = auto()         # block
    END_BLOCK = auto()     # end_block
    RUN = auto()           # run
    
    # Control Flow & Jumps
    GOTO = auto()          # goto
    LABEL = auto()         # ~label_name
    STOP = auto()          # stop
    STOP_ERROR = auto()    # stop:error
    
    # File I/O Namespaces
    FILE_OPEN = auto()     # file:open
    FILE_WRITE = auto()    # file:write
    FILE_READ = auto()     # file:read
    FILE_CLOSE = auto()    # file:close

    # Functions (params + return value)
    RETURN = auto()        # return expr

    # Built-in string/number conversion & string ops
    TO_STRING = auto()     # to_string(expr)
    TO_NUMBER = auto()     # to_number(expr)
    STR_LENGTH = auto()    # str:length(expr)
    STR_CHAR_AT = auto()   # str:char_at(expr, idx)
    STR_SLICE = auto()     # str:slice(expr, start, end)
    RANDOM = auto()         # random(min, max)
    ARRAY_LENGTH = auto()   # array:length(expr)

    # Dictionaries
    DICT = auto()           # dict   (empty dict constructor)
    DICT_GET = auto()       # dict:get(d, key)
    DICT_SET = auto()       # dict:set d, key = expr
    DICT_HAS = auto()       # dict:has(d, key)
    DICT_DELETE = auto()    # dict:delete d, key
    DICT_LENGTH = auto()    # dict:length(d)
    DICT_KEY_AT = auto()    # dict:key_at(d, i)

    # Error handling
    TRY = auto()            # try
    CATCH = auto()          # catch err
    END_TRY = auto()        # end_try
    RAISE = auto()          # raise expr

    # Arrays
    ARRAY = auto()         # array N   (array literal / constructor)
    LBRACKET = auto()      # [
    RBRACKET = auto()      # ]

    # Call-site / definition delimiters for params/args
    LPAREN = auto()        # (
    RPAREN = auto()        # )
    COMMA = auto()         # ,

    # Literals & Identifiers
    IDENTIFIER = auto()    # variable or block names
    NUMBER = auto()        # integer literals
    STRING = auto()        # string literals
    
    # Operators & Delimiters
    ASSIGN = auto()        # =
    PLUS_ASSIGN = auto()   # +=
    MINUS_ASSIGN = auto()  # -=
    STAR_ASSIGN = auto()   # *=
    SLASH_ASSIGN = auto()  # /=
    PLUS = auto()          # +
    MINUS = auto()         # -
    STAR = auto()          # *
    SLASH = auto()         # /
    PERCENT = auto()       # %
    
    # Comparisons
    EQ = auto()            # ==
    NEQ = auto()           # !=
    GT = auto()            # >
    LT = auto()            # <
    GTE = auto()           # >=
    LTE = auto()           # <=
    ARROW = auto()         # ->
    
    # Misc
    NEWLINE = auto()       # \n
    EOF = auto()           # End of file


class Token:
    """Represents a single token with its type, value, line, and column info."""
    def __init__(self, type_: TokenType, value: str, line: int, col: int):
        self.type = type_
        self.value = value
        self.line = line
        self.col = col

    def __repr__(self):
        return f"Token({self.type.name}, {repr(self.value)}, L{self.line}:C{self.col})"
