# MyLand/highlevel/test.py
import sys
import os
import subprocess
import tempfile

# Local imports from the same directory
from lexer import Lexer
from parser import Parser
from semantic import SemanticAnalyzer
from codegen import CodeGenerator

def run_highlevel_test(source_file_path: str):
    if not os.path.exists(source_file_path):
        print(f"\033[1;31mError\033[0m: Source file '{source_file_path}' not found!")
        return

    with open(source_file_path, "r", encoding="utf-8") as f:
        source_code = f.read()

    print(f"\033[1;34m[MyLand High-Level Engine]\033[0m Running test on '{source_file_path}'...\n")

    # 1. Lexical Analysis
    try:
        lexer = Lexer(source_code, filename=source_file_path)
        tokens = lexer.tokenize()
        print("  \033[1;32m[1/4]\033[0m Lexing completed successfully.")
    except SyntaxError as e:
        print(f"\033[1;31m{e}\033[0m")
        return

    # 2. Parsing (AST Construction)
    try:
        parser = Parser(tokens, filename=source_file_path)
        ast = parser.parse()
        print("  \033[1;32m[2/4]\033[0m Parsing completed (AST generated).")
    except SyntaxError as e:
        print(f"\033[1;31m{e}\033[0m")
        return

    # 3. Semantic Analysis
    analyzer = SemanticAnalyzer(filename=source_file_path)
    if not analyzer.analyze(ast):
        print("\033[1;31mSemantic Analysis Failed.\033[0m")
        return
    print("  \033[1;32m[3/4]\033[0m Semantic Analysis passed.")

    # 4. Assembly Generation & Execution
    codegen = CodeGenerator(filename=source_file_path)
    asm_code = codegen.generate(ast)
    print("  \033[1;32m[4/4]\033[0m x86_64 NASM Assembly generated.")

    # Temp build and execution
    base_name = os.path.splitext(os.path.basename(source_file_path))[0]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        asm_path = os.path.join(tmpdir, f"{base_name}.asm")
        obj_path = os.path.join(tmpdir, f"{base_name}.o")
        bin_path = os.path.join(tmpdir, f"{base_name}_exec")

        with open(asm_path, "w", encoding="utf-8") as f:
            f.write(asm_code)

        try:
            # Assemble and link natively
            subprocess.run(["nasm", "-f", "elf64", asm_path, "-o", obj_path], check=True, capture_output=True)
            subprocess.run(["ld", obj_path, "-o", bin_path], check=True, capture_output=True)
            os.chmod(bin_path, 0o755)

            print("\n\033[1;35m================= OUTPUT =================\033[0m")
            subprocess.run([bin_path])
            print("\033[1;35m==========================================\033[0m\n")

        except subprocess.CalledProcessError as e:
            print(f"\033[1;31mBuild Error\033[0m:\n{e.stderr.decode()}")
        except FileNotFoundError:
            print("\033[1;31mError\033[0m: 'nasm' or 'ld' not installed. Install via `sudo apt install nasm binutils`.")

if __name__ == "__main__":
    # Pointing to test.acl in parent directory (MyLand/test.acl)
    target_acl = os.path.join(os.path.dirname(__file__), "..", "test.acl")
    
    if len(sys.argv) > 1:
        target_acl = sys.argv[1]

    run_highlevel_test(target_acl)
