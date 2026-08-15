# MyLand/compiler.py
import sys
import os
import subprocess
import tempfile
import argparse

# Subfolders ko import path mein add kar rahe hain
sys.path.append(os.path.join(os.path.dirname(__file__), "highlevel"))
sys.path.append(os.path.join(os.path.dirname(__file__), "lowlevel"))

# High-Level Pipeline Imports
from highlevel.lexer import Lexer as HighLexer
from highlevel.parser import Parser as HighParser
from highlevel.semantic import SemanticAnalyzer as HighSemantic
from highlevel.codegen import CodeGenerator as HighCodeGen

# Low-Level & Baremetal Pipeline Imports
from lowlevel.lexer import LowLexer
from lowlevel.parser import LowParser
from lowlevel.semantic import LowSemanticAnalyzer as LowSemantic
from lowlevel.codegen import LowCodeGenerator


class MyLandCompiler:
    def __init__(self, source_path: str):
        self.source_path = source_path
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source file '{source_path}' not found!")

        with open(source_path, "r", encoding="utf-8") as f:
            self.source_code = f.read()

    def compile_and_run(self, output_bin: str = None, keep_asm: bool = False):
        print(f"\033[1;34m[MyLand Master Compiler]\033[0m Processing '{self.source_path}'...")

        # Mode Detect Karein (@mode: high, @mode: low, @mode: baremetal)
        blocks = self._parse_mode_blocks(self.source_code)
        
        combined_asm = []
        is_baremetal = False

        for idx, (mode, code_chunk) in enumerate(blocks):
            if not code_chunk.strip():
                continue

            if mode == "baremetal":
                is_baremetal = True

            print(f"  \033[1;33m[Block {idx+1}]\033[0m Compiling in \033[1;36m@{mode}\033[0m mode...")

            if mode == "high":
                asm = self._compile_high(code_chunk)
            else:
                asm = self._compile_low(code_chunk, mode)

            if asm is None:
                print("\033[1;31mCompilation aborted due to errors.\033[0m")
                return None, False

            combined_asm.append(asm)

        final_asm = "\n\n".join(combined_asm)

        # Output Executable ya Binary Path
        base_name = os.path.splitext(os.path.basename(self.source_path))[0]
        if output_bin:
            # Absolute paths aur paths jinme already a directory separator ho
            # unhe as-is use karo; sirf bare filenames ko './' se prefix karo.
            if os.path.isabs(output_bin) or os.path.dirname(output_bin):
                out_binary_name = output_bin
            else:
                out_binary_name = f"./{output_bin}"
        else:
           out_binary_name = f"./{base_name}.bin" if is_baremetal else f"./{base_name}"

        with tempfile.TemporaryDirectory() as tmpdir:
            asm_file = os.path.join(tmpdir, f"{base_name}.asm")
            obj_file = os.path.join(tmpdir, f"{base_name}.o")

            with open(asm_file, "w", encoding="utf-8") as f:
                f.write(final_asm)

            if keep_asm:
                saved_asm = f"{base_name}.asm"
                with open(saved_asm, "w", encoding="utf-8") as f:
                    f.write(final_asm)
                print(f"  \033[1;32m[ASM Saved]\033[0m Assembly written to '{saved_asm}'")

            try:
                if is_baremetal:
                    # Bare-metal compilation (32-bit Multiboot ELF/Binary)
                    subprocess.run(["nasm", "-f", "elf32", asm_file, "-o", obj_file], check=True, capture_output=True)
                    subprocess.run(["ld", "-m", "elf_i386", "-Ttext", "0x100000", obj_file, "-o", out_binary_name], check=True, capture_output=True)
                    print(f"  \033[1;32m[Build Success]\033[0m Bare-metal Kernel created: \033[1;32m{out_binary_name}\033[0m")
                else:
                    # Linux x86_64 compilation
                    subprocess.run(["nasm", "-f", "elf64", asm_file, "-o", obj_file], check=True, capture_output=True)
                    subprocess.run(["ld", obj_file, "-o", out_binary_name], check=True, capture_output=True)
                    os.chmod(out_binary_name, 0o755)
                    print(f"  \033[1;32m[Build Success]\033[0m Executable created: \033[1;32m{out_binary_name}\033[0m")

                return out_binary_name, is_baremetal

            except subprocess.CalledProcessError as e:
                print(f"\033[1;31mBuild Error\033[0m:\n{e.stderr.decode()}")
                return None, False
            except FileNotFoundError:
                print("\033[1;31mError\033[0m: 'nasm' or 'ld' not found. Install using `sudo apt install nasm binutils`.")
                return None, False

    def _parse_mode_blocks(self, source: str):
        lines = source.splitlines()
        blocks = []
        current_mode = "high"
        current_chunk = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("@mode:"):
                if current_chunk:
                    blocks.append((current_mode, "\n".join(current_chunk)))
                    current_chunk = []
                mode_val = stripped.split(":")[1].strip().lower()
                if "baremetal" in mode_val:
                    current_mode = "baremetal"
                elif "low" in mode_val:
                    current_mode = "low"
                else:
                    current_mode = "high"
            else:
                current_chunk.append(line)

        if current_chunk:
            blocks.append((current_mode, "\n".join(current_chunk)))

        return blocks

    def _compile_high(self, code: str) -> str:
        try:
            tokens = HighLexer(code, filename=self.source_path).tokenize()
            ast = HighParser(tokens, filename=self.source_path).parse()
            if not HighSemantic(filename=self.source_path).analyze(ast):
                return None
            return HighCodeGen(filename=self.source_path).generate(ast)
        except SyntaxError as e:
            print(f"\033[1;31m{e}\033[0m")
            return None

    def _compile_low(self, code: str, mode: str) -> str:
        try:
            tokens = LowLexer(code, filename=self.source_path).tokenize()
            parser = LowParser(tokens, filename=self.source_path)
            parser.mode = mode
            ast = parser.parse()
            if not LowSemantic(filename=self.source_path).analyze(ast):
                return None
            return LowCodeGenerator(filename=self.source_path).generate(ast)
        except SyntaxError as e:
            print(f"\033[1;31m{e}\033[0m")
            return None


def main():
    parser = argparse.ArgumentParser(description="MyLand Master Compiler Engine")
    parser.add_argument("command", choices=["run", "build"], help="'run' to execute, 'build' to create binary")
    parser.add_argument("file", help="Path to your .acl source file")
    parser.add_argument("-o", "--output", help="Output file name")
    parser.add_argument("--save-asm", action="store_true", help="Save generated NASM assembly file")

    args = parser.parse_args()

    compiler = MyLandCompiler(args.file)

    if args.command == "run":
        exe_path, is_bm = compiler.compile_and_run(output_bin=args.output, keep_asm=args.save_asm)
        if exe_path:
            if is_bm:
                print("\n\033[1;35m[QEMU Booting OS Kernel...]\033[0m")
                subprocess.run(["qemu-system-i386", "-kernel", exe_path])
            else:
                print("\n\033[1;35m================= OUTPUT =================\033[0m")
                subprocess.run([exe_path])
                print("\033[1;35m==========================================\033[0m\n")
                if os.path.exists(exe_path) and not args.output:
                    os.remove(exe_path)

    elif args.command == "build":
        compiler.compile_and_run(output_bin=args.output, keep_asm=args.save_asm)

if __name__ == "__main__":
    main()
