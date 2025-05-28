import os
import nltk
from nltk.tokenize import sent_tokenize
import openai
from openai import OpenAI
from lark import Lark, Transformer
from sympy import symbols, And, Or, Not, Implies, Equivalent
from sympy.logic.boolalg import to_cnf, simplify_logic
from functools import reduce
import csv
nltk.download('punkt')
nltk.download('punkt_tab')
###############################################################################
# 1. Define the Lark Grammar & Transformer
###############################################################################
grammar = r"""
start: logic_expr

?logic_expr: func_call
           | SYMBOL

func_call: NAME "(" [expr_list] ")"

?expr_list: logic_expr ("," logic_expr)*

SYMBOL: /[A-Za-z][A-Za-z0-9_]*/

NAME: "And" | "Or" | "Not" | "Implies" | "Equivalent"

%import common.WS
%ignore WS
"""

parser = Lark(grammar, start='start')

class LogicTransformer(Transformer):
    def __init__(self):
        super().__init__()
        self.vars = {}
        self.functions = {
            'And': And,
            'Or': Or,
            'Not': Not,
            'Implies': Implies,
            'Equivalent': Equivalent
        }

    def SYMBOL(self, token):
        # Turn a single token (e.g. "P") into a Sympy symbol
        name = str(token)
        if name not in self.vars:
            self.vars[name] = symbols(name)
        return self.vars[name]

    def start(self, items):
        return items[0]

    def logic_expr(self, items):
        if len(items) == 1:
            return items[0]
        raise ValueError(f"Unexpected parse structure in logic_expr: {items}")

    def func_call(self, items):
        # items[0] = function name (e.g., "And"), items[1] = list of arguments
        func_name = str(items[0])
        func = self.functions.get(func_name)
        if func is None:
            raise ValueError(f"Unknown function '{func_name}'")
        args = items[1] if len(items) > 1 else []
        if not isinstance(args, list):
            args = [args]
        return func(*args)

    def expr_list(self, items):
        return items
transformer = LogicTransformer()

###############################################################################
# 2. CNF Simplification Function
###############################################################################
def simplify_cnf_expression(expr):
    """
    Given a Sympy boolean expression (already in CNF or convertible to CNF),
    use Sympy's simplify_logic to produce a simplified CNF form.
    """
    if expr is None:
        return None
    # 'simplify_logic' can produce a minimal CNF form when form='cnf' and force=True
    simplified_expr = simplify_logic(expr, force=True, form='cnf')
    return simplified_expr

###############################################################################
# 3. Single-Prompt Function
###############################################################################
def text_to_cnf_all_at_once(text):
    strings = ''
    cnf_exprs = []
    results = []  # Store (sentence, CNF) for debugging
    lines = eval(text[1])
    for i in lines: 
        strings += i.strip() + '\n'
    #Parse each line => Lark => Sympy => CNF
    strings = strings.splitlines()
    for i, line in enumerate(strings):
        line_stripped = line.strip()
        if not line_stripped:
            continue

        print(f"Model Output: {line_stripped}")

        try:
            # Parse with Lark
            parse_tree = parser.parse(line_stripped)
            expr = transformer.transform(parse_tree)
            # Convert to CNF
            cnf_expr = to_cnf(expr, simplify=True, force=True)
            cnf_exprs.append(cnf_expr)

            # Log debug info
            print(f"Parsed Expression: {expr}")
            print(f"CNF Expression: {cnf_expr}")
            print("-----")

            results.append((cnf_expr))

        except Exception as e:
            print(f"Error processing line: '{line_stripped}'")
            print(e)
            print("-----")

    # Combine all CNF expressions with logical And
    if cnf_exprs:
        combined_cnf = reduce(And, cnf_exprs)
    else:
        combined_cnf = None

    # Simplify the final CNF
    simplified_cnf = simplify_cnf_expression(combined_cnf)

    return simplified_cnf, results

###############################################################################
# 4. Example main
###############################################################################
if __name__ == "__main__":
    # Make sure you have your OPENAI_API_KEY set
    openai.api_key = os.getenv("OPENAI_API_KEY")  # or directly: openai.api_key = "sk-..."

    # Ensure nltk 'punkt' is downloaded for sentence tokenization
    nltk.download("punkt")
    with open('train.csv', 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row_num, row in enumerate(reader):
            print(row)
            if row_num > 0:  # Skip header row
                final_cnf, expressions = text_to_cnf_all_at_once(row)

                print("\nFinal CNF Expression (Simplified):")
                print(final_cnf)

