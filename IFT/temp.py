import re

def tokenize(expression):
    tokens = re.findall(r'[a-zA-Z_]\w*(?:\.\w+)*|[\+\-\*/()]', expression)
    return tokens

def infix_to_postfix(tokens):
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
    output = []
    operators = []
    
    for token in tokens:
        if re.match(r'[a-zA-Z_]\w*(?:\.\w+)*', token):
            output.append(token)
        elif token in precedence:
            while (operators and operators[-1] in precedence and
                   precedence[token] <= precedence[operators[-1]]):
                output.append(operators.pop())
            operators.append(token)
        elif token == '(':
            operators.append(token)
        elif token == ')':
            while operators and operators[-1] != '(':
                output.append(operators.pop())
            operators.pop()
    
    while operators:
        output.append(operators.pop())
    
    return output

def evaluate_postfix(postfix_tokens, values):
    stack = []
    for token in postfix_tokens:
        if re.match(r'[a-zA-Z_]\w*(?:\.\w+)*', token):
            stack.append(values[token])
        else:
            b = stack.pop()
            a = stack.pop()
            if token == '+':
                stack.append(a + b)
            elif token == '-':
                stack.append(a - b)
            elif token == '*':
                stack.append(a * b)
            elif token == '/':
                stack.append(a / b)
    return stack[0]

# Define the variable values, including complex identifiers
values = {
    'fundamental_data.npm': 10,
    'fundamental_data.dte': 5
}

# Expression to evaluate
expr_string =  "fundamental_data.npm + fundamental_data.dte - (fundamental_data.dte * fundamental_data.npm)"
tokens = tokenize(expr_string)
postfix_tokens = infix_to_postfix(tokens)
result = evaluate_postfix(postfix_tokens, values)

print(f"The result of the expression is: {result}")  # Output: 15.0

