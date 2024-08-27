def alpha_example_6(fundamental_data):
    alpha_expression = """
    npm_rank = rank(fundamental_data.npm)
    dte_rank = rank(fundamental_data.dte)
    npm_rank + dte_rank
    """
    
    return {
        'alpha': alpha_expression.strip(),
        'neutralisation': 'market',
        'decay': 5,
    }

print(alpha_example_6(2))