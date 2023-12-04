# Utility functions - can be for random one offs that may need to be repeated

def block_print(*args):
    """function that pretty prints a list of statements in a block. Useful if you want multiple print statements without writing print a bunch of times
    :param statements: all the text strings you want to print
    """
    for statement in args:
        if statement == args[-1]:
            statement = f'{statement}\n'
        print(statement)