import inspect


class A(object):
    def __init__(self):
        print("hello world!")

print(inspect.isclass(A))