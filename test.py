import threading

class test():
    def __init__(self):
        self.a = 100

    def change(self, char):
        self.a = char


if __name__ == "__main__":
    A = test
    timer = threading.Timer(5, A.change(A,100))
    timer.start()