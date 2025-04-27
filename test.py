class Test:
    def __init__(self, arg1, arg2):
        self.arg1 = arg1
        self.arg2 = arg2

    def m1(self):
        x = 3
        y = 4
        return x, y
    
    def m2(self):
        w = self.m1()[0]
        z = self.m2()[1]
        return w, z
    


if __name__ == "__main__":
    t = Test(4, 5)
    w, z = t.m1()
    print(w, z)