import time
def timeprofile(fun):
    def tmp(*args, **kwargs):
        s = time.time()
        r = fun(*args, **kwargs)
        print(fun.__name__ + " " + str(time.time() - s))
        return r
    return tmp
    