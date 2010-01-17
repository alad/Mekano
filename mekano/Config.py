class Parameters:
    """Holds configuration parameters
    
    p = Parameters()
    p.x = 10
    p.y = "fdfd"
    print p.params
    """
    def __init__(self):
        self.__dict__["params"] = {}
    
    def __getattr__(self, key):
        try:
            return self.params[key]
        except KeyError:
            raise AttributeError
    
    def __setattr__(self, key, val):
        self.params[key] = val
    