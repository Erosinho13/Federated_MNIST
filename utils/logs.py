import pickle

class Logger:
        
    def save(self, obj, path):
        with open(path, 'wb') as logfile:
            pickle.dump(obj, logfile)
    
    def load(self, path):
        with open(path, 'rb') as logfile:
            new_instance = pickle.load(logfile)
        return new_instance
    
class Logs:
    
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)