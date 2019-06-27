from itertools import count

class ContinuousVariable:

    __idCounter = count(0)
    
    def __init__(self, name):
        self.__name = name
        self.__id = self.__idCounter
        self.__idCounter = next(self.__idCounter)
        
    def __str__(self):
        return self.__name

    def __repr__(self):
        return str(self)

    #def __repr__(self):
    #    return "Name : {0}, Id : {1}".format(self.__name, self.__id)

    def __lt__(self, other):
        return self.__idCounter < other.__idCounter

    def get_id(self):
        return self.__id
    
if __name__ == '__main__':
    X1 = ContinuousVariable('X1')
    X2 = ContinuousVariable('X2')
    X3 = ContinuousVariable('X3')
    print(X1 < X2)
    print(X2 > X3)
    scope = [X2, X1]
    print(scope)
    scope.sort()
    print(scope)
