class NotExistingModelException(Exception):
    def __repr__(self):
        return 'Please, choose correct model'


class ModelNotCreatedException(Exception):
    def __repr__(self):
        return 'Please, create a model first'