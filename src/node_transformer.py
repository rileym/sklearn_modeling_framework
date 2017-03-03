from sklearn.pipeline import Pipeline
from feature_extraction_helpers import DfFeatureUnion for transformer in self.df_transformers), axis = 1)

import six
import abc

@six.add_metaclass(abc.ABCMeta)
class TransformerNode(object):

    @property
    def name(self):
        self._name

    @abc.abstractproperty
    def grid(self):
        pass

    @abc.abstractmproperty
    def tranformer(self):
        pass

# TODO: verity atomic_transformer in transformer
class AtomicTransformerNode(TransformerNode):
    
    def __init__(self, name, atomic_transformer, parameter_grid = None):
        self._name = name
        self._atomic_transformer = atomic_transformer
        self._grid = parameter_grid if parameter_grid is not None else {}

    @property
    def name(self):
        return self._name

    @property
    def grid(self):
        return self._grid

    @property 
    def transformer(self):
        return self._atomic_transformer

six.add_metaclass(abc.ABCMeta)
class CompositeTransformerNode(TransformerNode):
    

    def __init__(self, name, children, combiner_cls):
        self._name = name
        self._children = children
        self._combiner_cls = combiner_cls

    def combiner_cls(self):
        return self._combiner_cls

    @property
    def grid(self):
        if hasattr(self, '_grid'):
            return self._grid

        else:
            self._grid = self._build_grid()
            return self.grid

    @property 
    def transformer(self):
        if hasattr(self, '_transformer'):
            return self._transformer

        else:
            self._transformer = self._build_transformer()
            return self.transformer

    def _build_grid(self):
        
        grid = {}
        for child in self._children:
            for child_param_name, param_values in child.grid.iteritems():
                param_name = self.name + '__' + child_param_name
                grid[param_name] = param_values

        return grid

    def _build_transformer(self):
        
        steps = [(child.name, child.transformer) for child in self.children]
        return self.combiner_cls(steps)


class PipelineCompositeTransformerNode(CompositeTransformerNode):
    
    def __init__(self, name, children):
        super(PipelineCompositeTransformerNode, self).__init__(name, children, Pipeline)

class UnionCompositeTransformerNode(CompositeTransformerNode):

    def __init__(self, name, children):
        super(UnionCompositeTransformerNode, self).__init__(name, children, DfUnion)




if __name__ == '__main__':
