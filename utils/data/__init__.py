def get_dataset(dataset_name):
    if dataset_name == 'cornell':
        from .cornell_data import CornellDataset
        return CornellDataset
    elif dataset_name == 'jacquard':
        from .jacquard_data import JacquardDataset
        return JacquardDataset
    elif dataset_name == 'cornell_class':
        from .cornell_class import CornellCocoDataset
        return CornellCocoDataset
    elif dataset_name == 'cornell_rot':
        from .cornell_class_rot import CornellCocoDataset
        return CornellCocoDataset
    else:
        raise NotImplementedError('Dataset Type {} is Not implemented'.format(dataset_name))