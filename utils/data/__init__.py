def get_dataset(dataset_name):
    if dataset_name == 'cornell':
        from .cornell_data import CornellDataset
        return CornellDataset
    elif dataset_name == 'jacquard':
        from .jacquard_data import JacquardDataset
        return JacquardDataset
    elif dataset_name == 'cornell_skfold':
        from .cornell_kfold import CornellKDataset
        return CornellKDataset
    elif dataset_name == 'jacquard_kfold':
        from .jacquard_data_kfold import JacquardDataset
        return JacquardDataset
    elif dataset_name == 'jacquard_skfold':
        from .jacquard_kfold import JacquardKDataset
        return JacquardKDataset
    elif dataset_name == 'jacquard_depth':
        from .jacquard_kfold_depth import JacquardKDataset
        return JacquardKDataset
    elif dataset_name == 'cornell_class':
        from .cornell_class import CornellCocoDataset
        return CornellCocoDataset
    elif dataset_name == 'jacquard_sal':
        from .jacquard_sal import JacquardSalDataset
        return JacquardSalDataset
    elif dataset_name == 'cornell_sal':
        from .cornell_sal import CornellSalDataset
        return CornellSalDataset
    else:
        raise NotImplementedError('Dataset Type {} is Not implemented'.format(dataset_name))