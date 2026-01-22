from .grabcut import GrabCutDataset


class CVCDataset(GrabCutDataset):
    def __init__(self, dataset_path, **kwargs):
        super().__init__(dataset_path, images_dir_name='img', masks_dir_name='gt', **kwargs)
        self.name = 'CVC'
