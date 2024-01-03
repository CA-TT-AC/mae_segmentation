# from .builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmseg.datasets.builder import DATASETS

@DATASETS.register_module()
class NyuDataset(CustomDataset):
    CLASSES = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror', 'floor mat', 'clothes', 'ceiling', 'books', 'refridgerator', 'television', 'paper', 'towel', 'shower curtain', 'box', 'whiteboard', 'person', 'night stand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop')
    
    PALETTE = [[74, 60, 161], [196, 158, 25], [106, 46, 129], [49, 54, 209], [40, 27, 233], [174, 222, 58], [103, 186, 101], [222, 26, 139], [232, 14, 170], [156, 111, 165], [198, 12, 100], [126, 106, 18], [149, 34, 238], [10, 108, 42], [91, 127, 134], [174, 40, 82], [232, 251, 55], [157, 194, 55], [173, 98, 111], [61, 14, 31], [81, 179, 161], [25, 59, 245], [91, 178, 191], [107, 26, 60], [0, 115, 11], [175, 126, 130], [29, 229, 152], [248, 246, 87], [75, 176, 241], [0, 119, 35], [205, 203, 185], [78, 252, 186], [158, 188, 40], [179, 227, 237], [210, 2, 223], [190, 21, 173], [54, 208, 160], [161, 88, 207], [58, 209, 159], [77, 106, 252]]

    def __init__(self, **kwargs):
        super(NyuDataset, self).__init__(
            reduce_zero_label=True,
            **kwargs
        )
