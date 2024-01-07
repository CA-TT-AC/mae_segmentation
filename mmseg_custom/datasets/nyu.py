# from .builder import DATASETS
import json
import os
from mmseg.datasets.custom import CustomDataset
from mmseg.datasets.builder import DATASETS
from torch import Tensor
@DATASETS.register_module()
class NyuDataset_al(CustomDataset):
    CLASSES = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror', 'floor mat', 'clothes', 'ceiling', 'books', 'refridgerator', 'television', 'paper', 'towel', 'shower curtain', 'box', 'whiteboard', 'person', 'night stand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop')
    
    PALETTE = [[74, 60, 161], [196, 158, 25], [106, 46, 129], [49, 54, 209], [40, 27, 233], [174, 222, 58], [103, 186, 101], [222, 26, 139], [232, 14, 170], [156, 111, 165], [198, 12, 100], [126, 106, 18], [149, 34, 238], [10, 108, 42], [91, 127, 134], [174, 40, 82], [232, 251, 55], [157, 194, 55], [173, 98, 111], [61, 14, 31], [81, 179, 161], [25, 59, 245], [91, 178, 191], [107, 26, 60], [0, 115, 11], [175, 126, 130], [29, 229, 152], [248, 246, 87], [75, 176, 241], [0, 119, 35], [205, 203, 185], [78, 252, 186], [158, 188, 40], [179, 227, 237], [210, 2, 223], [190, 21, 173], [54, 208, 160], [161, 88, 207], [58, 209, 159], [77, 106, 252]]

    def __init__(self, kp_infor_dir=None, test_mode=False, **kwargs):
        super(NyuDataset_al, self).__init__(
            reduce_zero_label=True,
            **kwargs
        )
        if not test_mode:
            print(kp_infor_dir)
            print(len(self.img_infos))
            print(self.img_infos[0])
            # 读取json为dict
            if os.path.exists(kp_infor_dir):
                with open(kp_infor_dir, 'r') as f:
                    json_dict = json.load(f)
                self.kp_infors = json_dict
                # in this json_dict, key is file name without suffix, value is what should be return while calling __getitem__
            
        
    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            data = self.prepare_train_img(idx)
            
            file_name = data['img_metas'].data['ori_filename']
            ori_filename = file_name.split('.')[0]
            # print(data)
            kp_infor = self.kp_infors[ori_filename]['patch']
            # print(kp_infor)
            data['img_metas'].data['kp'] = Tensor(kp_infor)
            return data
            # print(data)
            # exit()


@DATASETS.register_module()
class NyuDataset(CustomDataset):
    CLASSES = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror', 'floor mat', 'clothes', 'ceiling', 'books', 'refridgerator', 'television', 'paper', 'towel', 'shower curtain', 'box', 'whiteboard', 'person', 'night stand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop')
    
    PALETTE = [[74, 60, 161], [196, 158, 25], [106, 46, 129], [49, 54, 209], [40, 27, 233], [174, 222, 58], [103, 186, 101], [222, 26, 139], [232, 14, 170], [156, 111, 165], [198, 12, 100], [126, 106, 18], [149, 34, 238], [10, 108, 42], [91, 127, 134], [174, 40, 82], [232, 251, 55], [157, 194, 55], [173, 98, 111], [61, 14, 31], [81, 179, 161], [25, 59, 245], [91, 178, 191], [107, 26, 60], [0, 115, 11], [175, 126, 130], [29, 229, 152], [248, 246, 87], [75, 176, 241], [0, 119, 35], [205, 203, 185], [78, 252, 186], [158, 188, 40], [179, 227, 237], [210, 2, 223], [190, 21, 173], [54, 208, 160], [161, 88, 207], [58, 209, 159], [77, 106, 252]]

    def __init__(self, kp_infor_dir, **kwargs):
        super(NyuDataset, self).__init__(
            reduce_zero_label=True,
            **kwargs
        )