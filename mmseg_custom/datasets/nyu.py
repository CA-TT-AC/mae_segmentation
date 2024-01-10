# from .builder import DATASETS
import json
import os
from mmseg.datasets.custom import CustomDataset
from mmseg.datasets.builder import DATASETS
import torch
from mmcv.parallel import DataContainer

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
    def coord2patchId(img_size, coord, ps=16):
        """Get training/test data after pipeline.
        Args:
            img_size (tuple): input image size.
            coord (tuple): query coord

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """
        n_w = img_size[0] // ps
        patch_coord = (coord[0]//16)+1, (coord[1]//16)+1
        
        
        id = patch_coord[1] * n_w + patch_coord[0]
        return id
    
    def coord2patchId_tensor(self, img_size, coords, ps=16):
        """
        Convert a 3D tensor of coordinates to corresponding patch ids.

        Args:
            img_size (tuple): input image size (height, width).
            coords (tensor): A 3D tensor of shape [x, y, 2] where coords[x, y] is the coordinate.
            ps (int): patch size.

        Returns:
            Tensor: A 2D tensor of shape [x, y] containing patch ids.
        """
        n_w = img_size[1] // ps
        patch_coords = torch.div(coords, ps, rounding_mode='trunc')
        ids = (patch_coords[..., 0] ) * n_w + (patch_coords[..., 1] )
        return ids
    
    def create_coordinate_tensor(self, img_size):
        h, w = img_size
        y_coords, x_coords = torch.meshgrid(torch.arange(w), torch.arange(h))
        return torch.stack((y_coords, x_coords), dim=-1)
    
    def scale2x_patch_ids(self, ids, w_ori=224, w_cur=448, ps=16):
        n_w_ori = w_ori // ps
        n_w_cur = w_cur // ps
        ids_1 = torch.div(ids, n_w_ori, rounding_mode='trunc') * n_w_cur + (ids % n_w_ori) * (n_w_cur / n_w_ori)
        ids_1 = ids_1.type(torch.int32)
        ids_2 = ids_1 + 1 
        ids_3 = ids_1 + n_w_cur
        ids_4 = ids_3 + 1
        ret = torch.concat([ids_1, ids_2, ids_3, ids_4], dim=-1)
        return ret
        
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
            
            img_size = data['img_metas'].data['img_shape']
            img_size = img_size[0], img_size[1]
            # create a tensor of shape [h, w, 2] where ts[y, x, :] = [y, x]
            coord_tensor = self.create_coordinate_tensor(img_size)
            # shape of [h, w], save its patch ids of every pixel
            patch_ids_tensor = self.coord2patchId_tensor(img_size, coord_tensor)
            file_name = data['img_metas'].data['ori_filename']
            ori_filename = file_name.split('.')[0]
            kp_infor_ori = torch.LongTensor(self.kp_infors[ori_filename]['patch'])
            kp_infor_ori = kp_infor_ori[:10]
            kp_infor_scaled = self.scale2x_patch_ids(kp_infor_ori)
            mask = torch.isin(patch_ids_tensor, kp_infor_scaled)
            
            # print(mask)
            # print(mask.shape)
            # print(kp_infor_scaled)
            # print('kp:', kp_infor_tensor.shape)
            # print("ori")
            # print(kp_infor_ori)
            # print("scaled")
            # print(kp_infor_scaled)
            
            ann = data['gt_semantic_seg'].data
            # print(data['gt_semantic_seg'].size(), data['gt_semantic_seg'].dim(), data['gt_semantic_seg'].stack)
            # print("ori")
            # print(ann)
            ann[:, ~mask] = 0
            data['gt_semantic_seg'] = DataContainer(ann, stack=True)

            # print("ann")
            # print(ann)
            
            # exit()
            return data


@DATASETS.register_module()
class NyuDataset(CustomDataset):
    CLASSES = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror', 'floor mat', 'clothes', 'ceiling', 'books', 'refridgerator', 'television', 'paper', 'towel', 'shower curtain', 'box', 'whiteboard', 'person', 'night stand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop')
    
    PALETTE = [[74, 60, 161], [196, 158, 25], [106, 46, 129], [49, 54, 209], [40, 27, 233], [174, 222, 58], [103, 186, 101], [222, 26, 139], [232, 14, 170], [156, 111, 165], [198, 12, 100], [126, 106, 18], [149, 34, 238], [10, 108, 42], [91, 127, 134], [174, 40, 82], [232, 251, 55], [157, 194, 55], [173, 98, 111], [61, 14, 31], [81, 179, 161], [25, 59, 245], [91, 178, 191], [107, 26, 60], [0, 115, 11], [175, 126, 130], [29, 229, 152], [248, 246, 87], [75, 176, 241], [0, 119, 35], [205, 203, 185], [78, 252, 186], [158, 188, 40], [179, 227, 237], [210, 2, 223], [190, 21, 173], [54, 208, 160], [161, 88, 207], [58, 209, 159], [77, 106, 252]]

    def __init__(self, kp_infor_dir, **kwargs):
        super(NyuDataset, self).__init__(
            reduce_zero_label=True,
            **kwargs
        )