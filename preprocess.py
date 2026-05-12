import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
from plyfile import PlyData
from tqdm import tqdm
from pathlib import Path
import torch_geometric.transforms as T

class SHREC_Dataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        if pre_transform is None:
            pre_transform = T.FaceToEdge(remove_faces=True)
            
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['shrec_intr_coarsened_data.pt']

    def process(self):
        root_path = Path(self.root)
        ply_files = list(root_path.rglob('*.ply'))

        if not ply_files:
            raise FileNotFoundError(f"No .ply files found in {self.root} or its subdirectories.")

        categories = sorted(list(set([path.parent.parent.name for path in ply_files])))
        class_to_idx = {cat: i for i, cat in enumerate(categories)}
        print(f"Found {len(categories)} classes. Mapping: {class_to_idx}")

        data_list = []

        for raw_path in tqdm(ply_files, desc="Processing PLY files"):
            ply_data = PlyData.read(str(raw_path))

            # Extract Pos
            pos_x = ply_data['vertex']['x']
            pos_y = ply_data['vertex']['y']
            pos_z = ply_data['vertex']['z']
            pos = torch.tensor(np.vstack([pos_x, pos_y, pos_z]).T, dtype=torch.float32)

            # Extract Node Features
            curvatures = ply_data['vertex']['vertex_gaussian_curvature']
            x = torch.tensor(curvatures, dtype=torch.float32).view(-1, 1)

            # Extract Faces
            faces = ply_data['face']['vertex_indices']
            face_tensor = torch.tensor(np.vstack(faces), dtype=torch.long).t().contiguous()

            # Extract Label from the folder structure
            category = raw_path.parent.parent.name
            y = torch.tensor([class_to_idx[category]], dtype=torch.long)

            # Build Data Object
            data = Data(x=x, pos=pos, face=face_tensor, y=y)
            
            # Apply the transform to convert data.face -> data.edge_index
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            
            data_list.append(data)

        self.save(data_list, self.processed_paths[0])
