import torch
from PIL import Image
from torch.utils.data import Dataset


class UnderWaterDataset(Dataset):

    def __init__(self, df, split_size, boxes, classes, transform=None):
        self.df = df
        self.S = split_size
        self.B = boxes
        self.C = classes
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        filename = self.df.iloc[index]['filename']
        boxes = []
        with open(filename) as f:
            for line in f.readlines():
                label, x, y, width, height = line.split()
                boxes.append([int(label), float(x), float(y), float(width), float(height)])

        label_matrix = torch.zeros((self.S, self.S, 5 + self.C))
        filename_path = filename.split("/")
        images_path = [
            subsection if subsection != "labels" else "images"
            for subsection in filename_path
        ]
        images_filename = "/".join(images_path)
        image = Image.open(f"{images_filename[:-3]}jpg")

        for box in boxes:
            label, x, y, width, height = box
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            width_cell, height_cell = width * self.S, height * self.S

            if label_matrix[i, j, self.C] == 0:
                label_matrix[i, j, self.C] = 1
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                label_matrix[i, j, self.C+1:self.C+5] = box_coordinates
                label_matrix[i, j, label] = 1

        if self.transform is not None:
            image, label_matrix = self.transform(image, label_matrix)

        return image, label_matrix