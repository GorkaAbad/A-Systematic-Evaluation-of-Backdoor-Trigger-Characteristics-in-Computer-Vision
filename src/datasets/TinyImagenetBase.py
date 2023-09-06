from torchvision.datasets import VisionDataset
from typing import Any, Tuple


class TinyImagenetBase(VisionDataset):

    def __init__(
            self,
            root=None,
            train='train',
            transform=None,
            target_transform=None,
            data=None,
            targets=None,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set

        self.data = data
        self.targets = targets

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, target
