import torch
import torch.nn.functional as F


class FGSM:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def attack(self, images, labels, epsilon):
        images = images.clone().detach().requires_grad_(True)

        outputs = self.model(images)
        loss = F.cross_entropy(outputs, labels)

        self.model.zero_grad()
        loss.backward()

        grad_sign = images.grad.sign()
        adv_images = images + epsilon * grad_sign
        adv_images = torch.clamp(adv_images, 0, 1)

        return adv_images
