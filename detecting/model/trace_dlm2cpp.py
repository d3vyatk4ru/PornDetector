import torch
import torchvision
import os


class ModuleForCpp(torch.nn.Module):

    def __init__(self) -> None:

        super(ModuleForCpp, self).__init__()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        self.resnet34 = torch.load(os.path.abspath("./vgg11_porn_model.pth"),
                                    map_location=device)
        self.resnet34.eval()

        self.resnet34 = torch.jit.trace(self.resnet34,
                                      torch.rand((1, 3, 224, 224),
                                      dtype=torch.float))

    def forward(self, img):

        return self.resnet34(img)
        

if __name__ == "__main__":

    module = ModuleForCpp()
    sm = torch.jit.script(module)

    sm.save('annotation_vgg11_porn_model.pt')
