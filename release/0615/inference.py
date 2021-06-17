import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((50, 50)),
    transforms.Grayscale(3),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225])
    ])

class FTModel(torch.nn.Module):

    def __init__(self, num_class, model_type="resnet18", **kwargs):
        super().__init__()
        self.num_class = num_class        
        self.model = getattr(models, model_type)()
        in_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(in_features, num_class)
        

    def forward(self, img, **kwargs):
        
        logits = self.model(img, **kwargs)

        return logits



def inference(img, model_path1, model_path2, model_type="resnet18"):

    device = torch.device("cpu")
    model1 = FTModel(num_class=2, model_type=model_type)
    model2 = FTModel(num_class=800, model_type=model_type)
    model1.load_state_dict(torch.load(model_path1, map_location=device))
    model2.load_state_dict(torch.load(model_path2, map_location=device))
    model1.eval()
    model2.eval()
    prediction = 0

    with torch.no_grad():
        img = transform(img)
        img = img.view(1, 3, 50, 50)
        pred_null = model1(img)
        pred_null = pred_null.argmax(1).item()

        if pred_null == 1:
            prediction = 801 # isNull class
        else:
            logits = model2(img)
            prediction = logits.argmax(1).item() + 1
            # +1 since model: 0-indexing, folder: 1-indexing

    return prediction

if __name__ == '__main__':

    import PIL
    
    img_path = "test.png"
    model_path1 = "model_null.pth"
    model_path2 = "model_word.pth"
    model_type = "resnet18"
    img = PIL.Image.open(img_path).convert("RGB")
    pred = inference(img, model_path1, model_path2, model_type)
    print(pred)



    

    
