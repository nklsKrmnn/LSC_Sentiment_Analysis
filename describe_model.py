import torch

def describe_model(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f'{name} has trainable parameters: {sum(p.numel() for p in param if p.requires_grad)}')


    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters (total): {total_params}")
    print(f"Number of parameters (trainale): {trainable_params}")

def main():
    model = torch.load("./runs/model_saves/best_model_Test0706.pt", map_location=torch.device('cpu'))
    describe_model(model)

if __name__ == '__main__':
    main()