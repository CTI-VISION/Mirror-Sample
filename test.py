# -*- coding: utf-8 -*-
import torch
from Models.Mirror_Model import Get_Model
from Data.prepare_data import generate_dataloader
from trainer import validate
from opts import opts
import os

local_code_root = os.path.dirname(os.path.abspath(__file__))


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = opts()
    model = Get_Model(args)
    model = model.to(device)
    state_dict = model.state_dict()
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # load model weights
    trained_model = torch.load(os.path.join(local_code_root, 'checkpoint', '{}_{}.pth'.format(args.src, args.tar_t)),
                               map_location=device)

    state_dict.update(trained_model)
    model.load_state_dict(state_dict)

    _, _, _, target_test_loader = generate_dataloader(args, local_code_root)

    acc = validate(device, target_test_loader, model, criterion, args)

    print acc


if __name__ == '__main__':
    main()