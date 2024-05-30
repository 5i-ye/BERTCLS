import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dataset import TextDataset
from config import ExtractOptions
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

if __name__ == '__main__':

    # load config
    opt = ExtractOptions().parse_args()
    device = torch.device(opt.device)
    tokenizer = AutoTokenizer.from_pretrained(opt.tokenizer_path)
    model = torch.load(opt.model_path, map_location=device)
    data = pd.read_csv(opt.data_path, index_col=0)

    # load dataset
    dataset = TextDataset(data, tokenizer)
    dataloader = DataLoader(dataset, batch_size = opt.batch_size, shuffle = False)
    
    # setup
    model.to(device)
    model.eval()

    extracted_features = {}
    def hook(module, input, output):
        extracted_features['features'] = output

    # ------ change CLS token position
    hook_handle = model.module.electra.encoder.layer[-1].output.LayerNorm.register_forward_hook(hook)

    # save vector
    real_label = np.empty((0, 1))
    model_label = np.empty((0, 1))
    result = np.empty((0,768))

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
                outputs = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
                logits = outputs.logits
                label = batch['targets'].to(device)
                pred = torch.argmax(F.softmax(logits, dim=1), dim=1)
                CLS_vector = extracted_features['features'][:, 0, :]

                result = np.concatenate([result, CLS_vector.cpu().detach().numpy()])
                real_label = np.concatenate([real_label, batch['targets'].cpu().detach().numpy().reshape(-1, 1)])
                model_label = np.concatenate([model_label, pred.cpu().detach().numpy().reshape(-1, 1)])
        
        hook_handle.remove()

        np.save("./cls_vector.npy", result)
        np.save("./real_label.npy", real_label)
        np.save("./model_label.npy", model_label)

