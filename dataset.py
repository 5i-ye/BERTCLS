from torch.utils.data import Dataset

class TextDataset(Dataset):

    def __init__(self, df, tokenizer):
      super(TextDataset, self).__init__()
      self.df = df
      self.tokenizer = tokenizer

    def __len__(self):
      return len(self.df)

    def __getitem__(self, idx):
      text = self.df.iloc[idx, 0]
      label = self.df.iloc[idx, 1]
      inputs = self.tokenizer(
          text,
          return_tensors='pt',
          truncation=True,
          padding='max_length',
          max_length=512,
          pad_to_max_length=True,
          add_special_tokens=True
      )
      input_ids = inputs['input_ids'][0]
      attention_mask = inputs['attention_mask'][0]
      target = label


      return{
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'targets': target
      }