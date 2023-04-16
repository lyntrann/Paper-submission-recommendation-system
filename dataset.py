import torch

class DualPaperDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, n_classes) -> None:
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.n_classes = n_classes
    
    def __getitem__(self, index):
        label = torch.tensor(self.data['Label'][index])
        title = self.data['title'][index].strip() if isinstance(self.data['title'][index], str) else ""
        title = f"main idea : {title} . "
        abstract = self.data['abstract'][index].strip() if isinstance(self.data['abstract'][index], str) else ""
        abstract = f"concise summary : {abstract} . "
        keywords = self.data['keywords'][index].strip() if isinstance(self.data['keywords'][index], str) else ""
        keywords = f"important words : {keywords} ."
        context = title + abstract + keywords
        context = self.tokenizer(context, max_length=352, truncation=True, padding=False)
        context['label'] = label
        return context
    
    def __len__(self):
        return len(self.data)