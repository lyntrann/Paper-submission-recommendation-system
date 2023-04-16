import torch
import pickle

# Utils
def save_parameter(save_object, save_file):
    with open(save_file, 'wb') as f:
        pickle.dump(save_object, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_parameter(load_file):
    with open(load_file, 'rb') as f:
        output = pickle.load(f)
    return output

def sim_matrix(a, b, eps=1e-8):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_metrics(logits, labels):
    topks = (1, 3, 5, 10)
    scores = {k: 0 for k in topks}
    sorted_logits = torch.argsort(torch.exp(logits), axis=1, descending=True)
    num_cnt = {k: 0 for k in topks}
    for k in topks:
        batch_num_correct = 0
        n_points = len(labels)
        for idx in range(n_points):
            if labels[idx] in sorted_logits[idx, 0:k]:
                batch_num_correct += 1
        scores[k] = batch_num_correct/n_points
        num_cnt[k]+= batch_num_correct
    return scores, num_cnt

def cosine_distance(x):
    dist = 1-torch.mm(x, x.T)
    dist = dist.triu(diagonal=1).mean()
    return dist

def preprocess_keywords(keyword):
    keyword=  keyword.split(",")
    keyword = [ele.strip() for ele in keyword if len(ele.strip())>3]
    keyword = ", ".join(keyword)
    return keyword

def mean_pooling(model_output, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
    return torch.sum(model_output * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
