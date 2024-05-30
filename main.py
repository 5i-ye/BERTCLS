from tqdm import tqdm
import pandas as pd
from config import MainOptions
from dataset import TextDataset
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW

if __name__ == '__main__':
    # load config
    opt = MainOptions().parse_args()
    tokenizer = AutoTokenizer.from_pretrained(opt.tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(opt.tokenizer_path, num_labels=opt.num_label)

    train_data = pd.read_csv(opt.train_path, index_col=0)
    test_data = pd.read_csv(opt.test_path, index_col=0)

    # load dataset
    train_dataset = TextDataset(train_data, tokenizer)
    test_dataset = TextDataset(test_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size = opt.batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = opt.batch_size, shuffle = False)

    # setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model= nn.DataParallel(model)
    else:
        device = torch.device("cpu")
    
    print("Using PyTorch Ver", torch.__version__)
    model.to(device)
    optimizer = AdamW(model.parameters(), correct_bias=False)

    # start train
    train_loss=[]
    train_accuracy=[]
    train_f1score=[]

    EPOCH = opt.num_epoch
    min_loss = 1e9

    for epoch in range(EPOCH):

        print('start Epoch ', epoch+1)

        model.train()
        total_loss, total_correct, total_len, total_f1 = 0, 0, 0, 0
        total_precision, total_recall = 0, 0

        for idx, batch in enumerate(tqdm(train_loader)):

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['targets'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask, labels=label)
            loss, logits = outputs.loss, outputs.logits

            if isinstance(loss, torch.Tensor):
                loss = loss.mean()

            pred = torch.argmax(F.softmax(logits, dim=1), dim=1)
            correct = pred.eq(label)
            total_correct += correct.sum().item()
            total_len += len(label)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            ## label==2
            # f1 = f1_score(label.cpu().numpy(), pred.cpu().numpy(), zero_division=1)
            # total_f1 += f1
            # precision = precision_score(label.cpu().numpy(), pred.cpu().numpy(), zero_division=1)
            # total_precision += precision
            # recall = recall_score(label.cpu().numpy(), pred.cpu().numpy(), zero_division=1)
            # total_recall += recall

            train_loss.append(loss.item())
            train_accuracy.append(total_correct/total_len)
            # train_f1score.append(f1)

        # print('[Epoch {}/{}] -> Train Loss: {:.5f}, Accuracy: {:.5f}, F1_score: {:.5f}, Precision: {:.5f}, Recall: {:.5f}'.format(epoch+1, EPOCH, total_loss/len(train_loader), total_correct/total_len, total_f1/len(train_loader), total_precision/len(train_loader), total_recall/len(train_loader)))
        print('[Epoch {}/{}] -> Train Loss: {:.5f}, Accuracy: {:.5f}'.format(epoch+1, EPOCH, total_loss/len(train_loader), total_correct/total_len))


        model.eval()
        with torch.no_grad():
            total_loss, total_correct, total_len, total_f1 = 0, 0, 0, 0
            total_precision, total_recall = 0, 0

            for idx, batch in enumerate(tqdm(test_loader)):

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                label = batch['targets'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=label)
                loss, logits = outputs.loss, outputs.logits

                if isinstance(loss, torch.Tensor):
                    loss = loss.mean()

                pred = torch.argmax(F.softmax(logits, dim=1), dim=1)
                correct = pred.eq(label)
                total_correct += correct.sum().item()
                total_len += len(label)
                total_loss += loss.item()

                # f1 = f1_score(label.cpu().numpy(), pred.cpu().numpy(), zero_division=1)
                # total_f1 += f1
                # precision = precision_score(label.cpu().numpy(), pred.cpu().numpy(), zero_division=1)
                # total_precision += precision
                # recall = recall_score(label.cpu().numpy(), pred.cpu().numpy(), zero_division=1)
                # total_recall += recall


            # print('[Epoch {}/{}] -> Test Loss: {:.5f}, Accuracy: {:.5f}, F1_score: {:.5f}, Precision: {:.5f}, Recall: {:.5f}'.format(epoch+1, EPOCH, total_loss/len(test_loader), total_correct/total_len, total_f1/len(test_loader), total_precision/len(test_loader), total_recall/len(test_loader)))
            print('[Epoch {}/{}] -> Test Loss: {:.5f}, Accuracy: {:.5f}'.format(epoch+1, EPOCH, total_loss/len(test_loader), total_correct/total_len))

            # save model
            if total_loss/len(test_loader) < min_loss:
                torch.save(model, 'model.pth')
                print(epoch+1, "모델 저장")
                min_loss = total_loss/len(test_loader)