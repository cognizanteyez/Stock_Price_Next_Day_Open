from LSTMModel import Lstm
from dataset import cleanData
from common_parsers import args
import torch
import numpy as np


def predict(company, x):
    model = Lstm(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size=1)
    model.to(args.device)
    checkpoint = torch.load(f"{args.save_file}{company}")
    model.load_state_dict(checkpoint['state_dict'])
    preds = []
    labels = []
    # end_max, end_min, train_loader, test_loader = cleanData(f"data\{company}_hist_1mo.csv", args.sequence_length, args.batch_size)
    # for idx, (x, label) in enumerate(test_loader):
    #     if args.useGPU:
    #         x = x.squeeze(1).cuda()  # batch_size, seq_len, input_size
    #     else:
    #         x = x.squeeze(1)
    pred = model(x)
    list = pred.data.squeeze(1).tolist()
    preds.extend(list[-1])
    # labels.extend(label.tolist())
    print("/n n/")
    print("preds:", preds)
    # for i in range(len(preds)):
    #     print('Predict value is%.2f,True Value is%.2f' % (
    #     preds[i][0] * (end_max - end_min) + end_min, labels[i] * (end_max - end_min) + end_min))
# Open,High,Low,Close,Volume
x = [449.0,453.6000061035156,448.19000244140625,452.1600036621094,16507000]
x = np.array(x).reshape(1, -1)
# x = torch.tensor(x, dtype=torch.float32)
x_tensor = torch.tensor([x], dtype=torch.float32)  # Ensure the dtype matches your model's requirements
for i in ["MSFT", "TSLA", "AMZN", "META", "AAPL", "GOOGL", "NVDA"]:
    print("\n \n \n")
    print("company",i)
    predict(i, x_tensor)

# # Construct X and Y
# # Based on the data of the previous n days, predict the closing price (close) of the next day
# sequence = sequence_length
# X = []
# Y = []
# for i in range(df.shape[0] - sequence):
#     X.append(np.array(df.iloc[i:(i + sequence), ].values, dtype=np.float32))
#     test_X = np.array(X)
#     # print("shape_of_X", X.shape)
#     # print("X_Values:", X)
#     Y.append(np.array(df.iloc[(i + sequence), 3], dtype=np.float32))

# test_X = np.array(X)
# print("shape_of_X", test_X.shape)
# print("shape_of_X", test_X)

# # batch
# total_len = len(Y)
# # print(total_len)

# train_loader = DataLoader(dataset=Mydataset(trainx, trainy, transform=transforms.ToTensor()), batch_size=batchSize, shuffle=True)