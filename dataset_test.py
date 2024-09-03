from dataset import cleanData

from common_parsers import args

for i in ["MSFT", "TSLA", "AMZN", "META", "AAPL", "GOOGL", "NVDA"]:
    print("\n \n \n")
    print("company",i)
    end_max, end_min, train_loader, test_loader = cleanData(f"data\{i}_hist_1mo.csv",args.sequence_length,args.batch_size)

    print("end_max:", end_max)
    print("\n \n")

    print("end_min:", end_min)
    print("\n \n")

    print("train_loader:", train_loader)
    print("\n \n")

    print("test_loader:", test_loader)