import csv
import pandas as pd

threshold = 0.5
outputfile = (
    f"./final-project-challenge-1-ilovedlcv/evalai_test/output/tmp/pred_{threshold}.csv"
)
pred = []
with open(
    "./final-project-challenge-1-ilovedlcv/evalai_test/output/result/pred.csv",
    mode="r",
) as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in csv_reader:
        if line_count == 0:  # skip header
            line_count += 1
        pred.append([row["Id"], row["Predicted"]])
for i in range(len(pred)):
    if int(pred[i][1]) >= threshold:
        pred[i][1] = 1
    else:
        pred[i][1] = 0
pred = pd.DataFrame(pred)
pred.columns = ["Id", "Predicted"]
pred.to_csv(outputfile, index=False)
